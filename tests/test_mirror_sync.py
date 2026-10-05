"""Mirroring + incremental cache tests: TAP fake on 127.0.0.1, remote HATS over HTTP.

Covers the ``xmatch sync`` data plane behind :func:`xmatch.mirror.sync_catalogue`:

* cold TAP sync → local HATS catalogue (proven readable via
  :func:`xmatch.hats_native.list_hats_pixels`)
* incremental re-sync with zero page downloads (probe-only),
* :code:`--force` re-fetch,
* window-shrink re-sync (one refetched page),
* append-only continuation (new tail page),
* remote HATS mirror over ``http://127.0.0.1`` (``partition_info.parquet``
  listing + per-partition GETs + ``_metadata``/``properties`` HEADs).
"""

from __future__ import annotations

import functools
import threading
import urllib.error
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl
import pytest

from xmatch import hats_native, mirror
from xmatch.exceptions import CrossMatchError
from xmatch.sources import CatalogueSource

from .tap_fake import FakeTAPServer, make_rows

PAGE = 100


def test_mirrored_source_preserves_measurement_and_motion_metadata(tmp_path):
    from xmatch.storage import LocalStorage

    source = CatalogueSource(
        name="pilot",
        is_local=False,
        access_method="tap",
        tap_url="https://example.org/tap",
        release_namespace="pilot:dr1",
        release_metadata={"citation": "upstream"},
        photometry=[{"passband": "g", "value_column": "flux", "unit": "Jy"}],
        property_evidence=[
            {
                "property": "type.native",
                "value_column": "class",
                "category": "literature_assertion",
                "method": "native",
            }
        ],
        frame="galactic",
        ra_column="l",
        dec_column="b",
        ra_err_column="e_l",
        dec_err_column="e_b",
        pos_err_units="mas",
        default_pos_error_arcsec=0.1,
        epoch=2016.0,
        pm_ra_column="pm_l",
        pm_dec_column="pm_b",
    )
    cache = LocalStorage(tmp_path)
    mirrored = mirror._mirrored_source(source, cache, "pilot/version", str(tmp_path))
    assert mirrored.release_namespace == "pilot:dr1"
    assert mirrored.photometry == source.photometry
    assert mirrored.property_evidence == source.property_evidence
    assert mirrored.release_metadata == {"citation": "upstream"}
    assert mirrored.frame == "galactic" and mirrored.pos_err_units == "mas"
    assert mirrored.ra_err_column == "e_l" and mirrored.default_pos_error_arcsec == 0.1
    assert mirrored.path == tmp_path / "pilot/version"
    assert mirrored.access_method == "hats"
    assert source.access_method == "tap" and source.path is None


@pytest.fixture()
def tap_server() -> Iterator[FakeTAPServer]:
    srv = FakeTAPServer(make_rows(200))
    yield srv
    srv.shutdown()


def _tap_source(srv: FakeTAPServer, name: str = "probe") -> CatalogueSource:
    return CatalogueSource(
        name=name,
        is_local=False,
        access_method="tap",
        access_identifier="tap.probe",
        tap_url=srv.url,
        ra_column="ra",
        dec_column="dec",
        id_column="id",
        default_columns=["id", "ra", "dec"],
    )


def _mirror_rows(cache_root: str, src: CatalogueSource) -> pl.DataFrame:
    root, rel = mirror.locate_mirrored(src, cache_root=cache_root)
    pixels = hats_native.list_hats_pixels(Path(root) / rel)
    frames = [pl.read_parquet(p) for _, _, p in pixels]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _sync(src: CatalogueSource, cache_root: str, **kw) -> mirror.SyncStats:
    return mirror.sync_catalogue(
        src,
        cache_root=cache_root,
        page_size=100,
        estimated_size=300,
        rate_limit_rps=0.0,
        **kw,
    )


def test_tap_cold_then_incremental_zero_download(tmp_path: Path, tap_server) -> None:
    src = _tap_source(tap_server)
    cache = str(tmp_path / "cache")

    cold = _sync(src, cache)
    assert cold.pages == 2  # 200 rows / 100 per page
    assert cold.files_downloaded == 2
    assert cold.converted is True

    rows = _mirror_rows(cache, src)
    assert rows.height == 200
    assert set(rows.columns) >= {"id", "ra", "dec"}

    # incremental: no pages, no bytes, no rebuild
    snap_len = len(tap_server.queries)
    again = _sync(src, cache)
    assert again.pages == 0
    assert again.bytes_downloaded == 0
    assert again.converted is False
    # only key/count probes went out (no LIMIT page fetches)
    assert any('ORDER BY t."id" DESC' in q for q in tap_server.queries[snap_len:])
    assert tap_server.count_queries("COUNT(*)") >= 2


def test_tap_force_refetches(tmp_path: Path, tap_server) -> None:
    src = _tap_source(tap_server)
    cache = str(tmp_path / "cache")
    _sync(src, cache)
    forced = _sync(src, cache, force=True)
    # 200 rows @ page_size=100 -> two non-empty pages (the OFFSET 200 probe
    # returns an empty frame and is not counted)
    assert forced.pages == 2
    assert forced.files_downloaded == 2
    assert forced.converted is True


def test_tap_window_shrink_refetches_only_changed_page(tmp_path: Path, tap_server) -> None:
    src = _tap_source(tap_server)
    cache = str(tmp_path / "cache")
    _sync(src, cache)

    # shrink the table: [0..199] -> [0..179]; the stored [100,199] window probe
    # (COUNT(*)) returns 80 != 100, so exactly that window is refetched
    tap_server.update(make_rows(180))
    st = _sync(src, cache)
    assert st.pages == 1  # single refetch of the shrunk window
    assert st.files_downloaded == 1
    rows = mirror_rows(cache, src)
    assert rows.height == 180


def test_tap_append_adds_tail_page(tmp_path: Path, tap_server) -> None:
    src = _tap_source(tap_server)
    cache = str(tmp_path / "cache")
    _sync(src, cache)

    tap_server.update(make_rows(250))  # append-only growth
    st = _sync(src, cache)
    assert st.pages == 1  # keyset continuation past the stored last key
    assert st.files_downloaded == 1
    rows = mirror_rows(cache, src)
    assert rows.height == 250
    assert int(rows["id"].max()) == 249


def test_hats_over_http_mirror(tmp_path: Path) -> None:
    """A HATS catalogue served over plain HTTP mirrors file-for-file."""
    frame = pl.DataFrame(
        {
            "id": list(range(120)),
            "ra": [10.0 + (i % 40) * 0.001 for i in range(120)],
            "dec": [-5.0 + (i % 30) * 0.001 for i in range(120)],
        }
    )
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    mirror._write_hats_native(
        frame, serve_dir / "cat", ra_column="ra", dec_column="dec", threshold=50
    )
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        functools.partial(SimpleHTTPRequestHandler, directory=str(serve_dir)),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        src = CatalogueSource(
            name="remote-hats",
            is_local=False,
            access_method="hats",
            access_identifier=f"http://127.0.0.1:{httpd.server_address[1]}/cat",
            ra_column="ra",
            dec_column="dec",
        )
        cache = str(tmp_path / "cache")
        st = mirror.sync_catalogue(src, cache_root=cache, rate_limit_rps=0.0, force=False)
        assert st.failed == 0
        assert st.files_downloaded >= 2  # partition files + properties/metadata
        rows = mirror_rows(cache, src)
        assert rows.height == 120
        # second run: everything is skipped by the manifest
        st2 = mirror.sync_catalogue(src, cache_root=cache, rate_limit_rps=0.0, force=False)
        assert st2.files_skipped == st.files_downloaded
        assert st2.files_downloaded == 0
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_hats_over_vos_mirror(tmp_path: Path, monkeypatch) -> None:
    """A ``vos:`` HATS source mirrors node-for-node through the Storage path.

    Regression: ``do_fetch`` routed every source through the HTTP fetcher,
    which hard-raises for ``vos:`` identifiers — every file failed and the
    mirror stayed empty.  A fake ``vos:`` backend (monkeypatched
    ``open_storage``) exercises the real listing + fetch branches."""
    frame = pl.DataFrame(
        {
            "id": list(range(120)),
            "ra": [10.0 + (i % 40) * 0.001 for i in range(120)],
            "dec": [-5.0 + (i % 30) * 0.001 for i in range(120)],
        }
    )
    vos_root = tmp_path / "vos"
    mirror._write_hats_native(
        frame, vos_root / "cat", ra_column="ra", dec_column="dec", threshold=50
    )

    from xmatch.storage import LocalStorage

    class _VlsLikeStorage(LocalStorage):
        """LocalStorage with the real ``vls`` quirks _walk_storage must handle:

        direct-children basenames, no trailing slash on directories, and a
        leaf file echoing its own basename when listed.
        """

        def list(self, rel: str):
            p = self._path(rel)
            if not p.is_dir():
                return [p.name]  # vls on a leaf file echoes the file itself
            return sorted(e.name for e in p.iterdir())

    def fake_open_storage(root):
        if isinstance(root, str) and root.startswith("vos:"):
            return _VlsLikeStorage(str(vos_root / "cat"))
        return LocalStorage(root)

    monkeypatch.setattr(mirror, "open_storage", fake_open_storage)

    src = CatalogueSource(
        name="vos-hats",
        is_local=False,
        access_method="hats",
        access_identifier="vos:fake/cat",
        ra_column="ra",
        dec_column="dec",
    )
    cache = str(tmp_path / "cache")
    st = mirror.sync_catalogue(src, cache_root=cache, rate_limit_rps=0.0, force=False)
    assert st.failed == 0
    assert st.files_downloaded >= 2  # partition files + properties/metadata
    rows = mirror_rows(cache, src)
    assert rows.height == 120
    # second run: everything is skipped by the manifest
    st2 = mirror.sync_catalogue(src, cache_root=cache, rate_limit_rps=0.0, force=False)
    assert st2.files_skipped == st.files_downloaded
    assert st2.files_downloaded == 0


def test_ensure_mirrored_local_conversion(tmp_path: Path) -> None:
    """Plain local parquet inputs are converted once into the cache as HATS."""
    frame = list(make_rows(37, ra0=200.0, dec0=10.0))
    df = pl.DataFrame(frame)
    path = tmp_path / "local.parquet"
    df.write_parquet(path)
    src = CatalogueSource(
        name="local-cat",
        is_local=True,
        access_method="local",
        path=path,
        ra_column="ra",
        dec_column="dec",
        default_columns=["id", "ra", "dec"],
    )
    cache = str(tmp_path / "cache")
    mirrored = mirror.ensure_mirrored(src, cache_root=cache, hats_threshold=10)
    assert mirrored.access_method == "hats"
    assert mirrored.hats_cache_rel
    rows = mirror_rows(cache, mirrored)
    assert rows.height == 37
    root, rel = mirror.locate_mirrored(mirrored, cache_root=cache)
    n_parts_first = len(hats_native.list_hats_pixels(Path(root) / rel))
    # second call reuses the conversion: with hats_threshold=1 a rebuild would
    # produce far more (smaller) HEALPix partitions — pin the partition count.
    again = mirror.ensure_mirrored(src, cache_root=cache, hats_threshold=1)
    assert again.hats_cache_rel == mirrored.hats_cache_rel
    root2, rel2 = mirror.locate_mirrored(again, cache_root=cache)
    assert len(hats_native.list_hats_pixels(Path(root2) / rel2)) == n_parts_first


def mirror_rows(cache_root: str, src: CatalogueSource) -> pl.DataFrame:
    root, rel = mirror.locate_mirrored(src, cache_root=cache_root)
    pixels = hats_native.list_hats_pixels(Path(root) / rel)
    frames = [pl.read_parquet(p) for _, _, p in pixels]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _backdate_manifest(cache: str, name: str, days: float, raw: bool = True) -> None:
    """Rewind the manifest's fetched_at timestamp (simulates an older sync)."""
    import json
    import time as _t

    rel = Path(cache) / name / ("raw" if raw else "") / "sync.json"
    man = json.loads(rel.read_text())
    man["fetched_at"] = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(_t.time() - days * 86400))
    rel.write_text(json.dumps(man))


def test_fresh_after_skips_probe_and_network(tmp_path: Path, tap_server, monkeypatch) -> None:
    """A mirror synced within --fresh-after days: zero TAP queries, no probe."""
    src = _tap_source(tap_server)
    cache = str(tmp_path / "cache")
    assert _sync(src, cache).pages == 2  # cold sync builds the HATS mirror
    queries_before = len(tap_server.queries)
    _backdate_manifest(cache, "probe", days=1)

    def boom(*args, **kwargs):  # the probe path must not run at all
        raise AssertionError("_mirror_tap ran despite a fresh mirror")

    monkeypatch.setattr(mirror, "_mirror_tap", boom)
    stats = _sync(src, cache, fresh_after=30)
    assert stats.pages == 0
    assert stats.files_downloaded == 0
    assert len(tap_server.queries) == queries_before  # truly zero network


def test_fresh_after_still_probes_when_stale(tmp_path: Path, tap_server) -> None:
    """Older than --fresh-after: the incremental probe runs (COUNT(*), no refetch)."""
    src = _tap_source(tap_server)
    cache = str(tmp_path / "cache")
    assert _sync(src, cache).pages == 2
    _backdate_manifest(cache, "probe", days=40)
    queries_before = len(tap_server.queries)

    stats = _sync(src, cache, fresh_after=30)
    assert stats.pages == 0  # table unchanged: no page refetched
    assert len(tap_server.queries) > queries_before  # probe queries did run


def test_headroom_gate_blocks_sync_when_disk_full(tmp_path: Path, tap_server, monkeypatch) -> None:
    """--min-free-gb: fail fast before any fetch when the cache root is nearly full."""
    src = _tap_source(tap_server)
    cache = str(tmp_path / "cache")
    import collections

    DU = collections.namedtuple("DU", "total used free")
    monkeypatch.setattr(
        "xmatch.storage.shutil.disk_usage",
        lambda p: DU(1000, 900, 5.0),  # 5 bytes free
    )
    with pytest.raises(mirror.CrossMatchError, match=r"min-free-gb"):
        _sync(src, cache, min_free_gb=10)


def test_sync_falls_back_to_alternate_endpoint(tmp_path: Path, monkeypatch) -> None:
    """A dead primary TAP endpoint fails over to a same-schema fallback.

    Every query targeting the primary raises an endpoint error; the sync
    must come back served entirely by the fallback (distinct RA offsets
    prove the rows are B's), land in the primary's cache identity, and
    never reach the primary's HTTP layer.
    """
    srv_a = FakeTAPServer(make_rows(200))
    srv_b = FakeTAPServer(make_rows(200, ra0=30.0))
    try:
        src = _tap_source(srv_a)
        src.fallbacks = [_tap_source(srv_b, name="probe-b")]

        real_tap_run = mirror._tap_run

        def flaky(fetch, service, query, **kw):
            if fetch.tap_url == srv_a.url:
                raise urllib.error.URLError("primary endpoint down")
            return real_tap_run(fetch, service, query, **kw)

        monkeypatch.setattr(mirror, "_tap_run", flaky)
        cache = str(tmp_path / "cache")
        stats = _sync(src, cache)
        assert stats.pages == 2
        assert stats.failed == 0

        rows = mirror_rows(cache, src)
        assert rows.height == 200
        assert rows["ra"].min() >= 30.0  # rows came from B
        assert srv_a.queries == []  # primary never served anything
    finally:
        srv_a.shutdown()
        srv_b.shutdown()


def test_sync_hats_partial_transfer_fails_over(tmp_path: Path, monkeypatch) -> None:
    """A HATS endpoint that died mid-transfer is retried on the fallback.

    Per-file fetch errors are swallowed into ``stats.failed`` (incremental
    mirroring), so the zero-transfer check alone would keep the broken
    partial mirror; the candidate must be judged by its failure delta.
    """
    calls: list = []

    def flaky(src, cache, *, bucket, force, workers, progress_cb, stats, endpoint):
        calls.append(endpoint.access_identifier)
        if len(calls) == 1:
            stats.files_downloaded += 3
            stats.failed += 2  # primary died partway through the transfer
        else:
            stats.files_skipped += 5  # fallback completes the mirror

    src = CatalogueSource(
        name="probe-hats",
        is_local=False,
        access_method="hats",
        access_identifier="https://a.example.org/cat",
        default_columns=["id", "ra", "dec"],
        fallbacks=["https://b.example.org/cat"],
    )
    monkeypatch.setattr(mirror, "_mirror_remote_hats", flaky)
    stats = _sync(src, str(tmp_path / "cache"))
    assert calls == [
        "https://a.example.org/cat",
        "https://b.example.org/cat",
    ]
    assert stats.files_downloaded == 3
    assert stats.files_skipped == 5
    assert stats.failed == 2  # partial-failure facts preserved for the caller


def test_sync_fallback_column_mismatch_refuses(tmp_path: Path, tap_server) -> None:
    """A fallback with different default_columns is refused before any fetch."""
    srv_b = FakeTAPServer(make_rows(50))
    try:
        src = _tap_source(tap_server)
        fb = _tap_source(srv_b, name="probe-b")
        fb.default_columns = ["id", "ra"]  # differs from the primary's
        src.fallbacks = [fb]

        with pytest.raises(CrossMatchError, match="different columns"):
            _sync(src, str(tmp_path / "cache"))
        # the gate fires before any candidate runs: neither server is hit
        assert srv_b.queries == []
        assert len(tap_server.queries) == 0
    finally:
        srv_b.shutdown()


def test_locate_mirrored_probes_roots_in_order(tmp_path: Path, tap_server) -> None:
    """locate_mirrored(cache_roots=[...]) returns the first surviving copy."""
    r1 = str(tmp_path / "root1")
    r2 = str(tmp_path / "root2")
    src = _tap_source(tap_server)
    _sync(src, r2)  # mirrored into root2 only

    root, rel = mirror.locate_mirrored(src, cache_roots=[r1, r2])
    assert root == r2
    assert rel == f"probe/{mirror._version_dir(src)}"

    with pytest.raises(CrossMatchError) as ei:
        mirror.locate_mirrored(src, cache_roots=[r1, str(tmp_path / "root3")])
    msg = str(ei.value)
    assert "not mirrored yet" in msg
    assert r1 in msg and str(tmp_path / "root3") in msg  # lists every probed root


def test_write_hats_native_empty_frame_produces_readable_catalogue(tmp_path: Path) -> None:
    """An empty input (a region/query with no rows) must still mirror as a
    valid, readable HATS catalogue.

    Regression: ``pl.DataFrame([])`` has no columns, so the
    ``partition_info`` ``.select([...])`` raised ``ColumnNotFoundError`` — a
    local input file with zero rows crashed ``--engine ray-union`` with an
    opaque polars error instead of writing an empty catalogue.
    """
    import hats

    empty = pl.DataFrame(schema={"ra": pl.Float64, "dec": pl.Float64, "id": pl.Utf8})
    out = tmp_path / "empty"
    mirror._write_hats_native(empty, out, ra_column="ra", dec_column="dec", threshold=10)

    info = pl.read_csv(out / "partition_info.csv")
    assert info.height == 0
    assert info.columns == [
        "Norder",
        "Dir",
        "Npix",
        "Nfiles",
        "file_loc",
        "file_size",
        "count",
    ]
    props = (out / "properties").read_text()
    assert "hats_nrows=0" in props
    read = hats.read_hats(out)
    assert read.catalog_info.total_rows == 0
    assert read.get_healpix_pixels() == []
