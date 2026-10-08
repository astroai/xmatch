"""CLI wiring tests for ``xmatcher sync`` and the ray-union match route.

* ``xmatcher sync NAME --config cfg.yaml --cache-root dir`` mirrors a TAP
  catalogue through the real CLI; a second run is a zero-download
  incremental (the fake server records every query).
* ``xmatcher match a.hats b.hats --engine ray-union -o out.hats`` runs the
  distributed pipeline end to end from the CLI and yields a hats-readable
  joined catalogue.
* Unknown catalogue handles in ``sync`` exit 1 with an error message.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from xmatcher import hats_native, mirror
from xmatcher.cli import main
from xmatcher.sources import CatalogueSource

from .tap_fake import FakeTAPServer, make_rows


@pytest.fixture()
def tap_server() -> Iterator[FakeTAPServer]:
    srv = FakeTAPServer(make_rows(200))
    yield srv
    srv.shutdown()


def _cat_yaml(tap_url: str) -> str:
    return f"""\
archives:
  probe_archive:
    tap_service:
      access_url: "{tap_url}"
      access_method: tap
      column_selection_supported: true
    service_priority: [tap_service]
catalogues:
  probe:
    archive: probe_archive
    service_id: tap_service
    access_identifier: tap.probe
    table_name: tap.probe
    ra_column: ra
    dec_column: dec
    id_column: id
    default_columns: [id, ra, dec]
"""


def _config_path(tmp_path: Path, tap_url: str) -> Path:
    cfg = tmp_path / "xmatcher.yaml"
    cfg.write_text(_cat_yaml(tap_url))
    return cfg


def _rows(cache: Path, tap_url: str) -> pl.DataFrame:
    src = CatalogueSource(
        name="probe",
        is_local=False,
        access_method="tap",
        access_identifier="tap.probe",
        tap_url=tap_url,
        ra_column="ra",
        dec_column="dec",
        id_column="id",
    )
    root, rel = mirror.locate_mirrored(src, cache_root=str(cache))
    pixels = hats_native.list_hats_pixels(Path(root) / rel)
    frames = [pl.read_parquet(p) for _, _, p in pixels]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _mirrored_props(cache: Path, tap_url: str) -> Path:
    src = CatalogueSource(
        name="probe",
        is_local=False,
        access_method="tap",
        access_identifier="tap.probe",
        tap_url=tap_url,
        ra_column="ra",
        dec_column="dec",
        id_column="id",
    )
    root, rel = mirror.locate_mirrored(src, cache_root=str(cache))
    return Path(root) / rel


def test_cli_sync_tap_cold_then_incremental_zero_download(tmp_path, tap_server) -> None:
    cfg = _config_path(tmp_path, tap_server.url)
    cache = tmp_path / "cache"

    rc = main(["sync", "probe", "--config", str(cfg), "--cache-root", str(cache)])
    assert rc == 0
    rows = _rows(cache, tap_server.url)
    assert rows.height == 200
    cold_queries = list(tap_server.queries)
    mirrored = _mirrored_props(cache, tap_server.url)
    part_files = list((mirrored / "dataset").rglob("Npix=*.parquet"))
    mtimes = {p.name: p.stat().st_mtime_ns for p in part_files}

    rc = main(["sync", "probe", "--config", str(cfg), "--cache-root", str(cache)])
    assert rc == 0
    # incremental: probe-only — no rewritten partitions, unchanged rows
    assert tap_server.queries[len(cold_queries) :], "incremental run should have probed"
    for p in (mirrored / "dataset").rglob("Npix=*.parquet"):
        assert p.stat().st_mtime_ns == mtimes[p.name]
    assert _rows(cache, tap_server.url).height == 200


def test_cli_sync_unknown_catalogue_returns_one(tmp_path, tap_server, capsys) -> None:
    cfg = _config_path(tmp_path, tap_server.url)
    cache = tmp_path / "cache"
    rc = main(["sync", "not_a_catalogue", "--config", str(cfg), "--cache-root", str(cache)])
    assert rc == 1
    assert "not_a_catalogue" in capsys.readouterr().err


def _hats_dir(path: Path, name: str, df: pl.DataFrame) -> Path:
    root = path / name
    (root / "dataset" / "Norder=0" / "Dir=0").mkdir(parents=True)
    df.write_parquet(root / "dataset" / "Norder=0" / "Dir=0" / "Npix=0.parquet")
    (root / "properties").write_text(
        "dataproduct_type=object\nobs_collection=xmatcher-test\n"
        "hats_col_ra=ra\nhats_col_dec=dec\nhats_ordering=NESTED\nhats_nrows=%d\n" % df.height
    )
    return root


def test_cli_match_ray_union_from_local_hats(tmp_path) -> None:
    a = _hats_dir(
        tmp_path,
        "a",
        pl.DataFrame(
            {
                "id": [f"A{i}" for i in range(20)],
                "ra": [10.0 + i * 0.001 for i in range(20)],
                "dec": [-5.0 + i * 0.001 for i in range(20)],
            }
        ),
    )
    b = _hats_dir(
        tmp_path,
        "b",
        pl.DataFrame(
            {
                "id": [f"B{i}" for i in range(12)],
                "ra": [10.0 + i * 0.001 for i in range(12)],
                "dec": [-5.0 + i * 0.001 for i in range(12)],
            }
        ),
    )
    out = tmp_path / "out.hats"
    rc = main(
        [
            "match",
            str(a),
            str(b),
            "--engine",
            "ray-union",
            "-r",
            "5.0",
            "-o",
            str(out),
            "--hats-threshold",
            "1000000",
        ]
    )
    assert rc == 0
    pixels = hats_native.list_hats_pixels(out)
    total = sum(pl.read_parquet(p).height for _, _, p in pixels)
    # B rows are exact twins of A rows: 8 unmatched '1' singles + 12 '1+2'
    # pairs — the union oracle never emits a singleton for a matched row.
    assert total == 20

    import hats  # noqa: PLC0415

    cat = hats.read_hats(out)  # valid tree, no overlap
    assert len(cat.get_healpix_pixels()) == len(pixels)
    df = pl.concat(
        [pl.read_parquet(p) for _, _, p in hats_native.list_hats_pixels(out)],
        how="diagonal_relaxed",
    )
    assert sorted(df["_src_cats"].unique().to_list()) == ["1", "1+2"]
    assert (df["_src_cats"] == "1").sum() == 8


def test_cli_resolves_bare_output_under_platform_root(monkeypatch) -> None:
    """On the platform a bare -o name lands under the output root; explicit
    paths and off-platform runs are untouched (XMATCHER_OUTPUT_ROOT override
    rides through default_output_root)."""
    from xmatcher.cli import _resolve_output_path

    monkeypatch.setattr("xmatcher.cli.default_output_root", lambda: "/arc/projects/hats/xmatcher")
    assert _resolve_output_path("full.hats") == "/arc/projects/hats/xmatcher/full.hats"
    assert _resolve_output_path("out.parquet") == "/arc/projects/hats/xmatcher/out.parquet"
    assert _resolve_output_path("/abs/full.hats") == "/abs/full.hats"
    assert _resolve_output_path("sub/full.hats") == "sub/full.hats"
    assert _resolve_output_path(None) is None

    monkeypatch.setattr("xmatcher.cli.default_output_root", lambda: None)
    assert _resolve_output_path("full.hats") == "full.hats"
