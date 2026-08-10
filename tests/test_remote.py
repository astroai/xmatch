"""Remote backends exercised with mocks (no network access)."""

import sys
import types
from pathlib import Path

import polars as pl
import pytest

from xmatch import CrossMatch
from xmatch.matchers import MatchSpec
from xmatch.request import MatchRequest
from xmatch.sources import CatalogueSource


@pytest.fixture
def cm():
    return CrossMatch()


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """Pin the durable cache root to an empty temp dir.

    The pairwise cone-download path now serves from a mirrored HATS copy
    when one exists; without this fixture a mirror on the developer's
    machine (~/.cache/xmatch) would silently divert the mocked-download
    tests.
    """
    monkeypatch.setenv("XMATCH_CACHE_ROOT", str(tmp_path / "cache"))


def test_local_vs_remote_tap_downloads_then_matches(cm, monkeypatch):
    """A local frame vs a TAP catalogue: download is mocked, match runs locally."""
    import xmatch.remote_tap as rt

    def fake_download(src, **kwargs):
        # Returns rows near the local source so a match is found.
        return pl.DataFrame({"ra": [10.00005], "dec": [5.00005], "source_id": [999]})

    monkeypatch.setattr(rt, "download_from_tap", fake_download)

    local = pl.DataFrame({"ra": [10.0], "dec": [5.0], "my_id": [1]})
    out = cm.crossmatch(local, "gaia_esa", radius_arcsec=1.0)
    assert out.height == 1
    assert "source_id" in out.columns


def test_local_vs_remote_forwards_correct_side_columns(cm, monkeypatch):
    """columns_2 (remote is side 2) must reach the download, not columns_1."""
    import xmatch.remote_tap as rt

    captured = {}

    def fake_download(src, **kwargs):
        captured["columns"] = kwargs.get("columns")
        return pl.DataFrame({"ra": [10.00005], "dec": [5.00005], "source_id": [1]})

    monkeypatch.setattr(rt, "download_from_tap", fake_download)
    local = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    cm.crossmatch(local, "gaia_esa", radius_arcsec=1.0, columns_2=["ra", "dec", "source_id"])
    assert captured["columns"] == ["ra", "dec", "source_id"]


def test_target_epoch_forces_remote_space_motion_columns(cm, monkeypatch):
    import xmatch.remote_tap as rt

    captured = {}

    def fake_download(src, **kwargs):
        captured["columns"] = kwargs.get("columns")
        return pl.DataFrame()

    monkeypatch.setattr(rt, "download_from_tap", fake_download)
    src = cm.resolve_source("gaia_esa", {})
    req = MatchRequest(
        "gaia_esa",
        "gaia_esa",
        spec=MatchSpec(target_epoch=2025.0),
        ra=10.0,
        dec=5.0,
        radius_deg=0.01,
    )
    req.side1.columns = ["source_id"]

    cm._download_remote(src, req, prefix="1")

    assert set(captured["columns"]) == {
        "source_id",
        "ra",
        "dec",
        "pmra",
        "pmdec",
        "ref_epoch",
        "parallax",
        "radial_velocity",
    }


def test_target_epoch_skyellipse_forces_remote_astrometric_covariance(cm, monkeypatch):
    import xmatch.remote_tap as rt

    captured = {}

    def fake_download(src, **kwargs):
        captured["columns"] = kwargs.get("columns")
        return pl.DataFrame()

    monkeypatch.setattr(rt, "download_from_tap", fake_download)
    src = cm.resolve_source("gaia_esa", {})
    req = MatchRequest(
        "gaia_esa",
        "gaia_esa",
        spec=MatchSpec(target_epoch=2025.0, matcher="skyellipse"),
        ra=10.0,
        dec=5.0,
        radius_deg=0.01,
    )
    req.side1.columns = ["source_id"]

    cm._download_remote(src, req, prefix="1")

    assert src.astrometric_covariance_columns is not None
    assert set(src.astrometric_covariance_columns.values()).issubset(captured["columns"])
    assert {"ra", "dec", "pmra", "pmdec", "parallax", "radial_velocity"}.issubset(
        captured["columns"]
    )


def test_target_epoch_bypasses_unpropagated_tap_self_join(cm, monkeypatch):
    import xmatch.remote_tap as rt

    src1 = cm.resolve_source("gaia_esa", {})
    src2 = cm.resolve_source("gaia_esa", {})
    req = MatchRequest(
        "gaia_esa",
        "gaia_esa",
        spec=MatchSpec(target_epoch=2025.0),
        ra=10.0,
        dec=5.0,
        radius_deg=0.01,
    )
    downloads = []

    def fail_self_join(*args, **kwargs):
        raise AssertionError("target-epoch matching must not use server-side coordinates")

    def fake_download(src, req, **kwargs):
        downloads.append(src.name)
        return pl.DataFrame({"ra": [], "dec": []})

    def fake_local_match(*args, **kwargs):
        return pl.DataFrame({"ok": [True]}).lazy()

    monkeypatch.setattr(rt, "tap_self_join", fail_self_join)
    monkeypatch.setattr(cm, "_download_remote", fake_download)
    monkeypatch.setattr(cm, "_local_match", fake_local_match)

    result = cm._remote_vs_remote(src1, src2, req).collect()

    assert result["ok"].to_list() == [True]
    assert downloads == ["gaia_esa", "gaia_esa"]


def test_target_epoch_bypasses_unpropagated_cds_xmatch(cm, monkeypatch):
    import xmatch.remote_cds as rc

    local = cm.resolve_source(pl.DataFrame({"ra": [10.0], "dec": [5.0]}), {})
    remote = cm.resolve_source("gaia_cds", {})
    request = MatchRequest(
        pl.DataFrame({"ra": [10.0], "dec": [5.0]}),
        "gaia_cds",
        spec=MatchSpec(target_epoch=2025.0),
        ra=10.0,
        dec=5.0,
        radius_deg=0.01,
    )
    downloads = []

    def fail_cds_xmatch(*args, **kwargs):
        raise AssertionError("target-epoch matching must not use server-side coordinates")

    def fake_download(src, req, **kwargs):
        downloads.append(src.name)
        return pl.DataFrame({"RA_ICRS": [], "DE_ICRS": []})

    def fake_local_match(*args, **kwargs):
        return pl.DataFrame({"ok": [True]}).lazy()

    monkeypatch.setattr(rc, "cds_xmatch_local_remote", fail_cds_xmatch)
    monkeypatch.setattr(cm, "_download_remote", fake_download)
    monkeypatch.setattr(cm, "_local_match", fake_local_match)

    result = cm._local_vs_remote(local, remote, request).collect()

    assert result["ok"].to_list() == [True]
    assert downloads == ["gaia_cds"]


def test_download_remote_requires_region_when_no_local(cm):
    from xmatch.exceptions import CrossMatchError

    src = cm.resolve_source("gaia_esa", {})
    req = MatchRequest("gaia", "gaia")  # no ra/dec/radius set
    with pytest.raises(CrossMatchError):
        cm._download_remote(src, req, prefix="1")  # no ra/dec/radius and no local extent


def test_tap_self_join_builds_query_and_parses(monkeypatch):
    from astropy.table import Table

    import xmatch.remote_tap as rt

    captured = {}

    def fake_get_service(url, auth_session=None):
        return object()

    def fake_execute(service, query, maxrec=None):
        captured["query"] = query
        return Table({"a_ra": [10.0], "b_ra": [10.0], "best_sep_arcsec": [0.1]})

    monkeypatch.setattr(rt, "get_tap_service", fake_get_service)
    monkeypatch.setattr(rt, "execute_tap_query", fake_execute)

    s1 = CatalogueSource(
        name="c1",
        is_local=False,
        access_method="tap",
        ra_column="ra",
        dec_column="dec",
        tap_url="http://x/tap",
        access_identifier="t1",
        default_columns=["ra", "dec"],
    )
    s2 = CatalogueSource(
        name="c2",
        is_local=False,
        access_method="tap",
        ra_column="ra",
        dec_column="dec",
        tap_url="http://x/tap",
        access_identifier="t2",
        default_columns=["ra", "dec"],
    )
    out = rt.tap_self_join(s1, s2, MatchSpec(radius_arcsec=1.0))
    assert "CONTAINS" in captured["query"]
    assert out.height == 1


def test_cds_xmatch_local_remote_rejoins_on_surrogate_id(monkeypatch):
    """CDS XMatch result is joined back to the full local rows via a surrogate id."""
    from astropy.table import Table

    import xmatch.remote_cds as rc

    class FakeXMatch:
        def query(self, cat1, cat2, max_distance, colRA1, colDec1):
            # Echo the surrogate id for the first local row plus a remote column.
            key_col = rc._XMATCH_KEY
            return Table({key_col: [0], colRA1: [10.0], colDec1: [5.0], "remote_mag": [21.0]})

    fake_module = types.ModuleType("astroquery.xmatch")
    fake_module.XMatch = FakeXMatch
    monkeypatch.setitem(sys.modules, "astroquery.xmatch", fake_module)

    local_src = CatalogueSource(name="loc", is_local=True, ra_column="ra", dec_column="dec")
    remote_src = CatalogueSource(
        name="viz", is_local=False, access_method="cds_xmatch", access_identifier="I/355/gaiadr3"
    )
    local = pl.DataFrame({"ra": [10.0, 80.0], "dec": [5.0, 5.0], "my_id": [101, 102]})

    out = rc.cds_xmatch_local_remote(
        local_src, remote_src, local.lazy(), MatchSpec(radius_arcsec=2.0)
    )
    assert out.height == 1
    assert out["my_id"][0] == 101  # original local column preserved
    assert "remote_mag" in out.columns


def _write_fake_mirror(cm, tmp_path, rows):
    """Build a minimal mirrored HATS copy for ``gaia_cds`` under tmp_path cache."""
    from xmatch.mirror import _safe_name, _version_dir

    src = cm.resolve_source("gaia_cds", {})
    hats = Path(tmp_path / "cache") / f"{_safe_name(src.name)}/{_version_dir(src)}"
    part = hats / "dataset" / "Norder=0" / "Dir=2" / "Npix=0"
    part.mkdir(parents=True)
    rows.write_parquet(part / "Npix=0.parquet")
    (hats / "properties").write_text("hats_col_ra=RA_ICRS\nhats_col_dec=DE_ICRS\n")


def test_mirrored_hats_cone_serves_download_without_tap(cm, monkeypatch, tmp_path):
    """A mirrored HATS copy in the cache serves the pairwise cone download:
    no TAP call happens and rows come from the local partitions."""
    import xmatch.remote_tap as rt

    def _no_tap(*args, **kwargs):
        raise AssertionError("mirrored cone must not hit TAP")

    monkeypatch.setattr(rt, "download_from_tap", _no_tap)
    _write_fake_mirror(
        cm,
        tmp_path,
        pl.DataFrame(
            {
                "RA_ICRS": [10.0, 10.00015, 80.0],
                "DE_ICRS": [5.0, 5.00015, -30.0],
                "Source": [1, 999, 2],
            }
        ),
    )

    local = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    out = cm.crossmatch(local, "gaia_cds", radius_arcsec=1.5)

    assert out.height == 1
    assert "Source" in out.columns  # mirror columns served, not the TAP schema
    assert out["Source"][0] == 1  # nearest mirror row, far row filtered by the cone


def test_mirrored_cone_falls_back_to_tap_when_column_missing(cm, monkeypatch, tmp_path):
    """Requested columns the mirror does not hold fall back to the live TAP."""
    import xmatch.remote_tap as rt

    captured = {}

    def fake_download(src, **kwargs):
        captured["columns"] = kwargs.get("columns")
        return pl.DataFrame({"RA_ICRS": [10.00005], "DE_ICRS": [5.00005], "Source": [1]})

    monkeypatch.setattr(rt, "download_from_tap", fake_download)
    _write_fake_mirror(
        cm,
        tmp_path,
        pl.DataFrame({"RA_ICRS": [10.0], "DE_ICRS": [5.0], "Source": [1]}),
    )

    local = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    cm.crossmatch(local, "gaia_cds", radius_arcsec=1.0, columns_2=["ra", "dec", "source_id"])

    assert captured["columns"] == ["ra", "dec", "source_id"]
