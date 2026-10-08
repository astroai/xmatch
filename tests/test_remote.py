"""Remote backends exercised with mocks (no network access)."""

import re
import sys
import types
from pathlib import Path

import astropy.units as u
import numpy as np
import polars as pl
import pytest
from astropy.coordinates import SkyCoord

from xmatcher import CrossMatch
from xmatcher.exceptions import CrossMatchError
from xmatcher.matchers import MatchSpec
from xmatcher.request import MatchRequest
from xmatcher.sources import CatalogueSource


@pytest.fixture
def cm():
    return CrossMatch()


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """Pin the durable cache root to an empty temp dir.

    The pairwise cone-download path now serves from a mirrored HATS copy
    when one exists; without this fixture a mirror on the developer's
    machine (~/.cache/xmatcher) would silently divert the mocked-download
    tests.
    """
    monkeypatch.setenv("XMATCHER_CACHE_ROOT", str(tmp_path / "cache"))


def test_local_vs_remote_tap_downloads_then_matches(cm, monkeypatch):
    """A local frame vs a TAP catalogue: download is mocked, match runs locally."""
    import xmatcher.remote_tap as rt

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
    import xmatcher.remote_tap as rt

    captured = {}

    def fake_download(src, **kwargs):
        captured["columns"] = kwargs.get("columns")
        return pl.DataFrame({"ra": [10.00005], "dec": [5.00005], "source_id": [1]})

    monkeypatch.setattr(rt, "download_from_tap", fake_download)
    local = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    cm.crossmatch(local, "gaia_esa", radius_arcsec=1.0, columns_2=["ra", "dec", "source_id"])
    assert captured["columns"] == ["ra", "dec", "source_id"]


def test_target_epoch_forces_remote_space_motion_columns(cm, monkeypatch):
    import xmatcher.remote_tap as rt

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
    import xmatcher.remote_tap as rt

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
    import xmatcher.remote_tap as rt

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
    import xmatcher.remote_cds as rc

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
    from xmatcher.exceptions import CrossMatchError

    src = cm.resolve_source("gaia_esa", {})
    req = MatchRequest("gaia", "gaia")  # no ra/dec/radius set
    with pytest.raises(CrossMatchError):
        cm._download_remote(src, req, prefix="1")  # no ra/dec/radius and no local extent


def test_datalab_cone_box_is_a_conservative_spherical_bound():
    from xmatcher.remote_tap import _cone_predicate

    # Independent spherical geometry check: this source is inside the 1 deg
    # cone despite being 180 deg away in RA because the cone reaches the pole.
    centre = SkyCoord(0.0 * u.deg, 89.9 * u.deg, frame="icrs")
    near_pole = SkyCoord(180.0 * u.deg, 89.5 * u.deg, frame="icrs")
    assert centre.separation(near_pole).deg < 1.0
    pole_query = _cone_predicate('"ra"', '"dec"', 0.0, 89.9, 1.0, box=True)
    assert 't."ra"' not in pole_query
    assert 't."dec" BETWEEN 88.9 AND 90.0' in pole_query

    # Sample the boundary with Astropy, independently checking the exact
    # tangent-meridian longitude bound for a cap that does not reach a pole.
    centre = SkyCoord(10.0 * u.deg, 60.0 * u.deg, frame="icrs")
    bearings = np.linspace(0.0, 360.0, 3601) * u.deg
    boundary = centre.directional_offset_by(bearings, 20.0 * u.deg)
    offsets = (boundary.ra.deg - centre.ra.deg + 180.0) % 360.0 - 180.0
    max_offset = float(np.max(np.abs(offsets)))
    tangent_bound = np.degrees(np.arcsin(np.sin(np.radians(20.0)) / np.cos(np.radians(60.0))))
    assert max_offset <= tangent_bound + 1e-6
    box_query = _cone_predicate('"ra"', '"dec"', 10.0, 60.0, 20.0, box=True)
    limits = re.search(r'>= ([\d.]+) OR t\."ra" <= ([\d.]+)', box_query)
    assert limits is not None
    ra_lo, ra_hi = map(float, limits.groups())
    assert np.all((boundary.ra.deg >= ra_lo) | (boundary.ra.deg <= ra_hi))

    wrap_query = _cone_predicate('"ra"', '"dec"', 0.1, 0.0, 1.0, box=True)
    assert 't."ra" >= 359.1 OR t."ra" <= 1.1' in wrap_query


@pytest.mark.parametrize(
    "region",
    [
        {"ra": 10.0},
        {"ra": "invalid", "dec": 0.0, "radius_deg": 1.0},
        {"ra": float("nan"), "dec": 0.0, "radius_deg": 1.0},
        {"ra": 10.0, "dec": 91.0, "radius_deg": 1.0},
        {"ra": 10.0, "dec": 0.0, "radius_deg": -1.0},
    ],
)
def test_tap_download_rejects_partial_or_invalid_cone_before_network(monkeypatch, region):
    import xmatcher.remote_tap as rt

    def no_network(*args, **kwargs):
        raise AssertionError("invalid cone input reached the TAP service")

    monkeypatch.setattr(rt, "get_tap_service", no_network)
    src = CatalogueSource(
        name="test",
        is_local=False,
        tap_url="https://example.invalid/tap",
        access_identifier="schema.table",
        ra_column="ra",
        dec_column="dec",
    )
    with pytest.raises(CrossMatchError, match="TAP cone search"):
        rt.download_from_tap(src, **region)


@pytest.mark.parametrize(
    "region",
    [
        {"ra": None, "dec": 0.0, "radius_arcsec": 1.0},
        {"ra": "invalid", "dec": 0.0, "radius_arcsec": 1.0},
        {"ra": 10.0, "dec": 91.0, "radius_arcsec": 1.0},
        {"ra": 10.0, "dec": 0.0, "radius_arcsec": np.inf},
        {"ra": 10.0, "dec": 0.0, "radius_arcsec": 648_001.0},
    ],
)
def test_cds_download_rejects_invalid_region_before_network(region):
    import xmatcher.remote_cds as rc

    src = CatalogueSource(
        name="viz",
        is_local=False,
        access_identifier="I/355/gaiadr3",
        access_method="cds",
    )
    with pytest.raises(CrossMatchError, match="CDS|spatial region"):
        rc.download_from_cds(src, **region)


def test_cds_download_normalizes_ra_before_query(monkeypatch):
    from astropy.table import Table

    import xmatcher.remote_cds as rc

    captured = {}

    class FakeVizier:
        def __init__(self, **kwargs):
            pass

        def query_region(self, center, *, radius, catalog):
            captured["ra"] = center.ra.deg
            captured["radius"] = radius.to_value(u.arcsec)
            captured["catalog"] = catalog
            return [Table({"source_id": [1]})]

    fake_module = types.ModuleType("astroquery.vizier")
    fake_module.Vizier = FakeVizier
    monkeypatch.setitem(sys.modules, "astroquery.vizier", fake_module)
    src = CatalogueSource(
        name="viz",
        is_local=False,
        access_identifier="I/355/gaiadr3",
        access_method="cds",
    )

    out = rc.download_from_cds(src, ra=370.0, dec=0.0, radius_arcsec=1.0)

    assert captured == {"ra": 10.0, "radius": 1.0, "catalog": "I/355/gaiadr3"}
    assert out["source_id"].to_list() == [1]


def test_tap_self_join_builds_query_and_parses(monkeypatch):
    from astropy.table import Table

    import xmatcher.remote_tap as rt

    captured = {}

    def fake_get_service(url, auth_session=None):
        return object()

    def fake_execute(service, query, maxrec=None):
        captured["query"] = query
        return Table({"a_ra": [10.0], "b_ra": [10.0], "all_sep_arcsec": [0.1]})

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
    out = rt.tap_self_join(s1, s2, MatchSpec(radius_arcsec=1.0, find="all"))
    assert "CONTAINS" in captured["query"]
    assert "all_sep_arcsec" in captured["query"]
    assert out.height == 1


def test_cds_xmatch_local_remote_rejoins_on_surrogate_id(monkeypatch):
    """CDS XMatch result is joined back to the full local rows via a surrogate id."""
    from astropy.table import Table

    import xmatcher.remote_cds as rc

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
        local_src, remote_src, local.lazy(), MatchSpec(radius_arcsec=2.0, find="all")
    )
    assert out.height == 1
    assert out["my_id"][0] == 101  # original local column preserved
    assert "remote_mag" in out.columns


@pytest.mark.parametrize(
    "spec",
    [
        MatchSpec(radius_arcsec=1.0),
        MatchSpec(radius_arcsec=1.0, find="all", matcher="skyerr"),
        MatchSpec(radius_arcsec=1.0, find="all", join_type="1or2"),
        MatchSpec(radius_arcsec=1.0, find="all", prior_columns=["mag"]),
        MatchSpec(radius_arcsec=1.0, find="all", probabilistic=True),
    ],
)
def test_tap_self_join_rejects_unsupported_semantics_before_network(monkeypatch, spec):
    import xmatcher.remote_tap as rt

    def no_network(*args, **kwargs):
        raise AssertionError("unsupported TAP self-join reached the service")

    monkeypatch.setattr(rt, "get_tap_service", no_network)
    src = CatalogueSource(
        name="tap",
        is_local=False,
        tap_url="https://example.invalid/tap",
        access_identifier="schema.table",
        ra_column="ra",
        dec_column="dec",
    )
    with pytest.raises(CrossMatchError, match="tap_self_join cannot honor"):
        rt.tap_self_join(src, src, spec)


def test_cds_xmatch_rejects_unsupported_semantics_before_importing_optional_client():
    import xmatcher.remote_cds as rc

    local_src = CatalogueSource(name="local", is_local=True, ra_column="ra", dec_column="dec")
    remote_src = CatalogueSource(name="viz", is_local=False, access_identifier="I/355/gaiadr3")
    with pytest.raises(CrossMatchError, match="CDS XMatch cannot honor"):
        rc.cds_xmatch_local_remote(
            local_src,
            remote_src,
            pl.DataFrame({"ra": [1.0], "dec": [0.0]}).lazy(),
            MatchSpec(radius_arcsec=1.0),
        )


def test_tap_without_projection_selects_all_columns():
    from xmatcher.remote_tap import _select_columns

    src = CatalogueSource(
        name="tap",
        is_local=False,
        tap_url="https://example.invalid/tap",
        access_identifier="schema.table",
        ra_column="ra",
        dec_column="dec",
    )
    assert _select_columns(src, None) == "*"


def _write_fake_mirror(cm, tmp_path, rows):
    """Build a minimal mirrored HATS copy for ``gaia_cds`` under tmp_path cache."""
    from xmatcher.mirror import _safe_name, _version_dir

    src = cm.resolve_source("gaia_cds", {})
    hats = Path(tmp_path / "cache") / f"{_safe_name(src.name)}/{_version_dir(src)}"
    part = hats / "dataset" / "Norder=0" / "Dir=2" / "Npix=0"
    part.mkdir(parents=True)
    rows.write_parquet(part / "Npix=0.parquet")
    (hats / "properties").write_text("hats_col_ra=RA_ICRS\nhats_col_dec=DE_ICRS\n")


def test_mirrored_hats_cone_serves_download_without_tap(cm, monkeypatch, tmp_path):
    """A mirrored HATS copy in the cache serves the pairwise cone download:
    no TAP call happens and rows come from the local partitions."""
    import xmatcher.remote_tap as rt

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
                "pmRA": [0.0, 0.0, 0.0],
                "pmDE": [0.0, 0.0, 0.0],
                "Plx": [0.0, 0.0, 0.0],
                "RV": [0.0, 0.0, 0.0],
                "Gmag": [18.0, 19.0, 20.0],
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
    import xmatcher.remote_tap as rt

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

    assert {"ra", "dec", "source_id", "RA_ICRS", "DE_ICRS"}.issubset(captured["columns"])
