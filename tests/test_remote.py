"""Remote backends exercised with mocks (no network access)."""

import sys
import types

import polars as pl
import pytest

from xmatch import CrossMatch
from xmatch.matchers import MatchSpec
from xmatch.request import MatchRequest
from xmatch.sources import CatalogueSource


@pytest.fixture
def cm():
    return CrossMatch()


def test_local_vs_remote_tap_downloads_then_matches(cm, monkeypatch):
    """A local frame vs a TAP catalogue: download is mocked, match runs locally."""
    import xmatch.remote_tap as rt

    def fake_download(src, **kwargs):
        # Returns rows near the local source so a match is found.
        return pl.DataFrame({"ra": [10.00005], "dec": [5.00005], "source_id": [999]})

    monkeypatch.setattr(rt, "download_from_tap", fake_download)

    local = pl.DataFrame({"ra": [10.0], "dec": [5.0], "my_id": [1]})
    out = cm.crossmatch(local, "gaia", radius_arcsec=1.0)
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
    cm.crossmatch(local, "gaia", radius_arcsec=1.0, columns_2=["ra", "dec", "source_id"])
    assert captured["columns"] == ["ra", "dec", "source_id"]


def test_download_remote_requires_region_when_no_local(cm):
    from xmatch.exceptions import CrossMatchError

    src = cm.resolve_source("gaia", {})
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
