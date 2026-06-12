import numpy as np
import polars as pl
import pytest

from xmatch import stilts
from xmatch.matchers import MatchSpec, id_join, sky_match
from xmatch.sources import CatalogueSource


def _src(name, ra="ra", dec="dec", **kw):
    return CatalogueSource(name=name, is_local=True, ra_column=ra, dec_column=dec, **kw)


# Engines to test: always astropy; add STILTS when a command is discoverable.
_ENGINES = ["astropy"]
if stilts.stilts_available():
    _ENGINES.append("stilts")


def test_sky_match_best_within_radius():
    left = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 6.0]})
    right = pl.DataFrame({"ra": [10.00005, 50.0], "dec": [5.00005, 5.0]})
    spec = MatchSpec(radius_arcsec=1.0)
    out = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="astropy"
    ).collect()
    assert out.height == 1
    assert out["sep_arcsec"][0] < 1.0


def test_sky_match_radius_is_arcsec_not_degrees():
    # 0.5 deg apart must NOT match at radius 1 arcsec (regression for the 3600x bug).
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.5], "dec": [5.0]})
    spec = MatchSpec(radius_arcsec=1.0)
    out = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="astropy"
    ).collect()
    assert out.height == 0


def test_sky_match_find_all():
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005, 10.0001, 50.0], "dec": [5.00005, 5.0, 5.0]})
    spec = MatchSpec(radius_arcsec=1.0, find="all")
    out = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="astropy"
    ).collect()
    assert out.height == 2


def test_sky_match_outer_join_keeps_unmatched():
    left = pl.DataFrame({"ra": [10.0, 80.0], "dec": [5.0, 5.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005]})
    spec = MatchSpec(radius_arcsec=1.0, join_type="all1")
    out = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="astropy"
    ).collect()
    assert out.height == 2  # one match + one unmatched-left


def test_skyerr_sigma_criterion():
    # 0.5 arcsec apart; with tiny errors and 3-sigma it should be rejected.
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "rae": [0.01], "dee": [0.01]})
    right = pl.DataFrame({"ra": [10.0001388], "dec": [5.0], "rae": [0.01], "dee": [0.01]})
    a = _src("a", ra_err_column="rae", dec_err_column="dee", pos_err_units="arcsec")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", pos_err_units="arcsec")
    spec = MatchSpec(radius_arcsec=2.0, matcher="skyerr", max_error=3.0)
    out = sky_match(a, b, left.lazy(), right.lazy(), spec, engine="astropy").collect()
    assert out.height == 0
    # With a large sigma allowance it matches.
    spec_big = MatchSpec(radius_arcsec=2.0, matcher="skyerr", max_error=100.0)
    out2 = sky_match(a, b, left.lazy(), right.lazy(), spec_big, engine="astropy").collect()
    assert out2.height == 1


@pytest.mark.skipif("stilts" not in _ENGINES, reason="STILTS not available")
def test_sky_engine_parity():
    """STILTS and astropy must agree on matches and sep_arcsec for a sky match."""
    a = pl.DataFrame({"id": [1, 2, 3], "ra": [10.0, 20.0, 30.0], "dec": [5.0, 6.0, 7.0]})
    b = pl.DataFrame(
        {"bid": [1, 2, 3], "ra": [10.00005, 20.5, 30.00002], "dec": [5.00005, 6.5, 7.00001]}
    )
    spec = MatchSpec(radius_arcsec=1.0)
    out = {
        eng: sky_match(_src("a"), _src("b"), a.lazy(), b.lazy(), spec, engine=eng)
        .collect()
        .sort("id")
        for eng in ("astropy", "stilts")
    }
    assert out["astropy"]["id"].to_list() == out["stilts"]["id"].to_list()
    assert np.allclose(
        out["astropy"]["sep_arcsec"].to_numpy(), out["stilts"]["sep_arcsec"].to_numpy(), atol=1e-6
    )


@pytest.mark.skipif("stilts" not in _ENGINES, reason="STILTS not available")
def test_skyerr_engine_parity():
    cos = np.cos(np.radians(5.0))
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "rae": [0.1], "dee": [0.1]})
    right = pl.DataFrame(
        {"ra": [10.0 + 0.5 / 3600 / cos], "dec": [5.0], "rae": [0.1], "dee": [0.1]}
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee")
    b = _src("b", ra_err_column="rae", dec_err_column="dee")
    # e=hypot(.1,.1)=.1414, combined=.2828; matches iff 0.5 <= max_error*0.2828
    for me, expected in [(1.0, 0), (2.0, 1)]:
        rows = {
            eng: sky_match(
                a,
                b,
                left.lazy(),
                right.lazy(),
                MatchSpec(matcher="skyerr", max_error=me),
                engine=eng,
            )
            .collect()
            .height
            for eng in ("astropy", "stilts")
        }
        assert rows["astropy"] == rows["stilts"] == expected, (me, rows)


def test_id_join_inner_and_anti():
    a = pl.DataFrame({"oid": [1, 2, 3], "g": [1.0, 2.0, 3.0]})
    b = pl.DataFrame({"oid": [2, 3, 4], "r": [9.0, 8.0, 7.0]})
    assert id_join(a.lazy(), b.lazy(), "oid", "oid", "1and2").collect().height == 2
    assert id_join(a.lazy(), b.lazy(), "oid", "oid", "1not2").collect().height == 1
    assert id_join(a.lazy(), b.lazy(), "oid", "oid", "2not1").collect().height == 1
    assert id_join(a.lazy(), b.lazy(), "oid", "oid", "1or2").collect().height == 4
