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


# ------------------------------------------------------------------- tier 1
def test_sky_engine_fast_matches_astropy_within_tolerance():
    """Tier 1 (scipy cKDTree) must agree with astropy within 1e-6 arcsec."""
    left = pl.DataFrame({"ra": [10.0, 20.0, 30.0], "dec": [5.0, 6.0, 7.0]})
    right = pl.DataFrame({"ra": [10.00005, 20.5, 30.00002], "dec": [5.00005, 6.5, 7.00001]})
    spec = MatchSpec(radius_arcsec=1.0)
    ast = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="astropy"
    ).collect()
    fast = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert ast.height == fast.height == 2
    sep_ast = np.sort(ast["sep_arcsec"].to_numpy())
    sep_fast = np.sort(fast["sep_arcsec"].to_numpy())
    assert np.allclose(sep_ast, sep_fast, atol=1e-6)


def test_sky_engine_fast_find_all():
    """Tier 1 with find='all' must yield every within-radius pair."""
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005, 10.0001, 50.0], "dec": [5.00005, 5.0, 5.0]})
    spec = MatchSpec(radius_arcsec=1.0, find="all")
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 2


def test_sky_engine_zone_matches_fast_within_tolerance():
    """Tier 2 falls back to Tier 1 when cdshealpix is unavailable. Parity
    must hold in either code path so this test exercises the dispatch."""
    left = pl.DataFrame({"ra": [10.0, 20.0, 30.0], "dec": [5.0, 6.0, 7.0]})
    right = pl.DataFrame({"ra": [10.00005, 20.5, 30.00002], "dec": [5.00005, 6.5, 7.00001]})
    spec = MatchSpec(radius_arcsec=1.0)
    fast = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    zone = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="zone").collect()
    assert fast.height == zone.height == 2
    sep_fast = np.sort(fast["sep_arcsec"].to_numpy())
    sep_zone = np.sort(zone["sep_arcsec"].to_numpy())
    assert np.allclose(sep_fast, sep_zone, atol=1e-6)


# ------------------------------------------------------------------- tier 3
def test_probabilistic_pmatch_in_unit_range():
    """Tier 3: p_match must always be in [0, 1]; column present when prior_columns set."""
    left = pl.DataFrame(
        {
            "ra": [10.0, 10.00005, 10.5],
            "dec": [5.0, 5.0, 5.0],
            "mag": [10.0, 10.5, 14.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.0, 10.0, 10.0],
            "dec": [5.0, 5.0, 5.0],
            "mag": [10.0, 12.0, 15.0],
        }
    )
    spec = MatchSpec(radius_arcsec=60.0, prior_columns=["mag"])
    # Fixture: left row #2 is 0.5 deg from the right cluster, far outside
    # radius=60 arcsec, so it must NOT match. Only the first two rows fit.
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 2
    assert "p_match" in out.columns
    p = out["p_match"].to_numpy()
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_probabilistic_pmatch_pos_v_sep_distinguishes():
    """The Budavári hierarchical posterior must be sensitive to astrometric
    separation. Fixture: one pair at sep~0.36 arcsec, the other at
    sep=60 arcsec; magnitude KDE contributes equally. The closer pair
    must beat the farther pair by at least 0.2 in p_match (Budavári's
    posterior is sensitive but not strictly monotonic once KDE softens)."""
    left = pl.DataFrame(
        {
            "ra": [10.0001, 10.01667],  # ~0.36 arcsec and 60 arcsec from right[0]
            "dec": [5.0, 5.0],
            "mag": [10.0, 10.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.0, 10.0],
            "dec": [5.0, 5.0],
            "mag": [10.0, 11.0],
        }
    )
    spec = MatchSpec(radius_arcsec=60.0, prior_columns=["mag"])
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 2
    assert "p_match" in out.columns
    p = out["p_match"].to_numpy()
    sep = out["sep_arcsec"].to_numpy()
    closer_p = p[int(np.argmin(sep))]
    farther_p = p[int(np.argmax(sep))]
    assert closer_p - farther_p >= 0.2, (
        f"Expected sep-driven posterior gap >= 0.2; got closer={closer_p} "
        f"farther={farther_p} seps={sorted(sep.tolist())}"
    )
