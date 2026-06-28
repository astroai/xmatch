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


# ------------------------------------------------------------------- proper motion
@pytest.fixture
def pm_frames():
    """Gaia-like frames with proper motion and epoch columns."""
    # Two stars: one with significant PM, one with zero PM.
    left = pl.DataFrame({
        "ra": [10.0, 20.0],
        "dec": [5.0, 6.0],
        "pmra": [100.0, 0.0],   # mas/yr (already cosDec-scaled)
        "pmdec": [50.0, 0.0],
        "ref_epoch": [2015.5, 2015.5],
    })
    right = pl.DataFrame({
        "ra": [10.0, 20.0],
        "dec": [5.0, 6.0],
        "pmra": [-50.0, 50.0],
        "pmdec": [-25.0, 25.0],
        "ref_epoch": [2015.5, 2015.5],
    })
    return left, right


def test_proper_motion_propagates_both_sides(pm_frames):
    """Both sides with PM columns should have coordinates shifted."""
    from xmatch.matchers import _apply_proper_motion

    left, right = pm_frames
    l_src = _src("a", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")
    r_src = _src("b", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")

    new_left, new_right = _apply_proper_motion(
        left, right, l_src, r_src, target_epoch=2016.0,
    )
    # First left star has 100 mas/yr PM for 0.5 yr → ~50 mas → ~0.0139 deg RA shift
    delta_ra_left = abs(new_left["ra"][0] - left["ra"][0])
    assert delta_ra_left > 1e-7, f"Expected RA shift from PM, got {delta_ra_left}"
    # Second left star has zero PM → no shift
    assert abs(new_left["ra"][1] - left["ra"][1]) < 1e-10
    # Right side should also shift
    delta_ra_right = abs(new_right["ra"][0] - right["ra"][0])
    assert delta_ra_right > 1e-7, f"Expected RA shift on right side, got {delta_ra_right}"


def test_proper_motion_nan_pm_treated_as_zero(pm_frames):
    """NaN proper motions should be treated as zero (no propagation)."""
    from xmatch.matchers import _apply_proper_motion

    left, right = pm_frames
    # Set PM to NaN for first star
    left = left.with_columns(pl.Series("pmra", [np.nan, 0.0]))
    left = left.with_columns(pl.Series("pmdec", [np.nan, 0.0]))

    l_src = _src("a", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")
    r_src = _src("b")  # no PM on right

    new_left, new_right = _apply_proper_motion(
        left, right, l_src, r_src, target_epoch=2016.0,
    )
    # NaN PM → no shift
    assert abs(new_left["ra"][0] - left["ra"][0]) < 1e-10


def test_proper_motion_no_pm_columns_returns_unchanged(pm_frames):
    """When neither source has PM columns, frames are returned as-is."""
    from xmatch.matchers import _apply_proper_motion

    left, right = pm_frames
    l_src = _src("a")  # no PM columns
    r_src = _src("b")  # no PM columns

    new_left, new_right = _apply_proper_motion(
        left, right, l_src, r_src, target_epoch=2016.0,
    )
    assert new_left is left
    assert new_right is right


def test_proper_motion_catalogue_level_epoch(pm_frames):
    """Catalogue-level epoch attribute should be used when epoch_column is absent."""
    from xmatch.matchers import _apply_proper_motion

    left, right = pm_frames
    # Drop the ref_epoch column; use catalogue-level epoch instead.
    left = left.drop("ref_epoch")
    right = right.drop("ref_epoch")

    l_src = _src("a", pm_ra_column="pmra", pm_dec_column="pmdec", epoch=2015.5)
    r_src = _src("b", pm_ra_column="pmra", pm_dec_column="pmdec", epoch=2015.5)

    new_left, new_right = _apply_proper_motion(
        left, right, l_src, r_src, target_epoch=2016.0,
    )
    delta_ra = abs(new_left["ra"][0] - left["ra"][0])
    assert delta_ra > 1e-7, f"Expected RA shift using catalogue-level epoch, got {delta_ra}"


def test_proper_motion_one_side_only(pm_frames):
    """When only one side has PM info, only that side is propagated."""
    from xmatch.matchers import _apply_proper_motion

    left, right = pm_frames
    l_src = _src("a", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")
    r_src = _src("b")  # no PM

    new_left, new_right = _apply_proper_motion(
        left, right, l_src, r_src, target_epoch=2016.0,
    )
    # Left should shift
    assert abs(new_left["ra"][0] - left["ra"][0]) > 1e-7
    # Right unchanged
    assert abs(new_right["ra"][0] - right["ra"][0]) < 1e-10


def test_proper_motion_missing_epoch_skips(pm_frames):
    """When no epoch info is available, PM propagation is skipped."""
    from xmatch.matchers import _apply_proper_motion

    left, right = pm_frames
    # Drop epoch column and don't set catalogue-level epoch.
    left = left.drop("ref_epoch")
    right = right.drop("ref_epoch")

    l_src = _src("a", pm_ra_column="pmra", pm_dec_column="pmdec")  # no epoch!
    r_src = _src("b")

    new_left, new_right = _apply_proper_motion(
        left, right, l_src, r_src, target_epoch=2016.0,
    )
    assert new_left is left  # unchanged


def test_proper_motion_end_to_end_via_sky_match():
    """End-to-end: PM-corrected match should give different results than uncorrected.
    Propagate to a different epoch (20 years later) and confirm coordinates shift."""
    import numpy as np

    left = pl.DataFrame({
        "ra": [10.0],
        "dec": [5.0],
        "pmra": [100.0],   # 100 mas/yr
        "pmdec": [50.0],
        "ref_epoch": [2000.0],
    })
    right = pl.DataFrame({
        "ra": [10.0],
        "dec": [5.0],
        "pmra": [0.0],
        "pmdec": [0.0],
        "ref_epoch": [2000.0],
    })

    l_src = _src("a", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")
    r_src = _src("b", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")

    # Without PM correction: same position, 0 arcsec separation
    spec_no_pm = MatchSpec(radius_arcsec=1.0)
    out_no_pm = sky_match(l_src, r_src, left.lazy(), right.lazy(), spec_no_pm, engine="fast").collect()
    assert out_no_pm.height == 1
    sep_no_pm = out_no_pm["sep_arcsec"][0]

    # With PM correction to 2020.0 (20-year baseline): left star moves ~2000 mas = 2 arcsec
    spec_pm = MatchSpec(radius_arcsec=1.0, target_epoch=2020.0)
    out_pm = sky_match(l_src, r_src, left.lazy(), right.lazy(), spec_pm, engine="fast").collect()
    # After PM propagation, left star moves away → should NOT match within 1 arcsec
    assert out_pm.height == 0, (
        f"Expected 0 matches after 20yr PM propagation (star moved {100*20/1000:.1f} arcsec), "
        f"got {out_pm.height}"
    )


# ------------------------------------------------------------------- filter expression
def test_filter_expr_keeps_matching_pairs():
    """filter_expr should keep only pairs satisfying the SQL WHERE clause."""
    left = pl.DataFrame({"ra": [10.0, 10.0], "dec": [5.0, 5.0], "mag": [10.0, 14.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005], "mag": [10.1]})
    spec = MatchSpec(radius_arcsec=1.0, filter_expr="abs(mag - mag_2) < 1.0", find="all")
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    # Only the first left star (mag=10.0) matches right (mag=10.1) → diff=0.1 < 1.0
    # Second left star (mag=14.0) → diff=3.9 >= 1.0 → filtered out
    assert out.height == 1
    assert out["mag"][0] == 10.0


def test_filter_expr_empty_result():
    """When filter excludes all pairs, result should be empty."""
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "mag": [10.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005], "mag": [20.0]})
    spec = MatchSpec(radius_arcsec=1.0, filter_expr="abs(mag - mag_2) < 0.1", find="best")
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 0


def test_filter_expr_all_pass():
    """When filter is always true, all pairs survive."""
    left = pl.DataFrame({"ra": [10.0, 10.0], "dec": [5.0, 5.0]})
    right = pl.DataFrame({"ra": [10.00005, 10.0001], "dec": [5.00005, 5.0]})
    spec = MatchSpec(radius_arcsec=1.0, filter_expr="1 = 1", find="all")
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 4  # 2 left × 2 right = 4 matches within 1 arcsec


def test_filter_expr_invalid_sql_falls_back():
    """Invalid SQL should be caught and all pairs kept with a warning."""
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005]})
    spec = MatchSpec(radius_arcsec=1.0, filter_expr="THIS IS NOT VALID SQL !!!", find="best")
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    # Should keep the pair despite invalid filter (graceful fallback)
    assert out.height == 1


def test_filter_expr_with_find_best_reduces_after_filter():
    """filter_expr applied before find='best' reduction: multiple candidates
    get filtered, then the best among survivors is picked."""
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "mag": [10.0]})
    right = pl.DataFrame({
        "ra": [10.00002, 10.0001],
        "dec": [5.00002, 5.0],
        "mag": [10.05, 20.0],  # second has bad photometry
    })
    # Both right stars are within 1 arcsec spatially.
    # filter_expr removes the bad-photometry match (mag diff = 10.0 > 1.0).
    # Then find="best" picks the sole survivor.
    spec = MatchSpec(
        radius_arcsec=1.0, find="best",
        filter_expr="abs(mag - mag_2) < 1.0",
    )
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 1
    assert abs(out["mag_2"][0] - 10.05) < 0.01


def test_filter_expr_engine_zone_supported():
    """filter_expr should work with the zone engine."""
    left = pl.DataFrame({"ra": [10.0, 10.0], "dec": [5.0, 5.0], "mag": [10.0, 14.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005], "mag": [10.1]})
    spec = MatchSpec(radius_arcsec=1.0, filter_expr="abs(mag - mag_2) < 1.0", find="all")
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="zone").collect()
    assert out.height == 1


# ------------------------------------------------------------------- N-dimensional cKDTree
def test_nd_match_identical_photometry_same_as_spatial():
    """When extra columns are identical on both sides, N-d matching should
    produce the same results as spatial-only matching."""
    left = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 6.0], "mag": [10.0, 10.0]})
    # Right: both stars are within 1 arcsec of their left counterparts
    right = pl.DataFrame({
        "ra": [10.00005, 20.00005],
        "dec": [5.00005, 6.00005],
        "mag": [10.0, 10.0],
    })

    spec_spatial = MatchSpec(radius_arcsec=1.0, find="best")
    spec_nd = MatchSpec(radius_arcsec=1.0, find="best", extra_distance_cols={"mag": 1.0})

    out_sp = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec_spatial, engine="fast").collect()
    out_nd = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec_nd, engine="fast").collect()

    assert out_sp.height == out_nd.height == 2
    assert np.allclose(sorted(out_sp["sep_arcsec"].to_list()),
                       sorted(out_nd["sep_arcsec"].to_list()), atol=1e-6)


def test_nd_match_picks_photometrically_closer_star():
    """Given two spatial candidates, the N-d matcher should pick the one with
    more similar photometry even if it's spatially slightly farther."""
    # Star A: spatially closer (~0.18 arcsec) but photometrically off (diff=5.0)
    # Star B: spatially farther (~0.29 arcsec) but photometrically perfect (diff=0.05)
    # Spatial-only engine picks A (closer spatially).
    # N-d engine should pick B (better overall N-d distance).
    left = pl.DataFrame({
        "ra": [10.0, 10.0],
        "dec": [5.0, 5.0],
        "mag": [10.0, 10.0],
    })
    right = pl.DataFrame({
        "ra": [10.00005, 10.00008],  # first is closer spatially
        "dec": [5.00005, 5.0],
        "mag": [15.0, 10.05],       # second is photometrically closer
    })

    spec_spatial = MatchSpec(radius_arcsec=1.0, find="best")
    spec_nd = MatchSpec(radius_arcsec=1.0, find="best", extra_distance_cols={"mag": 1.0})

    out_spatial = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                            spec_spatial, engine="fast").collect()
    out_nd = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                       spec_nd, engine="fast").collect()

    assert out_spatial.height == 2  # both left stars match within 1 arcsec
    assert out_nd.height == 2
    # Spatial-only: both left stars pick the spatially closer right star (mag=15)
    assert all(abs(out_spatial["mag_2"].to_numpy() - 15.0) < 0.01)
    # N-d: both left stars should pick the photometrically similar star (mag=10.05)
    # because the N-d distance (spatial chord diff + z-score normalized mag diff)
    # favors the photometric match over the tiny spatial advantage
    assert all(abs(out_nd["mag_2"].to_numpy() - 10.05) < 0.01)


def test_nd_chunk_size_parity():
    """Chunked _scipy_match_nd must produce identical results regardless of
    _ND_CHUNK_SIZE.  Test with extreme chunk_size=1 (one row per chunk) vs
    the default 50_000 on a small catalogue to verify index-offset
    correctness."""
    from xmatch import matchers

    left = pl.DataFrame({
        "ra": [10.0, 10.0, 20.0, 20.0],
        "dec": [5.0, 5.0, 6.0, 6.0],
        "mag": [10.0, 10.0, 12.0, 12.0],
    })
    right = pl.DataFrame({
        "ra": [10.00005, 10.00008, 20.00005, 20.00008],
        "dec": [5.00005, 5.0, 6.00005, 6.0],
        "mag": [10.0, 15.0, 12.0, 16.0],
    })
    spec = MatchSpec(radius_arcsec=1.0, find="best",
                     extra_distance_cols={"mag": 1.0})

    orig_chunk = matchers._ND_CHUNK_SIZE
    try:
        matchers._ND_CHUNK_SIZE = 1  # extreme: one row per chunk
        out_1 = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                          spec, engine="fast").collect()

        matchers._ND_CHUNK_SIZE = 50_000
        out_50k = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                            spec, engine="fast").collect()
    finally:
        matchers._ND_CHUNK_SIZE = orig_chunk

    assert out_1.height == out_50k.height == 4
    assert np.allclose(
        sorted(out_1["sep_arcsec"].to_list()),
        sorted(out_50k["sep_arcsec"].to_list()),
        atol=1e-6,
    )
    # Verify N-d ranking: all 4 left stars should pick the photometrically
    # similar right star (mag_2 = left mag, not 15 or 16)
    for i in range(4):
        assert abs(out_1["mag_2"][i] - out_1["mag"][i]) < 0.01


def test_nd_match_missing_column_warns_but_matches():
    """When extra_distance_cols references a column not in the data,
    the engine should warn and fall back to spatial-only."""
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005]})
    spec = MatchSpec(radius_arcsec=1.0, find="best",
                     extra_distance_cols={"nonexistent_col": 1.0})
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 1  # still matches spatially


def test_nd_match_find_all_unaffected():
    """extra_distance_cols should not affect find='all' mode — it only
    affects best-match ranking."""
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "mag": [10.0]})
    right = pl.DataFrame({
        "ra": [10.00005, 10.0001],
        "dec": [5.00005, 5.0],
        "mag": [10.5, 15.0],
    })
    spec = MatchSpec(radius_arcsec=1.0, find="all", extra_distance_cols={"mag": 1.0})
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    # Both right stars are within 1 arcsec; find="all" returns both
    assert out.height == 2


def test_nd_match_multiple_extra_columns():
    """Multiple extra_distance_cols should all contribute to N-d ranking.
    Verify N-d picks a photometrically better star even when it's spatially
    slightly farther."""
    left = pl.DataFrame({"ra": [10.0, 10.0], "dec": [5.0, 5.0], "g": [10.0, 10.0], "r": [9.0, 9.0]})
    right = pl.DataFrame({
        "ra": [10.00005, 10.00008],
        "dec": [5.00005, 5.0],
        "g": [16.0, 10.1],     # first is photometrically wrong
        "r": [15.0, 9.1],       # second is photometrically close
    })

    spec_spatial = MatchSpec(radius_arcsec=1.0, find="best")
    spec_nd = MatchSpec(radius_arcsec=1.0, find="best",
                        extra_distance_cols={"g": 0.5, "r": 0.5})

    out_spatial = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                            spec_spatial, engine="fast").collect()
    out_nd = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                       spec_nd, engine="fast").collect()

    assert out_spatial.height == out_nd.height == 2
    # Spatial-only picks the spatially closer right star (mag g=16, r=15)
    assert all(abs(out_spatial["g_2"].to_numpy() - 16.0) < 0.01)
    # N-d picks the photometrically similar star (g=10.1, r=9.1)
    assert all(abs(out_nd["g_2"].to_numpy() - 10.1) < 0.01)


# ------------------------------------------------------------------- batch_size / out-of-core
def test_batch_size_parity_with_no_batching():
    """batch_size must produce identical results to no-batching."""
    rng = np.random.default_rng(42)
    left = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.1, 200),
        "dec": rng.uniform(5.0, 5.1, 200),
    })
    right = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.1, 300),
        "dec": rng.uniform(5.0, 5.1, 300),
    })

    spec_full = MatchSpec(radius_arcsec=10.0, find="best")
    spec_batch = MatchSpec(radius_arcsec=10.0, find="best", batch_size=5)

    out_full = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                         spec_full, engine="zone").collect()
    out_batch = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                          spec_batch, engine="zone").collect()

    assert out_full.height == out_batch.height
    assert np.allclose(
        sorted(out_full["sep_arcsec"].to_list()),
        sorted(out_batch["sep_arcsec"].to_list()),
        atol=1e-6,
    )


def test_batch_size_one_pixel_per_batch():
    """batch_size=1 should process one pixel group at a time and produce
    correct results."""
    rng = np.random.default_rng(123)
    left = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.05, 100),
        "dec": rng.uniform(5.0, 5.05, 100),
    })
    right = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.05, 150),
        "dec": rng.uniform(5.0, 5.05, 150),
    })

    spec = MatchSpec(radius_arcsec=5.0, find="best", batch_size=1)
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="zone").collect()
    # Just verify it doesn't crash and returns valid results
    assert out.height >= 1
    assert "sep_arcsec" in out.columns


def test_batch_size_larger_than_pixel_count():
    """batch_size larger than available pixel groups should degrade
    gracefully to all-at-once."""
    rng = np.random.default_rng(99)
    left = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.02, 50),
        "dec": rng.uniform(5.0, 5.02, 50),
    })
    right = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.02, 60),
        "dec": rng.uniform(5.0, 5.02, 60),
    })

    spec = MatchSpec(radius_arcsec=5.0, find="best", batch_size=10_000)
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="zone").collect()
    assert out.height >= 1


def test_batch_size_find_all():
    """batch_size should work with find='all'."""
    rng = np.random.default_rng(42)
    left = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.05, 80),
        "dec": rng.uniform(5.0, 5.05, 80),
    })
    right = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.05, 100),
        "dec": rng.uniform(5.0, 5.05, 100),
    })

    spec_full = MatchSpec(radius_arcsec=5.0, find="all")
    spec_batch = MatchSpec(radius_arcsec=5.0, find="all", batch_size=3)

    out_full = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                         spec_full, engine="zone").collect()
    out_batch = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                          spec_batch, engine="zone").collect()

    assert out_full.height == out_batch.height


# ------------------------------------------------------------------- ray engine

def test_ray_engine_parity_with_zone():
    """Ray-parallelised zone engine must produce identical match results
    to the single-machine zone engine on synthetic data.

    Spawns a local Ray cluster, matches a random catalogue with both
    ``engine='zone'`` and ``engine='ray'``, and asserts:
    * same number of matched pairs
    * identical left/right indices (sorted)
    * separations agree within floating-point tolerance
    """
    ray = pytest.importorskip("ray")

    # Generate random stars in a ~0.1×0.1 degree patch.
    rng = np.random.default_rng(42)
    n_left, n_right = 500, 800
    left = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.1, n_left),
        "dec": rng.uniform(5.0, 5.1, n_left),
    })
    right = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.1, n_right),
        "dec": rng.uniform(5.0, 5.1, n_right),
    })

    spec = MatchSpec(radius_arcsec=15.0, find="best")

    # Single-machine zone baseline.
    out_zone = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(),
        spec, engine="zone",
    ).collect()

    # Start a local Ray cluster and run the distributed engine.
    # ray.init handles re-init gracefully; shut down when done.
    ray.init(ignore_reinit_error=True, logging_level=40)
    try:
        out_ray = sky_match(
            _src("a"), _src("b"), left.lazy(), right.lazy(),
            spec, engine="ray",
        ).collect()
    finally:
        ray.shutdown()

    # Same number of matches.
    assert out_zone.height == out_ray.height, (
        f"zone={out_zone.height} vs ray={out_ray.height} matches"
    )
    if out_zone.height == 0:
        return  # both empty — parity holds vacuously

    # Match pair identity: sort by (left_ra, right_ra_2, sep_arcsec) so we
    # compare the same logical pairs regardless of internal ordering.
    def _sort_key(df):
        return sorted(
            zip(df["ra"].to_list(), df["ra_2"].to_list(),
                df["sep_arcsec"].to_list()),
        )

    assert _sort_key(out_zone) == _sort_key(out_ray), (
        "zone and ray produced different matched pairs"
    )

    # Separations agree.
    sep_zone = np.sort(out_zone["sep_arcsec"].to_numpy())
    sep_ray = np.sort(out_ray["sep_arcsec"].to_numpy())
    assert np.allclose(sep_zone, sep_ray, atol=1e-6), (
        f"Separation mismatch; max diff={np.max(np.abs(sep_zone - sep_ray))}"
    )


def test_ray_engine_find_all_parity_with_zone():
    """Ray engine must also agree with zone when find='all'."""
    ray = pytest.importorskip("ray")

    rng = np.random.default_rng(99)
    left = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.05, 300),
        "dec": rng.uniform(5.0, 5.05, 300),
    })
    right = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.05, 500),
        "dec": rng.uniform(5.0, 5.05, 500),
    })

    spec = MatchSpec(radius_arcsec=10.0, find="all")

    out_zone = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(),
        spec, engine="zone",
    ).collect()

    ray.init(ignore_reinit_error=True, logging_level=40)
    try:
        out_ray = sky_match(
            _src("a"), _src("b"), left.lazy(), right.lazy(),
            spec, engine="ray",
        ).collect()
    finally:
        ray.shutdown()

    assert out_zone.height == out_ray.height, (
        f"find=all: zone={out_zone.height} vs ray={out_ray.height}"
    )
    if out_zone.height == 0:
        return

    # Match pair identity (same as the find='best' test).
    def _sort_key(df):
        return sorted(
            zip(df["ra"].to_list(), df["ra_2"].to_list(),
                df["sep_arcsec"].to_list()),
        )
    assert _sort_key(out_zone) == _sort_key(out_ray), (
        "find=all: zone and ray produced different matched pairs"
    )

    sep_zone = np.sort(out_zone["sep_arcsec"].to_numpy())
    sep_ray = np.sort(out_ray["sep_arcsec"].to_numpy())
    assert np.allclose(sep_zone, sep_ray, atol=1e-6)


def test_ray_engine_graceful_fallback_when_unavailable(monkeypatch):
    """When Ray is not installed, the ray engine must fall back to zone
    (or fast) transparently and still produce correct results."""
    # Directly patch the availability check so the fallback path is
    # exercised regardless of whether Ray is actually installed.
    monkeypatch.setattr("xmatch.ray_engine.ray_available", lambda: False)
    # Clear the lazy-initialised remote function cache.
    monkeypatch.setattr("xmatch.ray_engine._RAY_PIXEL_BATCH", None)

    left = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 6.0]})
    right = pl.DataFrame({
        "ra": [10.00005, 20.5, 30.00002],
        "dec": [5.00005, 6.5, 7.00001],
    })
    spec = MatchSpec(radius_arcsec=1.0)

    # Should not raise; falls back internally.
    out = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(),
        spec, engine="ray",
    ).collect()
    assert out.height >= 1  # at least one pair should match within 1 arcsec


def test_ray_engine_graceful_fallback_zone_to_fast(monkeypatch):
    """When cdshealpix is unavailable, ray→zone→fast chain must still
    produce correct results via the final fast fallback."""
    # Simulate no cdshealpix so zone_match falls back to _scipy_match.
    # Ray is also unavailable for this test (chain: ray → zone → fast).
    monkeypatch.setattr("xmatch.ray_engine.ray_available", lambda: False)
    monkeypatch.setattr("xmatch.ray_engine._RAY_PIXEL_BATCH", None)

    import sys

    # Remove cdshealpix from sys.modules so the import inside _zone_match
    # actually triggers the __import__ patch below (avoids cache hit).
    monkeypatch.delitem(sys.modules, "cdshealpix", raising=False)

    # Save original __import__ so _fake_import doesn't call itself.
    import builtins
    _orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "cdshealpix":
            raise ImportError("cdshealpix not available")
        return _orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    left = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 6.0]})
    right = pl.DataFrame({"ra": [10.00005, 50.0], "dec": [5.00005, 5.0]})
    spec = MatchSpec(radius_arcsec=1.0)

    # Should fall through ray→zone→fast and still produce correct matches.
    out = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(),
        spec, engine="ray",
    ).collect()
    assert out.height == 1
    assert out["sep_arcsec"][0] < 1.0


def test_margin_caching_correctness_vs_fast():
    """The margin-cached zone engine must produce the same matches as the
    fast (scipy cKDTree) engine."""
    rng = np.random.default_rng(7)
    left = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.1, 150),
        "dec": rng.uniform(5.0, 5.1, 150),
    })
    right = pl.DataFrame({
        "ra": rng.uniform(10.0, 10.1, 200),
        "dec": rng.uniform(5.0, 5.1, 200),
    })

    spec = MatchSpec(radius_arcsec=8.0, find="best")

    out_fast = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                         spec, engine="fast").collect()
    out_zone = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(),
                         spec, engine="zone").collect()

    assert out_fast.height == out_zone.height
    assert np.allclose(
        sorted(out_fast["sep_arcsec"].to_list()),
        sorted(out_zone["sep_arcsec"].to_list()),
        atol=1e-6,
    )
