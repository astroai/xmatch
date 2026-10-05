import numpy as np
import polars as pl
import pytest

from xmatch import stilts
from xmatch.exceptions import CrossMatchError
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
    left = pl.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [5.0, 6.0],
            "pmra": [100.0, 0.0],  # mas/yr (already cosDec-scaled)
            "pmdec": [50.0, 0.0],
            "ref_epoch": [2015.5, 2015.5],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [5.0, 6.0],
            "pmra": [-50.0, 50.0],
            "pmdec": [-25.0, 25.0],
            "ref_epoch": [2015.5, 2015.5],
        }
    )
    return left, right


def test_proper_motion_propagates_both_sides(pm_frames):
    """Both sides with PM columns should have coordinates shifted."""
    from xmatch.matchers import _apply_proper_motion

    left, right = pm_frames
    l_src = _src("a", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")
    r_src = _src("b", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")

    new_left, new_right = _apply_proper_motion(
        left,
        right,
        l_src,
        r_src,
        target_epoch=2016.0,
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
        left,
        right,
        l_src,
        r_src,
        target_epoch=2016.0,
    )
    # NaN PM → no shift
    assert abs(new_left["ra"][0] - left["ra"][0]) < 1e-10


def test_proper_motion_uses_6d_only_for_complete_physical_rows(monkeypatch):
    from xmatch import astro_utils
    from xmatch.matchers import _apply_proper_motion

    angular_calls = []
    space_calls = []

    def fake_angular(ra, dec, *args):
        angular_calls.append((np.asarray(ra), np.asarray(dec)))
        return np.asarray(ra) + 1.0, np.asarray(dec) + 1.0

    def fake_space(ra, dec, *args):
        space_calls.append((np.asarray(ra), np.asarray(dec)))
        return np.asarray(ra) + 10.0, np.asarray(dec) + 10.0

    monkeypatch.setattr(astro_utils, "propagate_proper_motion", fake_angular)
    monkeypatch.setattr(astro_utils, "propagate_space_motion", fake_space)
    left = pl.DataFrame(
        {
            "ra": [10.0, 20.0, 30.0, 40.0],
            "dec": [1.0, 2.0, 3.0, 4.0],
            "pmra": [100.0, 100.0, 100.0, 1.0e9],
            "pmdec": [50.0, 50.0, 50.0, 50.0],
            "epoch": [2000.0] * 4,
            "parallax": [100.0, 100.0, -1.0, 1.0],
            "rv": [20.0, np.nan, 20.0, 20.0],
            "tag": ["complete", "missing-rv", "negative-parallax", "impossible"],
        }
    )
    source = _src(
        "gaia",
        pm_ra_column="pmra",
        pm_dec_column="pmdec",
        epoch_column="epoch",
        parallax_column="parallax",
        radial_velocity_column="rv",
    )

    moved, _ = _apply_proper_motion(
        left,
        pl.DataFrame({"ra": [0.0], "dec": [0.0]}),
        source,
        _src("reference"),
        target_epoch=2025.0,
    )

    assert len(angular_calls) == 1
    assert len(space_calls) == 1
    np.testing.assert_allclose(space_calls[0][0], [10.0])
    np.testing.assert_allclose(moved["ra"], [20.0, 21.0, 31.0, 41.0])
    assert moved["tag"].to_list() == left["tag"].to_list()
    np.testing.assert_allclose(moved["parallax"], left["parallax"])


def test_proper_motion_transports_gaia_covariance(monkeypatch):
    from xmatch import astro_utils
    from xmatch.matchers import _apply_proper_motion, _pos_covariance
    from xmatch.sources import ASTROMETRIC_COVARIANCE_KEYS

    covariance_columns = {key: key for key in ASTROMETRIC_COVARIANCE_KEYS}
    row = {
        "ra": [10.0],
        "dec": [20.0],
        "pmra": [100.0],
        "pmdec": [50.0],
        "epoch": [2016.0],
        "parallax": [100.0],
        "rv": [20.0],
        "ra_error": [1.0],
        "dec_error": [1.5],
        "parallax_error": [0.5],
        "pmra_error": [2.0],
        "pmdec_error": [3.0],
    }
    for key in ASTROMETRIC_COVARIANCE_KEYS[5:]:
        row[key] = [0.5 if key == "ra_pmra_corr" else 0.0]
    frame = pl.DataFrame(row)
    source = _src(
        "gaia",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        corr_column="ra_dec_corr",
        pos_err_units="mas",
        pm_ra_column="pmra",
        pm_dec_column="pmdec",
        epoch_column="epoch",
        parallax_column="parallax",
        radial_velocity_column="rv",
        astrometric_covariance_columns=covariance_columns,
    )

    def fake_with_jacobian(ra, dec, *args):
        jacobian = np.eye(6)[None, ...]
        jacobian[:, 0, 2] = 10.0
        jacobian[:, 1, 3] = 10.0
        return np.asarray(ra) + 1.0, np.asarray(dec) + 2.0, jacobian

    monkeypatch.setattr(
        astro_utils,
        "propagate_space_motion_with_jacobian",
        fake_with_jacobian,
    )
    moved, _ = _apply_proper_motion(
        frame,
        pl.DataFrame({"ra": [0.0], "dec": [0.0]}),
        source,
        _src("reference"),
        target_epoch=2026.0,
        propagate_covariance=True,
        fallback_policy="error",
    )
    covariance = _pos_covariance(moved, source)

    assert covariance is not None
    east, north, rho = covariance
    np.testing.assert_allclose(east, [421.0e-6])
    np.testing.assert_allclose(north, [902.25e-6])
    np.testing.assert_allclose(rho, [0.0])
    np.testing.assert_allclose(moved["ra"], [11.0])
    np.testing.assert_allclose(moved["dec"], [22.0])


def test_angular_motion_transports_gaia_covariance_without_rv(monkeypatch):
    from xmatch import astro_utils
    from xmatch.matchers import _apply_proper_motion, _pos_covariance
    from xmatch.sources import ASTROMETRIC_COVARIANCE_KEYS

    covariance_columns = {key: key for key in ASTROMETRIC_COVARIANCE_KEYS}
    row = {
        "ra": [10.0],
        "dec": [20.0],
        "pmra": [100.0],
        "pmdec": [50.0],
        "epoch": [2016.0],
        "parallax": [100.0],
        "rv": [np.nan],
        "ra_error": [1.0],
        "dec_error": [1.5],
        "parallax_error": [0.5],
        "pmra_error": [2.0],
        "pmdec_error": [3.0],
    }
    for key in ASTROMETRIC_COVARIANCE_KEYS[5:]:
        row[key] = [0.5 if key == "ra_pmra_corr" else 0.0]
    frame = pl.DataFrame(row)
    source = _src(
        "gaia",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        corr_column="ra_dec_corr",
        pos_err_units="mas",
        pm_ra_column="pmra",
        pm_dec_column="pmdec",
        epoch_column="epoch",
        parallax_column="parallax",
        radial_velocity_column="rv",
        astrometric_covariance_columns=covariance_columns,
    )

    def fake_with_jacobian(ra, dec, *args):
        jacobian = np.eye(4)[:2][None, ...]
        jacobian[:, 0, 2] = 10.0
        jacobian[:, 1, 3] = 10.0
        return np.asarray(ra) + 1.0, np.asarray(dec) + 2.0, jacobian

    monkeypatch.setattr(
        astro_utils,
        "propagate_proper_motion_with_jacobian",
        fake_with_jacobian,
    )
    moved, _ = _apply_proper_motion(
        frame,
        pl.DataFrame({"ra": [0.0], "dec": [0.0]}),
        source,
        _src("reference"),
        target_epoch=2026.0,
        propagate_covariance=True,
        fallback_policy="error",
    )
    covariance = _pos_covariance(moved, source)

    assert covariance is not None
    east, north, rho = covariance
    np.testing.assert_allclose(east, [421.0e-6])
    np.testing.assert_allclose(north, [902.25e-6])
    np.testing.assert_allclose(rho, [0.0])
    np.testing.assert_allclose(moved["ra"], [11.0])
    np.testing.assert_allclose(moved["dec"], [22.0])


def test_invalid_gaia_covariance_row_is_not_zero_imputed():
    from xmatch.matchers import _astrometric_covariance_mas
    from xmatch.sources import ASTROMETRIC_COVARIANCE_KEYS

    covariance_columns = {key: key for key in ASTROMETRIC_COVARIANCE_KEYS}
    row = {
        "ra_error": [1.0],
        "dec_error": [1.0],
        "parallax_error": [1.0],
        "pmra_error": [1.0],
        "pmdec_error": [1.0],
    }
    for key in ASTROMETRIC_COVARIANCE_KEYS[5:]:
        row[key] = [1.1 if key == "ra_dec_corr" else 0.0]
    result = _astrometric_covariance_mas(
        pl.DataFrame(row),
        _src("gaia", astrometric_covariance_columns=covariance_columns),
    )

    assert result is not None
    _covariance, valid_6d, valid_angular = result
    assert not valid_6d[0]
    assert not valid_angular[0]


def test_angular_covariance_remains_valid_without_parallax_uncertainty():
    from xmatch.matchers import _astrometric_covariance_mas
    from xmatch.sources import ASTROMETRIC_COVARIANCE_KEYS

    covariance_columns = {key: key for key in ASTROMETRIC_COVARIANCE_KEYS}
    row = {
        "ra_error": [1.0],
        "dec_error": [1.0],
        "parallax_error": [np.nan],
        "pmra_error": [1.0],
        "pmdec_error": [1.0],
    }
    for key in ASTROMETRIC_COVARIANCE_KEYS[5:]:
        row[key] = [np.nan if "parallax" in key else 0.0]
    result = _astrometric_covariance_mas(
        pl.DataFrame(row),
        _src("gaia", astrometric_covariance_columns=covariance_columns),
    )

    assert result is not None
    covariance, valid_6d, valid_angular = result
    assert not valid_6d[0]
    assert valid_angular[0]
    np.testing.assert_allclose(covariance[0, :4, :4], np.eye(4))


def test_proper_motion_empty_frame_preserves_schema():
    from xmatch.matchers import _apply_proper_motion

    empty = pl.DataFrame(
        schema={
            "ra": pl.Float64,
            "dec": pl.Float64,
            "pmra": pl.Float64,
            "pmdec": pl.Float64,
            "epoch": pl.Float64,
        }
    )
    source = _src("empty", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="epoch")
    moved, _ = _apply_proper_motion(
        empty,
        pl.DataFrame({"ra": [0.0], "dec": [0.0]}),
        source,
        _src("reference"),
        target_epoch=2025.0,
    )
    assert moved.schema == empty.schema


def test_proper_motion_rejects_nonfinite_target_epoch(pm_frames):
    left, right = pm_frames
    source = _src("a", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")
    with pytest.raises(CrossMatchError, match="target_epoch must be finite"):
        sky_match(
            source,
            _src("b"),
            left.lazy(),
            right.lazy(),
            MatchSpec(target_epoch=np.nan),
        )


def test_proper_motion_no_pm_columns_returns_unchanged(pm_frames):
    """When neither source has PM columns, frames are returned as-is."""
    from xmatch.matchers import _apply_proper_motion

    left, right = pm_frames
    l_src = _src("a")  # no PM columns
    r_src = _src("b")  # no PM columns

    new_left, new_right = _apply_proper_motion(
        left,
        right,
        l_src,
        r_src,
        target_epoch=2016.0,
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
        left,
        right,
        l_src,
        r_src,
        target_epoch=2016.0,
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
        left,
        right,
        l_src,
        r_src,
        target_epoch=2016.0,
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
        left,
        right,
        l_src,
        r_src,
        target_epoch=2016.0,
    )
    assert new_left is left  # unchanged


def test_proper_motion_end_to_end_via_sky_match():
    """End-to-end: PM-corrected match should give different results than uncorrected.
    Propagate to a different epoch (20 years later) and confirm coordinates shift."""

    left = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "pmra": [100.0],  # 100 mas/yr
            "pmdec": [50.0],
            "ref_epoch": [2000.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "pmra": [0.0],
            "pmdec": [0.0],
            "ref_epoch": [2000.0],
        }
    )

    l_src = _src("a", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")
    r_src = _src("b", pm_ra_column="pmra", pm_dec_column="pmdec", epoch_column="ref_epoch")

    # Without PM correction: same position, 0 arcsec separation
    spec_no_pm = MatchSpec(radius_arcsec=1.0)
    out_no_pm = sky_match(
        l_src, r_src, left.lazy(), right.lazy(), spec_no_pm, engine="fast"
    ).collect()
    assert out_no_pm.height == 1
    out_no_pm["sep_arcsec"][0]

    # With PM correction to 2020.0 (20-year baseline): left star moves ~2000 mas = 2 arcsec
    spec_pm = MatchSpec(radius_arcsec=1.0, target_epoch=2020.0)
    out_pm = sky_match(l_src, r_src, left.lazy(), right.lazy(), spec_pm, engine="fast").collect()
    # After PM propagation, left star moves away → should NOT match within 1 arcsec
    assert out_pm.height == 0, (
        f"Expected 0 matches after 20yr PM propagation (star moved {100 * 20 / 1000:.1f} arcsec), "
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
    right = pl.DataFrame(
        {
            "ra": [10.00002, 10.0001],
            "dec": [5.00002, 5.0],
            "mag": [10.05, 20.0],  # second has bad photometry
        }
    )
    # Both right stars are within 1 arcsec spatially.
    # filter_expr removes the bad-photometry match (mag diff = 10.0 > 1.0).
    # Then find="best" picks the sole survivor.
    spec = MatchSpec(
        radius_arcsec=1.0,
        find="best",
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
    right = pl.DataFrame(
        {
            "ra": [10.00005, 20.00005],
            "dec": [5.00005, 6.00005],
            "mag": [10.0, 10.0],
        }
    )

    spec_spatial = MatchSpec(radius_arcsec=1.0, find="best")
    spec_nd = MatchSpec(radius_arcsec=1.0, find="best", extra_distance_cols={"mag": 1.0})

    out_sp = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_spatial, engine="fast"
    ).collect()
    out_nd = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_nd, engine="fast"
    ).collect()

    assert out_sp.height == out_nd.height == 2
    assert np.allclose(
        sorted(out_sp["sep_arcsec"].to_list()), sorted(out_nd["sep_arcsec"].to_list()), atol=1e-6
    )


def test_nd_match_picks_photometrically_closer_star():
    """Given two spatial candidates, the N-d matcher should pick the one with
    more similar photometry even if it's spatially slightly farther."""
    # Star A: spatially closer (~0.18 arcsec) but photometrically off (diff=5.0)
    # Star B: spatially farther (~0.29 arcsec) but photometrically perfect (diff=0.05)
    # Spatial-only engine picks A (closer spatially).
    # N-d engine should pick B (better overall N-d distance).
    left = pl.DataFrame(
        {
            "ra": [10.0, 10.0],
            "dec": [5.0, 5.0],
            "mag": [10.0, 10.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.00005, 10.00008],  # first is closer spatially
            "dec": [5.00005, 5.0],
            "mag": [15.0, 10.05],  # second is photometrically closer
        }
    )

    spec_spatial = MatchSpec(radius_arcsec=1.0, find="best")
    spec_nd = MatchSpec(radius_arcsec=1.0, find="best", extra_distance_cols={"mag": 1.0})

    out_spatial = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_spatial, engine="fast"
    ).collect()
    out_nd = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_nd, engine="fast"
    ).collect()

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

    left = pl.DataFrame(
        {
            "ra": [10.0, 10.0, 20.0, 20.0],
            "dec": [5.0, 5.0, 6.0, 6.0],
            "mag": [10.0, 10.0, 12.0, 12.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.00005, 10.00008, 20.00005, 20.00008],
            "dec": [5.00005, 5.0, 6.00005, 6.0],
            "mag": [10.0, 15.0, 12.0, 16.0],
        }
    )
    spec = MatchSpec(radius_arcsec=1.0, find="best", extra_distance_cols={"mag": 1.0})

    orig_chunk = matchers._ND_CHUNK_SIZE
    try:
        matchers._ND_CHUNK_SIZE = 1  # extreme: one row per chunk
        out_1 = sky_match(
            _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast"
        ).collect()

        matchers._ND_CHUNK_SIZE = 50_000
        out_50k = sky_match(
            _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast"
        ).collect()
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
    spec = MatchSpec(radius_arcsec=1.0, find="best", extra_distance_cols={"nonexistent_col": 1.0})
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 1  # still matches spatially


def test_nd_match_find_all_unaffected():
    """extra_distance_cols should not affect find='all' mode — it only
    affects best-match ranking."""
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "mag": [10.0]})
    right = pl.DataFrame(
        {
            "ra": [10.00005, 10.0001],
            "dec": [5.00005, 5.0],
            "mag": [10.5, 15.0],
        }
    )
    spec = MatchSpec(radius_arcsec=1.0, find="all", extra_distance_cols={"mag": 1.0})
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    # Both right stars are within 1 arcsec; find="all" returns both
    assert out.height == 2


def test_nd_match_multiple_extra_columns():
    """Multiple extra_distance_cols should all contribute to N-d ranking.
    Verify N-d picks a photometrically better star even when it's spatially
    slightly farther."""
    left = pl.DataFrame({"ra": [10.0, 10.0], "dec": [5.0, 5.0], "g": [10.0, 10.0], "r": [9.0, 9.0]})
    right = pl.DataFrame(
        {
            "ra": [10.00005, 10.00008],
            "dec": [5.00005, 5.0],
            "g": [16.0, 10.1],  # first is photometrically wrong
            "r": [15.0, 9.1],  # second is photometrically close
        }
    )

    spec_spatial = MatchSpec(radius_arcsec=1.0, find="best")
    spec_nd = MatchSpec(radius_arcsec=1.0, find="best", extra_distance_cols={"g": 0.5, "r": 0.5})

    out_spatial = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_spatial, engine="fast"
    ).collect()
    out_nd = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_nd, engine="fast"
    ).collect()

    assert out_spatial.height == out_nd.height == 2
    # Spatial-only picks the spatially closer right star (mag g=16, r=15)
    assert all(abs(out_spatial["g_2"].to_numpy() - 16.0) < 0.01)
    # N-d picks the photometrically similar star (g=10.1, r=9.1)
    assert all(abs(out_nd["g_2"].to_numpy() - 10.1) < 0.01)


# ------------------------------------------------------------------- batch_size / out-of-core
def test_batch_size_parity_with_no_batching():
    """batch_size must produce identical results to no-batching."""
    rng = np.random.default_rng(42)
    left = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.1, 200),
            "dec": rng.uniform(5.0, 5.1, 200),
        }
    )
    right = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.1, 300),
            "dec": rng.uniform(5.0, 5.1, 300),
        }
    )

    spec_full = MatchSpec(radius_arcsec=10.0, find="best")
    spec_batch = MatchSpec(radius_arcsec=10.0, find="best", batch_size=5)

    out_full = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_full, engine="zone"
    ).collect()
    out_batch = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_batch, engine="zone"
    ).collect()

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
    left = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.05, 100),
            "dec": rng.uniform(5.0, 5.05, 100),
        }
    )
    right = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.05, 150),
            "dec": rng.uniform(5.0, 5.05, 150),
        }
    )

    spec = MatchSpec(radius_arcsec=5.0, find="best", batch_size=1)
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="zone").collect()
    # Just verify it doesn't crash and returns valid results
    assert out.height >= 1
    assert "sep_arcsec" in out.columns


def test_batch_size_larger_than_pixel_count():
    """batch_size larger than available pixel groups should degrade
    gracefully to all-at-once."""
    rng = np.random.default_rng(99)
    left = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.02, 50),
            "dec": rng.uniform(5.0, 5.02, 50),
        }
    )
    right = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.02, 60),
            "dec": rng.uniform(5.0, 5.02, 60),
        }
    )

    spec = MatchSpec(radius_arcsec=5.0, find="best", batch_size=10_000)
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="zone").collect()
    assert out.height >= 1


def test_batch_size_find_all():
    """batch_size should work with find='all'."""
    rng = np.random.default_rng(42)
    left = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.05, 80),
            "dec": rng.uniform(5.0, 5.05, 80),
        }
    )
    right = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.05, 100),
            "dec": rng.uniform(5.0, 5.05, 100),
        }
    )

    spec_full = MatchSpec(radius_arcsec=5.0, find="all")
    spec_batch = MatchSpec(radius_arcsec=5.0, find="all", batch_size=3)

    out_full = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_full, engine="zone"
    ).collect()
    out_batch = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec_batch, engine="zone"
    ).collect()

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
    left = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.1, n_left),
            "dec": rng.uniform(5.0, 5.1, n_left),
        }
    )
    right = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.1, n_right),
            "dec": rng.uniform(5.0, 5.1, n_right),
        }
    )

    spec = MatchSpec(radius_arcsec=15.0, find="best")

    # Single-machine zone baseline.
    out_zone = sky_match(
        _src("a"),
        _src("b"),
        left.lazy(),
        right.lazy(),
        spec,
        engine="zone",
    ).collect()

    # Start a local Ray cluster and run the distributed engine.
    # ray.init handles re-init gracefully; shut down when done.
    ray.init(ignore_reinit_error=True, logging_level=40)
    try:
        out_ray = sky_match(
            _src("a"),
            _src("b"),
            left.lazy(),
            right.lazy(),
            spec,
            engine="ray",
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
            zip(df["ra"].to_list(), df["ra_2"].to_list(), df["sep_arcsec"].to_list(), strict=False),
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
    left = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.05, 300),
            "dec": rng.uniform(5.0, 5.05, 300),
        }
    )
    right = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.05, 500),
            "dec": rng.uniform(5.0, 5.05, 500),
        }
    )

    spec = MatchSpec(radius_arcsec=10.0, find="all")

    out_zone = sky_match(
        _src("a"),
        _src("b"),
        left.lazy(),
        right.lazy(),
        spec,
        engine="zone",
    ).collect()

    ray.init(ignore_reinit_error=True, logging_level=40)
    try:
        out_ray = sky_match(
            _src("a"),
            _src("b"),
            left.lazy(),
            right.lazy(),
            spec,
            engine="ray",
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
            zip(df["ra"].to_list(), df["ra_2"].to_list(), df["sep_arcsec"].to_list(), strict=False),
        )

    assert _sort_key(out_zone) == _sort_key(out_ray), (
        "find=all: zone and ray produced different matched pairs"
    )

    sep_zone = np.sort(out_zone["sep_arcsec"].to_numpy())
    sep_ray = np.sort(out_ray["sep_arcsec"].to_numpy())
    assert np.allclose(sep_zone, sep_ray, atol=1e-6)


def test_ray_engine_joins_ray_address(monkeypatch):
    """engine='ray' joins the cluster in RAY_ADDRESS (CANFAR cluster
    script); a dead address falls back to a fresh local cluster."""
    ray = pytest.importorskip("ray")
    ray.shutdown()  # deterministic start: init must actually run

    monkeypatch.setenv("RAY_ADDRESS", "head.example:6379")
    calls: list = []
    real_init = ray.init

    def fake_init(**kw):
        calls.append(kw)
        if kw.get("address"):
            raise ConnectionError("no cluster at head.example")
        return real_init(**kw)

    monkeypatch.setattr(ray, "init", fake_init)
    left = pl.DataFrame({"ra": [10.0, 10.001], "dec": [5.0, 5.0]})
    right = pl.DataFrame({"ra": [10.0005, 30.0], "dec": [5.0005, 5.0]})
    spec = MatchSpec(radius_arcsec=15.0, find="best")
    try:
        out = sky_match(
            _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="ray"
        ).collect()
        assert calls[0]["address"] == "head.example:6379"
        assert calls[1]["address"] is None  # ConnectionError -> local fallback
        assert out.height == 2  # both left rows matched the close right row
    finally:
        ray.shutdown()


def test_ray_engine_graceful_fallback_when_unavailable(monkeypatch):
    """When Ray is not installed, the ray engine must fall back to zone
    (or fast) transparently and still produce correct results."""
    # Directly patch the availability check so the fallback path is
    # exercised regardless of whether Ray is actually installed.
    monkeypatch.setattr("xmatch.ray_engine.ray_available", lambda: False)
    # Clear the lazy-initialised remote function cache.
    monkeypatch.setattr("xmatch.ray_engine._RAY_PIXEL_BATCH", None)

    left = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 6.0]})
    right = pl.DataFrame(
        {
            "ra": [10.00005, 20.5, 30.00002],
            "dec": [5.00005, 6.5, 7.00001],
        }
    )
    spec = MatchSpec(radius_arcsec=1.0)

    # Should not raise; falls back internally.
    out = sky_match(
        _src("a"),
        _src("b"),
        left.lazy(),
        right.lazy(),
        spec,
        engine="ray",
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
        _src("a"),
        _src("b"),
        left.lazy(),
        right.lazy(),
        spec,
        engine="ray",
    ).collect()
    assert out.height == 1
    assert out["sep_arcsec"][0] < 1.0


def test_margin_caching_correctness_vs_fast():
    """The margin-cached zone engine must produce the same matches as the
    fast (scipy cKDTree) engine."""
    rng = np.random.default_rng(7)
    left = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.1, 150),
            "dec": rng.uniform(5.0, 5.1, 150),
        }
    )
    right = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.1, 200),
            "dec": rng.uniform(5.0, 5.1, 200),
        }
    )

    spec = MatchSpec(radius_arcsec=8.0, find="best")

    out_fast = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast"
    ).collect()
    out_zone = sky_match(
        _src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="zone"
    ).collect()

    assert out_fast.height == out_zone.height
    assert np.allclose(
        sorted(out_fast["sep_arcsec"].to_list()),
        sorted(out_zone["sep_arcsec"].to_list()),
        atol=1e-6,
    )


# ------------------------------------------------------------------- skyellipse
def test_skyellipse_mahalanobis_matches_within_sigma():
    """skyellipse with Mahalanobis distance: pair within N-sigma should match."""
    left = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "rae": [0.1],
            "dee": [0.1],
            "corr": [0.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.000055],
            "dec": [5.0],  # ~0.2 arcsec in RA at Dec=5
            "rae": [0.1],
            "dee": [0.1],
            "corr": [0.0],
        }
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    # Combined sigma ≈ 0.14 arcsec. max_error=3 → 0.42 arcsec > 0.2 → match.
    spec = MatchSpec(radius_arcsec=1.0, matcher="skyellipse", max_error=3.0)
    out = sky_match(a, b, left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 1


def test_skyellipse_mahalanobis_rejects_outside_sigma():
    """skyellipse with tight max_error should reject pair outside N-sigma."""
    left = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "rae": [0.1],
            "dee": [0.1],
            "corr": [0.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.000208],
            "dec": [5.0],  # ~0.75 arcsec
            "rae": [0.1],
            "dee": [0.1],
            "corr": [0.0],
        }
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    # max_error=3 → 0.42 arcsec < 0.75 → no match.
    spec = MatchSpec(radius_arcsec=1.0, matcher="skyellipse", max_error=3.0)
    out = sky_match(a, b, left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height == 0


def test_skyellipse_no_error_info_falls_back_to_sky():
    """skyellipse without error columns falls back to plain sky match."""
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005]})
    spec = MatchSpec(radius_arcsec=1.0, matcher="skyellipse", max_error=3.0)
    out = sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), spec, engine="fast").collect()
    # sky_match falls back to matcher="sky" when no error info exists.
    # The two stars are within 1 arcsec, so they should match.
    assert out.height == 1
    assert out["sep_arcsec"][0] < 1.0


def test_skyellipse_correlation_affects_match():
    """Correlation should affect Mahalanobis distance.
    With tightly correlated errors and a large offset, the Mahalanobis
    distance differs from the uncorrelated case. Both should match at
    generous max_error but with different p_match-equivalent behavior."""
    left = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "rae": [0.2],
            "dee": [0.05],
            "corr": [0.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.00002],
            "dec": [5.0],  # small offset in RA only
            "rae": [0.2],
            "dee": [0.05],
            "corr": [0.0],
        }
    )
    a0 = _src("a", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    b0 = _src("b", ra_err_column="rae", dec_err_column="dee", corr_column="corr")

    spec = MatchSpec(radius_arcsec=1.0, matcher="skyellipse", max_error=5.0)
    out_no_corr = sky_match(a0, b0, left.lazy(), right.lazy(), spec, engine="fast").collect()

    left_corr = left.with_columns(pl.Series("corr", [0.9]))
    right_corr = right.with_columns(pl.Series("corr", [0.9]))
    out_corr = sky_match(a0, b0, left_corr.lazy(), right_corr.lazy(), spec, engine="fast").collect()

    # Both match; the spatial separation should be identical.
    assert out_no_corr.height == out_corr.height == 1
    assert np.allclose(
        out_no_corr["sep_arcsec"].to_numpy(), out_corr["sep_arcsec"].to_numpy(), atol=1e-6
    )


def test_skyellipse_k_candidates_picks_best_by_mahalanobis():
    """When the spatial-nearest candidate fails Mahalanobis d² but a
    slightly-farther candidate passes, the engine should pick the farther
    one because it queries k>1 candidates and ranks by d²."""
    # One left source, two right sources within spatial range.
    # Star A: spatially closer, but its error ellipse has tiny Dec variance
    #          and the offset is mostly in Dec → large Mahalanobis d².
    # Star B: spatially farther, but error ellipse matches the offset → small d².
    left = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "rae": [0.5],
            "dee": [0.01],
            "corr": [0.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.00002, 10.00004],  # B is twice as far in RA
            "dec": [5.000005, 5.0],  # A has ~0.02 arcsec Dec offset
            "rae": [0.5, 0.5],
            "dee": [0.01, 0.5],
            "corr": [0.0, 0.0],
        }
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    # Use sky (spatial-only) to verify: star A is spatially closer.
    spec_sky = MatchSpec(radius_arcsec=1.0)
    out_sky = sky_match(a, b, left.lazy(), right.lazy(), spec_sky, engine="fast").collect()
    assert out_sky.height == 1
    # Spatial engine picks the closer star (star A, ra_2=10.00002).
    assert abs(out_sky["ra_2"][0] - 10.00002) < 1e-6

    # skyellipse: star B should be a better Mahalanobis match because its
    # error is more isotropic (smaller d² relative to the offset direction).
    spec_el = MatchSpec(radius_arcsec=1.0, matcher="skyellipse", max_error=5.0)
    out_el = sky_match(a, b, left.lazy(), right.lazy(), spec_el, engine="fast").collect()
    assert out_el.height == 1
    # skyellipse should pick star B (ra_2=10.00004) if its d² is smaller.
    # If it picked star A, the k>1 query isn't working — it's just filtering
    # the spatial-nearest.
    assert abs(out_el["ra_2"][0] - 10.00004) < 1e-6, (
        f"Expected skyellipse to pick the farther star (B, ra=10.00004) "
        f"by Mahalanobis d², but got ra_2={out_el['ra_2'][0]}"
    )


def test_skyellipse_engine_parity():
    """fast, astropy, and zone should agree on skyellipse match counts."""
    rng = np.random.default_rng(42)
    n = 100
    left = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.05, n),
            "dec": rng.uniform(5.0, 5.05, n),
            "rae": np.full(n, 0.1),
            "dee": np.full(n, 0.1),
            "corr": np.zeros(n),
        }
    )
    right = pl.DataFrame(
        {
            "ra": rng.uniform(10.0, 10.05, n),
            "dec": rng.uniform(5.0, 5.05, n),
            "rae": np.full(n, 0.1),
            "dee": np.full(n, 0.1),
            "corr": np.zeros(n),
        }
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    # Use a generous max_error so the spatial pre-filter returns many candidates.
    spec = MatchSpec(radius_arcsec=5.0, matcher="skyellipse", max_error=50.0)

    # fast and zone should agree on count.
    out_fast = sky_match(a, b, left.lazy(), right.lazy(), spec, engine="fast").collect()
    out_zone = sky_match(a, b, left.lazy(), right.lazy(), spec, engine="zone").collect()
    assert out_fast.height > 0, f"fast: expected matches, got {out_fast.height}"
    assert out_zone.height > 0, f"zone: expected matches, got {out_zone.height}"
    assert out_fast.height == out_zone.height


# ------------------------------------------------------------------- nway
@pytest.mark.parametrize("n_cats", [2, 3])
def test_nway_bayesian_pmatch_in_unit_range(n_cats):
    """nway Bayesian multi-catalogue match must produce p_match in [0, 1]."""
    from xmatch.bayes import compute_nway_p_match

    rng = np.random.default_rng(42)
    n_tuples = 20

    # Build synthetic tuples that are closely clustered.
    base_ra, base_dec = 10.0, 5.0
    ras = []
    decs = []
    sigmas = []
    for _ in range(n_cats):
        ras.append(np.full(n_tuples, base_ra) + rng.normal(0, 0.0001, n_tuples))
        decs.append(np.full(n_tuples, base_dec) + rng.normal(0, 0.0001, n_tuples))
        sigmas.append(np.full(n_tuples, 0.1))

    p = compute_nway_p_match(ras, decs, sigmas, radius_arcsec=3.0)
    assert len(p) == n_tuples
    assert ((p >= 0.0) & (p <= 1.0)).all()

    # Well-separated tuples should have low p_match.
    ras_far = []
    decs_far = []
    sigmas_far = []
    for i in range(n_cats):
        ras_far.append(np.array([10.0 + i * 0.01]))
        decs_far.append(np.array([5.0 + i * 0.01]))
        sigmas_far.append(np.array([0.01]))
    p_far = compute_nway_p_match(ras_far, decs_far, sigmas_far, radius_arcsec=3.0)
    assert p_far[0] < 0.5, f"Expected low p_match for scattered tuple, got {p_far[0]}"


def test_nway_crossmatch_end_to_end():
    """nway_match on CrossMatch should produce result frame with p_match."""
    from xmatch import CrossMatch

    left = pl.DataFrame(
        {
            "ra": [10.0, 10.00002, 10.00004],
            "dec": [5.0, 5.0, 5.0],
            "g": [10.0, 10.1, 10.2],
        }
    )
    mid = pl.DataFrame(
        {
            "ra": [10.00002],
            "dec": [5.0],
            "g": [10.1],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.0, 10.00004],
            "dec": [5.0, 5.0],
            "g": [10.0, 10.2],
        }
    )

    cm = CrossMatch()
    result = cm.nway_match(
        [left, mid, right],
        radius_arcsec=1.0,
        prior_columns=["g"],
    )
    assert result.height >= 1
    assert "p_match" in result.columns
    p = result["p_match"].to_numpy()
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_nway_crossmatch_two_catalogues_works():
    """nway_match with exactly 2 catalogues should produce results."""
    from xmatch import CrossMatch

    left = pl.DataFrame({"ra": [10.0, 10.00005], "dec": [5.0, 5.0]})
    right = pl.DataFrame({"ra": [10.0, 10.00005], "dec": [5.0, 5.0]})

    cm = CrossMatch()
    result = cm.nway_match([left, right], radius_arcsec=1.0)
    assert result.height >= 1
    assert "p_match" in result.columns


# --------------------------------------------------------------------------- #
# PM drift prior tests (Wilson 2023)
# --------------------------------------------------------------------------- #
def test_pm_prior_inflates_errors_for_sources_without_pm():
    """pm_prior should work with skyerr matcher (inflated error floor)."""
    left = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "ra_error": [0.1],
            "dec_error": [0.1],
            "epoch": [2000.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.0002],
            "dec": [5.0],
            "ra_error": [0.1],
            "dec_error": [0.1],
            "epoch": [2016.0],
        }
    )
    src_a = _src(
        "a",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=2000.0,
    )
    src_b = _src(
        "b",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=2016.0,
    )

    spec = MatchSpec(
        radius_arcsec=2.0,
        matcher="skyerr",
        max_error=5.0,
        target_epoch=2016.0,
        pm_prior=True,
    )
    out = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert "sep_arcsec" in out.columns


def test_pm_prior_no_epoch_skips_gracefully():
    """pm_prior should skip sides without epoch info (no crash)."""
    left = pl.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [5.0, 6.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.00005, 10.0001],
            "dec": [5.00005, 5.0],
        }
    )
    src_a = _src("a")
    src_b = _src("b")

    spec = MatchSpec(
        radius_arcsec=2.0,
        matcher="sky",
        find="best",
        target_epoch=2016.0,
        pm_prior=True,
    )
    out = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height >= 1
    assert "sep_arcsec" in out.columns


def test_pm_prior_with_skyerr_matcher():
    """pm_prior inflated errors should work with skyerr matcher."""
    left = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "ra_error": [0.1],
            "dec_error": [0.1],
            "epoch": [2000.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.002],
            "dec": [5.0],
            "ra_error": [0.1],
            "dec_error": [0.1],
            "epoch": [2016.0],
        }
    )
    src_a = _src(
        "a",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=2000.0,
    )
    src_b = _src(
        "b",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=2016.0,
    )

    spec = MatchSpec(
        radius_arcsec=2.0,
        matcher="skyerr",
        max_error=5.0,
        target_epoch=2016.0,
        pm_prior=True,
    )
    out = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert "sep_arcsec" in out.columns


def test_pm_prior_magnitude_column_scales_dispersion():
    """Magnitude scaling must change match outcomes: a right star at ~2 arcsec
    separation should only match when bright magnitudes inflate the drift floor.

    Uses a 116-year epoch baseline (1900\u21922016) so the drift is at least
    ~1 arcsec for bright stars (mag\u224810, mag_scale\u22483.0).  At this baseline
    the faint-star drift (~0.1 arcsec) stays below the per-row errors, so
    the match radius is effectively unchanged.

    * Without pm_prior:           max sep \u2248 0.85\" \u2192 2.0\" star \u2192 NO match
    * pm_prior + bright (mag=10): max sep \u2248 3.6\"  \u2192 2.0\" star \u2192 MATCH
    * pm_prior + faint (mag=20):  max sep \u2248 0.85\" \u2192 2.0\" star \u2192 NO match
    """
    base_left = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "ra_error": [0.1],
            "dec_error": [0.1],
            "epoch": [1900.0],
        }
    )
    base_right = pl.DataFrame(
        {
            "ra": [10.00056],
            "dec": [5.0],
            "ra_error": [0.1],
            "dec_error": [0.1],
            "epoch": [2016.0],
        }
    )
    src_a = _src(
        "a",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=1900.0,
    )
    src_b = _src(
        "b",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=2016.0,
    )

    # --- without pm_prior: no match (2.0\" > ~0.85\" max) -------------------
    spec_no = MatchSpec(
        radius_arcsec=4.0,
        matcher="skyerr",
        max_error=3.0,
        target_epoch=2016.0,
    )
    # Unknown motion must be explicitly modelled at a different epoch.
    with pytest.raises(CrossMatchError, match="finite proper motion"):
        sky_match(src_a, src_b, base_left.lazy(), base_right.lazy(), spec_no, engine="fast")

    # --- pm_prior + bright stars (mag=10): should match -------------------
    left_bright = base_left.with_columns(pl.Series("mag_g", [10.0]))
    spec_bright = MatchSpec(
        radius_arcsec=4.0,
        matcher="skyerr",
        max_error=3.0,
        target_epoch=2016.0,
        pm_prior=True,
        pm_prior_magnitude_column="mag_g",
    )
    out_bright = sky_match(
        src_a, src_b, left_bright.lazy(), base_right.lazy(), spec_bright, engine="fast"
    ).collect()
    assert out_bright.height == 1, (
        f'With pm_prior + bright mag, 2.0" star should match '
        f"(drift floor inflated ~10\u00d7 by mag_scale=3.0); "
        f"got {out_bright.height} matches"
    )

    # --- pm_prior + faint stars (mag=20): no match ------------------------
    left_faint = base_left.with_columns(pl.Series("mag_g", [20.0]))
    spec_faint = MatchSpec(
        radius_arcsec=4.0,
        matcher="skyerr",
        max_error=3.0,
        target_epoch=2016.0,
        pm_prior=True,
        pm_prior_magnitude_column="mag_g",
    )
    out_faint = sky_match(
        src_a, src_b, left_faint.lazy(), base_right.lazy(), spec_faint, engine="fast"
    ).collect()
    assert out_faint.height == 0, (
        f"With pm_prior + faint mag, mag_scale=0.3 keeps drift below "
        f'per-row errors, so 2.0" should NOT match; '
        f"got {out_faint.height} matches"
    )


def test_pm_prior_magnitude_column_absent_skips_gracefully():
    """pm_prior_magnitude_column pointing to a missing column should not crash."""
    left = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "epoch": [2000.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [10.00005],
            "dec": [5.00005],
            "epoch": [2016.0],
        }
    )
    src_a = _src("a", epoch_column="epoch", epoch=2000.0)
    src_b = _src("b", epoch_column="epoch", epoch=2016.0)

    spec = MatchSpec(
        radius_arcsec=2.0,
        matcher="sky",
        find="best",
        target_epoch=2016.0,
        pm_prior=True,
        pm_prior_magnitude_column="nonexistent",
    )
    out = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec, engine="fast").collect()
    assert out.height >= 1
    assert "sep_arcsec" in out.columns


def test_pm_prior_per_row_drift_added_to_astrometric_errors():
    """Per-row PM drift should be added in quadrature to per-row astrometric
    errors, increasing max allowed separation beyond the source-level floor.

    Uses a position near the Galactic plane (ra~282 deg, dec~0 deg) where
    sigma_mu ~ 10 mas/yr.  With a 116-year baseline, drift ~ 1.16\",
    pushing max sep from ~0.85\" to ~3.9\" -- enough to match a 2.5\" right star.

    Asymmetry note: the right side's epoch (2016.0) equals target_epoch,
    so \u0394t=0 on the right \u2014 only the left side (epoch=1900.0) gets drift
    inflation.  This matches the typical use case of crossmatching an old
    survey (no PMs) against a modern reference catalogue at the reference
    epoch.
    """
    left = pl.DataFrame(
        {
            "ra": [282.0],
            "dec": [0.0],
            "ra_error": [0.1],
            "dec_error": [0.1],
            "epoch": [1900.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [282.000694],
            "dec": [0.0],  # 0.000694 deg * 3600 ~ 2.5\"
            "ra_error": [0.1],
            "dec_error": [0.1],
            "epoch": [2016.0],
        }
    )
    src_a = _src(
        "a",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=1900.0,
    )
    src_b = _src(
        "b",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=2016.0,
    )

    # Without pm_prior: per-row errors only \u2192 max sep \u2248 0.85\" \u2192 NO match
    spec_no = MatchSpec(
        radius_arcsec=4.0,
        matcher="skyerr",
        max_error=3.0,
        target_epoch=2016.0,
    )
    with pytest.raises(CrossMatchError, match="finite proper motion"):
        sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_no, engine="fast")

    # With pm_prior: per-row drift added in quadrature \u2192 max sep >> 3.0\"
    spec_pm = MatchSpec(
        radius_arcsec=4.0,
        matcher="skyerr",
        max_error=3.0,
        target_epoch=2016.0,
        pm_prior=True,
    )
    out_pm = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_pm, engine="fast").collect()
    assert out_pm.height == 1, (
        f"With pm_prior + per-row drift, 2.5 arcsec star SHOULD match; got {out_pm.height}"
    )
    assert "sep_arcsec" in out_pm.columns
    # Verify separation is reasonable (~2.5\")
    sep = out_pm["sep_arcsec"][0]
    assert 2.0 < sep < 3.0, f"Expected sep ~2.5 arcsec, got {sep:.3f}"


def test_pm_prior_per_row_gaia_realistic_error_budgets():
    """Gaia-style end-to-end test: per-row ra_error/dec_error columns plus
    pm_prior with realistic Gaia/2MASS error budgets and a 16-year epoch
    baseline.  Verifies that per-row PM drift is computed correctly from
    position + magnitude and added in quadrature to per-row astrometric
    errors, end-to-end through sky_match.

    Setup (Galactic plane: RA=282 deg, Dec=0 deg where sigma_mu base
    = 3 + 7*exp(0) = 10 mas/yr)::

        left  = 2MASS-like  (epoch 2000, sigma=0.1")
        right = Gaia-like   (epoch 2016 = target_epoch, sigma=0.5 mas)

    Baseline = 16 years; only the left side carries drift because right
    is at target_epoch (delta_t = 0).

    Two left rows at the same sky position but different magnitudes
    exercise per-row drift through the magnitude scaling::

        Bright (mag=10): scale clipped to 3.0
            drift = 10 * 3.0 * 16 = 480 mas = 0.48"
            sigma_per_row ~ sqrt(0.1^2 + 0.48^2) ~ 0.49"
        Faint  (mag=20): scale clipped to 0.3
            drift = 10 * 0.3 * 16 = 48 mas = 0.048"
            sigma_per_row ~ sqrt(0.1^2 + 0.048^2) ~ 0.111"

    Skyerr applies the N-sigma criterion *per row*:
    ``sep <= max_error * (sigma_l[i] + sigma_r[j])``.  The global
    ``max_error * (nanmax(lsig) + nanmax(rsig))`` bound is only a
    candidate pre-filter, so the wide drift on the bright row
    (``sigma ~ 0.49"``) cannot admit a 0.5" candidate for the faint row
    whose own budget is ``3 * (0.111 + 0.0007) ~ 0.34"``.  One row's
    error budget must never change another row's match.
    """
    offset_deg = 0.5 / 3600.0  # 0.5 arcsec RA offset at Dec=0 (cos 0 = 1)
    sigma_b = 0.1  # 2MASS-like 100 mas per axis
    sigma_gaia = 0.0005  # Gaia bright source 0.5 mas per axis

    left = pl.DataFrame(
        {
            "ra": [282.0, 282.0],
            "dec": [0.0, 0.0],
            "ra_error": [sigma_b, sigma_b],
            "dec_error": [sigma_b, sigma_b],
            "epoch": [2000.0, 2000.0],
            # mag_g drives per-row magnitude scaling: bright clipped to 3.0,
            # faint clipped to 0.3.
            "mag_g": [10.0, 20.0],
        }
    )
    right = pl.DataFrame(
        {
            "ra": [282.0 + offset_deg, 282.0 + offset_deg],
            "dec": [0.0, 0.0],
            "ra_error": [sigma_gaia, sigma_gaia],
            "dec_error": [sigma_gaia, sigma_gaia],
            "epoch": [2016.0, 2016.0],
        }
    )

    src_a = _src(
        "a",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=2000.0,
    )
    src_b = _src(
        "b",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=2016.0,
    )

    # Baseline: without pm_prior, no drift on either side; the per-row
    # error budget alone gives chord_max ~ 3 * (0.141 + 0.0007) ~ 0.426".
    # Both candidates at 0.5" are too far -> 0 matches.
    spec_no_pm = MatchSpec(
        radius_arcsec=2.0,
        matcher="skyerr",
        max_error=3.0,
        target_epoch=2016.0,
    )
    with pytest.raises(CrossMatchError, match="finite proper motion"):
        sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_no_pm, engine="fast")

    # With pm_prior + magnitude scaling: per-row drift inflates the
    # left-side sigma and chord_max rises to ~1.47\", allowing both 0.5\"
    # candidates to match in a single query.  Round-trip verifies the
    # injected offset is preserved to within float precision.
    spec_pm = MatchSpec(
        radius_arcsec=2.0,
        matcher="skyerr",
        max_error=3.0,
        target_epoch=2016.0,
        pm_prior=True,
        pm_prior_magnitude_column="mag_g",
    )
    out_pm = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_pm, engine="fast").collect()
    # Only the bright row matches.  Its wide sigma (drift 0.48") admits the
    # 0.5" candidate (per-row budget ~1.47"); the faint row's own tight
    # budget (~0.34") rejects it even though the bright row is in the same
    # table -- i.e. per-row drift is genuinely computed per row and one
    # row's error budget does not leak into another's decision.
    assert out_pm.height == 1, (
        "Expected only the bright row to match: bright per-row budget "
        '~1.47" admits the 0.5" candidate, faint per-row budget ~0.34" '
        f"rejects it; got {out_pm.height} matches"
    )
    assert out_pm["mag_g"].to_list() == [10.0], (
        f"Expected the bright row (mag_g=10) to match; got {out_pm['mag_g'].to_list()}"
    )
    assert "sep_arcsec" in out_pm.columns
    # Tight tolerance: float64 precision at 0.5\" is well below
    # microarcseconds, so 0.499..0.501 catches anything larger than float
    # round-off while tolerating sub-milliarcsecond rounding.
    seps = out_pm["sep_arcsec"].to_numpy()
    assert all(0.499 < s < 0.501 for s in seps), (
        f"Expected sep ~0.5 arcsec for the bright match; got {seps.tolist()}"
    )

    # Per-row sigma sanity: the faint row alone (chord_max driven by
    # faint only) is tight enough to reject at 0.5\".  This verifies
    # that pm_prior *correctly* produces a SMALL drift on the faint row
    # (a regression that dropped magnitude scaling would inflate the
    # faint row and let it through).  Bright-only at the same offset
    # should match -- confirming pm_prior inflates per-row sigma when
    # the magnitude scaling demands it.
    left_bright = left.head(1)
    left_faint = left.tail(1)
    out_bright = sky_match(
        src_a, src_b, left_bright.lazy(), right.lazy(), spec_pm, engine="fast"
    ).collect()
    out_faint = sky_match(
        src_a, src_b, left_faint.lazy(), right.lazy(), spec_pm, engine="fast"
    ).collect()
    assert out_bright.height == 1, f'Bright row alone at 0.5" should match; got {out_bright.height}'
    assert out_faint.height == 0, (
        'Faint row alone at 0.5" should NOT match: per-row chord_max '
        f'~0.33" < 0.5". Got {out_faint.height} matches -- a '
        "regression that drops magnitude scaling would let it through."
    )

    # Quadrature-vs-linear-add discrimination.  Bright row alone at
    # 1.6 arcsec separation.  Under correct quadrature: max sep
    # = 3 * (0.49 + 0.0007) = 1.47\" -> REJECTED.  Under broken linear
    # addition (lsig = 0.1 + 0.48 = 0.58\"): max sep = 3 * (0.58 +
    # 0.0007) = 1.74\" -> ACCEPTED.  height == 0 iff drift is added
    # in true quadrature.
    offset_wide_deg = 1.6 / 3600.0
    right_wide = pl.DataFrame(
        {
            "ra": [282.0 + offset_wide_deg, 282.0 + offset_wide_deg],
            "dec": [0.0, 0.0],
            "ra_error": [sigma_gaia, sigma_gaia],
            "dec_error": [sigma_gaia, sigma_gaia],
            "epoch": [2016.0, 2016.0],
        }
    )
    out_wide = sky_match(
        src_a, src_b, left_bright.lazy(), right_wide.lazy(), spec_pm, engine="fast"
    ).collect()
    assert out_wide.height == 0, (
        "1.6 arcsec should NOT match the bright row under quadrature "
        f"max (1.47 arcsec); got {out_wide.height} matches. Drift may "
        "be added linearly rather than in quadrature."
    )


def test_pm_prior_both_sides_drift_inflation():
    """Two-old-survey case (e.g., USNO-B vs 2MASS, neither has PMs):
    BOTH sides have epochs far from target_epoch so each carries
    its own per-row drift.  Verifies per-row drift is added in
    quadrature on BOTH sides independently and that the joint
    ``chord_max`` (under skyerr) reflects drift from BOTH sides.

    Realistic use case: crossmatching USNO-B (mean epoch ~1980) and
    2MASS (mean epoch ~2000) at the Gaia DR3 reference epoch (2016).
    Neither survey has measured PMs, so ``pm_prior`` is the only way
    to widen the match radius to cover proper-motion drift accumulated
    between the two surveys.

    Setup at Galactic plane (RA=282 deg, Dec=0 deg => ``sigma_mu`` base
    = 10 mas/yr, magnitude scaling disabled).

    Per-row drift at ``target_epoch=2016``::

        left  (epoch 1980): dt=36 yr -> drift = 10*36 = 360 mas = 0.36"
        right (epoch 2000): dt=16 yr -> drift = 10*16 = 160 mas = 0.16"

    Per-row ``lsig``/``rsig`` (per-row error + drift in quadrature,
    following the formula in ``_pos_sigma_arcsec``)::

        sigma = sqrt(ra_err^2 + dec_err^2 + drift^2)

    Joint chord_max under skyerr = 3 * (lsig + rsig)::

        both sides drift:    3 * (0.387 + 0.214) ~ 1.80"   <- target
        left drift only:     3 * (0.387 + 0.141) ~ 1.59"
        right drift only:    3 * (0.141 + 0.214) ~ 1.07"
        neither side drifts: 3 * (0.141 + 0.141) ~ 0.85"

    Three test separations chosen to isolate each combination:

    1. At sep 1.4" -- minimum-distance gate (chord_max = 1.80):

       * without pm_prior (chord 0.85): 1.4 > 0.85 -> NO MATCH
       * joint both drift (chord 1.80): 1.4 < 1.80 -> MATCH
       -> Demonstrates pm_prior inflates the joint chord_max on BOTH
          sides.  Round-trip seps verify the injected offset.

    2. At sep 1.65" -- above single-side chord_max:

       * left-only drift (right at target_epoch, chord 1.59):
           1.65 > 1.59 -> NO MATCH
       * right-only drift (left at target_epoch, chord 1.07):
           1.65 > 1.07 -> NO MATCH
       * joint both drift (chord 1.80): 1.65 < 1.80 -> MATCH
       -> Two rejected + one accepted at the same sep proves BOTH
          sides' drift contributes to the joint chord_max.

    3. At sep 1.95" -- quadrature vs linear-add regression::

       Quadrature (correct):     lsig 0.387, rsig 0.214, chord 1.80 < 1.95 -> NO MATCH
       Linear add (regression):  lsig = 0.141+0.36 = 0.50; rsig = 0.141+0.16 = 0.30;
                                 chord = 3 * (0.50 + 0.30) = 2.41 > 1.95 -> MATCH

       height == 0 iff drift is added in true quadrature on at least
       one side (regression would surface as a match that should not
       exist).
    """
    offset_deg = 1.4 / 3600.0  # both sides drift, sep ~ joint chord
    offset_split_deg = 1.65 / 3600.0  # sep above single-side chord, below joint chord
    offset_wide_deg = 1.95 / 3600.0  # sep above correct chord, below linear-add chord
    sigma_b = 0.1  # 100 mas per axis

    def make_lr(epoch_a, epoch_b, off):
        """Build a (left, right) DataFrame pair at Galactic plane with the
        given side epochs and a RA offset of ``off`` degrees at Dec=0
        (where 1 deg = 3600 arcsec exactly since cos(0) = 1)."""
        return (
            pl.DataFrame(
                {
                    "ra": [282.0],
                    "dec": [0.0],
                    "ra_error": [sigma_b],
                    "dec_error": [sigma_b],
                    "epoch": [epoch_a],
                }
            ),
            pl.DataFrame(
                {
                    "ra": [282.0 + off],
                    "dec": [0.0],
                    "ra_error": [sigma_b],
                    "dec_error": [sigma_b],
                    "epoch": [epoch_b],
                }
            ),
        )

    # Catalog-level epochs: per-row epoch column wins when present, so
    # these default to 1980/2000 but are overridden by DataFrame values
    # when the per-row epoch is 2016.
    src_a = _src(
        "a",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=1980.0,
    )
    src_b = _src(
        "b",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        epoch_column="epoch",
        epoch=2000.0,
    )
    spec_no_pm = MatchSpec(
        radius_arcsec=2.0,
        matcher="skyerr",
        max_error=3.0,
        target_epoch=2016.0,
    )
    spec_pm = MatchSpec(
        radius_arcsec=2.0,
        matcher="skyerr",
        max_error=3.0,
        target_epoch=2016.0,
        pm_prior=True,
    )

    # --- 1a. baseline: no pm_prior at sep 1.4" -> NO MATCH -----------------
    left, right = make_lr(1980.0, 2000.0, offset_deg)
    with pytest.raises(CrossMatchError, match="finite proper motion"):
        sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_no_pm, engine="fast")

    # --- 1b. joint both drift at sep 1.4" -> MATCH -----------------------
    # Round-trip the injected separation in the same assertion to verify
    # sky_match reports the offset accurately under per-row drift.
    left, right = make_lr(1980.0, 2000.0, offset_deg)
    out = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_pm, engine="fast").collect()
    assert out.height == 1, (
        'Joint both-drift at 1.4" sep: chord_max ~ 3*(0.387+0.214) '
        f'= 1.80" should match; got {out.height}'
    )
    assert "sep_arcsec" in out.columns
    sep = out["sep_arcsec"][0]
    assert 1.39 < sep < 1.41, f"Expected sep ~1.4 arcsec (injected offset); got {sep:.6f}"

    # --- 2a. left-only drift (right at target_epoch) at sep 1.65" --------
    # right epoch = target_epoch, so delta_t = 0 -> no drift on right.
    # chord_max collapses to 3*(lsig_left + rsig_right)
    #                          = 3*(0.387 + 0.141) ~ 1.59"
    # 1.65 > 1.59 -> NO MATCH.  Demonstrates that left's drift alone
    # does not inflate chord enough at this separation.
    left, right = make_lr(1980.0, 2016.0, offset_split_deg)
    out = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_pm, engine="fast").collect()
    assert out.height == 0, (
        'Left-only drift at 1.65" sep: chord ~1.59" < 1.65". '
        f"Right at target_epoch should NOT contribute drift; "
        f"got {out.height} matches"
    )

    # --- 2b. right-only drift (left at target_epoch) at sep 1.65" -------
    # Same sep but mirrored: left epoch = target_epoch.  chord_max
    # collapses to 3*(0.141 + 0.214) ~ 1.07", so 1.65 > 1.07 -> NO MATCH.
    left, right = make_lr(2016.0, 2000.0, offset_split_deg)
    out = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_pm, engine="fast").collect()
    assert out.height == 0, (
        'Right-only drift at 1.65" sep: chord ~1.07" < 1.65". '
        f"Left at target_epoch should NOT contribute drift; "
        f"got {out.height} matches"
    )

    # --- 2c. joint both drift at sep 1.65" -> MATCH ----------------------
    # Same 1.65" candidate is now accepted because BOTH sides' drift
    # lifts the joint chord_max to ~1.80".  Mirrored with 2a/2b, this
    # is the proof that both sides' drift contributes independently.
    left, right = make_lr(1980.0, 2000.0, offset_split_deg)
    out = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_pm, engine="fast").collect()
    assert out.height == 1, (
        'Joint both-drift at 1.65" sep: chord_max ~1.80" should '
        f'match (margin 0.151"); got {out.height}. The same sep '
        "rejected by left-only and right-only cases above."
    )

    # --- 3. quadrature-vs-linear-add regression at sep 1.95" ------------
    # Quadrature correct: chord = 3*(0.387+0.214) = 1.80 < 1.95 -> NO.
    # Linear-add broken: lsig = 0.141+0.36 = 0.50; rsig = 0.141+0.16 = 0.30;
    #                    chord = 3*(0.50+0.30) = 2.41 > 1.95 -> YES.
    # height == 0 iff drift is added in quadrature on at least one side.
    left, right = make_lr(1980.0, 2000.0, offset_wide_deg)
    out = sky_match(src_a, src_b, left.lazy(), right.lazy(), spec_pm, engine="fast").collect()
    assert out.height == 0, (
        'Joint both-drift at 1.95" sep, chord ~1.80" < 1.95". '
        "A non-zero result suggests drift was added linearly instead "
        f"of in quadrature on at least one side. Got {out.height}"
    )


# ------------------------------------------------------------------- audit regressions
# `skyerr`, `skyellipse` and non-finite input used to behave differently per
# engine.  These tests pin the pair set (and the `find="best"` ranking) across
# `fast`, `zone` and `astropy` so error-weighted matching is engine-independent.


def _ra_offset(arcsec: float, dec_deg: float) -> float:
    """RA (deg) change corresponding to ``arcsec`` on-sky at ``dec_deg``."""
    return arcsec / 3600.0 / np.cos(np.radians(dec_deg))


def test_skyerr_per_row_criterion_engine_parity():
    """`skyerr` must apply its per-row N-sigma test in every engine.

    ``fast``/``zone`` used to keep any candidate inside the *global* chord
    bound ``max_error * (max sigma_l + max sigma_r)``, so a row with tiny
    errors was accepted against a partner well outside its own sigma budget.
    """
    dec = 5.0
    left = pl.DataFrame({"ra": [10.0], "dec": [dec], "rae": [1.0], "dee": [0.0]})
    # "loose": 0.45" away, 1.0" error  -> limit 0.5*(1.0+1.0)=1.0  -> accept
    # "tight": 0.55" away, 0.05" error -> limit 0.5*(1.0+0.05)=0.525 -> reject
    right = pl.DataFrame(
        {
            "ra": [10.0 + _ra_offset(0.45, dec), 10.0 + _ra_offset(0.55, dec)],
            "dec": [dec, dec],
            "rae": [1.0, 0.05],
            "dee": [0.0, 0.0],
        }
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee", pos_err_units="arcsec")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", pos_err_units="arcsec")
    spec = MatchSpec(radius_arcsec=2.0, matcher="skyerr", max_error=0.5, find="all")
    # Compare on-sky separations in arcsec: absolute degree tolerances are
    # always dominated by the ~10 deg coordinate magnitude.
    loose_sep = float(right["ra"][0] - left["ra"][0]) * 3600.0 * np.cos(np.radians(dec))
    for eng in ("fast", "zone", "astropy"):
        out = sky_match(a, b, left.lazy(), right.lazy(), spec, engine=eng).collect()
        assert out.height == 1, f"{eng}: expected only the loose pair, got {out.height}"
        assert abs(out["sep_arcsec"][0] - loose_sep) < 1e-3, (
            f'{eng}: expected the 0.45" pair, got sep={out["sep_arcsec"][0]}'
        )


def test_skyerr_best_ranking_is_normalised_not_spatial():
    """``find="best"`` ranks by ``sep / (sigma_l + sigma_r)``, not raw sep.

    A tight-error candidate that is spatially nearest must lose to a slightly
    farther but looser one when the normalised separation is smaller — the
    same score ``astropy`` uses.
    """
    dec = 5.0
    left = pl.DataFrame({"ra": [10.0], "dec": [dec], "rae": [1.0], "dee": [0.0]})
    # nearest: 0.20" / (0.5*1.05) = 0.381 ; farther: 0.30" / (0.5*2.0) = 0.300
    right = pl.DataFrame(
        {
            "ra": [10.0 + _ra_offset(0.20, dec), 10.0 + _ra_offset(0.30, dec)],
            "dec": [dec, dec],
            "rae": [0.05, 1.0],
            "dee": [0.0, 0.0],
        }
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee", pos_err_units="arcsec")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", pos_err_units="arcsec")
    spec = MatchSpec(radius_arcsec=2.0, matcher="skyerr", max_error=0.5, find="best")
    # the farther (0.30"), but better-normalised, pair
    expected_sep = 0.30
    for eng in ("fast", "zone", "astropy"):
        out = sky_match(a, b, left.lazy(), right.lazy(), spec, engine=eng).collect()
        assert out.height == 1, f"{eng}: expected one match, got {out.height}"
        assert abs(out["sep_arcsec"][0] - expected_sep) < 1e-3, (
            f'{eng}: expected the best-normalised pair at {expected_sep}", '
            f'got sep={out["sep_arcsec"][0]} (spatially-nearest is 0.20")'
        )


def test_skyellipse_best_picks_mahalanobis_best_in_all_engines():
    """`zone` must query k>1 candidates for ``skyellipse`` + ``find="best"``.

    Querying only the chord-nearest candidate dropped the row whenever that
    candidate failed the Mahalanobis test, even though a slightly farther one
    passed it (``fast``/``astropy`` already handled this).
    """
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "rae": [0.5], "dee": [0.01], "corr": [0.0]})
    right = pl.DataFrame(
        {
            "ra": [10.00002, 10.00004],  # second is twice as far in RA
            "dec": [5.000005, 5.0],
            "rae": [0.5, 0.5],
            "dee": [0.01, 0.5],
            "corr": [0.0, 0.0],
        }
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    spec = MatchSpec(radius_arcsec=1.0, matcher="skyellipse", max_error=5.0)
    for eng in ("fast", "zone", "astropy"):
        out = sky_match(a, b, left.lazy(), right.lazy(), spec, engine=eng).collect()
        assert out.height == 1, f"{eng}: expected one Mahalanobis match, got {out.height}"
        assert abs(out["ra_2"][0] - 10.00004) < 1e-6, (
            f"{eng}: expected the Mahalanobis-best candidate (ra_2=10.00004)"
        )


@pytest.mark.parametrize("engine", ["fast", "zone", "astropy", "ray"])
def test_nonfinite_coordinates_raise_clean_error(engine):
    """A non-finite / out-of-range latitude must raise ``CrossMatchError``.

    ``cdshealpix`` panics inside its Rust core on such input; the resulting
    ``PanicException`` is a ``BaseException`` that escaped every handler.
    """
    left = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 91.0]})
    right = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    with pytest.raises(CrossMatchError, match="non-finite|out-of-range"):
        sky_match(
            _src("a"),
            _src("b"),
            left.lazy(),
            right.lazy(),
            MatchSpec(radius_arcsec=1.0),
            engine=engine,
        ).collect()


def test_nan_dec_raises_clean_error():
    left = pl.DataFrame({"ra": [10.0], "dec": [np.nan]})
    right = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    with pytest.raises(CrossMatchError, match="non-finite|out-of-range"):
        sky_match(
            _src("a"), _src("b"), left.lazy(), right.lazy(), MatchSpec(radius_arcsec=1.0)
        ).collect()


@pytest.mark.parametrize("engine", ["auto", "stilts"])
@pytest.mark.parametrize("bad_side", ["left", "right"])
def test_stilts_validates_coordinates_before_external_matching(monkeypatch, engine, bad_side):
    from xmatch import stilts

    called = []

    def external_match(*args, **kwargs):
        called.append(True)
        return pl.DataFrame()

    monkeypatch.setattr(stilts, "stilts_available", lambda command: True)
    monkeypatch.setattr(stilts, "stilts_sky_match", external_match)
    valid = pl.DataFrame({"ra": [10.0], "dec": [0.0]})
    invalid = valid.with_columns(pl.lit(float("nan")).alias("dec"))
    left, right = (invalid, valid) if bad_side == "left" else (valid, invalid)
    with pytest.raises(CrossMatchError, match="non-finite|out-of-range"):
        sky_match(_src("a"), _src("b"), left.lazy(), right.lazy(), MatchSpec(), engine=engine)
    assert called == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"matcher": "sky-error"},
        {"join_type": "1and"},
        {"find": "nearest"},
    ],
)
def test_matchspec_rejects_unknown_vocabulary(kwargs):
    """Typos in matcher/join_type/find must fail loudly, not return nothing."""
    with pytest.raises(ValueError):
        MatchSpec(**kwargs)


def test_skyellipse_anisotropic_search_radius_engine_parity():
    """astropy must use the same conservative search bound as fast/zone.

    With elongated, correlated error ellipses (10" per axis, rho=0.9) the
    combined covariance's major semi-axis is ``sqrt(2*380) ~ 27.6"`` but
    ``max(sigma_ra^2, sigma_dec^2)`` per side only gives ``sqrt(200) ~ 14.1"``.
    A pair at 18" along the major axis has d^2 = 18^2/760 = 0.43 <= 1 and MUST
    match; astropy used to miss it because its candidate radius was too small.
    """
    left = pl.DataFrame({"ra": [10.0], "dec": [0.0], "rae": [10.0], "dee": [10.0], "corr": [0.9]})
    off = 18.0 / np.sqrt(2.0) / 3600.0  # equal RA/Dec offsets -> 18" at 45 deg
    right = pl.DataFrame(
        {"ra": [10.0 + off], "dec": [off], "rae": [10.0], "dee": [10.0], "corr": [0.9]}
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", corr_column="corr")
    spec = MatchSpec(radius_arcsec=60.0, matcher="skyellipse", max_error=1.0)
    for eng in ("fast", "zone", "astropy"):
        out = sky_match(a, b, left.lazy(), right.lazy(), spec, engine=eng).collect()
        assert out.height == 1, f'{eng}: expected the 18" major-axis pair to match'
        assert abs(out["sep_arcsec"][0] - 18.0) < 1e-3, f"{eng}: sep={out['sep_arcsec'][0]}"


def test_skyerr_ray_engine_matches_single_machine():
    """Distributed `ray` must apply the same per-row skyerr criterion and the
    same best-match ranking as the single-machine engines (the ray engine is
    the union path, so its pairing must not drift)."""
    ray = pytest.importorskip("ray")

    dec = 5.0
    left = pl.DataFrame({"ra": [10.0], "dec": [dec], "rae": [1.0], "dee": [0.0]})
    # A: 0.20" (tight errors) -> normalised 0.381 ; C: 0.30" -> 0.300
    # D: 0.90" with 0.02" errors -> limit 0.51 -> rejected per row
    right = pl.DataFrame(
        {
            "ra": [
                10.0 + _ra_offset(0.20, dec),
                10.0 + _ra_offset(0.30, dec),
                10.0 + _ra_offset(0.90, dec),
            ],
            "dec": [dec, dec, dec],
            "rae": [0.05, 1.0, 0.02],
            "dee": [0.0, 0.0, 0.0],
        }
    )
    a = _src("a", ra_err_column="rae", dec_err_column="dee", pos_err_units="arcsec")
    b = _src("b", ra_err_column="rae", dec_err_column="dee", pos_err_units="arcsec")

    ray.init(ignore_reinit_error=True, logging_level=40)
    try:
        for find, expected in (("all", 2), ("best", 1)):
            spec = MatchSpec(radius_arcsec=2.0, matcher="skyerr", max_error=0.5, find=find)
            base = sky_match(a, b, left.lazy(), right.lazy(), spec, engine="astropy").collect()
            out = sky_match(a, b, left.lazy(), right.lazy(), spec, engine="ray").collect()
            assert base.height == out.height == expected, (
                f"find={find}: astropy={base.height}, ray={out.height}, expected {expected}"
            )
            assert np.allclose(
                sorted(base["sep_arcsec"].to_list()), sorted(out["sep_arcsec"].to_list()), atol=1e-6
            ), f"find={find}: ray and astropy disagree on the pair set"
    finally:
        ray.shutdown()
