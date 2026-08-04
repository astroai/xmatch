import astropy.units as u
import numpy as np
import polars as pl
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from xmatch import astro_utils
from xmatch.astro_utils import (
    find_coord_columns,
    propagate_proper_motion,
    propagate_proper_motion_with_jacobian,
    propagate_space_motion,
    propagate_space_motion_with_jacobian,
    sky_extent,
    sky_extent_from_frame,
    validate_coordinates,
)


def test_find_coord_columns_basic():
    assert find_coord_columns(["id", "ra", "dec", "g"]) == ("ra", "dec")


def test_find_coord_columns_aliases_and_separators():
    assert find_coord_columns(["RA_ICRS", "DE_ICRS"]) == ("RA_ICRS", "DE_ICRS")
    assert find_coord_columns(["RAJ2000", "DEJ2000"]) == ("RAJ2000", "DEJ2000")
    assert find_coord_columns(["RA (deg)", "Dec (deg)"]) == ("RA (deg)", "Dec (deg)")


def test_find_coord_columns_missing():
    assert find_coord_columns(["foo", "bar"]) == (None, None)


def test_validate_coordinates_rejects_nan_and_range():
    with pytest.raises(ValueError):
        validate_coordinates(np.array([np.nan]), np.array([0.0]))
    with pytest.raises(ValueError):
        validate_coordinates(np.array([400.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        validate_coordinates(np.array([10.0]), np.array([95.0]))
    validate_coordinates(np.array([10.0, 359.9]), np.array([-89.0, 5.0]))


def test_sky_extent_center_and_radius():
    ra = np.array([10.0, 10.2])
    dec = np.array([5.0, 5.0])
    extent = sky_extent(ra, dec)
    assert extent is not None
    assert 9.9 < extent["ra_center_deg"] < 10.3
    assert extent["radius_deg"] > 0


def test_sky_extent_handles_ra_wrap():
    ra = np.array([359.9, 0.1])
    dec = np.array([0.0, 0.0])
    extent = sky_extent(ra, dec)
    # Center should be near 0/360, not ~180.
    assert extent["ra_center_deg"] > 350 or extent["ra_center_deg"] < 10


def test_sky_extent_lazy_matches_eager_numpy():
    """The lazy polars version must agree with the eager numpy version."""
    ra = [10.0, 10.1, 9.9, 10.2, 10.0]
    dec = [5.0, 5.05, 4.95, 5.1, 5.0]
    frame = pl.DataFrame({"ra": ra, "dec": dec})
    eager = sky_extent(np.asarray(ra), np.asarray(dec))
    lazy = sky_extent_from_frame(frame, "ra", "dec")
    assert lazy is not None and eager is not None
    assert abs(lazy["ra_center_deg"] - eager["ra_center_deg"]) < 1e-9
    assert abs(lazy["dec_center_deg"] - eager["dec_center_deg"]) < 1e-9
    # Max-separation uses cos(sep).min() → arccos; stay within ~1 arcsec of numpy.
    assert abs(lazy["radius_deg"] - eager["radius_deg"]) < 1e-3
    assert 0.0 < eager["radius_deg"] < 2.0
    assert 0.0 < lazy["radius_deg"] < 2.0


def test_sky_extent_lazy_accepts_lazyframe():
    frame = pl.LazyFrame({"ra": [1.0, 2.0, 3.0], "dec": [0.0, 0.1, -0.1]})
    ext = sky_extent_from_frame(frame, "ra", "dec")
    assert ext is not None
    assert ext["radius_deg"] > 0


def test_sky_extent_lazy_empty_frame():
    frame = pl.DataFrame({"ra": [], "dec": []}, schema={"ra": pl.Float64, "dec": pl.Float64})
    assert sky_extent_from_frame(frame, "ra", "dec") is None


def test_proper_motion_prefers_torchsky_and_preserves_nan_semantics(monkeypatch):
    """Torchsky receives the established finite-PM and finite-epoch contract."""
    received = {}

    def fake_torchsky(ra, dec, pmra, pmdec, *, source_epoch_jyear, target_epoch_jyear):
        received.update(
            ra=ra,
            dec=dec,
            pmra=pmra,
            pmdec=pmdec,
            source_epoch=source_epoch_jyear,
            target_epoch=target_epoch_jyear,
        )
        return np.asarray(ra) + 0.25, np.asarray(dec) - 0.5

    monkeypatch.setattr(
        astro_utils, "_load_torchsky_propagate_proper_motion", lambda: fake_torchsky
    )
    ra, dec = propagate_proper_motion(
        np.array([10.0, 20.0]),
        np.array([89.999, -89.999]),
        np.array([np.nan, 100.0]),
        np.array([50.0, np.nan]),
        np.array([np.nan, 2000.0]),
        2025.0,
    )

    np.testing.assert_allclose(ra, [10.25, 20.25])
    np.testing.assert_allclose(dec, [89.499, -90.499])
    np.testing.assert_allclose(received["pmra"], [0.0, 100.0])
    np.testing.assert_allclose(received["pmdec"], [50.0, 0.0])
    np.testing.assert_allclose(received["source_epoch"], [2025.0, 2000.0])
    assert received["target_epoch"] == 2025.0


def test_proper_motion_jacobian_adapter_preserves_batch(monkeypatch):
    expected_jacobian = np.broadcast_to(np.eye(4)[:2], (2, 2, 4)).copy()

    def fake_torchsky(*values, source_epoch_jyear, target_epoch_jyear):
        ra, dec = values[:2]
        return (np.asarray(ra) + 1.0, np.asarray(dec) - 2.0), expected_jacobian

    monkeypatch.setattr(
        astro_utils,
        "_load_torchsky_propagate_proper_motion_with_jacobian",
        lambda: fake_torchsky,
    )
    result = propagate_proper_motion_with_jacobian(
        np.array([10.0, 20.0]),
        np.array([30.0, 40.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([2016.0, 2016.0]),
        2026.0,
    )

    assert result is not None
    ra, dec, jacobian = result
    np.testing.assert_allclose(ra, [11.0, 21.0])
    np.testing.assert_allclose(dec, [28.0, 38.0])
    np.testing.assert_allclose(jacobian, expected_jacobian)


def test_proper_motion_astropy_fallback_matches_high_declination_reference(monkeypatch):
    """The optional-backend fallback remains accurate near both celestial poles."""
    monkeypatch.setattr(astro_utils, "_load_torchsky_propagate_proper_motion", lambda: None)
    ra = np.array([359.9, 0.1])
    dec = np.array([89.999, -89.999])
    pmra = np.array([10_000.0, -10_000.0])
    pmdec = np.array([-5_000.0, 5_000.0])
    source_epoch = np.array([2000.0, 2000.0])

    moved_ra, moved_dec = propagate_proper_motion(ra, dec, pmra, pmdec, source_epoch, 2100.0)
    reference = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=Time(source_epoch, format="jyear", scale="tcb"),
        frame="icrs",
    ).apply_space_motion(new_obstime=Time(2100.0, format="jyear", scale="tcb"))

    np.testing.assert_allclose(
        ((moved_ra - reference.ra.deg + 180.0) % 360.0) - 180.0, 0.0, atol=1e-10
    )
    np.testing.assert_allclose(moved_dec, reference.dec.deg, atol=1e-10)


def test_space_motion_prefers_torchsky(monkeypatch):
    received = {}

    def fake_torchsky(*values, source_epoch_jyear, target_epoch_jyear):
        received["values"] = values
        received["source_epoch"] = source_epoch_jyear
        received["target_epoch"] = target_epoch_jyear
        ra, dec = values[:2]
        return np.asarray(ra) + 1.0, np.asarray(dec) - 2.0, *values[2:6]

    monkeypatch.setattr(astro_utils, "_load_torchsky_propagate_space_motion", lambda: fake_torchsky)
    ra, dec = propagate_space_motion(
        np.array([10.0]),
        np.array([20.0]),
        np.array([100.0]),
        np.array([50.0]),
        np.array([100.0]),
        np.array([20.0]),
        np.array([2000.0]),
        2025.0,
    )

    np.testing.assert_allclose(ra, [11.0])
    np.testing.assert_allclose(dec, [18.0])
    np.testing.assert_allclose(received["values"][4], [100.0])
    np.testing.assert_allclose(received["values"][5], [20.0])
    assert received["target_epoch"] == 2025.0


def test_space_motion_jacobian_adapter_preserves_batch(monkeypatch):
    expected_jacobian = np.broadcast_to(np.eye(6), (2, 6, 6)).copy()

    def fake_torchsky(*values, source_epoch_jyear, target_epoch_jyear):
        ra, dec = values[:2]
        propagated = (np.asarray(ra) + 1.0, np.asarray(dec) - 2.0, *values[2:6])
        return propagated, expected_jacobian

    monkeypatch.setattr(
        astro_utils,
        "_load_torchsky_propagate_space_motion_with_jacobian",
        lambda: fake_torchsky,
    )
    result = propagate_space_motion_with_jacobian(
        np.array([10.0, 20.0]),
        np.array([30.0, 40.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0]),
        np.array([7.0, 8.0]),
        np.array([2016.0, 2016.0]),
        2026.0,
    )

    assert result is not None
    ra, dec, jacobian = result
    np.testing.assert_allclose(ra, [11.0, 21.0])
    np.testing.assert_allclose(dec, [28.0, 38.0])
    np.testing.assert_allclose(jacobian, expected_jacobian)


def test_space_motion_astropy_fallback_matches_reference(monkeypatch):
    monkeypatch.setattr(astro_utils, "_load_torchsky_propagate_space_motion", lambda: None)
    ra = np.array([269.452075])
    dec = np.array([4.693391])
    pmra = np.array([-801.551])
    pmdec = np.array([10_362.394])
    parallax = np.array([548.31])
    radial_velocity = np.array([-110.6])
    source_epoch = np.array([2000.0])

    moved_ra, moved_dec = propagate_space_motion(
        ra, dec, pmra, pmdec, parallax, radial_velocity, source_epoch, 2025.0
    )
    reference = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        distance=(1000.0 / parallax) * u.pc,
        radial_velocity=radial_velocity * u.km / u.s,
        obstime=Time(source_epoch, format="jyear", scale="tcb"),
        frame="icrs",
    ).apply_space_motion(new_obstime=Time(2025.0, format="jyear", scale="tcb"))

    np.testing.assert_allclose(
        ((moved_ra - reference.ra.deg + 180.0) % 360.0) - 180.0, 0.0, atol=1e-10
    )
    np.testing.assert_allclose(moved_dec, reference.dec.deg, atol=1e-10)
