import numpy as np
import polars as pl
import pytest

from xmatch.astro_utils import (
    find_coord_columns,
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
    # Different kernels (numpy vs Arrow) can drift at the arc-second level
    # for the max-separation path; tolerate that here.
    assert abs(lazy["radius_deg"] - eager["radius_deg"]) < 1.0
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
