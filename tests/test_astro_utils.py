import numpy as np
import pytest

from xmatch.astro_utils import find_coord_columns, sky_extent, validate_coordinates


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
