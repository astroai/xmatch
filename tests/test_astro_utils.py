import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np

# Adjust the import path based on your project structure
from xmatch.astro_utils import (
    validate_coordinates,
    find_coord_columns,
    create_skycoord,
    calculate_separation,
    crossmatch_coords,
    apply_epoch_propagation
)


# --- Tests for validate_coordinates ---

def test_validate_coordinates_valid():
    """Test validate_coordinates with valid data. Should not raise an error."""
    data = {'ra': [10.0, 20.0], 'dec': [5.0, -5.0]}
    df = pd.DataFrame(data)
    # No assertion needed, test passes if no exception is raised
    validate_coordinates(df, 'ra', 'dec')

def test_validate_coordinates_invalid_ra():
    """Test validate_coordinates with invalid RA values."""
    data = {'ra': [370.0, 20.0], 'dec': [5.0, -5.0]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="invalid RA values"):
        validate_coordinates(df, 'ra', 'dec')

def test_validate_coordinates_invalid_dec():
    """Test validate_coordinates with invalid Dec values."""
    data = {'ra': [10.0, 20.0], 'dec': [95.0, -5.0]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="invalid Dec values"):
        validate_coordinates(df, 'ra', 'dec')

def test_validate_coordinates_missing_col():
    """Test validate_coordinates with missing coordinate columns."""
    data = {'ra': [10.0, 20.0]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Dec column 'dec' not found"):
        validate_coordinates(df, 'ra', 'dec')

def test_validate_coordinates_nan_values():
    """Test validate_coordinates with NaN values in coordinates."""
    data = {'ra': [10.0, None], 'dec': [5.0, -5.0]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="NaN value\(s\) in RA column"):
        validate_coordinates(df, 'ra', 'dec')


# --- Tests for find_coord_columns ---

def test_find_coord_columns_standard():
    """Test find_coord_columns with standard 'ra', 'dec' names."""
    df = pd.DataFrame({'ra': [1], 'dec': [2]})
    ra_col, dec_col = find_coord_columns(df)
    assert ra_col == 'ra'
    assert dec_col == 'dec'

def test_find_coord_columns_alternative():
    """Test find_coord_columns with alternative names like 'RAJ2000'."""
    df = pd.DataFrame({'RAJ2000': [1], 'DEJ2000': [2]})
    ra_col, dec_col = find_coord_columns(df)
    assert ra_col == 'RAJ2000'
    assert dec_col == 'DEJ2000'

def test_find_coord_columns_case_insensitive():
    """Test find_coord_columns is case-insensitive."""
    df = pd.DataFrame({'RA': [1], 'Dec': [2]})
    ra_col, dec_col = find_coord_columns(df)
    assert ra_col == 'RA'
    assert dec_col == 'Dec'

def test_find_coord_columns_partial_match():
    """Test find_coord_columns with partial matches like 'ra_deg'."""
    df = pd.DataFrame({'ra_deg': [1], 'dec_deg': [2]})
    ra_col, dec_col = find_coord_columns(df)
    assert ra_col == 'ra_deg'
    assert dec_col == 'dec_deg'


def test_find_coord_columns_no_match():
    """Test find_coord_columns when no coordinate columns are found."""
    df = pd.DataFrame({'col1': [1], 'col2': [2]})
    ra_col, dec_col = find_coord_columns(df)
    assert ra_col is None
    assert dec_col is None


# --- Fixtures for coordinate tests ---

@pytest.fixture
def sample_coord_data():
    """Provides a sample DataFrame for coordinate tests."""
    data = {
        'id': [1, 2, 3],
        'ra': [10.0, 20.0, 30.0],
        'dec': [5.0, -5.0, 50.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_skycoord1(sample_coord_data):
    """Provides a SkyCoord object based on sample_coord_data."""
    return create_skycoord(sample_coord_data, 'ra', 'dec')

@pytest.fixture
def sample_skycoord2():
    """Provides a second SkyCoord object for separation/matching tests."""
    # Slightly offset coordinates
    ra = [10.0001, 20.0002, 30.0000]
    dec = [5.0001, -5.0002, 50.0003]
    return SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

@pytest.fixture
def sample_skycoord3_large_sep():
    """Provides SkyCoord with larger separation."""
    ra = [180.0, 190.0, 200.0]
    dec = [0.0, -10.0, 10.0]
    return SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')


# --- Tests for create_skycoord ---

def test_create_skycoord_success(sample_coord_data):
    """Test successful creation of SkyCoord object."""
    coords = create_skycoord(sample_coord_data, 'ra', 'dec')
    assert isinstance(coords, SkyCoord)
    assert len(coords) == len(sample_coord_data)
    assert coords.frame.name == 'icrs'
    assert coords.ra.unit == u.deg
    assert coords.dec.unit == u.deg
    # Check one value for sanity
    assert np.isclose(coords[0].ra.deg, 10.0)
    assert np.isclose(coords[0].dec.deg, 5.0)

def test_create_skycoord_invalid_coords(sample_coord_data):
    """Test create_skycoord raises ValueError for invalid input coordinates."""
    invalid_data = sample_coord_data.copy()
    invalid_data.loc[0, 'ra'] = 400 # Invalid RA
    with pytest.raises(ValueError):
        create_skycoord(invalid_data, 'ra', 'dec')

def test_create_skycoord_custom_units_frame(sample_coord_data):
    """Test create_skycoord with custom units and frame."""
    # Convert sample data RA/Dec to radians for testing
    data_rad = sample_coord_data.copy()
    data_rad['ra_rad'] = np.deg2rad(data_rad['ra'])
    data_rad['dec_rad'] = np.deg2rad(data_rad['dec'])

    coords = create_skycoord(data_rad, 'ra_rad', 'dec_rad', frame='fk5', unit=(u.rad, u.rad))
    assert isinstance(coords, SkyCoord)
    assert coords.frame.name == 'fk5'
    # Check the *values* in the expected units, not the default representation unit
    assert np.allclose(coords.ra.rad, data_rad['ra_rad'])
    assert np.allclose(coords.dec.rad, data_rad['dec_rad'])


# --- Tests for calculate_separation ---

def test_calculate_separation_same_coords(sample_skycoord1):
    """Test separation calculation with identical coordinate sets."""
    sep = calculate_separation(sample_skycoord1, sample_skycoord1)
    assert isinstance(sep, np.ndarray)
    assert len(sep) == len(sample_skycoord1)
    # Separation should be close to zero
    assert np.allclose(sep, 0.0, atol=1e-9)

def test_calculate_separation_small_offset(sample_skycoord1, sample_skycoord2):
    """Test separation calculation with slightly offset coordinates."""
    sep = calculate_separation(sample_skycoord1, sample_skycoord2)
    assert isinstance(sep, np.ndarray)
    assert len(sep) == len(sample_skycoord1)
    # Separations should be small but non-zero (a few arcsec)
    assert np.all(sep > 1e-6) # Should be greater than zero
    # Adjust threshold based on actual offsets
    assert np.all(sep < 1.1) # Example: all should be < 1.1 arcsec for these offsets

def test_calculate_separation_zero(sample_skycoord1):
    """Test separation calculation with identical coordinates."""
    sep = calculate_separation(sample_skycoord1, sample_skycoord1)
    assert isinstance(sep, np.ndarray)
    assert len(sep) == len(sample_skycoord1)
    # Separation should be close to zero
    assert np.allclose(sep, 0.0, atol=1e-9)

def test_calculate_separation_mismatched_length(sample_skycoord1):
    """Test separation calculation raises ValueError for mismatched input lengths."""
    coords_shorter = sample_skycoord1[:-1] # Create a shorter version
    with pytest.raises(ValueError):
        calculate_separation(sample_skycoord1, coords_shorter)


# --- Tests for crossmatch_coords ---

def test_crossmatch_coords_close_match(sample_skycoord1, sample_skycoord2):
    """Test crossmatch finds matches within a small separation."""
    # Adjust max_sep based on separations calculated in test_calculate_separation_small_offset
    max_sep_arcsec = 1.1 # arcseconds 
    idx, sep = crossmatch_coords(sample_skycoord1, sample_skycoord2, max_sep=max_sep_arcsec)

    assert isinstance(idx, np.ndarray)
    assert isinstance(sep, np.ndarray)
    # All 3 coords in sample_skycoord1 should match the corresponding ones in sample_skycoord2
    assert len(idx) == 3
    assert len(sep) == 3
    # Indices should correspond to the closest match (in this case, 0->0, 1->1, 2->2)
    assert np.array_equal(idx, [0, 1, 2])
    # All separations should be less than max_sep
    assert np.all(sep <= max_sep_arcsec)
    assert np.all(sep > 0)

def test_crossmatch_coords_no_match(sample_skycoord1, sample_skycoord3_large_sep):
    """Test crossmatch finds no matches when separations are large."""
    max_sep_arcsec = 1.0 # arcseconds
    idx, sep = crossmatch_coords(sample_skycoord1, sample_skycoord3_large_sep, max_sep=max_sep_arcsec)

    assert isinstance(idx, np.ndarray)
    assert isinstance(sep, np.ndarray)
    # No matches expected within 1 arcsecond
    assert len(idx) == 0
    assert len(sep) == 0

def test_crossmatch_coords_partial_match(sample_skycoord1, sample_skycoord2):
    """Test crossmatch finds only matches within the specified max_sep."""
    # Calculate actual separations first
    actual_sep = calculate_separation(sample_skycoord1, sample_skycoord2)
    # Set max_sep to exclude the largest separation
    max_sep_arcsec = np.sort(actual_sep)[-2] # Second largest separation

    idx, sep = crossmatch_coords(sample_skycoord1, sample_skycoord2, max_sep=max_sep_arcsec)

    # Should find 2 matches now
    assert len(idx) == 2
    assert len(sep) == 2
    # All returned separations should be <= max_sep_arcsec
    assert np.all(sep <= max_sep_arcsec)

# Remove the TODO as tests are now added
# TODO: Add tests for create_skycoord, calculate_separation, crossmatch_coords 

# --- Add tests for propagate_coordinates_to_epoch --- 

@pytest.fixture
def propagation_df():
    """DataFrame fixture for epoch propagation tests."""
    # Simple data: source at (10, 20) deg, epoch 2016.0
    # PM: +10 mas/yr in RA*cos(Dec), -5 mas/yr in Dec
    # Target epoch J2000.0 (delta_t = -16 yr)
    # Expected RA change: +10 mas/yr * -16 yr / cos(20deg) = -160 / cos(20) mas = -170.27 mas = -4.73e-5 deg
    # Expected Dec change: -5 mas/yr * -16 yr = +80 mas = +2.22e-5 deg
    # Expected RA_2000 = 10 - 4.73e-5 = 9.9999527
    # Expected Dec_2000 = 20 + 2.22e-5 = 20.0000222
    data = {
        'source_id': [1, 2, 3, 4],
        'ra_deg':    [10.0, 15.0, 20.0, 25.0],
        'dec_deg':   [20.0, 25.0, 30.0, 35.0],
        'pm_ra':     [10.0, np.nan, -5.0, 20.0], # mas/yr
        'pm_dec':    [-5.0, 10.0, np.nan, -15.0], # mas/yr
        'epoch':     [2016.0, 2016.0, 2015.0, np.nan] # Julian Year
    }
    return pd.DataFrame(data)

def test_propagate_coordinates_basic(propagation_df):
    """Test basic coordinate propagation."""
    df = propagation_df.copy()
    target_epoch = 2000.0
    propagated_df = apply_epoch_propagation(
        df, 
        ra_col='ra_deg', 
        dec_col='dec_deg', 
        pm_ra_col='pm_ra', 
        pm_dec_col='pm_dec', 
        epoch_col='epoch', 
        target_epoch=target_epoch
    )
    assert 'ra_propagated' in propagated_df.columns
    assert 'dec_propagated' in propagated_df.columns
    # Check the first row (calculated manually above)
    pd.testing.assert_series_equal(
        propagated_df['ra_propagated'], 
        pd.Series([9.9999527, 15.0, 20.0000148, 25.0], name='ra_propagated'),
        check_exact=False, 
        rtol=1e-6
    )
    pd.testing.assert_series_equal(
        propagated_df['dec_propagated'], 
        pd.Series([20.0000222, 24.9999556, 30.0, 35.0], name='dec_propagated'),
        check_exact=False, 
        rtol=1e-6
    )

def test_propagate_coordinates_nan_pm(propagation_df):
    """Test propagation with NaN in proper motion (should treat as zero PM)."""
    df = propagation_df.copy()
    target_epoch = 2000.0
    propagated_df = apply_epoch_propagation(
        df, 
        ra_col='ra_deg', 
        dec_col='dec_deg', 
        pm_ra_col='pm_ra', 
        pm_dec_col='pm_dec', 
        epoch_col='epoch', 
        target_epoch=target_epoch
    )
    # Row 1 (index 1) had pm_ra=NaN -> RA should not change significantly
    # Row 2 (index 2) had pm_dec=NaN -> Dec should not change significantly
    assert np.isclose(propagated_df.loc[1, 'ra_propagated'], df.loc[1, 'ra_deg'])
    assert np.isclose(propagated_df.loc[2, 'dec_propagated'], df.loc[2, 'dec_deg'])

def test_propagate_coordinates_nan_epoch(propagation_df):
    """Test propagation with NaN in epoch (should effectively not propagate)."""
    df = propagation_df.copy()
    target_epoch = 2000.0
    propagated_df = apply_epoch_propagation(
        df, 
        ra_col='ra_deg', 
        dec_col='dec_deg', 
        pm_ra_col='pm_ra', 
        pm_dec_col='pm_dec', 
        epoch_col='epoch', 
        target_epoch=target_epoch
    )
    # Row 3 (index 3) had epoch=NaN -> RA/Dec should not change
    assert np.isclose(propagated_df.loc[3, 'ra_propagated'], df.loc[3, 'ra_deg'])
    assert np.isclose(propagated_df.loc[3, 'dec_propagated'], df.loc[3, 'dec_deg'])

def test_propagate_coordinates_missing_cols(propagation_df):
    """Test propagation when required columns are missing."""
    df = propagation_df.copy()
    target_epoch = 2000.0
    # Missing pm_ra_col
    propagated_df = apply_epoch_propagation(
        df, 
        ra_col='ra_deg', 
        dec_col='dec_deg', 
        pm_ra_col='pm_ra_missing', # Non-existent column 
        pm_dec_col='pm_dec', 
        epoch_col='epoch', 
        target_epoch=target_epoch
    )
    # Should return original coordinates renamed
    assert 'ra_propagated' in propagated_df.columns
    assert 'dec_propagated' in propagated_df.columns
    pd.testing.assert_series_equal(propagated_df['ra_propagated'], df['ra_deg'])
    pd.testing.assert_series_equal(propagated_df['dec_propagated'], df['dec_deg']) 