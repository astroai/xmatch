import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

# Adjust import based on project structure
from src.xmatch.crossmatch import CrossMatch, CrossMatchError

# --- Mock Configurations ---

@pytest.fixture
def minimal_valid_config():
    return {
        "archives": {
            "cds": {
                "description": "CDS Archive",
                "tap_service": {
                    "access_url": "http://cda.cfa.harvard.edu/csctap",
                    "access_method": "tap",
                    "param_A": "service_default"
                },
                "xmatch_service": {
                    "access_method": "cds_xmatch",
                     "param_B": "service_default"
                 }
            }
        },
        "catalogues": {
            "gaia_cds": {
                "archive": "cds",
                "service_id": "tap_service",
                "release": "DR3",
                "access_identifier": "I/355/gaiadr3",
                "ra_column": "RA_ICRS",
                "dec_column": "DE_ICRS",
                "param_A": "catalogue_override"
            }
        },
        "crossmatch_methods": {
            "stilts_tmatch2_sky": {"executor": "stilts"}
        },
        "stilts_config": {"java_opts": "-Xmx1g"}
    }

# --- Helper to instantiate CrossMatch with mocked config ---

def get_cm_with_mock_config(mock_config):
    # Patch _load_config to return our mock config
    # Patch _validate_config to prevent it running on potentially invalid mock configs during tests
    # Patch auth.load_auth_config as it's not needed for these config tests
    with patch.object(CrossMatch, '_load_config', return_value=mock_config), \
         patch.object(CrossMatch, '_validate_config', return_value=None), \
         patch('src.xmatch.auth.load_auth_config', return_value={}):
        # The config_file path doesn't matter here as load is mocked
        cm = CrossMatch(config_file="dummy_path.yaml")
        # Manually set the config attributes that __init__ would normally set
        cm.config = mock_config
        cm.archives_config = mock_config.get("archives", {})
        cm.catalogues_config = mock_config.get("catalogues", {})
        cm.methods_config = mock_config.get("crossmatch_methods", {})
        cm.stilts_config = mock_config.get("stilts_config", {})
        return cm

# --- Tests for _validate_config ---
# We test this by calling it directly on mock configs, not via __init__

def test_validate_config_valid(minimal_valid_config):
    """Test validation with a minimal valid config."""
    # Instantiate with a dummy path, then manually assign and validate
    with patch('src.xmatch.auth.load_auth_config', return_value={}):
        cm = CrossMatch.__new__(CrossMatch) # Create instance without calling __init__
        cm.config = minimal_valid_config
        # Should run without errors
        cm._validate_config()

@pytest.mark.parametrize(
    "invalid_config",
    [
        ({}), # Missing top-level keys
        ({"archives": [], "catalogues": {}}), # Wrong type for top-level
        ({"archives": {"a": {}}, "catalogues": []}), # Wrong type for top-level
        ({"archives": {"a": []}, "catalogues": {"c": {}}}), # Wrong type for archive config
        ({"archives": {"a": {"s":[]}}, "catalogues": {"c": {"archive":"a","service_id":"s","access_identifier":"id","ra_column":"r","dec_column":"d"}}}), # Wrong type for service config
        ({"archives": {"a": {"s":{}}}, "catalogues": {"c": []}}), # Wrong type for catalogue config
        ({"archives": {"a": {"s":{}}}, "catalogues": {"c": {"archive":"b"}}}), # Missing required cat key (service_id etc)
        ({"archives": {"a": {"s":{}}}, "catalogues": {"c": {"archive":"b","service_id":"s"}}}), # Archive 'b' not defined
        ({"archives": {"a": {"s":{}}}, "catalogues": {"c": {"archive":"a","service_id":"t"}}}), # Service 't' not defined
    ]
)
def test_validate_config_invalid(invalid_config):
    """Test validation fails for various invalid config structures."""
    with patch('src.xmatch.auth.load_auth_config', return_value={}):
        cm = CrossMatch.__new__(CrossMatch)
        cm.config = invalid_config
        # Set dummy archives/catalogues to avoid KeyErrors before validation call
        cm.archives_config = invalid_config.get("archives", {})
        cm.catalogues_config = invalid_config.get("catalogues", {})
        with pytest.raises(CrossMatchError):
            cm._validate_config()

# --- Tests for get_catalogue_config ---

def test_get_catalogue_config_success(minimal_valid_config):
    """Test successful retrieval and merging of catalogue config."""
    cm = get_cm_with_mock_config(minimal_valid_config)
    resolved_config = cm.get_catalogue_config("gaia_cds")

    # Check core catalogue keys are present
    assert resolved_config["_catalogue_name"] == "gaia_cds"
    assert resolved_config["archive"] == "cds"
    assert resolved_config["service_id"] == "tap_service"
    assert resolved_config["release"] == "DR3"
    assert resolved_config["access_identifier"] == "I/355/gaiadr3"
    assert resolved_config["ra_column"] == "RA_ICRS"
    assert resolved_config["dec_column"] == "DE_ICRS"

    # Check merge: catalogue overrides service default
    assert resolved_config["param_A"] == "catalogue_override"

    # Check inherited service key is present
    assert resolved_config["access_url"] == "http://cda.cfa.harvard.edu/csctap"
    assert resolved_config["access_method"] == "tap"


def test_get_catalogue_config_not_found(minimal_valid_config):
    """Test error when catalogue name doesn't exist."""
    cm = get_cm_with_mock_config(minimal_valid_config)
    with pytest.raises(CrossMatchError, match="not found in configuration"):
        cm.get_catalogue_config("nonexistent_catalogue")


def test_get_catalogue_config_archive_not_found(minimal_valid_config):
    """Test error when catalogue's archive doesn't exist."""
    bad_config = minimal_valid_config.copy()
    bad_config["catalogues"]["gaia_cds"]["archive"] = "nonexistent_archive"
    cm = get_cm_with_mock_config(bad_config)
    with pytest.raises(CrossMatchError, match="Archive 'nonexistent_archive'.*not found"):
        cm.get_catalogue_config("gaia_cds")

def test_get_catalogue_config_service_not_found(minimal_valid_config):
    """Test error when catalogue's service doesn't exist in archive."""
    bad_config = minimal_valid_config.copy()
    bad_config["catalogues"]["gaia_cds"]["service_id"] = "nonexistent_service"
    cm = get_cm_with_mock_config(bad_config)
    with pytest.raises(CrossMatchError, match="Service 'nonexistent_service'.*not found"):
        cm.get_catalogue_config("gaia_cds")

def test_get_catalogue_config_missing_keys(minimal_valid_config):
    """Test error when catalogue definition is missing essential keys."""
    bad_config = minimal_valid_config.copy()
    del bad_config["catalogues"]["gaia_cds"]["archive"] # Remove a required key
    # Need to mock _validate_config differently here as it would normally catch this
    with patch.object(CrossMatch, '_load_config', return_value=bad_config), \
         patch('src.xmatch.auth.load_auth_config', return_value={}):
        cm = CrossMatch(config_file="dummy_path.yaml")
        cm.config = bad_config # Override config after init
        cm.catalogues_config = bad_config.get("catalogues", {})
        cm.archives_config = bad_config.get("archives", {})
        # We need to skip the normal validation to reach the get_catalogue_config error
        with patch.object(CrossMatch, '_validate_config', return_value=None):
             with pytest.raises(CrossMatchError, match="missing 'archive' or 'service_id'"):
                 cm.get_catalogue_config("gaia_cds")

# --- Tests for _get_config_for_input ---

@patch('pathlib.Path.is_file')
def test_get_config_for_input_local_file(mock_is_file, minimal_valid_config):
    """Test config generation for a local file input."""
    mock_is_file.return_value = True # Mock that the path is a file
    cm = get_cm_with_mock_config(minimal_valid_config)

    file_path = Path("data/my_cat.fits")
    local_config = cm._get_config_for_input(file_path)

    assert local_config["is_local"] is True
    assert local_config["_input_path"] == str(file_path)
    assert local_config["_catalogue_name"] == "my_cat"
    assert local_config["description"] == f"Local file: {file_path.name}"
    assert local_config["archive"] is None
    assert local_config["service_type"] == "local_file"
    assert local_config["ra_column"] == "ra" # Default assumption
    assert local_config["dec_column"] == "dec" # Default assumption

@patch('pathlib.Path.is_file')
def test_get_config_for_input_catalogue_name(mock_is_file, minimal_valid_config):
    """Test config retrieval for a configured catalogue name."""
    mock_is_file.return_value = False # Mock that the path is not a file
    cm = get_cm_with_mock_config(minimal_valid_config)

    # We expect get_catalogue_config to be called for a non-file string
    with patch.object(cm, 'get_catalogue_config', wraps=cm.get_catalogue_config) as mock_get_cat_conf:
        resolved_config = cm._get_config_for_input("gaia_cds")
        mock_get_cat_conf.assert_called_once_with("gaia_cds")

        # Check it returns the resolved config from the main config
        assert resolved_config["_catalogue_name"] == "gaia_cds"
        assert resolved_config["archive"] == "cds"
        assert resolved_config.get("is_local") is not True

@patch('pathlib.Path.is_file')
def test_get_config_for_input_unknown_string(mock_is_file, minimal_valid_config):
    """Test error when input string is neither a file nor a known catalogue."""
    mock_is_file.return_value = False
    cm = get_cm_with_mock_config(minimal_valid_config)

    with pytest.raises(CrossMatchError, match="Input string 'unknown_cat' is not a valid file path or known catalogue name"):
        cm._get_config_for_input("unknown_cat")

def test_get_config_for_input_dataframe(minimal_valid_config):
    """Test config generation for a DataFrame input."""
    cm = get_cm_with_mock_config(minimal_valid_config)
    df_input = pd.DataFrame({'ra': [1], 'dec': [1]}) # Dummy DataFrame

    df_config = cm._get_config_for_input(df_input)

    assert df_config["is_local"] is True
    assert df_config["_input_path"] is None # No path for DataFrame
    assert df_config["_catalogue_name"] == "dataframe_input"
    assert df_config["description"] == "DataFrame Input"
    assert df_config["archive"] is None
    assert df_config["service_type"] == "dataframe"
    assert df_config["ra_column"] == "ra"
    assert df_config["dec_column"] == "dec"

# --- Fixture for method determination tests ---

@pytest.fixture
def method_test_config():
    return {
        "archives": {
            "cds": {
                "description": "CDS Archive",
                "tap_service": {
                    "access_url": "tap", "access_method": "tap",
                    "column_selection_supported": True, "adql_features": ["table_join"]
                },
                "xmatch_service": {
                    "access_method": "cds_xmatch", "column_selection_supported": False,
                     "crossmatch_input_methods": ["remote_table_query"]
                 },
                "service_priority": ["tap_service", "xmatch_service"]
            },
            "noao": {
                 "description": "NOAO Archive",
                 "tap_service": {
                    "access_url": "tap", "access_method": "tap",
                    "column_selection_supported": True, "adql_features": ["table_join", "table_upload"]
                 },
                 "service_priority": ["tap_service"]
            }
        },
        "catalogues": {
            "gaia_cds_tap": {"archive": "cds", "service_id": "tap_service", "release": "DR3", "access_identifier": "gaia", "ra_column": "ra", "dec_column": "dec", "best_method": "tap_join"},
            "gaia_cds_xmatch": {"archive": "cds", "service_id": "xmatch_service", "release": "DR3", "access_identifier": "gaia_xmatch", "ra_column": "ra", "dec_column": "dec", "best_method": "cds_xmatch"},
            "ukidss_noao": {"archive": "noao", "service_id": "tap_service", "release": "DR11", "access_identifier": "ukidss", "ra_column": "ra", "dec_column": "dec", "best_method": "tap_join"}
        },
        "crossmatch_methods": {
            "stilts_tmatch2_sky": {"executor": "stilts", "description": "STILTS Sky Match", "params": {"matcher": "sky"}},
            "stilts_tmatchn_id": {"executor": "stilts", "description": "STILTS ID Match", "params": {"matcher": "exact"}},
            "tap_join": {"executor": "tap", "description": "TAP ADQL Join"},
            "cds_xmatch": {"executor": "cds", "description": "CDS X-Match Service"}
        },
        "stilts_config": {}
    }

# --- Tests for _determine_best_method ---

@pytest.mark.parametrize(
    "cat1_type, cat2_type, user_method, id_match, expected_method_name",
    [
        # User override always wins
        ("local", "local", "tap_join", False, "tap_join"),
        ("remote_tap", "remote_cds", "stilts_tmatch2_sky", False, "stilts_tmatch2_sky"),
        ("local", "local", "stilts_tmatchn_id", True, "stilts_tmatchn_id"), # User override ID

        # No override - defaults based on type
        ("local", "local", None, False, "stilts_tmatch2_sky"), # Local vs Local -> STILTS sky
        ("local", "local", None, True, "stilts_tmatchn_id"), # Local vs Local ID -> STILTS ID
        ("local", "remote_tap", None, False, "tap_join"), # Local vs Remote TAP -> TAP Upload/Join
        ("local", "remote_cds", None, False, "cds_xmatch"), # Local vs Remote CDS -> CDS XMatch Service
        ("remote_tap", "remote_tap", None, False, "tap_join"), # Remote TAP vs Remote TAP -> TAP Join
        ("remote_tap", "remote_cds", None, False, "tap_join"), # TAP preferred over CDS if available for primary
        ("remote_cds", "remote_tap", None, False, "tap_join"), # TAP preferred even if cat2 is TAP
        ("remote_cds", "remote_cds", None, False, "cds_xmatch"), # Both CDS -> CDS XMatch (assuming no TAP preference)

        # ID matching without user override
        ("local", "remote_tap", None, True, "tap_join"), # ID match via TAP
        ("remote_tap", "remote_tap", None, True, "tap_join"), # ID match via TAP

    ]
)
def test_determine_best_method(cat1_type, cat2_type, user_method, id_match, expected_method_name, method_test_config):
    """Test determination of the best crossmatch method."""
    cm = get_cm_with_mock_config(method_test_config)

    # Create mock configs based on type
    if cat1_type == "local":
        config1 = cm._get_config_for_input(Path("local1.fits"))
    elif cat1_type == "remote_tap":
        config1 = cm.get_catalogue_config("gaia_cds_tap")
    else: # remote_cds
        config1 = cm.get_catalogue_config("gaia_cds_xmatch")

    if cat2_type == "local":
        config2 = cm._get_config_for_input(Path("local2.fits"))
    elif cat2_type == "remote_tap":
        config2 = cm.get_catalogue_config("ukidss_noao")
    else: # remote_cds
        config2 = cm.get_catalogue_config("gaia_cds_xmatch")

    # Mock Path.is_file for local file checks within _determine_best_method if needed
    with patch('pathlib.Path.is_file', return_value=True):
        method_name, method_config = cm._determine_best_method(
            config1,
            config2,
            method_override=user_method,
            id_column_1="id1" if id_match else None,
            id_column_2="id2" if id_match else None
        )

    assert method_name == expected_method_name
    assert method_config["description"] == method_test_config["crossmatch_methods"][expected_method_name]["description"]

def test_determine_best_method_no_suitable(method_test_config):
    """Test error when no suitable method can be found (e.g., invalid user override)."""
    cm = get_cm_with_mock_config(method_test_config)
    config1 = cm._get_config_for_input(Path("local1.fits"))
    config2 = cm._get_config_for_input(Path("local2.fits"))

    with patch('pathlib.Path.is_file', return_value=True):
        with pytest.raises(CrossMatchError, match="No suitable cross-match method found"):
            cm._determine_best_method(config1, config2, method_override="nonexistent_method")

        # Test case where ID match is requested but no ID method defined
        simplified_config = method_test_config.copy()
        simplified_config["crossmatch_methods"] = {
            "stilts_tmatch2_sky": simplified_config["crossmatch_methods"]["stilts_tmatch2_sky"]
        }
        cm_simple = get_cm_with_mock_config(simplified_config)
        with pytest.raises(CrossMatchError, match="No suitable cross-match method found for ID matching"):
            cm_simple._determine_best_method(config1, config2, id_column_1="id1", id_column_2="id2")

# --- Fixtures ---

@pytest.fixture
def mock_astroquery():
    """Mocks astroquery.xmatch.XMatch for testing."""
    with patch('xmatch.crossmatch.XMatch') as mock_xmatch_cls:
        mock_instance = MagicMock()
        # Simulate a successful query returning an Astropy Table
        mock_result_table = Table({
            'angDist': [0.5, 1.0] * u.arcsec,
            'ra_local': [10.0, 20.1], 
            'dec_local': [5.0, 6.1],
            'id_local': [101, 102],
            'ra_remote': [10.0001, 20.1001],
            'dec_remote': [5.0001, 6.1001],
            'ID_remote': ['viz1', 'viz2']
        })
        mock_instance.query_async.return_value = mock_result_table
        mock_xmatch_cls.return_value = mock_instance
        yield mock_xmatch_cls # Yield the class mock itself


@pytest.fixture
def local_df_config(tmp_path):
    """Config fixture for a local DataFrame input."""
    df = pd.DataFrame({
        'id_local': [101, 102, 103],
        'ra_local': [10.0, 20.1, 30.2],
        'dec_local': [5.0, 6.1, 7.2]
    })
    return {
        "_catalogue_name": "local_df",
        "_input_dataframe": df,
        "description": "Local DataFrame",
        "archive": None,
        "service_type": "local_file", # Use same type for simplicity
        "access_method": "file_system",
        "is_local": True,
        "ra_column": "ra_local",
        "dec_column": "dec_local",
        "epoch": 2000.0 # Example epoch
        # Add error columns if testing skyerr etc.
    }

@pytest.fixture
def remote_cds_config():
    """Config fixture for a remote catalogue using CDS XMatch."""
    return {
        "_catalogue_name": "vizier_cat",
        "description": "Remote VizieR Catalogue (CDS)",
        "archive": "cds",
        "service_id": "xmatch_service", # Corresponds to access_method: cds_xmatch
        "access_method": "cds_xmatch",
        "access_identifier": "I/355/gaiadr3", # Example VizieR ID
        "ra_column": "RA_ICRS", # These aren't used by CDS query but good practice
        "dec_column": "DE_ICRS",
        "epoch": 2016.0
        # Error columns etc. would go here if needed for other strategies
    }

# --- Tests ---

def test_crossmatch_strategy_cds_local_remote(crossmatcher, local_df_config, remote_cds_config):
    """Test strategy selection for local vs remote CDS."""
    strategy, params = crossmatcher._determine_crossmatch_strategy(
        local_df_config, remote_cds_config, radius_arcsec=2.0
    )
    assert strategy == "cds_xmatch_local_remote"

def test_crossmatch_cds_local_remote_success(crossmatcher, local_df_config, remote_cds_config, mock_astroquery):
    """Test successful execution of the CDS XMatch strategy."""
    # Mock _get_config_for_input to return our test configs
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[local_df_config, remote_cds_config]):
        result_df = crossmatcher.crossmatch(
            catalogue_1_input="local_df_placeholder", # Names don't matter due to patch
            catalogue_2_input="remote_cds_placeholder",
            radius_arcsec=2.0
        )
    
    # Check mock was called
    mock_astroquery.return_value.query_async.assert_called_once()
    call_args, call_kwargs = mock_astroquery.return_value.query_async.call_args
    assert isinstance(call_kwargs['cat1'], Table) # Input should be Astropy Table
    assert call_kwargs['cat2'] == f"vizier:{remote_cds_config['access_identifier']}"
    assert call_kwargs['max_distance'].value == 2.0
    assert call_kwargs['colRA1'] == local_df_config['ra_column']
    assert call_kwargs['colDec1'] == local_df_config['dec_column']

    # Check result type and basic content
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2 # Based on mock_result_table
    assert 'angDist' in result_df.columns
    assert 'id_local' in result_df.columns # From local input
    assert 'ID_remote' in result_df.columns # From mocked remote table

def test_crossmatch_cds_local_remote_missing_radius(crossmatcher, local_df_config, remote_cds_config, mock_astroquery):
    """Test CDS XMatch strategy failure when radius is missing."""
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[local_df_config, remote_cds_config]):
        with pytest.raises(ValueError, match="Missing required parameter 'radius_arcsec'"):
            crossmatcher.crossmatch(
                catalogue_1_input="local_df_placeholder",
                catalogue_2_input="remote_cds_placeholder",
                # No radius_arcsec provided
            )

def test_crossmatch_cds_local_remote_query_fails(crossmatcher, local_df_config, remote_cds_config, mock_astroquery):
    """Test CDS XMatch strategy when astroquery fails."""
    # Configure mock to raise an exception
    mock_astroquery.return_value.query_async.side_effect = Exception("CDS Query Failed")

    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[local_df_config, remote_cds_config]):
        with pytest.raises(CrossMatchError, match="Error during CDS XMatch query"):
            crossmatcher.crossmatch(
                catalogue_1_input="local_df_placeholder",
                catalogue_2_input="remote_cds_placeholder",
                radius_arcsec=2.0
            )

# --- Tests for Remote Spatial Chunked Match ---

@pytest.fixture
def remote_tap_config_1():
    """Config fixture for a remote TAP catalogue."""
    return {
        "_catalogue_name": "cat1_tap", "description": "Remote TAP Cat 1",
        "archive": "noao", "service_id": "tap_service", 
        "access_method": "tap", "access_url": "http://tap1.test",
        "access_identifier": "cat1.table", "ra_column": "ra1", "dec_column": "dec1",
        "epoch": 2000.0
    }

@pytest.fixture
def remote_tap_config_2():
    """Config fixture for another remote TAP catalogue."""
    return {
        "_catalogue_name": "cat2_tap", "description": "Remote TAP Cat 2",
        "archive": "esa", "service_id": "tap_service", # Different archive
        "access_method": "tap", "access_url": "http://tap2.test",
        "access_identifier": "cat2.table", "ra_column": "ra2", "dec_column": "dec2",
        "epoch": 2016.0, # Different epoch
        "pm_ra_column": "pmra", "pm_dec_column": "pmdec", "epoch_column": "ref_epoch"
    }

@pytest.fixture
def mock_healpix():
    """Mocks astropy_healpix.HEALPix and boundary functions."""
    # Define mock pixel boundaries (simplified)
    mock_boundaries = {
        0: SkyCoord(ra=[0, 1, 1, 0]*u.deg, dec=[0, 0, 1, 1]*u.deg, frame='icrs'),
        1: SkyCoord(ra=[1, 2, 2, 1]*u.deg, dec=[0, 0, 1, 1]*u.deg, frame='icrs'),
        2: SkyCoord(ra=[0, 1, 1, 0]*u.deg, dec=[1, 1, 2, 2]*u.deg, frame='icrs'),
    }

    with patch('xmatch.crossmatch.HEALPix') as mock_hp_cls, \
         patch('xmatch.crossmatch.boundaries_skycoord') as mock_bounds:
        
        mock_hp_instance = MagicMock()
        # Simulate cone search returning 3 pixels
        mock_hp_instance.cone_search_skycoord.return_value = [0, 1, 2]
        mock_hp_cls.return_value = mock_hp_instance

        # Mock boundaries function
        def boundaries_side_effect(pixels, nside):
            # Note: healpix boundaries_skycoord returns a list of SkyCoord objects
            return [mock_boundaries[p] for p in pixels]
        # boundaries_skycoord is imported directly, so patch it directly
        mock_bounds.side_effect = boundaries_side_effect
        
        yield {"class": mock_hp_cls, "boundaries": mock_bounds}


@pytest.fixture
def mock_fetch_remote_chunks():
    """Mocks _fetch_remote_catalogue to simulate chunk fetching."""
    # Return different data based on catalogue name and box params (simplified check)
    def fetch_side_effect(config, columns=None, query_constraints=None, cone_params=None, box_params=None):
        cat_name = config.get("_catalogue_name")
        # Use a simple check on box_params (e.g., min dec) to return different data
        dec_min = box_params.get('dec_min', -99) if box_params else -99

        #print(f"Mock fetch called for {cat_name} with dec_min={dec_min}") # Debug print

        if cat_name == "cat1_tap":
            if np.isclose(dec_min, 0.0): # Pixel 0 or 1
                return pd.DataFrame({'id1': [10, 11], 'ra1': [0.5, 1.5], 'dec1': [0.5, 0.5]})
            elif np.isclose(dec_min, 1.0): # Pixel 2
                return pd.DataFrame({'id1': [12], 'ra1': [0.5], 'dec1': [1.5]})
            else:
                return pd.DataFrame() # No data for other areas
        elif cat_name == "cat2_tap":
            if np.isclose(dec_min, 0.0): # Pixel 0 or 1
                 # Simulate no data in pixel 1 for cat2
                if box_params and np.isclose(box_params.get('ra_min', -99), 1.0):
                    return pd.DataFrame() 
                else: # Data for pixel 0
                    return pd.DataFrame({'id2': [20], 'ra2': [0.51], 'dec2': [0.51], 'pmra': [1], 'pmdec': [1], 'ref_epoch':[2016.0]})
            elif np.isclose(dec_min, 1.0): # Pixel 2
                 return pd.DataFrame({'id2': [21, 22], 'ra2': [0.51, 0.6], 'dec2': [1.51, 1.6], 'pmra': [1,1], 'pmdec': [1,1], 'ref_epoch':[2016.0]})
            else:
                 return pd.DataFrame()
            
    with patch.object(CrossMatch, '_fetch_remote_catalogue') as mock_fetch:
        mock_fetch.side_effect = fetch_side_effect
        yield mock_fetch

@pytest.fixture
def mock_stilts_chunks():
    """Mocks _execute_local_stilts for chunk processing."""
    call_count = 0
    def stilts_side_effect(config1, config2, **params):
        nonlocal call_count
        call_count += 1
        #print(f"Mock stilts called: Chunk {call_count}") # Debug print
        # Simulate successful match, return combined IDs (simplified)
        df1 = config1['_input_dataframe']
        df2 = config2['_input_dataframe']
        # Just return first ID from each as a dummy match result
        if not df1.empty and not df2.empty:
             # Use first column name heuristically as ID col
            id1_col = df1.columns[0]
            id2_col = df2.columns[0]
            return pd.DataFrame({f'{id1_col}_match': [df1[id1_col].iloc[0]], f'{id2_col}_match': [df2[id2_col].iloc[0]], 'chunk': [call_count]})
        else:
            return pd.DataFrame()

    with patch.object(CrossMatch, '_execute_local_stilts') as mock_stilts:
        mock_stilts.side_effect = stilts_side_effect
        yield mock_stilts


def test_crossmatch_strategy_remote_chunked(crossmatcher, remote_tap_config_1, remote_tap_config_2):
    """Test strategy selection for remote spatial chunked match."""
    strategy, params = crossmatcher._determine_crossmatch_strategy(
        remote_tap_config_1, remote_tap_config_2, 
        ra=1.0, dec=0.5, radius_arcsec=1800 # Provide spatial params
    )
    assert strategy == "remote_spatial_chunked_match"

def test_crossmatch_remote_chunked_success(
    crossmatcher, remote_tap_config_1, remote_tap_config_2, 
    mock_healpix, mock_fetch_remote_chunks, mock_stilts_chunks
):
    """Test successful execution of the remote spatial chunked match strategy."""
    # Need to patch _get_config_for_input to return the remote configs
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[remote_tap_config_1, remote_tap_config_2]):
        result_df = crossmatcher.crossmatch(
            catalogue_1_input="cat1_tap", 
            catalogue_2_input="cat2_tap",
            strategy="remote_spatial_chunked_match", # Force strategy for test clarity
            ra=1.0, dec=0.5, radius_arcsec=1800, # Spatial params required
            nside=32 # Example nside
        )

    # Check mocks
    # HEALPix cone search should be called once
    mock_healpix["class"].return_value.cone_search_skycoord.assert_called_once()
    # Boundaries should be called for the pixels found (3 in mock)
    # Patch boundaries_skycoord directly as it's imported
    mock_healpix["boundaries"].assert_called_once()
    assert mock_healpix["boundaries"].call_args[0][0] == [0, 1, 2] # Check pixels passed

    # Fetch should be called twice per pixel (3 pixels = 6 calls)
    assert mock_fetch_remote_chunks.call_count == 6
    # Stilts should be called only for pixels where *both* fetches returned data (pixels 0 and 2)
    assert mock_stilts_chunks.call_count == 2 

    # Check result (based on mock_stilts_chunks and mock_fetch_remote_chunks)
    # Chunk 1 (Pixel 0): Match id1=10 vs id2=20 -> call 1
    # Chunk 2 (Pixel 1): Cat2 fetch empty -> No stilts call
    # Chunk 3 (Pixel 2): Match id1=12 vs id2=21 -> call 2
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2
    assert list(result_df.columns) == ['id1_match', 'id2_match', 'chunk']
    assert result_df['id1_match'].tolist() == [10, 12]
    assert result_df['id2_match'].tolist() == [20, 21]
    assert result_df['chunk'].tolist() == [1, 2] # Reflects order stilts was called

def test_crossmatch_remote_chunked_missing_spatial_params(
    crossmatcher, remote_tap_config_1, remote_tap_config_2
):
    """Test remote chunked strategy fails if spatial params are missing."""
    # Mock strategy determination to force selection (as it would normally fallback)
    with patch.object(crossmatcher, '_determine_crossmatch_strategy', return_value=("remote_spatial_chunked_match", {"matcher": "sky"})), \
         patch.object(crossmatcher, '_get_config_for_input', side_effect=[remote_tap_config_1, remote_tap_config_2]):
            
        with pytest.raises(ValueError, match="Missing required parameters 'ra', 'dec', 'radius_arcsec'"):
            crossmatcher.crossmatch(
                catalogue_1_input="cat1_tap", 
                catalogue_2_input="cat2_tap",
                # Missing ra, dec, radius_arcsec
                # Strategy forced by mock, so crossmatch will call the execution function
                strategy="remote_spatial_chunked_match" 
            )

def test_crossmatch_remote_chunked_healpix_import_error(crossmatcher, remote_tap_config_1, remote_tap_config_2):
    """Test error handling if astropy-healpix is not installed."""
    # Patch sys.modules to simulate missing import
    # Also need to mock _get_config_for_input
    with patch.dict("sys.modules", {"astropy_healpix": None}), \
         patch.object(crossmatcher, '_get_config_for_input', side_effect=[remote_tap_config_1, remote_tap_config_2]):
        
        # Mock strategy determination to force selection
        with patch.object(crossmatcher, '_determine_crossmatch_strategy', return_value=("remote_spatial_chunked_match", {"matcher": "sky"})):
            with pytest.raises(CrossMatchError, match="'astropy-healpix' library is required"):
                crossmatcher.crossmatch(
                    catalogue_1_input="cat1_tap", 
                    catalogue_2_input="cat2_tap",
                    strategy="remote_spatial_chunked_match", 
                    ra=1.0, dec=0.5, radius_arcsec=1800,
                )

# --- Tests for Local File Chunking --- 

@pytest.fixture
def local_file_config_1(tmp_path):
    """Config fixture for a local file (CSV)."""
    f_path = tmp_path / "local1_chunk.csv"
    # Create a larger CSV file
    data = {
        'id1': range(25), 
        'ra1': np.linspace(0, 24, 25), 
        'dec1': np.linspace(0, 24, 25)
    }
    pd.DataFrame(data).to_csv(f_path, index=False)
    return {
        "_catalogue_name": "local1_file", "_input_path": str(f_path),
        "description": "Local File 1", "archive": None, 
        "access_method": "file_system", "is_local": True,
        "ra_column": "ra1", "dec_column": "dec1", "epoch": 2000.0
    }

@pytest.fixture
def local_file_config_2(tmp_path):
    """Config fixture for a second local file (CSV) - static target."""
    f_path = tmp_path / "local2_static.csv"
    data = {
        'id2': range(5), 
        'ra2': np.linspace(0.1, 4.1, 5), 
        'dec2': np.linspace(0.1, 4.1, 5)
    }
    pd.DataFrame(data).to_csv(f_path, index=False)
    return {
        "_catalogue_name": "local2_file", "_input_path": str(f_path),
        "description": "Local File 2 (Static)", "archive": None, 
        "access_method": "file_system", "is_local": True,
        "ra_column": "ra2", "dec_column": "dec2", "epoch": 2000.0
    }

@pytest.fixture
def mock_stilts_local_chunking():
    """Mocks _execute_local_stilts for local chunk processing."""
    call_count = 0
    def stilts_side_effect(config1, config2, **params):
        nonlocal call_count
        call_count += 1
        # Simulate match based on input DataFrames
        df1 = config1['_input_dataframe']
        df2 = config2['_input_dataframe']
        # Simple mock: return number of rows in chunk df1
        return pd.DataFrame({'match_id': range(len(df1)), 'chunk_num': [call_count] * len(df1)})

    with patch.object(CrossMatch, '_execute_local_stilts') as mock_stilts:
        mock_stilts.side_effect = stilts_side_effect
        yield mock_stilts

def test_crossmatch_chunked_local_file_success(
    crossmatcher, local_file_config_1, local_file_config_2, mock_stilts_local_chunking
):
    """Test successful execution of local file chunking (CSV)."""
    chunk_rows = 10
    # Mock _get_config_for_input to return file configs
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[local_file_config_1, local_file_config_2]):
        result_df = crossmatcher.crossmatch(
            catalogue_1_input=local_file_config_1["_input_path"],
            catalogue_2_input=local_file_config_2["_input_path"],
            strategy="chunked_local_match", # Force strategy
            chunk_rows=chunk_rows,
            chunk_input_index=1, # Chunk the first input (25 rows)
            radius_arcsec=1.0 # Dummy param for stilts mock
        )

    # Input 1 has 25 rows, chunk size 10 -> 3 chunks (10, 10, 5 rows)
    assert mock_stilts_local_chunking.call_count == 3
    assert len(result_df) == 25 # Total rows matched = total rows in chunked input
    assert result_df['chunk_num'].nunique() == 3
    assert result_df[result_df['chunk_num'] == 1].shape[0] == 10
    assert result_df[result_df['chunk_num'] == 2].shape[0] == 10
    assert result_df[result_df['chunk_num'] == 3].shape[0] == 5

def test_crossmatch_chunked_local_file_parquet_fallback(
    crossmatcher, tmp_path, local_file_config_2, mock_stilts_local_chunking
):
    """Test local file chunking fallback for Parquet (reads full file)."""
    # Create a Parquet file
    pq_path = tmp_path / "local1_chunk.parquet"
    data = {
        'id1': range(25), 
        'ra1': np.linspace(0, 24, 25), 
        'dec1': np.linspace(0, 24, 25)
    }
    pd.DataFrame(data).to_parquet(pq_path)
    local_file_config_1_pq = {
        "_catalogue_name": "local1_pq", "_input_path": str(pq_path),
        "description": "Local Parquet File", "archive": None, 
        "access_method": "file_system", "is_local": True,
        "ra_column": "ra1", "dec_column": "dec1", "epoch": 2000.0
    }

    chunk_rows = 10
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[local_file_config_1_pq, local_file_config_2]):
        # Expect a warning about inefficient fallback
        with pytest.warns(UserWarning, match="Direct file chunking for Parquet not fully implemented"):
             result_df = crossmatcher.crossmatch(
                catalogue_1_input=str(pq_path),
                catalogue_2_input=local_file_config_2["_input_path"],
                strategy="chunked_local_match", # Force strategy
                chunk_rows=chunk_rows,
                chunk_input_index=1, # Chunk the first input
                radius_arcsec=1.0 
            )

    # Even with fallback, the result should be processed in chunks
    assert mock_stilts_local_chunking.call_count == 3
    assert len(result_df) == 25
    assert result_df['chunk_num'].nunique() == 3

# --- Tests for Epoch Propagation Verification ---

@pytest.fixture
def gaia_df_config():
    """Config fixture for a Gaia-like DataFrame input."""
    df = pd.DataFrame({
        'id_g': [1], 'ra_g': [10.0], 'dec_g': [20.0],
        'pmra_g': [10.0], 'pmdec_g': [-5.0], 'epoch_g': [2016.0]
    })
    return {
        "_catalogue_name": "gaia_local_df", "_input_dataframe": df,
        "description": "Gaia DataFrame", "archive": None,
        "access_method": "file_system", "is_local": True,
        "ra_column": "ra_g", "dec_column": "dec_g", "epoch": 2016.0,
        "pm_ra_column": "pmra_g", "pm_dec_column": "pmdec_g", "epoch_column": "epoch_g"
    }

@pytest.fixture
def j2000_df_config():
    """Config fixture for a J2000.0 DataFrame input."""
    df = pd.DataFrame({
        'id_j': [101], 'ra_j': [10.0001], 'dec_j': [20.0001]
    })
    return {
        "_catalogue_name": "j2000_local_df", "_input_dataframe": df,
        "description": "J2000 DataFrame", "archive": None,
        "access_method": "file_system", "is_local": True,
        "ra_column": "ra_j", "dec_column": "dec_j", "epoch": 2000.0
    }

@pytest.fixture
def remote_gaia_config():
    """Config fixture for a remote Gaia catalogue (e.g., ESA)."""
    # Similar to remote_tap_config_2 but explicitly named
    return {
        "_catalogue_name": "gaia_remote", "description": "Remote Gaia TAP",
        "archive": "esa", "service_id": "tap_service", 
        "access_method": "tap", "access_url": "http://gaia.test",
        "access_identifier": "gaia.dr3", "ra_column": "ra_g", "dec_column": "dec_g",
        "epoch": 2016.0, "pm_ra_column": "pmra_g", "pm_dec_column": "pmdec_g", 
        "epoch_column": "epoch_g"
    }

@pytest.fixture
def mock_stilts_check_propagation():
    """Mocks _execute_local_stilts and checks for propagated columns."""
    call_args_list = []
    def stilts_side_effect(config1, config2, **params):
        # Store args for later inspection
        call_args_list.append({'config1': config1, 'config2': config2, 'params': params})
        # Return dummy result
        return pd.DataFrame({'match': [1]}) 

    with patch.object(CrossMatch, '_execute_local_stilts') as mock_stilts:
        mock_stilts.side_effect = stilts_side_effect
        # Yield the mock and the list to store calls
        yield {'mock': mock_stilts, 'calls': call_args_list}

def test_epoch_propagation_local_stilts(
    crossmatcher, gaia_df_config, j2000_df_config, mock_stilts_check_propagation
):
    """Verify propagation happens in local_stilts strategy."""
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[gaia_df_config, j2000_df_config]):
        crossmatcher.crossmatch(
            catalogue_1_input="gaia_local_df",
            catalogue_2_input="j2000_local_df",
            radius_arcsec=1.0
        )

    assert mock_stilts_check_propagation['mock'].call_count == 1
    call_info = mock_stilts_check_propagation['calls'][0]
    
    # Config1 should be Gaia, Config2 should be J2000
    conf1 = call_info['config1']
    conf2 = call_info['config2']
    df1 = conf1['_input_dataframe']

    # Check Gaia config/df passed to stilts
    assert conf1['_catalogue_name'] == "gaia_local_df"
    # Verify RA/Dec columns were updated to propagated versions
    assert conf1['ra_column'] == 'ra_propagated'
    assert conf1['dec_column'] == 'dec_propagated'
    # Verify propagated columns exist in the DataFrame
    assert 'ra_propagated' in df1.columns
    assert 'dec_propagated' in df1.columns
    # Verify original coordinates are different from propagated (within tolerance)
    assert not np.isclose(df1['ra_g'].iloc[0], df1['ra_propagated'].iloc[0])
    assert not np.isclose(df1['dec_g'].iloc[0], df1['dec_propagated'].iloc[0])

    # Check J2000 config/df (should not be propagated)
    assert conf2['_catalogue_name'] == "j2000_local_df"
    assert conf2['ra_column'] == 'ra_j'
    assert conf2['dec_column'] == 'dec_j'

def test_epoch_propagation_download_match(
    crossmatcher, remote_gaia_config, j2000_df_config, mock_stilts_check_propagation
):
    """Verify propagation happens in download_and_match strategy."""
    # Mock fetch to return Gaia data
    gaia_data = pd.DataFrame({
        'id_g': [1], 'ra_g': [10.0], 'dec_g': [20.0],
        'pmra_g': [10.0], 'pmdec_g': [-5.0], 'epoch_g': [2016.0]
    })
    with patch.object(crossmatcher, '_fetch_remote_catalogue', return_value=gaia_data), \
         patch.object(crossmatcher, '_get_config_for_input', side_effect=[remote_gaia_config, j2000_df_config]):
        
        crossmatcher.crossmatch(
            catalogue_1_input="gaia_remote",
            catalogue_2_input="j2000_local_df",
            radius_arcsec=1.0
        )

    assert mock_stilts_check_propagation['mock'].call_count == 1
    call_info = mock_stilts_check_propagation['calls'][0]
    
    # Config1 should be Gaia (now marked as local), Config2 should be J2000
    conf1 = call_info['config1'] 
    conf2 = call_info['config2']
    df1 = conf1['_input_dataframe']

    # Check Gaia config/df passed to stilts
    assert conf1['_catalogue_name'] == "gaia_remote"
    assert conf1['access_method'] == 'file_system' # Should be updated after download
    assert conf1['ra_column'] == 'ra_propagated'
    assert conf1['dec_column'] == 'dec_propagated'
    assert 'ra_propagated' in df1.columns
    assert 'dec_propagated' in df1.columns
    assert not np.isclose(df1['ra_g'].iloc[0], df1['ra_propagated'].iloc[0])
    assert not np.isclose(df1['dec_g'].iloc[0], df1['dec_propagated'].iloc[0])

    # Check J2000 config/df
    assert conf2['_catalogue_name'] == "j2000_local_df"
    assert conf2['ra_column'] == 'ra_j'
    assert conf2['dec_column'] == 'dec_j'

# --- Test Matcher Handling ---

@pytest.fixture
def local_err_config_1(tmp_path):
    """Config fixture for local df with error columns."""
    df = pd.DataFrame({
        'id1': [1], 'ra1': [10.0], 'dec1': [20.0],
        'err_ra1': [0.1], 'err_dec1': [0.1], 'corr1': [0.2]
    })
    return {
        "_catalogue_name": "local_err1", "_input_dataframe": df,
        "description": "Local with Errors 1", "archive": None,
        "access_method": "file_system", "is_local": True,
        "ra_column": "ra1", "dec_column": "dec1", "epoch": 2000.0,
        "ra_err_column": "err_ra1", "dec_err_column": "err_dec1", "corr_column": "corr1",
        "pos_err_units": "arcsec"
    }

@pytest.fixture
def local_err_config_2(tmp_path):
    """Config fixture for local df with error columns."""
    df = pd.DataFrame({
        'id2': [101], 'ra2': [10.0001], 'dec2': [20.0001],
        'err_ra2': [0.1], 'err_dec2': [0.1] # No correlation
    })
    return {
        "_catalogue_name": "local_err2", "_input_dataframe": df,
        "description": "Local with Errors 2", "archive": None,
        "access_method": "file_system", "is_local": True,
        "ra_column": "ra2", "dec_column": "dec2", "epoch": 2000.0,
        "ra_err_column": "err_ra2", "dec_err_column": "err_dec2", 
        "pos_err_units": "arcsec"
    }

def test_matcher_selection_local_stilts(
    crossmatcher, local_err_config_1, local_err_config_2, mock_stilts_check_propagation
):
    """Test different matchers are selected and params passed correctly."""
    
    # 1. Test default selection (should be skyerr)
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[local_err_config_1, local_err_config_2]):
        crossmatcher.crossmatch(
            catalogue_1_input="local_err1",
            catalogue_2_input="local_err2",
            # No radius or max_error, use defaults
        )
    assert mock_stilts_check_propagation['mock'].call_count == 1
    call_info1 = mock_stilts_check_propagation['calls'][0]
    # Auto-detection based on errors. Config 1 has corr, Config 2 doesn't -> should default to skyerr
    # Let's re-evaluate _determine_crossmatch_strategy logic: it requires BOTH to have errors for skyerr,
    # and BOTH to have correlation for skyellipse. Here, only config1 has corr, so it should be skyerr.
    assert call_info1['params'].get('matcher') == 'skyerr'
    assert 'max_error' in call_info1['params'] # Default max_error should be used
    assert 'radius_arcsec' not in call_info1['params'] # Radius not used for skyerr

    # Clear calls for next test
    mock_stilts_check_propagation['calls'].clear()

    # 2. Test user override to sky
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[local_err_config_1, local_err_config_2]):
        crossmatcher.crossmatch(
            catalogue_1_input="local_err1",
            catalogue_2_input="local_err2",
            matcher='sky',
            radius_arcsec=5.0 
        )
    assert mock_stilts_check_propagation['mock'].call_count == 2
    call_info2 = mock_stilts_check_propagation['calls'][0] # Index 0 as list was cleared
    assert call_info2['params'].get('matcher') == 'sky'
    assert call_info2['params'].get('radius_arcsec') == 5.0
    assert 'max_error' not in call_info2['params']

# Modify the mock fixture to patch the lower-level stilts function
@pytest.fixture
def mock_stilts_crossmatch_sky():
    """Mocks stilts.crossmatch_sky to inspect its arguments."""
    call_args_list = []
    # Need to patch the function in the module where it's *looked up*
    # which is crossmatch.py where stilts.crossmatch_sky is imported and used.
    with patch('xmatch.crossmatch.crossmatch_sky') as mock_cs:
        def side_effect(*args, **kwargs):
            # Store args/kwargs for inspection
            call_args_list.append({'args': args, 'kwargs': kwargs})
            # Return dummy result path (doesn't matter as we mock read_parquet)
            # Need to return *something* as _execute_local_stilts expects a path
            return "mock_output.parquet" 
        mock_cs.side_effect = side_effect
        # Also mock reading the result file
        with patch('pandas.read_parquet') as mock_read:
            mock_read.return_value = pd.DataFrame({'match': [1]}) 
            yield {'mock': mock_cs, 'calls': call_args_list}

# Test matcher parameter passing to stilts.crossmatch_sky
def test_stilts_matcher_param_passing(
    crossmatcher, local_err_config_1, local_err_config_2, mock_stilts_crossmatch_sky
):
    """Test that correct args (errors, params) are passed to stilts.crossmatch_sky."""
    
    # 1. Test skyerr (auto-detected)
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[local_err_config_1, local_err_config_2]):
        crossmatcher.crossmatch(
            catalogue_1_input="local_err1",
            catalogue_2_input="local_err2",
            # Use default matcher (skyerr)
            max_error=5.0 # Explicitly provide max_error
        )
    assert mock_stilts_crossmatch_sky['mock'].call_count == 1
    kwargs1 = mock_stilts_crossmatch_sky['calls'][0]['kwargs']
    assert kwargs1.get('matcher') == 'skyerr'
    assert kwargs1.get('max_error') == 5.0
    assert kwargs1.get('radius_arcsec') is None
    assert kwargs1.get('err_ra1') == local_err_config_1['ra_err_column']
    assert kwargs1.get('err_dec1') == local_err_config_1['dec_err_column']
    assert kwargs1.get('err_ra2') == local_err_config_2['ra_err_column']
    assert kwargs1.get('err_dec2') == local_err_config_2['dec_err_column']
    assert kwargs1.get('corr1') is None # Config 2 doesn't have corr, so skyellipse not chosen
    assert kwargs1.get('corr2') is None

    # Clear calls
    mock_stilts_crossmatch_sky['calls'].clear()

    # 2. Test skyellipse (force matcher, as config2 lacks corr)
    with patch.object(crossmatcher, '_get_config_for_input', side_effect=[local_err_config_1, local_err_config_2]):
        # Force skyellipse even though config2 lacks corr - crossmatch_sky should handle this internally
        crossmatcher.crossmatch(
            catalogue_1_input="local_err1",
            catalogue_2_input="local_err2",
            matcher='skyellipse', 
            max_error=3.0
        )
    assert mock_stilts_crossmatch_sky['mock'].call_count == 2
    kwargs2 = mock_stilts_crossmatch_sky['calls'][0]['kwargs']
    assert kwargs2.get('matcher') == 'skyellipse'
    assert kwargs2.get('max_error') == 3.0
    assert kwargs2.get('radius_arcsec') is None
    assert kwargs2.get('err_ra1') == local_err_config_1['ra_err_column']
    assert kwargs2.get('err_dec1') == local_err_config_1['dec_err_column']
    assert kwargs2.get('corr1') == local_err_config_1['corr_column']
    assert kwargs2.get('err_ra2') == local_err_config_2['ra_err_column']
    assert kwargs2.get('err_dec2') == local_err_config_2['dec_err_column']
    assert kwargs2.get('corr2') is None # Config 2 still lacks corr

# TODO: Add tests for different matcher types in local strategies 