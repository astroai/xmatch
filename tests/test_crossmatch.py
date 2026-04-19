from pathlib import Path
from unittest.mock import MagicMock, patch

import astropy.units as u
import pandas as pd
import pytest
from astropy.table import Table

# Adjust import based on project structure
from xmatch.crossmatch import CrossMatch, CrossMatchError

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
                    "param_A": "service_default",
                },
                "xmatch_service": {"access_method": "cds_xmatch", "param_B": "service_default"},
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
                "param_A": "catalogue_override",
            }
        },
        "crossmatch_methods": {"stilts_tmatch2_sky": {"executor": "stilts"}},
        "stilts_config": {"java_opts": "-Xmx1g"},
    }


# --- Helper to instantiate CrossMatch with mocked config ---


def get_cm_with_mock_config(mock_config):
    # Patch _load_config to return our mock config
    # Patch _validate_config to prevent it running on potentially invalid mock configs during tests
    # Patch auth.load_auth_config as it's not needed for these config tests
    with (
        patch.object(CrossMatch, "_load_config", return_value=mock_config),
        patch.object(CrossMatch, "_validate_config", return_value=None),
        patch("xmatch.auth.load_auth_config", return_value={}),
    ):
        # The config_file path doesn't matter here as load is mocked
        cm = CrossMatch(config_file="dummy_path.yaml")
        # Manually set the config attributes that __init__ would normally set
        cm.config = mock_config
        cm.archives_config = mock_config.get("archives", {})
        cm.catalogues_config = mock_config.get("catalogues", {})
        cm.methods_config = mock_config.get("crossmatch_methods", {})
        cm.stilts_config = mock_config.get("stilts_config", {})
        return cm


@pytest.fixture
def crossmatcher(minimal_valid_config):
    """Provides a CrossMatch instance with a minimal valid mock config."""
    return get_cm_with_mock_config(minimal_valid_config)


# --- Tests for _validate_config ---
# We test this by calling it directly on mock configs, not via __init__


def test_validate_config_valid(minimal_valid_config):
    """Test validation with a minimal valid config."""
    # Instantiate with a dummy path, then manually assign and validate
    with patch("xmatch.auth.load_auth_config", return_value={}):
        cm = CrossMatch.__new__(CrossMatch)  # Create instance without calling __init__
        cm.config = minimal_valid_config
        # Should run without errors
        cm._validate_config()


@pytest.mark.parametrize(
    "invalid_config",
    [
        ({}),  # Missing top-level keys
        ({"archives": [], "catalogues": {}}),  # Wrong type for top-level
        ({"archives": {"a": {}}, "catalogues": []}),  # Wrong type for top-level
        ({"archives": {"a": []}, "catalogues": {"c": {}}}),  # Wrong type for archive config
        (
            {
                "archives": {"a": {"s": []}},
                "catalogues": {
                    "c": {
                        "archive": "a",
                        "service_id": "s",
                        "access_identifier": "id",
                        "ra_column": "r",
                        "dec_column": "d",
                    }
                },
            }
        ),  # Wrong type for service config
        (
            {"archives": {"a": {"s": {}}}, "catalogues": {"c": []}}
        ),  # Wrong type for catalogue config
        (
            {"archives": {"a": {"s": {}}}, "catalogues": {"c": {"archive": "b"}}}
        ),  # Missing required cat key (service_id etc)
        (
            {"archives": {"a": {"s": {}}}, "catalogues": {"c": {"archive": "b", "service_id": "s"}}}
        ),  # Archive 'b' not defined
        (
            {"archives": {"a": {"s": {}}}, "catalogues": {"c": {"archive": "a", "service_id": "t"}}}
        ),  # Service 't' not defined
    ],
)
def test_validate_config_invalid(invalid_config):
    """Test validation fails for various invalid config structures."""
    with patch("xmatch.auth.load_auth_config", return_value={}):
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
    del bad_config["catalogues"]["gaia_cds"]["archive"]  # Remove a required key
    # Need to mock _validate_config differently here as it would normally catch this
    with (
        patch.object(CrossMatch, "_load_config", return_value=bad_config),
        patch("xmatch.auth.load_auth_config", return_value={}),
    ):
        cm = CrossMatch(config_file="dummy_path.yaml")
        cm.config = bad_config  # Override config after init
        cm.catalogues_config = bad_config.get("catalogues", {})
        cm.archives_config = bad_config.get("archives", {})
        # We need to skip the normal validation to reach the get_catalogue_config error
        with patch.object(CrossMatch, "_validate_config", return_value=None):
            with pytest.raises(CrossMatchError, match="missing 'archive' or 'service_id'"):
                cm.get_catalogue_config("gaia_cds")


# --- Tests for _get_config_for_input ---


@patch("pathlib.Path.is_file")
def test_get_config_for_input_local_file(mock_is_file, minimal_valid_config):
    """Test config generation for a local file input."""
    mock_is_file.return_value = True  # Mock that the path is a file
    cm = get_cm_with_mock_config(minimal_valid_config)

    file_path = Path("data/my_cat.fits")
    local_config = cm._get_config_for_input(file_path)

    assert local_config["is_local"] is True
    assert local_config["_input_path"] == str(file_path)
    assert local_config["_catalogue_name"] == "my_cat"
    assert local_config["description"] == f"Local file: {file_path.name}"
    assert local_config["archive"] is None
    assert local_config["service_type"] == "local_file"
    assert local_config["ra_column"] == "ra"  # Default assumption
    assert local_config["dec_column"] == "dec"  # Default assumption


@patch("pathlib.Path.is_file")
def test_get_config_for_input_catalogue_name(mock_is_file, minimal_valid_config):
    """Test config retrieval for a configured catalogue name."""
    mock_is_file.return_value = False  # Mock that the path is not a file
    cm = get_cm_with_mock_config(minimal_valid_config)

    # We expect get_catalogue_config to be called for a non-file string
    with patch.object(
        cm, "get_catalogue_config", wraps=cm.get_catalogue_config
    ) as mock_get_cat_conf:
        resolved_config = cm._get_config_for_input("gaia_cds")
        mock_get_cat_conf.assert_called_once_with("gaia_cds")

        # Check it returns the resolved config from the main config
        assert resolved_config["_catalogue_name"] == "gaia_cds"
        assert resolved_config["archive"] == "cds"
        assert resolved_config.get("is_local") is not True


@patch("pathlib.Path.is_file")
def test_get_config_for_input_unknown_string(mock_is_file, minimal_valid_config):
    """Test error when input string is neither a file nor a known catalogue."""
    mock_is_file.return_value = False
    cm = get_cm_with_mock_config(minimal_valid_config)

    with pytest.raises(
        CrossMatchError,
        match="Input string 'unknown_cat' is not a valid file path or known catalogue name",
    ):
        cm._get_config_for_input("unknown_cat")


def test_get_config_for_input_dataframe(minimal_valid_config):
    """Test config generation for a DataFrame input."""
    cm = get_cm_with_mock_config(minimal_valid_config)
    df_input = pd.DataFrame({"ra": [1], "dec": [1]})  # Dummy DataFrame

    df_config = cm._get_config_for_input(df_input)

    assert df_config["is_local"] is True
    assert df_config["_input_path"] is None  # No path for DataFrame
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
                    "access_url": "tap",
                    "access_method": "tap",
                    "column_selection_supported": True,
                    "adql_features": ["table_join"],
                },
                "xmatch_service": {
                    "access_method": "cds_xmatch",
                    "column_selection_supported": False,
                    "crossmatch_input_methods": ["remote_table_query"],
                },
                "service_priority": ["tap_service", "xmatch_service"],
            },
            "noao": {
                "description": "NOAO Archive",
                "tap_service": {
                    "access_url": "tap",
                    "access_method": "tap",
                    "column_selection_supported": True,
                    "adql_features": ["table_join", "table_upload"],
                },
                "service_priority": ["tap_service"],
            },
        },
        "catalogues": {
            "gaia_cds_tap": {
                "archive": "cds",
                "service_id": "tap_service",
                "release": "DR3",
                "access_identifier": "gaia",
                "ra_column": "ra",
                "dec_column": "dec",
                "best_method": "tap_join",
            },
            "gaia_cds_xmatch": {
                "archive": "cds",
                "service_id": "xmatch_service",
                "release": "DR3",
                "access_identifier": "gaia_xmatch",
                "ra_column": "ra",
                "dec_column": "dec",
                "best_method": "cds_xmatch",
            },
            "ukidss_noao": {
                "archive": "noao",
                "service_id": "tap_service",
                "release": "DR11",
                "access_identifier": "ukidss",
                "ra_column": "ra",
                "dec_column": "dec",
                "best_method": "tap_join",
            },
        },
        "crossmatch_methods": {
            "stilts_tmatch2_sky": {
                "executor": "stilts",
                "description": "STILTS Sky Match",
                "params": {"matcher": "sky"},
            },
            "stilts_tmatchn_id": {
                "executor": "stilts",
                "description": "STILTS ID Match",
                "params": {"matcher": "exact"},
            },
            "tap_join": {"executor": "tap", "description": "TAP ADQL Join"},
            "cds_xmatch": {"executor": "cds", "description": "CDS X-Match Service"},
        },
        "stilts_config": {},
    }


# --- Tests for _determine_best_method ---


@pytest.mark.parametrize(
    "cat1_type, cat2_type, user_method, id_match, expected_method_name",
    [
        # User override always wins
        ("local", "local", "tap_join", False, "tap_join"),
        ("remote_tap", "remote_cds", "stilts_tmatch2_sky", False, "stilts_tmatch2_sky"),
        ("local", "local", "stilts_tmatchn_id", True, "stilts_tmatchn_id"),  # User override ID
        # No override - defaults based on type
        ("local", "local", None, False, "stilts_tmatch2_sky"),  # Local vs Local -> STILTS sky
        ("local", "local", None, True, "stilts_tmatchn_id"),  # Local vs Local ID -> STILTS ID
        ("local", "remote_tap", None, False, "tap_join"),  # Local vs Remote TAP -> TAP Upload/Join
        (
            "local",
            "remote_cds",
            None,
            False,
            "cds_xmatch",
        ),  # Local vs Remote CDS -> CDS XMatch Service
        (
            "remote_tap",
            "remote_tap",
            None,
            False,
            "tap_join",
        ),  # Remote TAP vs Remote TAP -> TAP Join
        (
            "remote_tap",
            "remote_cds",
            None,
            False,
            "tap_join",
        ),  # TAP preferred over CDS if available for primary
        ("remote_cds", "remote_tap", None, False, "tap_join"),  # TAP preferred even if cat2 is TAP
        (
            "remote_cds",
            "remote_cds",
            None,
            False,
            "cds_xmatch",
        ),  # Both CDS -> CDS XMatch (assuming no TAP preference)
        # ID matching without user override
        ("local", "remote_tap", None, True, "tap_join"),  # ID match via TAP
        ("remote_tap", "remote_tap", None, True, "tap_join"),  # ID match via TAP
    ],
)
def test_determine_best_method(
    cat1_type, cat2_type, user_method, id_match, expected_method_name, method_test_config
):
    """Test determination of the best crossmatch method."""
    cm = get_cm_with_mock_config(method_test_config)

    # Create mock configs based on type
    if cat1_type == "local":
        config1 = cm._get_config_for_input(Path("local1.fits"))
    elif cat1_type == "remote_tap":
        config1 = cm.get_catalogue_config("gaia_cds_tap")
    else:  # remote_cds
        config1 = cm.get_catalogue_config("gaia_cds_xmatch")

    if cat2_type == "local":
        config2 = cm._get_config_for_input(Path("local2.fits"))
    elif cat2_type == "remote_tap":
        config2 = cm.get_catalogue_config("ukidss_noao")
    else:  # remote_cds
        config2 = cm.get_catalogue_config("gaia_cds_xmatch")

    # Mock Path.is_file for local file checks within _determine_best_method if needed
    with patch("pathlib.Path.is_file", return_value=True):
        method_name, method_config = cm._determine_best_method(
            config1,
            config2,
            method_override=user_method,
            id_column_1="id1" if id_match else None,
            id_column_2="id2" if id_match else None,
        )

    assert method_name == expected_method_name
    assert (
        method_config["description"]
        == method_test_config["crossmatch_methods"][expected_method_name]["description"]
    )


def test_determine_best_method_no_suitable(method_test_config):
    """Test error when no suitable method can be found (e.g., invalid user override)."""
    cm = get_cm_with_mock_config(method_test_config)
    config1 = cm._get_config_for_input(Path("local1.fits"))
    config2 = cm._get_config_for_input(Path("local2.fits"))

    with patch("pathlib.Path.is_file", return_value=True):
        with pytest.raises(CrossMatchError, match="No suitable cross-match method found"):
            cm._determine_best_method(config1, config2, method_override="nonexistent_method")

        # Test case where ID match is requested but no ID method defined
        simplified_config = method_test_config.copy()
        simplified_config["crossmatch_methods"] = {
            "stilts_tmatch2_sky": simplified_config["crossmatch_methods"]["stilts_tmatch2_sky"]
        }
        cm_simple = get_cm_with_mock_config(simplified_config)
        with pytest.raises(
            CrossMatchError, match="No suitable cross-match method found for ID matching"
        ):
            cm_simple._determine_best_method(config1, config2, id_column_1="id1", id_column_2="id2")


# --- Fixtures ---


@pytest.fixture
def mock_astroquery():
    """Mocks astroquery.xmatch.XMatch for testing."""
    with patch("xmatch.crossmatch.XMatch") as mock_xmatch_cls:
        mock_instance = MagicMock()
        # Simulate a successful query returning an Astropy Table
        mock_result_table = Table(
            {
                "angDist": [0.5, 1.0] * u.arcsec,
                "ra_local": [10.0, 20.1],
                "dec_local": [5.0, 6.1],
                "id_local": [101, 102],
                "ra_remote": [10.0001, 20.1001],
                "dec_remote": [5.0001, 6.1001],
                "ID_remote": ["viz1", "viz2"],
            }
        )
        mock_instance.query_async.return_value = mock_result_table
        mock_xmatch_cls.return_value = mock_instance
        yield mock_xmatch_cls  # Yield the class mock itself


@pytest.fixture
def local_df_config(tmp_path):
    """Config fixture for a local DataFrame input."""
    df = pd.DataFrame(
        {"id_local": [101, 102, 103], "ra_local": [10.0, 20.1, 30.2], "dec_local": [5.0, 6.1, 7.2]}
    )
    return {
        "_catalogue_name": "local_df",
        "_input_dataframe": df,
        "description": "Local DataFrame",
        "archive": None,
        "service_type": "local_file",  # Use same type for simplicity
        "access_method": "file_system",
        "is_local": True,
        "ra_column": "ra_local",
        "dec_column": "dec_local",
        "epoch": 2000.0,  # Example epoch
        # Add error columns if testing skyerr etc.
    }


@pytest.fixture
def remote_cds_config():
    """Config fixture for a remote catalogue using CDS XMatch."""
    return {
        "_catalogue_name": "vizier_cat",
        "description": "Remote VizieR Catalogue (CDS)",
        "archive": "cds",
        "service_id": "xmatch_service",  # Corresponds to access_method: cds_xmatch
        "access_method": "cds_xmatch",
        "access_identifier": "I/355/gaiadr3",  # Example VizieR ID
        "ra_column": "RA_ICRS",  # These aren't used by CDS query but good practice
        "dec_column": "DE_ICRS",
        "epoch": 2016.0,
        # Error columns etc. would go here if needed for other strategies
    }


# --- Tests ---


def test_crossmatch_strategy_cds_local_remote(crossmatcher, local_df_config, remote_cds_config):
    """Test strategy selection for local vs remote CDS."""
    strategy, params = crossmatcher._determine_crossmatch_strategy(
        local_df_config, remote_cds_config, radius_arcsec=2.0
    )
    assert strategy == "cds_xmatch_local_remote"


def test_crossmatch_cds_local_remote_success(
    crossmatcher, local_df_config, remote_cds_config, mock_astroquery
):
    """Test successful execution of the CDS XMatch strategy."""
    # Mock _get_config_for_input to return our test configs
    with patch.object(
        crossmatcher, "_get_config_for_input", side_effect=[local_df_config, remote_cds_config]
    ):
        result_df = crossmatcher.crossmatch(
            catalogue_1_input="local_df_placeholder",  # Names don't matter due to patch
            catalogue_2_input="remote_cds_placeholder",
            radius_arcsec=2.0,
        )

    # Check mock was called
    mock_astroquery.return_value.query_async.assert_called_once()
    call_args, call_kwargs = mock_astroquery.return_value.query_async.call_args
    assert isinstance(call_kwargs["cat1"], Table)  # Input should be Astropy Table
    assert call_kwargs["cat2"] == f"vizier:{remote_cds_config['access_identifier']}"
    assert call_kwargs["max_distance"].value == 2.0
    assert call_kwargs["colRA1"] == local_df_config["ra_column"]
    assert call_kwargs["colDec1"] == local_df_config["dec_column"]

    # Check result type and basic content
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2  # Based on mock_result_table
    assert "angDist" in result_df.columns
    assert "id_local" in result_df.columns  # From local input
    assert "ID_remote" in result_df.columns  # From mocked remote table


def test_crossmatch_cds_local_remote_missing_radius(
    crossmatcher, local_df_config, remote_cds_config, mock_astroquery
):
    """Test CDS XMatch strategy failure when radius is missing."""
    with patch.object(
        crossmatcher, "_get_config_for_input", side_effect=[local_df_config, remote_cds_config]
    ):
        with pytest.raises(ValueError, match="Missing required parameter 'radius_arcsec'"):
            crossmatcher.crossmatch(
                catalogue_1_input="local_df_placeholder",
                catalogue_2_input="remote_cds_placeholder",
                # No radius_arcsec provided
            )


def test_crossmatch_cds_local_remote_query_fails(
    crossmatcher, local_df_config, remote_cds_config, mock_astroquery
):
    """Test CDS XMatch strategy when astroquery fails."""
    # Configure mock to raise an exception
    mock_astroquery.return_value.query_async.side_effect = Exception("CDS Query Failed")

    with patch.object(
        crossmatcher, "_get_config_for_input", side_effect=[local_df_config, remote_cds_config]
    ):
        with pytest.raises(CrossMatchError, match="Error during CDS XMatch query"):
            crossmatcher.crossmatch(
                catalogue_1_input="local_df_placeholder",
                catalogue_2_input="remote_cds_placeholder",
                radius_arcsec=2.0,
            )


# --- Tests for handle_archive_override ---


def test_handle_archive_override(crossmatcher, minimal_valid_config):
    """Test the handle_archive_override function that appends archive prefixes."""
    # Get the crossmatch instance with our minimal config
    cm = crossmatcher

    # Add some test catalog with archive prefix
    cm.config["catalogues"]["test_cds"] = {"archive": "cds", "service_id": "tap_service"}
    cm.config["catalogues"]["test_esa"] = {"archive": "esa_gaia", "service_id": "tap_service"}

    # Test simple case (no override needed)
    result = cm.handle_archive_override("test_cds", "cds", cm)
    assert result == "test_cds"

    # Test when archive override applies (no existing match)
    result = cm.handle_archive_override("test", "cds", cm)
    assert result == "test_cds"

    # Test when override doesn't match any known catalog
    result = cm.handle_archive_override("unknown", "cds", cm)
    assert result == "unknown"  # Should return original when no match

    # Test with alias resolution
    cm.config["catalogue_aliases"] = {"test_alias": "test_cds"}
    result = cm.handle_archive_override("test_alias", "esa", cm)
    assert result == "test_alias"  # Alias resolved to test_cds, so override not needed


# --- Tests for archive override in CLI ---


@patch("xmatch.cli.CrossMatch")
def test_cli_archive_override(mock_cm_class, minimal_valid_config):
    """Test that CLI uses handle_archive_override correctly."""
    # Set up mock instance
    mock_cm = mock_cm_class.return_value
    mock_cm.config = minimal_valid_config
    mock_cm.crossmatch.return_value = pd.DataFrame()

    # Mock the handle_archive_override method
    mock_cm.handle_archive_override = MagicMock(side_effect=lambda name, arch, cm: f"{name}_{arch}")

    # Run CLI with archive override args
    with patch("sys.argv", ["xmatch", "cat1", "cat2", "--archive-1", "cds", "--archive-2", "esa"]):
        from xmatch.cli import main

        main()

    # Check that handle_archive_override was called with right args
    mock_cm.handle_archive_override.assert_any_call("cat1", "cds", mock_cm)
    mock_cm.handle_archive_override.assert_any_call("cat2", "esa", mock_cm)
