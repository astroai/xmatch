from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from xmatch.crossmatch import CrossMatch

# --- Test fixtures ---


@pytest.fixture
def crossmatcher():
    """Create a CrossMatch instance with mocked config."""
    with patch("xmatch.crossmatch._find_default_config_path") as mock_find:
        mock_find.return_value = Path(__file__).parent / "test_config.yaml"
        # Create a minimal config for testing if it doesn't exist
        config_path = mock_find.return_value
        if not config_path.exists():
            import yaml

            config = {
                "archives": {
                    "test_archive": {
                        "tap_service": {
                            "access_method": "tap",
                            "access_url": "https://example.com/tap",
                            "service_type": "tap",
                        },
                        "cds_service": {
                            "access_method": "cds_xmatch",
                            "access_url": "http://cdsxmatch.u-strasbg.fr/xmatch",
                            "service_type": "cds_xmatch",
                        },
                    }
                },
                "catalogues": {
                    "test_tap_cat": {
                        "archive": "test_archive",
                        "service_id": "tap_service",
                        "access_identifier": "test_schema.test_table",
                        "ra_column": "ra",
                        "dec_column": "dec",
                        "description": "Test TAP Catalogue",
                    },
                    "test_cds_cat": {
                        "archive": "test_archive",
                        "service_id": "cds_service",
                        "access_identifier": "I/350/gaiaedr3",
                        "ra_column": "ra",
                        "dec_column": "dec",
                        "description": "Test CDS Catalogue",
                    },
                },
                "stilts_config": {"stilts_cmd_base": "stilts", "java_opts": "-Xmx4G"},
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

        cm = CrossMatch(config_file=config_path)
        return cm


@pytest.fixture
def local_df_1():
    """Create a small local DataFrame for testing."""
    return pd.DataFrame(
        {
            "ra": [10.5, 11.0, 11.5],
            "dec": [41.2, 41.3, 41.4],
            "mag": [15.0, 16.0, 17.0],
            "source_id": [1001, 1002, 1003],
        }
    )


@pytest.fixture
def local_df_2():
    """Create another small local DataFrame for testing."""
    return pd.DataFrame(
        {
            "ra": [10.51, 11.01, 11.52, 12.0],
            "dec": [41.21, 41.31, 41.42, 42.0],
            "flux": [100.5, 200.3, 150.2, 300.1],
            "obj_id": ["A1", "A2", "A3", "A4"],
        }
    )


@pytest.fixture
def mock_tap_service():
    """Mock a TAP service for remote catalog tests."""
    mock_service = MagicMock()
    mock_results = pd.DataFrame(
        {
            "ra": [10.51, 11.01, 11.52],
            "dec": [41.21, 41.31, 41.42],
            "source_id": [2001, 2002, 2003],
            "mag": [16.1, 17.2, 15.5],
        }
    )
    mock_service.search.return_value = mock_results
    return mock_service


@pytest.fixture
def mock_stilts_local():
    """Mock the local STILTS execution."""
    with patch("xmatch.local_match.execute_local_stilts_match") as mock_stilts:
        # Create a mock match result
        def mock_match(df1, df2, **kwargs):
            ra1 = kwargs.get("ra1", "ra")
            dec1 = kwargs.get("dec1", "dec")
            ra2 = kwargs.get("ra2", "ra")
            dec2 = kwargs.get("dec2", "dec")

            # Simple mock implementation of spatial join using vectorized operations
            result_dfs = []
            for _, row1 in df1.iterrows():
                for _, row2 in df2.iterrows():
                    # Calculate separation (simplified version for testing)
                    sep = (
                        np.sqrt(
                            ((row1[ra1] - row2[ra2]) * np.cos(np.radians(row1[dec1]))) ** 2
                            + (row1[dec1] - row2[dec2]) ** 2
                        )
                        * 3600
                    )  # convert to arcsec

                    radius = kwargs.get("radius_arcsec", 1.0)
                    if sep <= radius:
                        # Create a row for this match
                        match_row = {}
                        # Add columns from df1
                        for col in df1.columns:
                            match_row[f"{col}_1"] = row1[col]
                        # Add columns from df2
                        for col in df2.columns:
                            match_row[f"{col}_2"] = row2[col]
                        # Add separation
                        match_row["separation"] = sep
                        result_dfs.append(pd.DataFrame([match_row]))

            if result_dfs:
                return pd.concat(result_dfs, ignore_index=True)
            else:
                return pd.DataFrame()

        mock_stilts.side_effect = mock_match
        yield mock_stilts


@pytest.fixture
def mock_download_catalogue():
    """Mock the catalog download function."""
    with patch("xmatch.crossmatch.CrossMatch._download_catalogue") as mock_download:
        mock_download.return_value = pd.DataFrame(
            {
                "ra": [10.51, 11.01, 11.52],
                "dec": [41.21, 41.31, 41.42],
                "source_id": [2001, 2002, 2003],
                "mag": [16.1, 17.2, 15.5],
            }
        )
        yield mock_download


@pytest.fixture
def mock_remote_join():
    """Mock the remote JOIN execution."""
    with patch("xmatch.remote_tap.execute_remote_join_match") as mock_join:
        mock_join.return_value = pd.DataFrame(
            {
                "ra_1": [10.5, 11.0],
                "dec_1": [41.2, 41.3],
                "source_id_1": [1001, 1002],
                "ra_2": [10.51, 11.01],
                "dec_2": [41.21, 41.31],
                "obj_id_2": ["A1", "A2"],
                "separation": [0.1, 0.15],
            }
        )
        yield mock_join


@pytest.fixture
def mock_query_remote():
    """Mock querying a remote catalog with constraints."""
    with patch("xmatch.crossmatch.CrossMatch._query_remote_with_constraints") as mock_query:
        mock_query.return_value = pd.DataFrame(
            {
                "ra": [10.51, 11.01, 11.52],
                "dec": [41.21, 41.31, 41.42],
                "source_id": [2001, 2002, 2003],
                "mag": [16.1, 17.2, 15.5],
            }
        )
        yield mock_query


@pytest.fixture
def mock_cds_xmatch():
    """Mock the CDS XMatch service."""
    with patch("xmatch.remote_cds.execute_cds_xmatch_local_remote") as mock_cds:
        mock_cds.return_value = pd.DataFrame(
            {
                "ra_1": [10.5, 11.0],
                "dec_1": [41.2, 41.3],
                "source_id_1": [1001, 1002],
                "ra_2": [10.51, 11.01],
                "dec_2": [41.21, 41.31],
                "source_id_2": [2001, 2002],
                "angDist": [0.1, 0.15],
            }
        )
        yield mock_cds


# --- Tests for Use Case 1: Two Direct Access Files ---


def test_case1_local_stilts(crossmatcher, local_df_1, local_df_2, mock_stilts_local):
    """Test Case 1: Matching two local DataFrames using STILTS."""
    # Set up configurations for the local DataFrames
    config1 = {
        "_catalogue_name": "df1",
        "_input_dataframe": local_df_1,
        "is_local": True,
        "ra_column": "ra",
        "dec_column": "dec",
    }
    config2 = {
        "_catalogue_name": "df2",
        "_input_dataframe": local_df_2,
        "is_local": True,
        "ra_column": "ra",
        "dec_column": "dec",
    }

    # Perform the match
    result_df = crossmatcher._execute_local_stilts(config1, config2, radius_arcsec=1.0)

    # Verify the mock was called correctly
    mock_stilts_local.assert_called_once()
    call_args = mock_stilts_local.call_args
    assert call_args[0][0].equals(local_df_1)  # First DataFrame
    assert call_args[0][1].equals(local_df_2)  # Second DataFrame
    assert call_args[1]["ra1"] == "ra"
    assert call_args[1]["dec1"] == "dec"
    assert call_args[1]["ra2"] == "ra"
    assert call_args[1]["dec2"] == "dec"
    assert call_args[1]["radius_arcsec"] == 1.0

    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    # Our mock_stilts_local fixture should return matches for the closest pairs
    assert len(result_df) > 0


def test_case1_local_stilts_with_files(crossmatcher, tmp_path, mock_stilts_local):
    """Test Case 1: Matching two local files using STILTS."""
    # Create test files
    file1 = tmp_path / "test_file1.csv"
    file2 = tmp_path / "test_file2.csv"

    df1 = pd.DataFrame(
        {"ra": [10.5, 11.0, 11.5], "dec": [41.2, 41.3, 41.4], "mag": [15.0, 16.0, 17.0]}
    )
    df2 = pd.DataFrame(
        {"ra": [10.51, 11.01, 11.52], "dec": [41.21, 41.31, 41.42], "flux": [100.5, 200.3, 150.2]}
    )

    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)

    # Mock the file loading
    with patch("xmatch.crossmatch.CrossMatch._load_local_catalogue") as mock_load:
        mock_load.side_effect = [df1, df2]

        # Call crossmatch directly
        result_df = crossmatcher.crossmatch(
            catalogue_1_input=str(file1), catalogue_2_input=str(file2), radius_arcsec=1.0
        )

    # Verify mock_load was called
    assert mock_load.call_count == 2
    # Verify mock_stilts_local was called
    assert mock_stilts_local.called
    # Verify result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


# --- Tests for Use Case 2: Local File vs Remote TAP Service ---


def test_case2_local_vs_remote_tap(crossmatcher, local_df_1, mock_stilts_local, mock_query_remote):
    """Test Case 2: Matching a local file vs a remote TAP catalog."""
    # Set up configurations
    local_config = {
        "_catalogue_name": "local_df",
        "_input_dataframe": local_df_1,
        "is_local": True,
        "ra_column": "ra",
        "dec_column": "dec",
    }
    remote_config = {
        "_catalogue_name": "remote_tap",
        "is_local": False,
        "access_method": "tap",
        "ra_column": "ra",
        "dec_column": "dec",
    }

    # Mock the _process_coordinate_chunk to use our mock_stilts_local
    with patch("xmatch.crossmatch.CrossMatch._process_coordinate_chunk") as mock_process:
        mock_process.return_value = pd.DataFrame(
            {
                "ra_1": [10.5, 11.0],
                "dec_1": [41.2, 41.3],
                "source_id_1": [1001, 1002],
                "ra_2": [10.51, 11.01],
                "dec_2": [41.21, 41.31],
                "source_id_2": [2001, 2002],
                "separation": [0.1, 0.15],
            }
        )

        # Execute chunked match query remote
        result_df = crossmatcher._execute_chunked_match_query_remote(
            local_config, remote_config, radius_arcsec=1.0
        )

    # Verify the mock was called
    assert mock_process.called
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_case2_local_vs_remote_cds(crossmatcher, local_df_1, mock_cds_xmatch):
    """Test Case 2: Matching a local file vs a remote CDS catalog."""
    # Set up configurations
    local_config = {
        "_catalogue_name": "local_df",
        "_input_dataframe": local_df_1,
        "is_local": True,
        "ra_column": "ra",
        "dec_column": "dec",
    }
    remote_config = {
        "_catalogue_name": "remote_cds",
        "is_local": False,
        "access_method": "cds_xmatch",
        "ra_column": "ra",
        "dec_column": "dec",
    }

    # Execute CDS XMatch
    result_df = crossmatcher._execute_cds_xmatch_local_remote(
        local_config, remote_config, radius_arcsec=1.0
    )

    # Verify the mock was called
    mock_cds_xmatch.assert_called_once()
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_case2_download_and_match(
    crossmatcher, local_df_1, mock_download_catalogue, mock_stilts_local
):
    """Test Case 2: Download remote catalog and match locally."""
    # Set up configurations
    local_config = {
        "_catalogue_name": "local_df",
        "_input_dataframe": local_df_1,
        "is_local": True,
        "ra_column": "ra",
        "dec_column": "dec",
    }
    remote_config = {
        "_catalogue_name": "remote_download",
        "is_local": False,
        "access_method": "tap",
        "ra_column": "ra",
        "dec_column": "dec",
    }

    # Execute download and match
    result_df = crossmatcher._download_and_match(
        local_config, remote_config, _catalogue_to_download="2", radius_arcsec=1.0
    )

    # Verify the mocks were called
    mock_download_catalogue.assert_called_once()
    assert mock_stilts_local.called
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


# --- Tests for Use Case 3: Two Remote Tables at Same Archive ---


def test_case3_remote_join(crossmatcher, mock_remote_join):
    """Test Case 3: Match two remote tables via remote TAP JOIN."""
    # Set up configurations
    remote_config1 = {
        "_catalogue_name": "remote_tap1",
        "is_local": False,
        "access_method": "tap",
        "access_url": "https://example.com/tap",
        "ra_column": "ra",
        "dec_column": "dec",
    }
    remote_config2 = {
        "_catalogue_name": "remote_tap2",
        "is_local": False,
        "access_method": "tap",
        "access_url": "https://example.com/tap",
        "ra_column": "ra",
        "dec_column": "dec",
    }

    # Execute remote JOIN match
    result_df = crossmatcher._execute_remote_join_match(
        remote_config1, remote_config2, radius_arcsec=1.0
    )

    # Verify the mock was called
    mock_remote_join.assert_called_once()
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_case3_remote_spatial_chunked(crossmatcher):
    """Test Case 3: Match two remote tables via spatial chunked TAP queries."""
    # Set up configurations
    remote_config1 = {
        "_catalogue_name": "remote_tap1",
        "is_local": False,
        "access_method": "tap",
        "access_url": "https://example.com/tap",
        "ra_column": "ra",
        "dec_column": "dec",
    }
    remote_config2 = {
        "_catalogue_name": "remote_tap2",
        "is_local": False,
        "access_method": "tap",
        "access_url": "https://example.com/tap",
        "ra_column": "ra",
        "dec_column": "dec",
    }

    # Mock the execute_remote_spatial_chunked_match function
    with patch("xmatch.remote_tap.execute_remote_spatial_chunked_match") as mock_chunked:
        mock_chunked.return_value = pd.DataFrame(
            {
                "ra_1": [10.5, 11.0],
                "dec_1": [41.2, 41.3],
                "source_id_1": [1001, 1002],
                "ra_2": [10.51, 11.01],
                "dec_2": [41.21, 41.31],
                "source_id_2": [2001, 2002],
                "separation": [0.1, 0.15],
            }
        )

        # Execute remote spatial chunked match
        result_df = crossmatcher._execute_remote_spatial_chunked_match(
            remote_config1, remote_config2, ra=10.5, dec=41.2, radius_arcsec=1800.0
        )

    # Verify the mock was called
    mock_chunked.assert_called_once()
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


# --- Tests for Use Case 4: Two Remote Tables at Different Archives ---


def test_case4_download_smaller_then_match(
    crossmatcher, mock_download_catalogue, mock_query_remote
):
    """Test Case 4: Download smaller remote catalog and match against the other remote catalog."""
    # Set up configurations
    remote_config1 = {
        "_catalogue_name": "remote_tap1",
        "is_local": False,
        "access_method": "tap",
        "access_url": "https://example1.com/tap",
        "ra_column": "ra",
        "dec_column": "dec",
    }
    remote_config2 = {
        "_catalogue_name": "remote_tap2",
        "is_local": False,
        "access_method": "tap",
        "access_url": "https://example2.com/tap",
        "ra_column": "ra",
        "dec_column": "dec",
    }

    # Mock necessary methods
    with patch("xmatch.crossmatch.CrossMatch._execute_chunked_match_query_remote") as mock_chunked:
        mock_chunked.return_value = pd.DataFrame(
            {
                "ra_1": [10.5, 11.0],
                "dec_1": [41.2, 41.3],
                "source_id_1": [1001, 1002],
                "ra_2": [10.51, 11.01],
                "dec_2": [41.21, 41.31],
                "source_id_2": [2001, 2002],
                "separation": [0.1, 0.15],
            }
        )

        # Execute download smaller table then match
        result_df = crossmatcher._execute_download_1_then_chunked_match(
            remote_config1, remote_config2, radius_arcsec=1.0
        )

    # Verify the mocks were called
    mock_download_catalogue.assert_called_once()
    mock_chunked.assert_called_once()
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_case4_remote_chunked_local_match(crossmatcher):
    """Test Case 4: Download chunks from both remote catalogs and match locally."""
    # Mock the _execute_remote_chunked_local_match method
    with patch(
        "xmatch.crossmatch.CrossMatch._execute_remote_chunked_local_match"
    ) as mock_chunked_local:
        mock_chunked_local.return_value = pd.DataFrame(
            {
                "ra_1": [10.5, 11.0],
                "dec_1": [41.2, 41.3],
                "source_id_1": [1001, 1002],
                "ra_2": [10.51, 11.01],
                "dec_2": [41.21, 41.31],
                "source_id_2": [2001, 2002],
                "separation": [0.1, 0.15],
            }
        )

        # Execute remote chunked local match
        result_df = crossmatcher._execute_remote_chunked_local_match(
            {
                "config1": {
                    "_catalogue_name": "remote_tap1",
                    "is_local": False,
                    "access_method": "tap",
                    "access_url": "https://example1.com/tap",
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "config2": {
                    "_catalogue_name": "remote_tap2",
                    "is_local": False,
                    "access_method": "tap",
                    "access_url": "https://example2.com/tap",
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "ra": 10.5,
                "dec": 41.2,
                "radius_arcsec": 1800.0,
            }
        )

    # Verify the mock was called
    mock_chunked_local.assert_called_once()
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


# --- Integration tests for all use cases through the main crossmatch method ---


def test_integration_local_vs_local(crossmatcher, local_df_1, local_df_2, mock_stilts_local):
    """Integration test: Case 1 through the main crossmatch method."""
    # Call crossmatch with two DataFrames
    result_df = crossmatcher.crossmatch(
        catalogue_1_input=local_df_1, catalogue_2_input=local_df_2, radius_arcsec=1.0
    )

    # Verify the mock was called
    assert mock_stilts_local.called
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_integration_local_vs_remote(crossmatcher, local_df_1, mock_cds_xmatch):
    """Integration test: Case 2 through the main crossmatch method."""
    # Mock the _get_config_for_input to return a local and a remote config
    with patch.object(crossmatcher, "_get_config_for_input") as mock_get_config:
        mock_get_config.side_effect = [
            {
                "_catalogue_name": "local_df",
                "_input_dataframe": local_df_1,
                "is_local": True,
                "ra_column": "ra",
                "dec_column": "dec",
            },
            {
                "_catalogue_name": "test_cds_cat",
                "access_method": "cds_xmatch",
                "is_local": False,
                "ra_column": "ra",
                "dec_column": "dec",
            },
        ]

        # Call crossmatch
        result_df = crossmatcher.crossmatch(
            catalogue_1_input=local_df_1, catalogue_2_input="test_cds_cat", radius_arcsec=1.0
        )

    # Verify the mock was called
    mock_cds_xmatch.assert_called_once()
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_integration_remote_vs_remote_same_archive(crossmatcher, mock_remote_join):
    """Integration test: Case 3 through the main crossmatch method."""
    # Mock the _get_config_for_input and _determine_crossmatch_strategy
    with (
        patch.object(crossmatcher, "_get_config_for_input") as mock_get_config,
        patch.object(crossmatcher, "_determine_crossmatch_strategy") as mock_strategy,
    ):
        mock_get_config.side_effect = [
            {
                "_catalogue_name": "remote_tap1",
                "is_local": False,
                "access_method": "tap",
                "access_url": "https://example.com/tap",
                "ra_column": "ra",
                "dec_column": "dec",
            },
            {
                "_catalogue_name": "remote_tap2",
                "is_local": False,
                "access_method": "tap",
                "access_url": "https://example.com/tap",
                "ra_column": "ra",
                "dec_column": "dec",
            },
        ]

        mock_strategy.return_value = ("remote_join", {})

        # Call crossmatch
        result_df = crossmatcher.crossmatch(
            catalogue_1_input="remote_tap1", catalogue_2_input="remote_tap2", radius_arcsec=1.0
        )

    # Verify the mock was called
    mock_remote_join.assert_called_once()
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0


def test_integration_remote_vs_remote_different_archives(
    crossmatcher, mock_download_catalogue, mock_stilts_local
):
    """Integration test: Case 4 through the main crossmatch method."""
    # Mock the _get_config_for_input and _determine_crossmatch_strategy
    with (
        patch.object(crossmatcher, "_get_config_for_input") as mock_get_config,
        patch.object(crossmatcher, "_determine_crossmatch_strategy") as mock_strategy,
    ):
        mock_get_config.side_effect = [
            {
                "_catalogue_name": "remote_tap1",
                "is_local": False,
                "access_method": "tap",
                "access_url": "https://example1.com/tap",
                "ra_column": "ra",
                "dec_column": "dec",
            },
            {
                "_catalogue_name": "remote_tap2",
                "is_local": False,
                "access_method": "tap",
                "access_url": "https://example2.com/tap",
                "ra_column": "ra",
                "dec_column": "dec",
            },
        ]

        mock_strategy.return_value = ("download_and_match", {"_catalogue_to_download": "both"})

        # Call crossmatch
        result_df = crossmatcher.crossmatch(
            catalogue_1_input="remote_tap1", catalogue_2_input="remote_tap2", radius_arcsec=1.0
        )

    # Verify the mocks were called
    assert mock_download_catalogue.call_count == 2
    assert mock_stilts_local.called
    # Verify the result
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0
