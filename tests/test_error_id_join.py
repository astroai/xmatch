#!/usr/bin/env python3
"""Tests for ID-based join error handling."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import logging

from xmatch.crossmatch import CrossMatch
from xmatch.cli import prepare_crossmatch_params

def test_id_join_missing_columns_warning():
    """Test that warning is issued when ID columns are missing but ID join is requested."""
    # Create mock args
    args = MagicMock()
    args.join_on_ids = True
    args.id_column_1 = None
    args.id_column_2 = None
    args.radius_arcsec = 1.0
    args.columns_1 = None
    args.columns_2 = None
    args.ra_column_1 = None
    args.dec_column_1 = None
    args.ra_column_2 = None
    args.dec_column_2 = None
    args.join_type = None
    args.ra = None
    args.dec = None
    args.matcher = None
    args.max_error = None
    args.strategy = None
    args.n_workers = None
    args.chunk_size = None

    # Mock logger and test warning is issued
    with patch('xmatch.cli.logger') as mock_logger:
        params = prepare_crossmatch_params(args)
        
        # Check that warning was logged with the improved error message
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "ID-based join requested but one or more ID columns not specified" in warning_msg
        assert "Will attempt to use default ID columns" in warning_msg
        assert "Specify with --id1 and --id2 for explicit control" in warning_msg
        
        # Check params contains empty join_on_ids dict
        assert 'join_on_ids' in params
        assert params['join_on_ids'] == {}


def test_id_join_specified_columns():
    """Test that ID columns are properly used when specified."""
    # Create mock args with ID columns
    args = MagicMock()
    args.join_on_ids = True
    args.id_column_1 = "source_id"
    args.id_column_2 = "gaia_id"
    args.radius_arcsec = None
    args.columns_1 = None
    args.columns_2 = None
    args.ra_column_1 = None
    args.dec_column_1 = None
    args.ra_column_2 = None
    args.dec_column_2 = None
    args.join_type = None
    args.ra = None
    args.dec = None
    args.matcher = None
    args.max_error = None
    args.strategy = None
    args.n_workers = None
    args.chunk_size = None

    # Check params format with ID columns
    with patch('xmatch.cli.logger'):
        params = prepare_crossmatch_params(args)
        
        # Check join_on_ids structure is correct
        assert 'join_on_ids' in params
        assert params['join_on_ids'] == {
            'cat1': 'source_id',
            'cat2': 'gaia_id'
        }


def test_cli_id_join_integration():
    """Integration test using the CLI to test ID join handling."""
    # Create test data files
    df1 = pd.DataFrame({
        'obj_id': [1, 2, 3],
        'ra': [10.0, 20.0, 30.0],
        'dec': [5.0, 15.0, 25.0]
    })
    
    df2 = pd.DataFrame({
        'gaia_id': [1, 2, 4],  # Note ID 3 missing, 4 added
        'ra': [10.01, 20.01, 40.0],
        'dec': [5.01, 15.01, 35.0]
    })
    
    # Write to temp files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "cat1.csv"
        file2 = Path(tmpdir) / "cat2.csv"
        output = Path(tmpdir) / "output.csv"
        
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        # Mock sys.argv for CLI test
        with patch('sys.argv', [
            'xmatch',
            str(file1),
            str(file2),
            '--join-on-ids',
            '--id1', 'obj_id',
            '--id2', 'gaia_id',
            '-o', str(output)
        ]), patch('xmatch.cli.logger'):
            from xmatch.cli import main
            exit_code = main()
            
            # Check successful execution
            assert exit_code == 0
            assert output.exists()
            
            # Check output contains correct joined data
            result = pd.read_csv(output)
            assert len(result) == 2  # Only IDs 1 and 2 should match
            assert set(result['obj_id'].tolist()) == {1, 2}
            assert set(result['gaia_id'].tolist()) == {1, 2}

def test_cli_id_join_missing_columns():
    """Test CLI behavior when ID join is requested but columns not specified."""
    # Setup
    df1 = pd.DataFrame({
        'source_id': [1, 2, 3],  # Use column name that matches default
        'ra': [10.0, 20.0, 30.0],
        'dec': [5.0, 15.0, 25.0]
    })
    
    df2 = pd.DataFrame({
        'id': [1, 2, 4],  # No matching default column name
        'ra': [10.01, 20.01, 40.0],
        'dec': [5.01, 15.01, 35.0]
    })
    
    # Write to temp files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "cat1.csv"
        file2 = Path(tmpdir) / "cat2.csv"
        
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        # Test with missing --id2 parameter
        with patch('sys.argv', [
            'xmatch',
            str(file1),
            str(file2),
            '--join-on-ids',
            '--id1', 'source_id',
            # Missing --id2
            '-v'  # Verbose to see warning
        ]), patch('xmatch.cli.logger') as mock_logger, \
           patch('xmatch.cli.print'):  # Silence print output
            from xmatch.cli import main
            exit_code = main()
            
            # Should log a warning
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert any("ID-based join requested but one or more ID columns not specified" in msg 
                       for msg in warning_calls)
            
            # Will attempt but fail to find match (should see error about missing ID column)
            assert exit_code != 0  # Should fail