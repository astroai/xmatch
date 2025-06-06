"""
Tests for the error ellipse matching functionality.

This module tests STILTS matching with both skyerr and skyellipse matchers,
verifying that position errors are properly handled with and without correlation.
"""
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from xmatch.crossmatch import CrossMatch, CrossMatchError
from xmatch.stilts import skymatch, StiltsError

# --- Fixtures ---

@pytest.fixture
def source_df_with_errors():
    """Create a DataFrame with simulated sources and position errors."""
    # Create a grid of positions with known errors
    n_sources = 5
    ra_base = 150.0
    dec_base = 30.0
    
    # Generate positions on a grid with varying errors
    df = pd.DataFrame({
        'source_id': [f"src{i+1}" for i in range(n_sources)],
        'ra': [ra_base + 0.01 * i for i in range(n_sources)],
        'dec': [dec_base + 0.01 * i for i in range(n_sources)],
        'ra_err': [0.05 + 0.01 * i for i in range(n_sources)],  # Position errors in arcsec
        'dec_err': [0.06 + 0.01 * i for i in range(n_sources)],  # Slightly different from ra_err
        'correlation': [0.0, 0.2, -0.3, 0.5, -0.7]  # Vary correlation values
    })
    
    return df

@pytest.fixture
def target_df_with_errors():
    """Create a target DataFrame with positions slightly offset from source_df."""
    # Create a grid of positions with known errors
    n_sources = 5
    ra_base = 150.0
    dec_base = 30.0
    
    # Generate positions that are slightly offset from source_df
    # but still within error ellipses for proper testing
    df = pd.DataFrame({
        'target_id': [f"tgt{i+1}" for i in range(n_sources)],
        'ra': [ra_base + 0.01 * i + 0.001 * (i+1) for i in range(n_sources)],  # Small offset
        'dec': [dec_base + 0.01 * i - 0.002 * (i+1) for i in range(n_sources)], # Small offset
        'ra_err': [0.04 + 0.015 * i for i in range(n_sources)],  # Different errors
        'dec_err': [0.05 + 0.012 * i for i in range(n_sources)],  # Different errors
        'correlation': [0.1, -0.1, 0.3, -0.4, 0.6]  # Different correlation values
    })
    
    # Add one source that is far away and shouldn't match
    df = pd.concat([df, pd.DataFrame({
        'target_id': ['no_match'],
        'ra': [ra_base + 1.0],  # Very far from any source
        'dec': [dec_base + 1.0],
        'ra_err': [0.1],
        'dec_err': [0.1],
        'correlation': [0.0]
    })], ignore_index=True)
    
    return df

# --- Direct STILTS Tests ---

def test_skymatch_sky_matching(source_df_with_errors, target_df_with_errors, tmp_path):
    """Test basic sky matching without errors."""
    # Use a radius large enough to find all matches except the one far away
    result_df = skymatch(
        in1=source_df_with_errors,
        in2=target_df_with_errors,
        ra1='ra',
        dec1='dec',
        ra2='ra',
        dec2='dec',
        error=0.05,  # 50 mas - should match sources near grid positions
        matcher='sky',  # Simple position match
        verbose=True
    )
    
    # Should find 5 matches (all sources except the deliberately far one)
    assert len(result_df) == 5
    
    # Verify basic columns are present
    assert 'source_id' in result_df.columns
    assert 'target_id' in result_df.columns
    
    # Verify all source_ids are present
    found_sources = set(result_df['source_id'])
    assert len(found_sources) == 5
    assert all(f"src{i+1}" in found_sources for i in range(5))
    
    # Verify the no_match target is not included
    assert 'no_match' not in set(result_df['target_id'])

def test_skymatch_skyerr_matching(source_df_with_errors, target_df_with_errors, tmp_path):
    """Test error ellipse matching without correlation."""
    # Use error ellipse matching (skyerr)
    result_df = skymatch(
        in1=source_df_with_errors,
        in2=target_df_with_errors,
        ra1='ra',
        dec1='dec',
        ra2='ra',
        dec2='dec',
        error=3.0,  # 3 sigma error scale factor
        matcher='skyerr',  # Error ellipse without correlation
        ra_err1='ra_err',
        dec_err1='dec_err',
        ra_err2='ra_err',
        dec_err2='dec_err',
        verbose=True
    )
    
    # Should find 5 matches (all sources except the deliberately far one)
    assert len(result_df) == 5
    
    # Verify that error columns are present in the output
    assert 'ra_err' in result_df.columns
    assert 'dec_err' in result_df.columns
    
    # Verify error columns from both catalogs are available
    # Names depend on STILTS column naming convention (might have _1, _2 suffixes)
    error_cols = [col for col in result_df.columns if 'err' in col]
    assert len(error_cols) >= 4  # Should have at least ra_err/dec_err for both cats
    
    # Verify separation column is present
    sep_col = [col for col in result_df.columns if col.lower() in ('separation', 'matchdist', 'distance')]
    assert len(sep_col) > 0

def test_skymatch_skyellipse_matching(source_df_with_errors, target_df_with_errors, tmp_path):
    """Test full error ellipse matching with correlation."""
    # Use full error ellipse matching (skyellipse)
    result_df = skymatch(
        in1=source_df_with_errors,
        in2=target_df_with_errors,
        ra1='ra',
        dec1='dec',
        ra2='ra',
        dec2='dec',
        error=3.0,  # 3 sigma error scale factor
        matcher='skyellipse',  # Full error ellipse with correlation
        ra_err1='ra_err',
        dec_err1='dec_err',
        ra_err2='ra_err',
        dec_err2='dec_err',
        ra_dec_corr1='correlation',
        ra_dec_corr2='correlation',
        verbose=True
    )
    
    # Should find 5 matches (all sources except the deliberately far one)
    assert len(result_df) == 5
    
    # Verify that error and correlation columns are present in the output
    assert any('corr' in col.lower() for col in result_df.columns)
    
    # Verify all source_ids are present
    found_sources = set(result_df['source_id'])
    assert len(found_sources) == 5
    assert all(f"src{i+1}" in found_sources for i in range(5))

def test_skymatch_skyellipse_validation(source_df_with_errors, target_df_with_errors):
    """Test validation of required parameters for skyellipse matcher."""
    # Try to use skyellipse without correlation columns
    with pytest.raises(ValueError, match="requires correlation columns"):
        skymatch(
            in1=source_df_with_errors,
            in2=target_df_with_errors,
            ra1='ra',
            dec1='dec',
            ra2='ra',
            dec2='dec',
            error=3.0,
            matcher='skyellipse',  # Full error ellipse with correlation
            ra_err1='ra_err',
            dec_err1='dec_err',
            ra_err2='ra_err',
            dec_err2='dec_err',
            # Missing ra_dec_corr1 and ra_dec_corr2
            verbose=True
        )

# --- Integration Tests with CrossMatch ---

def test_crossmatch_auto_selects_skyerr(source_df_with_errors, target_df_with_errors):
    """Test CrossMatch automatically selects skyerr matcher when error columns are available."""
    # Initialize CrossMatch with tmp config
    xm = CrossMatch()
    
    # Add mock for _execute_local_stilts to capture the selected matcher
    selected_matcher = None
    original_execute_local_stilts = xm._execute_local_stilts
    
    def mock_execute_local_stilts(config1, config2, **params):
        nonlocal selected_matcher
        selected_matcher = params.get("matcher")
        return original_execute_local_stilts(config1, config2, **params)
    
    with patch.object(xm, '_execute_local_stilts', side_effect=mock_execute_local_stilts):
        # Create configs with error columns but no correlation
        source_config = {
            "_catalogue_name": "source_df",
            "_input_dataframe": source_df_with_errors,
            "description": "Source with errors",
            "archive": None,
            "service_type": "local_file",
            "access_method": "file_system",
            "is_local": True,
            "ra_column": "ra",
            "dec_column": "dec",
            "ra_err_column": "ra_err",
            "dec_err_column": "dec_err",
            # No corr_column
        }
        
        target_config = {
            "_catalogue_name": "target_df",
            "_input_dataframe": target_df_with_errors,
            "description": "Target with errors",
            "archive": None,
            "service_type": "local_file",
            "access_method": "file_system",
            "is_local": True,
            "ra_column": "ra",
            "dec_column": "dec",
            "ra_err_column": "ra_err",
            "dec_err_column": "dec_err",
            # No corr_column
        }
        
        with patch.object(xm, '_get_config_for_input', side_effect=[source_config, target_config]):
            # Run crossmatch
            result = xm.crossmatch(
                catalogue_1_input=source_df_with_errors,
                catalogue_2_input=target_df_with_errors,
                radius_arcsec=5.0  # This will be ignored for skyerr matcher
            )
            
            # Check that skyerr matcher was selected
            assert selected_matcher == 'skyerr'
            assert len(result) == 5  # Same 5 matches as direct tests

def test_crossmatch_auto_selects_skyellipse(source_df_with_errors, target_df_with_errors):
    """Test CrossMatch automatically selects skyellipse matcher when error and correlation columns are available."""
    xm = CrossMatch()
    
    # Add mock for _execute_local_stilts to capture the selected matcher
    selected_matcher = None
    original_execute_local_stilts = xm._execute_local_stilts
    
    def mock_execute_local_stilts(config1, config2, **params):
        nonlocal selected_matcher
        selected_matcher = params.get("matcher")
        return original_execute_local_stilts(config1, config2, **params)
    
    with patch.object(xm, '_execute_local_stilts', side_effect=mock_execute_local_stilts):
        # Create configs with both error and correlation columns
        source_config = {
            "_catalogue_name": "source_df",
            "_input_dataframe": source_df_with_errors,
            "description": "Source with errors and correlation",
            "archive": None,
            "service_type": "local_file",
            "access_method": "file_system",
            "is_local": True,
            "ra_column": "ra",
            "dec_column": "dec",
            "ra_err_column": "ra_err",
            "dec_err_column": "dec_err",
            "corr_column": "correlation",
        }
        
        target_config = {
            "_catalogue_name": "target_df",
            "_input_dataframe": target_df_with_errors,
            "description": "Target with errors and correlation",
            "archive": None,
            "service_type": "local_file",
            "access_method": "file_system",
            "is_local": True,
            "ra_column": "ra",
            "dec_column": "dec",
            "ra_err_column": "ra_err",
            "dec_err_column": "dec_err",
            "corr_column": "correlation",
        }
        
        with patch.object(xm, '_get_config_for_input', side_effect=[source_config, target_config]):
            # Run crossmatch
            result = xm.crossmatch(
                catalogue_1_input=source_df_with_errors,
                catalogue_2_input=target_df_with_errors,
                radius_arcsec=5.0  # This will be ignored for skyellipse matcher
            )
            
            # Check that skyellipse matcher was selected
            assert selected_matcher == 'skyellipse'
            assert len(result) == 5  # Same 5 matches as direct tests

def test_crossmatch_user_override_matcher(source_df_with_errors, target_df_with_errors):
    """Test user can override the automatically selected matcher."""
    xm = CrossMatch()
    
    # Add mock for _execute_local_stilts to capture the selected matcher
    selected_matcher = None
    original_execute_local_stilts = xm._execute_local_stilts
    
    def mock_execute_local_stilts(config1, config2, **params):
        nonlocal selected_matcher
        selected_matcher = params.get("matcher")
        return original_execute_local_stilts(config1, config2, **params)
    
    with patch.object(xm, '_execute_local_stilts', side_effect=mock_execute_local_stilts):
        # Create configs with both error and correlation columns
        source_config = {
            "_catalogue_name": "source_df",
            "_input_dataframe": source_df_with_errors,
            "description": "Source with errors and correlation",
            "archive": None,
            "service_type": "local_file",
            "access_method": "file_system",
            "is_local": True,
            "ra_column": "ra",
            "dec_column": "dec",
            "ra_err_column": "ra_err",
            "dec_err_column": "dec_err",
            "corr_column": "correlation",
        }
        
        target_config = {
            "_catalogue_name": "target_df",
            "_input_dataframe": target_df_with_errors,
            "description": "Target with errors and correlation",
            "archive": None,
            "service_type": "local_file",
            "access_method": "file_system",
            "is_local": True,
            "ra_column": "ra",
            "dec_column": "dec",
            "ra_err_column": "ra_err",
            "dec_err_column": "dec_err",
            "corr_column": "correlation",
        }
        
        with patch.object(xm, '_get_config_for_input', side_effect=[source_config, target_config]):
            # Override to sky matcher despite having error/correlation columns
            result = xm.crossmatch(
                catalogue_1_input=source_df_with_errors,
                catalogue_2_input=target_df_with_errors,
                radius_arcsec=0.05,  # Small radius for precise matching
                matcher='sky'  # Override to simple sky matcher
            )
            
            # Check that sky matcher was selected as requested
            assert selected_matcher == 'sky'
            # With the small radius, we should still get matches
            assert len(result) > 0

def test_crossmatch_different_error_units(source_df_with_errors, target_df_with_errors):
    """Test CrossMatch handles different error units correctly."""
    xm = CrossMatch()
    
    # Create a modified dataframe with errors in milliarcseconds
    source_df_mas = source_df_with_errors.copy()
    source_df_mas['ra_err_mas'] = source_df_mas['ra_err'] * 1000  # Convert to mas
    source_df_mas['dec_err_mas'] = source_df_mas['dec_err'] * 1000  # Convert to mas
    
    # Mock config with milliarcsecond error units
    source_config = {
        "_catalogue_name": "source_df_mas",
        "_input_dataframe": source_df_mas,
        "description": "Source with errors in mas",
        "archive": None,
        "service_type": "local_file",
        "access_method": "file_system",
        "is_local": True,
        "ra_column": "ra",
        "dec_column": "dec",
        "ra_err_column": "ra_err_mas",
        "dec_err_column": "dec_err_mas",
        "pos_err_units": "mas",  # Specify milliarcsecond units
    }
    
    target_config = {
        "_catalogue_name": "target_df",
        "_input_dataframe": target_df_with_errors,
        "description": "Target with errors in arcsec",
        "archive": None,
        "service_type": "local_file",
        "access_method": "file_system",
        "is_local": True,
        "ra_column": "ra",
        "dec_column": "dec",
        "ra_err_column": "ra_err",
        "dec_err_column": "dec_err",
        "pos_err_units": "arcsec",  # Default arcsecond units
    }
    
    # Execute with mocked configs
    with patch.object(xm, '_get_config_for_input', side_effect=[source_config, target_config]):
        # Run crossmatch with skyerr
        result = xm.crossmatch(
            catalogue_1_input=source_df_mas,
            catalogue_2_input=target_df_with_errors,
            matcher='skyerr'
        )
        
        # We should still get the same matches despite different units
        assert len(result) == 5