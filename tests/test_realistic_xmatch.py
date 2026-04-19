"""
Tests for realistic astronomical catalog cross-matching scenarios.

This module uses realistic catalog data and error properties (based on common catalogs like Gaia)
to test our cross-matching functionality in real-world conditions.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from xmatch.crossmatch import CrossMatch
from xmatch.stilts import skymatch

# --- Test Data Fixtures ---


@pytest.fixture
def gaia_like_sources():
    """
    Create a DataFrame resembling Gaia catalog data with realistic astrometric errors.

    Error model based on Gaia DR3 documentation:
    - Position errors typically 0.01-0.5 mas depending on magnitude
    - Error ellipses with correlation coefficients
    """
    # Create 20 sources with realistic properties
    n_sources = 20
    np.random.seed(42)  # For reproducible tests

    # Generate some random positions around a specific sky region
    ra_base = 180.0  # degrees
    dec_base = 30.0  # degrees

    # Generate source positions with small scatter
    ra = ra_base + np.random.normal(0, 0.1, n_sources)  # scatter in 0.1 deg (~6 arcmin)
    dec = dec_base + np.random.normal(0, 0.1, n_sources)

    # Assign source IDs similar to Gaia style
    source_ids = [6000000000000000000 + i for i in range(n_sources)]

    # Generate realistic magnitudes (G band 12-20)
    g_mag = np.random.uniform(12, 20, n_sources)

    # Error model: brighter stars have smaller errors
    # Error scale in mas (milliarcseconds)
    error_scale = 0.01 + 0.04 * np.exp((g_mag - 12) / 3)  # Exponential increase with magnitude

    # Generate position errors and correlation coefficients
    ra_err_mas = error_scale * np.random.uniform(0.8, 1.2, n_sources)  # Small random variation
    dec_err_mas = error_scale * np.random.uniform(0.8, 1.2, n_sources)
    ra_dec_corr = np.random.uniform(-0.5, 0.5, n_sources)  # Realistic correlation range

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "source_id": source_ids,
            "ra": ra,
            "dec": dec,
            "phot_g_mean_mag": g_mag,
            "ra_error": ra_err_mas,  # mas
            "dec_error": dec_err_mas,  # mas
            "ra_dec_corr": ra_dec_corr,
            "parallax": np.random.normal(1, 0.5, n_sources),  # mas
            "pmra": np.random.normal(0, 5, n_sources),  # mas/year
            "pmdec": np.random.normal(0, 5, n_sources),  # mas/year
            "epoch": np.full(n_sources, 2016.0),  # Gaia DR3 reference epoch
        }
    )

    return df


@pytest.fixture
def wise_like_sources(gaia_like_sources):
    """
    Create a DataFrame resembling WISE catalog data with positions slightly offset from Gaia.

    - Error model based on WISE documentation
    - Position errors typically 80-300 mas, larger than Gaia
    - Most sources should match Gaia sources but with larger errors
    """
    # Get base positions from Gaia fixture but add small offsets
    # Some won't match due to larger offset or being absent in Gaia catalog
    n_wise = 25  # More WISE sources than Gaia
    np.random.seed(84)  # Different seed

    # Use Gaia positions but add small offsets (typical for cross-survey differences)
    n_gaia = len(gaia_like_sources)

    # For first n_gaia sources, use Gaia positions with small offsets
    ra_gaia = gaia_like_sources["ra"].values
    dec_gaia = gaia_like_sources["dec"].values

    # Create offset positions (typically within 0.5 arcsec)
    ra_offset_arcsec = np.random.normal(0, 0.2, n_gaia) / 3600  # Convert to degrees
    dec_offset_arcsec = np.random.normal(0, 0.2, n_gaia) / 3600

    ra_wise = ra_gaia + ra_offset_arcsec
    dec_wise = dec_gaia + dec_offset_arcsec

    # Add some sources that are completely new (not in Gaia)
    ra_base = 180.0
    dec_base = 30.0
    ra_new = ra_base + np.random.normal(0, 0.15, n_wise - n_gaia)
    dec_new = dec_base + np.random.normal(0, 0.15, n_wise - n_gaia)

    # Combine positions
    ra = np.concatenate([ra_wise, ra_new])
    dec = np.concatenate([dec_wise, dec_new])

    # WISE-like source designations
    designations = [
        f"J{int(ra * 100):06d}{'+' if dec >= 0 else '-'}{int(abs(dec) * 100):05d}"
        for ra, dec in zip(ra, dec)
    ]

    # WISE has larger position errors than Gaia
    w1_mag = np.random.uniform(10, 18, n_wise)

    # Error model: brighter stars have smaller errors, but overall larger than Gaia
    # Errors in milliarcseconds
    error_scale = 80 + 20 * np.exp((w1_mag - 10) / 4)  # 80-300 mas typical for WISE
    ra_err = error_scale * np.random.uniform(0.8, 1.2, n_wise)
    dec_err = error_scale * np.random.uniform(0.8, 1.2, n_wise)

    # WISE doesn't provide error correlation coefficients

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "designation": designations,
            "ra": ra,
            "dec": dec,
            "w1mpro": w1_mag,
            "w2mpro": w1_mag + np.random.normal(0, 0.5, n_wise),
            "ra_err": ra_err,  # mas
            "dec_err": dec_err,  # mas
            "cc_flags": np.random.choice(["0000", "0f00", "h000"], n_wise),
            "epoch": np.full(n_wise, 2010.5),  # WISE reference epoch
        }
    )

    return df


# --- Tests for Realistic Cross-matching Scenarios ---


def test_gaia_wise_xmatch_skyerr():
    """Test realistic Gaia-WISE cross-matching using skyerr matcher."""
    # Initialize CrossMatch
    CrossMatch()

    # Create test data
    gaia_df = gaia_like_sources()
    wise_df = wise_like_sources(gaia_df)

    # Use skyerr matcher (both catalogs have position errors but no correlation info for WISE)
    result_df = skymatch(
        in1=gaia_df,
        in2=wise_df,
        ra1="ra",
        dec1="dec",
        ra2="ra",
        dec2="dec",
        error=5.0,  # 5 sigma match criterion
        matcher="skyerr",  # Error ellipse without correlation
        ra_err1="ra_error",
        dec_err1="dec_error",
        ra_err2="ra_err",
        dec_err2="dec_err",
        # Correctly handle different units
        verbose=True,
    )

    # Since WISE errors are much larger than Gaia, matching should be driven by Gaia precision
    # Should match at least 80% of Gaia sources to WISE
    n_gaia = len(gaia_df)
    n_matched = len(result_df)

    # Check we get a reasonable number of matches (at least 80% of Gaia sources)
    assert n_matched >= 0.8 * n_gaia, f"Only matched {n_matched}/{n_gaia} sources"

    # Calculate match distance statistics
    if "separation" in result_df.columns:
        sep_col = "separation"
    elif "matchdist" in result_df.columns:
        sep_col = "matchdist"
    else:
        sep_cols = [c for c in result_df.columns if "dist" in c.lower()]
        assert len(sep_cols) > 0, "No separation column found in result"
        sep_col = sep_cols[0]

    # With error-based matching, separations should be reasonable
    # given the error ellipses (typically < 1 arcsec for most matches)
    assert result_df[sep_col].median() < 1.0, (
        f"Median separation too large: {result_df[sep_col].median():.3f} arcsec"
    )
    assert result_df[sep_col].max() < 5.0, (
        f"Maximum separation too large: {result_df[sep_col].max():.3f} arcsec"
    )


def test_gaia_proper_motion_propagation():
    """Test error ellipse matching with proper motion propagation."""
    # Initialize CrossMatch
    xm = CrossMatch()

    # Create test data
    gaia_df = gaia_like_sources()
    wise_df = wise_like_sources(gaia_df)

    # Enhance Gaia data with significant proper motions
    gaia_df["pmra"] = 100.0  # 100 mas/yr (large for testing)
    gaia_df["pmdec"] = -50.0  # -50 mas/yr

    # Calculate expected position shift from Gaia epoch (2016.0) to WISE epoch (2010.5)
    # Delta years: 2010.5 - 2016.0 = -5.5 years
    # Expected RA shift: pmra * -5.5 = -550 mas = -0.55 arcsec
    # Expected Dec shift: pmdec * -5.5 = 275 mas = 0.275 arcsec

    # Without propagation, we'd expect fewer matches due to ~0.6 arcsec offset
    # which is significant compared to Gaia error ellipses

    # With properly configured CrossMatch, it should automatically propagate
    # Gaia positions to the WISE epoch when matching

    # Mock configs that include proper motion and epoch info
    gaia_config = {
        "_catalogue_name": "gaia_test",
        "_input_dataframe": gaia_df,
        "description": "Gaia-like test catalog",
        "archive": None,
        "service_type": "local_file",
        "access_method": "file_system",
        "is_local": True,
        "ra_column": "ra",
        "dec_column": "dec",
        "ra_err_column": "ra_error",
        "dec_err_column": "dec_error",
        "corr_column": "ra_dec_corr",
        "pos_err_units": "mas",
        "epoch": 2016.0,
        "pm_ra_column": "pmra",
        "pm_dec_column": "pmdec",
        "pm_units": "mas/yr",
    }

    wise_config = {
        "_catalogue_name": "wise_test",
        "_input_dataframe": wise_df,
        "description": "WISE-like test catalog",
        "archive": None,
        "service_type": "local_file",
        "access_method": "file_system",
        "is_local": True,
        "ra_column": "ra",
        "dec_column": "dec",
        "ra_err_column": "ra_err",
        "dec_err_column": "dec_err",
        "pos_err_units": "mas",
        "epoch": 2010.5,
    }

    # Test with CrossMatch instance using mocked configs
    with patch.object(xm, "_get_config_for_input", side_effect=[gaia_config, wise_config]):
        result_with_pm = xm.crossmatch(
            catalogue_1_input=gaia_df, catalogue_2_input=wise_df, matcher="skyerr"
        )

    # Modify gaia_config to disable proper motion propagation by removing key columns
    gaia_config_no_pm = gaia_config.copy()
    gaia_config_no_pm.pop("pm_ra_column")
    gaia_config_no_pm.pop("pm_dec_column")

    # Test without proper motion propagation
    with patch.object(xm, "_get_config_for_input", side_effect=[gaia_config_no_pm, wise_config]):
        result_no_pm = xm.crossmatch(
            catalogue_1_input=gaia_df, catalogue_2_input=wise_df, matcher="skyerr"
        )

    # We expect more matches with proper motion propagation
    assert len(result_with_pm) > len(result_no_pm), (
        f"Expected more matches with PM propagation but got {len(result_with_pm)} vs {len(result_no_pm)}"
    )
