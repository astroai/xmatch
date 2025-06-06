import logging
import tempfile
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from astropy.table import Table

from .exceptions import CrossMatchError
from .stilts import StiltsError, _run_stilts # Assuming _run_stilts is accessible

logger = logging.getLogger(__name__)

def execute_local_stilts(config1: Dict[str, Any], config2: Dict[str, Any], crossmatch_instance: Any, **params) -> pd.DataFrame:
    """Executes a local match using STILTS when both catalogs are local."""
    logger.info("Executing local STILTS match strategy...")

    df1 = None
    df2 = None

    # Load first catalog
    if "_input_dataframe" in config1:
        df1 = config1["_input_dataframe"]
        logger.info(f"Using in-memory DataFrame for first catalog ({len(df1)} rows)")
    elif "_input_path" in config1:
        local_path = config1["_input_path"]
        logger.info(f"Loading first catalog from local file: {local_path}")
        df1 = crossmatch_instance._load_local_catalogue(local_path) # Use instance method
    else:
        raise CrossMatchError("First catalog marked as local but no data source provided")

    # Load second catalog
    if "_input_dataframe" in config2:
        df2 = config2["_input_dataframe"]
        logger.info(f"Using in-memory DataFrame for second catalog ({len(df2)} rows)")
    elif "_input_path" in config2:
        local_path = config2["_input_path"]
        logger.info(f"Loading second catalog from local file: {local_path}")
        df2 = crossmatch_instance._load_local_catalogue(local_path) # Use instance method
    else:
        raise CrossMatchError("Second catalog marked as local but no data source provided")

    if df1 is None or df2 is None:
         raise CrossMatchError("Failed to load data for one or both local catalogs.")

    if df1.empty or df2.empty:
        logger.warning("One or both catalogs are empty, returning empty result")
        return pd.DataFrame()

    # Prepare parameters, including STILTS config from instance
    local_params = params.copy()
    local_params['ra1'] = config1.get('ra_column')
    local_params['dec1'] = config1.get('dec_column')
    local_params['ra2'] = config2.get('ra_column')
    local_params['dec2'] = config2.get('dec_column')
    local_params['stilts_cmd_base'] = crossmatch_instance.stilts_cmd_base
    local_params['stilts_java_opts'] = crossmatch_instance.stilts_java_opts
    local_params['stilts_tmpdir'] = crossmatch_instance.stilts_tmpdir

    join_mode = local_params.get('join_mode', 'sky')

    if join_mode == 'id':
        join_keys = local_params.get('join_keys', {})
        if not join_keys or 'cat1' not in join_keys or 'cat2' not in join_keys:
            raise CrossMatchError("ID join requested but join_keys missing or incomplete")
        return execute_local_id_join(df1, df2, join_keys['cat1'], join_keys['cat2'], **local_params)
    else:
        return execute_local_stilts_match(df1, df2, **local_params)

def execute_local_stilts_match(df1: pd.DataFrame, df2: pd.DataFrame, **params) -> pd.DataFrame:
    """Executes a local match using STILTS between two pandas DataFrames."""
    # Extract position and error parameters
    ra1 = params.get('ra1')
    dec1 = params.get('dec1')
    ra2 = params.get('ra2')
    dec2 = params.get('dec2')
    
    # Check for position error columns
    ra_err1 = params.get('ra_err1')
    dec_err1 = params.get('dec_err1')
    ra_err2 = params.get('ra_err2')
    dec_err2 = params.get('dec_err2')
    ra_dec_corr1 = params.get('ra_dec_corr1')
    ra_dec_corr2 = params.get('ra_dec_corr2')
    
    # Get matching parameters
    radius_arcsec = params.get('radius_arcsec', 1.0)
    max_error = params.get('max_error', 3.0)  # Default error scale factor for skyerr
    join_type = params.get('join_type', '1and2')
    find = params.get('find', 'best')
    
    # Get STILTS configuration
    stilts_cmd_base = params.get('stilts_cmd_base')
    stilts_java_opts = params.get('stilts_java_opts')
    stilts_tmpdir = params.get('stilts_tmpdir')

    # Validate required parameters
    if not all([ra1, dec1, ra2, dec2]):
        raise CrossMatchError("Missing RA/Dec column names for STILTS match")

    # Determine matcher type based on available error information
    matcher = params.get('matcher')
    if not matcher:
        # Auto-detect matcher if not explicitly specified
        has_errors = all([ra_err1, dec_err1, ra_err2, dec_err2]) 
        has_correlation = ra_dec_corr1 is not None and ra_dec_corr2 is not None
        
        if has_errors:
            if has_correlation:
                matcher = 'skyellipse'  # Use skyellipse when correlation is available
                logger.info("Using 'skyellipse' matcher with error ellipses and correlation")
            else:
                matcher = 'skyerr'  # Use skyerr when only errors without correlation
                logger.info("Using 'skyerr' matcher with position errors")
        else:
            matcher = 'sky'  # Default to sky when no errors available
            logger.info("Using 'sky' matcher with simple positions")
    else:
        logger.info(f"Using '{matcher}' matcher as specified")

    try:
        # Set the error parameter based on matcher type
        error = radius_arcsec if matcher == 'sky' else max_error
        
        # Import skymatch here to avoid circular imports
        from .stilts import skymatch
        
        # Use the skymatch function from stilts.py
        result = skymatch(
            in1=df1,
            in2=df2,
            out=None,  # Return DataFrame directly
            ra1=ra1,
            dec1=dec1,
            ra2=ra2,
            dec2=dec2,
            error=error,  # This is radius_arcsec for sky and max_error for skyerr/skyellipse
            join_type=join_type,
            find=find,
            matcher=matcher,
            # Pass error parameters if available
            ra_err1=ra_err1,
            dec_err1=dec_err1,
            ra_err2=ra_err2,
            dec_err2=dec_err2,
            ra_dec_corr1=ra_dec_corr1,
            ra_dec_corr2=ra_dec_corr2,
            # Output format control
            output_format="parquet",
            # STILTS configuration
            stilts_cmd_base=stilts_cmd_base,
            java_opts=stilts_java_opts,
            tmpdir=stilts_tmpdir,
            # Logging
            verbose=(logger.getEffectiveLevel() <= logging.INFO)
        )
        
        if result is None:
            logger.info("STILTS match returned no results.")
            return pd.DataFrame()
            
        if isinstance(result, pd.DataFrame):
            logger.info(f"STILTS match successful, {len(result)} rows matched.")
            return result
        else:
            # This shouldn't happen as skymatch should return DataFrame when out=None
            logger.warning(f"Unexpected result type from skymatch: {type(result)}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error during local STILTS match: {e}", exc_info=True)
        raise CrossMatchError(f"Error during local STILTS match: {e}") from e

def execute_local_id_join(df1: pd.DataFrame, df2: pd.DataFrame, id_col1: str, id_col2: str, **params) -> pd.DataFrame:
    """Performs a local ID-based join between two DataFrames using pandas."""
    logger.info(f"Performing local ID join on {id_col1} = {id_col2}")

    if id_col1 not in df1.columns:
        raise CrossMatchError(f"ID column '{id_col1}' not found in first catalogue")
    if id_col2 not in df2.columns:
        raise CrossMatchError(f"ID column '{id_col2}' not found in second catalogue")

    try:
        # Determine join type (inner, left, right, outer)
        join_type = params.get('join_type', '1and2') # Default to inner
        how = 'inner'
        if join_type in ('1or2', 'all'): how = 'outer'
        elif join_type == '1not2': how = 'left'
        elif join_type == '2not1': how = 'right'
        # Note: '1not2'/'2not1' in STILTS implies keeping only non-matched rows.
        # pandas merge with how='left'/'right' keeps all rows from the left/right.
        # To replicate STILTS '1not2', use how='left', indicator=True, then filter.
        indicator = False
        if join_type in ('1not2', '2not1'):
             indicator = True

        # Add suffixes to avoid column name collisions
        suffixes = params.get('suffixes', ('_1', '_2'))

        result_df = pd.merge(df1, df2, left_on=id_col1, right_on=id_col2,
                             how=how, suffixes=suffixes, indicator=indicator)

        # Filter for STILTS-like '1not2' or '2not1'
        if join_type == '1not2':
             result_df = result_df[result_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        elif join_type == '2not1':
             result_df = result_df[result_df['_merge'] == 'right_only'].drop(columns=['_merge'])
        elif indicator: # Remove indicator column if not used for filtering
             result_df = result_df.drop(columns=['_merge'])


        logger.info(f"ID join ({how}) completed with {len(result_df)} matches")
        return result_df
    except Exception as e:
        logger.error(f"Error performing ID join: {e}", exc_info=True)
        raise CrossMatchError(f"Error performing ID join: {e}") from e
