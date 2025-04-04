import subprocess
import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Tuple
import pandas as pd
from astropy.table import Table
import functools
import time
import math # Added for unit conversion
from astropy import units as u

logger = logging.getLogger(__name__)

class StiltsError(Exception):
    """Custom exception for STILTS command errors."""
    pass

def stilts_retry(max_retries=3, delay=5):
    """Decorator that retries a STILTS function if it fails with certain recoverable errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except StiltsError as e:
                    # Only retry for specific error patterns that might be transient
                    if any(pattern in str(e) for pattern in ["connection reset", "timeout", "temporary failure"]):
                        retries += 1
                        if retries < max_retries:
                            sleep_time = delay * (2 ** (retries - 1))  # Exponential backoff
                            logger.warning(f"STILTS operation failed, retrying in {sleep_time}s: {e}")
                            time.sleep(sleep_time)
                        else:
                            logger.error(f"STILTS operation failed after {max_retries} retries: {e}")
                            raise
                    else:
                        # Not a retriable error
                        raise
        return wrapper
    return decorator

def _build_stilts_command(task: str, params: Dict[str, Any], 
                         java_opts: Optional[str] = None, 
                         tmpdir: Optional[str] = None,
                         stilts_cmd_base: Optional[str] = None) -> List[str]:
    """
    Constructs the full STILTS command line arguments.
    Uses stilts_cmd_base if provided, otherwise constructs from components.
    """
    if stilts_cmd_base:
        # Use the user-provided base command directly
        logger.debug(f"Using provided stilts_cmd_base: '{stilts_cmd_base}'")
        # Cache the split command if it's the same as last time
        if not hasattr(_build_stilts_command, "_last_cmd_base") or _build_stilts_command._last_cmd_base != stilts_cmd_base:
            _build_stilts_command._last_cmd_base = stilts_cmd_base
            _build_stilts_command._cached_cmd = stilts_cmd_base.split()
        cmd = _build_stilts_command._cached_cmd.copy()
    else:
        logger.debug("Constructing STILTS command from java_opts/tmpdir/STILTS_JAR.")
        cmd = ["java"]
        java_options = []
        if java_opts:
            java_options.extend(java_opts.split()) 
        if tmpdir:
             java_options.append(f"-Djava.io.tmpdir={tmpdir}")
        cmd.extend(java_options)
        
        stilts_jar = os.getenv("STILTS_JAR", "stilts.jar") 
        cmd.extend(["-jar", stilts_jar])
        cmd.extend(["-verbose", "-disk"]) 
    
    cmd.append(task)
    
    for key, value in params.items():
        if value is not None:
             if isinstance(value, list):
                 value_str = ' '.join(map(str, value))
             else:
                 value_str = str(value)
             cmd.append(f"{key}={value_str}")
             
    logger.debug(f"Constructed STILTS command args: {cmd}")
    return cmd

def _run_stilts(task: str, params: Dict[str, Any], 
               java_opts: Optional[str] = None, 
               tmpdir: Optional[str] = None,
               stilts_cmd_base: Optional[str] = None):
    """Runs a STILTS task using subprocess."""
    command_args = _build_stilts_command(task, params, java_opts, tmpdir, stilts_cmd_base)
    command_str = ' '.join(command_args)
    logger.info(f"Executing STILTS task: {task}...")
    logger.debug(f"Running command: {command_str}")
    
    try:
        process = subprocess.run(command_args, 
                                 capture_output=True, 
                                 text=True, 
                                 check=True,
                                 encoding='utf-8')
        logger.debug(f"STILTS stdout:\n{process.stdout}")
        if process.stderr:
            logger.warning(f"STILTS stderr:\n{process.stderr}")
        logger.info(f"STILTS task '{task}' completed successfully.")
        
    except FileNotFoundError:
         raise StiltsError(f"STILTS command failed. 'java' or '{os.getenv('STILTS_JAR', 'stilts.jar')}' not found. Ensure Java is installed and STILTS_JAR environment variable or stilts.jar is accessible.")
    except subprocess.CalledProcessError as e:
        error_message = f"STILTS task '{task}' failed with exit code {e.returncode}.\nFull command: {command_str}"
        if e.stderr:
            error_message += f"\nSTILTS stderr:\n{e.stderr}"
        if e.stdout:
             error_message += f"\nSTILTS stdout:\n{e.stdout}"
        logger.error(error_message)
        raise StiltsError(error_message) from e
    except Exception as e:
        error_message = f"An unexpected error occurred while running STILTS task '{task}': {e}\nFull command: {command_str}"
        logger.exception(error_message)
        raise StiltsError(error_message) from e

def _prepare_input_table(df: pd.DataFrame, temp_dir: str, filename: str = "input.fits") -> str:
    """Writes a DataFrame to a temporary FITS file for STILTS."""
    temp_file_path = Path(temp_dir) / filename
    try:
        table = Table.from_pandas(df)
        table.write(temp_file_path, format='fits', overwrite=True)
        logger.debug(f"Wrote temporary input table to: {temp_file_path}")
        return str(temp_file_path)
    except Exception as e:
        raise StiltsError(f"Failed to write temporary input FITS file {temp_file_path}: {e}") from e

@stilts_retry()
def stilts_cdsskymatch(
    catalogue_1_df: pd.DataFrame, 
    cds_id: str, 
    radius: float, 
    ra_column_1: str, 
    dec_column_1: str, 
    columns_2: Optional[List[str]] = None, 
    java_opts: Optional[str] = None, 
    tmpdir: Optional[str] = None,
    stilts_cmd_base: Optional[str] = None,
    **kwargs
) -> str:
    """Performs cross-match using STILTS cdsskymatch."""
    with tempfile.TemporaryDirectory(prefix='stilts_cds_') as temp_dir:
        input_fits = _prepare_input_table(catalogue_1_df, temp_dir, "catalogue_1_cds.fits")
        output_parquet = str(Path(temp_dir) / "output_cds.parquet")
        
        params = {
            "in": input_fits,
            "ifmt": "fits",
            "ra": ra_column_1,
            "dec": dec_column_1,
            "cdstable": cds_id,
            "radius": radius,
            "find": kwargs.get('find', 'best'),
            "ocmd": f'keepcols "*"' if not columns_2 else f'keepcols "* { " ".join(columns_2) }"',
            "out": output_parquet,
            "ofmt": "parquet-snappy"
        }
        params.update({k:v for k,v in kwargs.items() if k not in ['find', 'join']}) 

        _run_stilts("cdsskymatch", params, java_opts, tmpdir, stilts_cmd_base)
        
        final_output = tempfile.NamedTemporaryFile(suffix='_cds_result.parquet', delete=False).name
        shutil.copy2(output_parquet, final_output)
        logger.info(f"STILTS cdsskymatch result saved to: {final_output}")
        return final_output

def cdsskymatch(in1, out, ra, dec, cds_id, radius=1.0, find='best', cdscols=None,
              stilts_cmd_base=None, java_opts=None, tmpdir=None, verbose=True):
    """
    Cross-match with a VizieR catalog using the CDS XMatch service.
    
    Args:
        in1: Input file for the first catalog
        out: Output file for the result
        ra: RA column name in the first catalog
        dec: Dec column name in the first catalog
        cds_id: CDS catalog ID (e.g., "I/355/gaiadr3" or "simbad")
        radius: Match radius in arcseconds
        find: Finding mode ('best', 'all', 'each')
        cdscols: Comma-separated list of columns to retrieve from the VizieR catalog
        stilts_cmd_base: Base STILTS command
        java_opts: Java options
        tmpdir: Temporary directory
        verbose: Whether to output verbose information

    Returns:
        True if successful
    """
    try:
        # Build parameters dictionary for STILTS command
        params = {
            "in": in1,
            "ra": ra,
            "dec": dec,
            "radius": radius,
            "find": find,
            "out": out,
            "cdstable": cds_id
        }
        
        # Add optional columns if specified
        if cdscols:
            params["cdscols"] = cdscols
            
        if verbose:
            logger.info(f"Running STILTS cdsskymatch")
            
        # Run the command using existing function
        _run_stilts("cdsskymatch", params, java_opts, tmpdir, stilts_cmd_base)
            
        return True
        
    except Exception as e:
        logger.error(f"Error in STILTS cdsskymatch: {str(e)}")
        raise StiltsError(f"Error during STILTS cdsskymatch: {str(e)}")

def _get_error_config(config: Dict[str, Any], axis: str) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    """
    Extracts error configuration (column name, units, floor error) for a given axis (ra/dec).
    Prioritizes IVAR columns if available.

    Returns:
        Tuple: (error_col_name, ivar_col_name, floor_error_arcsec, units)
           - error_col_name: Name of the direct error column, or None.
           - ivar_col_name: Name of the inverse variance column, or None.
           - floor_error_arcsec: Default floor error from catalogue config, or None.
           - units: Units of the error/ivar column (before conversion), or 'arcsec' if only floor is used.
    """
    err_col = config.get(f'{axis}_err_column')
    ivar_col = config.get(f'{axis}_ivar_column')
    units = config.get('pos_err_units', 'arcsec') # Default to arcsec if not specified
    floor = config.get('default_pos_error_arcsec')

    if ivar_col:
        logger.debug(f"Using IVAR column '{ivar_col}' for {axis} axis.")
        # Units for ivar are typically arcsec^-2, resulting error is arcsec
        # But let config override if units are different (e.g., deg^-2)
        return None, ivar_col, floor, units 
    elif err_col:
        logger.debug(f"Using error column '{err_col}' for {axis} axis with units '{units}'.")
        return err_col, None, floor, units
    elif floor is not None:
        logger.debug(f"Using catalogue default floor error {floor} arcsec for {axis} axis.")
        return None, None, floor, 'arcsec' # Floor error is always arcsec
    else:
        logger.warning(f"No error, IVAR, or default floor error configuration found for axis '{axis}' in config: {config.get('name', 'Unknown')}. STILTS may fail if errors are required.")
        return None, None, None, 'arcsec' # No error info available


def _build_error_value_expression(err_col_name: Optional[str], 
                                 ivar_col_name: Optional[str],
                                 floor_error_arcsec: Optional[float],
                                 units: str) -> str:
    """
    Builds a STILTS expression to get the error value in degrees.
    Handles unit conversion, IVAR conversion, NULLs, and floor error fallback.
    """
    if not err_col_name and not ivar_col_name and floor_error_arcsec is None:
        # Should not happen if _get_error_config logic is sound, but defensively return 0
        logger.error("Error expression builder called with no error source! Returning 0.")
        return "0.0"

    # Default to catalogue's floor error if primary source is missing/invalid
    # If no catalogue floor error, use a tiny default? STILTS might handle this.
    # Let's rely on catalogue floor error first.
    fallback_floor_deg = (floor_error_arcsec * u.arcsec).to_value(u.deg) if floor_error_arcsec is not None else 1e-9 # Tiny fallback if no floor defined
    fallback_expr = f"{fallback_floor_deg}"

    if ivar_col_name:
        # Error = 1 / sqrt(ivar)
        # Handle units: ivar units are typically value/arcsec^2 or value/deg^2
        try:
            # Assuming units are like 'arcsec' or 'deg', inferring ivar units as unit^-2
            unit_power = -2
            base_unit = u.Unit(units)
            ivar_unit = base_unit**unit_power
            # Factor to convert error (sqrt(1/ivar)) from base_unit to degrees
            factor_to_deg = (1 * base_unit).to_value(u.deg) 
        except ValueError:
             logger.warning(f"Could not parse IVAR units '{units}'. Assuming arcsec^-2.")
             factor_to_deg = (1 * u.arcsec).to_value(u.deg)
        
        # Expression: Convert 1/sqrt(ivar) to degrees.
        # Handle NULL, zero, or negative IVAR values gracefully.
        # Use MAX(ivar_col, 1e-18) to avoid sqrt(0) or sqrt(<0). 1e-18 corresponds to ~1e9 arcsec error.
        # Use COALESCE to provide the floor error if IVAR is NULL.
        ivar_expr = f"({factor_to_deg} / sqrt(MAX({ivar_col_name}, 1e-18)))" # Error in degrees
        return f"COALESCE({ivar_expr}, {fallback_expr})"

    elif err_col_name:
        try:
            unit = u.Unit(units)
            factor_to_deg = (1 * unit).to_value(u.deg)
        except ValueError:
            logger.warning(f"Could not parse error units '{units}'. Assuming degrees.")
            factor_to_deg = 1.0

        # Use COALESCE to handle NULL error values, falling back to floor error.
        # Use ABS in case error is signed (unlikely but possible).
        err_expr = f"ABS({err_col_name}) * {factor_to_deg}"
        return f"COALESCE({err_expr}, {fallback_expr})"
    
    else: # Only floor error is available
        return fallback_expr

def _build_correlation_expression(col_name: Optional[str]) -> str:
    """Builds a STILTS expression for correlation, defaulting to 0 if missing."""
    if col_name:
        # Handle NULLs, default to 0 correlation
        return f"COALESCE({col_name}, 0.0)"
    else:
        return "0.0"

def crossmatch_sky(
    in1: str, 
    in2: str, 
    out: str, 
    ra1: str, dec1: str, 
    ra2: str, dec2: str,
    matcher: str, # 'sky', 'skyerr', 'skyellipse'
    config1: Dict[str, Any],
    config2: Dict[str, Any],
    max_error: float = 3.0, # Max separation in units of sigma for error matchers
    radius_arcsec: float = 1.0, # Fallback radius for 'sky' matcher
    join_type: str = '1and2', 
    stilts_cmd_base: Optional[str] = None, 
    java_opts: Optional[str] = None, 
    tmpdir: Optional[str] = None, 
    **kwargs
) -> str:
    """
    Performs spatial cross-matching between two catalogs using STILTS tmatch2.
    Supports different matchers: sky, skyerr, skyellipse.
    
    Args:
        in1: Input file path for table 1.
        in2: Input file path for table 2.
        out: Output file path for the result.
        ra1, dec1: RA/Dec column names in table 1.
        ra2, dec2: RA/Dec column names in table 2.
        matcher: The STILTS matcher to use ('sky', 'skyerr', 'skyellipse').
        config1: Configuration dictionary for table 1.
        config2: Configuration dictionary for table 2.
        max_error: Maximum separation in units of error (sigma) for skyerr/skyellipse.
        radius_arcsec: Match radius in arcseconds (used only for matcher='sky').
        join_type: Type of join ('1and2', '1or2', 'all1', 'all2', etc.).
        stilts_cmd_base: Base STILTS command.
        java_opts: Java options.
        tmpdir: Temporary directory.
        **kwargs: Additional parameters to pass directly to tmatch2.

    Returns:
        Output file path if successful.
    """
    params = {
        "in1": in1,
        "in2": in2,
        "out": out,
        "ofmt": kwargs.pop('ofmt', 'parquet-snappy'), # Default to parquet
        "join": join_type,
        "matcher": matcher
    }

    # --- Configure matcher-specific parameters --- 
    if matcher == 'skyerr' or matcher == 'skyellipse':
        ra_err_col1, ra_ivar_col1, floor1_ra, units1_ra = _get_error_config(config1, 'ra')
        dec_err_col1, dec_ivar_col1, floor1_dec, units1_dec = _get_error_config(config1, 'dec')
        # Use the same floor error and units for both axes if derived from ivar/err col
        floor1 = floor1_ra if floor1_ra is not None else floor1_dec
        units1 = units1_ra if ra_err_col1 or ra_ivar_col1 else units1_dec
        
        ra_err_col2, ra_ivar_col2, floor2_ra, units2_ra = _get_error_config(config2, 'ra')
        dec_err_col2, dec_ivar_col2, floor2_dec, units2_dec = _get_error_config(config2, 'dec')
        floor2 = floor2_ra if floor2_ra is not None else floor2_dec
        units2 = units2_ra if ra_err_col2 or ra_ivar_col2 else units2_dec

        # Get correlation column names
        corr_col1 = config1.get('corr_column')
        corr_col2 = config2.get('corr_column')

        # Build values expressions (RA, Dec, ErrRA, ErrDec, [Corr])
        # Errors must be converted to degrees
        err_expr_ra1 = _build_error_value_expression(ra_err_col1, ra_ivar_col1, floor1, units1)
        err_expr_dec1 = _build_error_value_expression(dec_err_col1, dec_ivar_col1, floor1, units1)
        err_expr_ra2 = _build_error_value_expression(ra_err_col2, ra_ivar_col2, floor2, units2)
        err_expr_dec2 = _build_error_value_expression(dec_err_col2, dec_ivar_col2, floor2, units2)
        
        # Add error parameters for skyerr and skyellipse
        if matcher == 'skyerr':
            params["values1"] = f"{ra1} {dec1} {err_expr_ra1} {err_expr_dec1}"
            params["values2"] = f"{ra2} {dec2} {err_expr_ra2} {err_expr_dec2}"
            params["params"] = str(max_error) # Separation in units of error
            logger.info(f"Using {matcher} matcher with max_error={max_error}. Units: deg.")
            logger.debug(f"  values1: {params['values1']}")
            logger.debug(f"  values2: {params['values2']}")
        elif matcher == 'skyellipse':
            params["values1"] = f"{ra1} {dec1} {err_expr_ra1} {err_expr_dec1} {_build_correlation_expression(corr_col1)}"
            params["values2"] = f"{ra2} {dec2} {err_expr_ra2} {err_expr_dec2} {_build_correlation_expression(corr_col2)}"
            params["params"] = str(max_error) # Separation in units of error
            logger.info(f"Using {matcher} matcher with max_error={max_error}. Units: deg.")
            logger.debug(f"  values1: {params['values1']}")
            logger.debug(f"  values2: {params['values2']}")

    elif matcher == 'sky':
        params["values1"] = f"{ra1} {dec1}"
        params["values2"] = f"{ra2} {dec2}"
        params["params"] = str(radius_arcsec) # Separation in arcsec
        logger.info(f"Using sky matcher with radius={radius_arcsec} arcsec.")
    else:
        raise StiltsError(f"Unsupported STILTS matcher specified: '{matcher}'")

    # Add any extra kwargs passed by the user
    params.update(kwargs)

    try:
        _run_stilts("tmatch2", params, java_opts, tmpdir, stilts_cmd_base)
        logger.info(f"STILTS tmatch2 ({matcher}) completed successfully. Output: {out}")
        return out
    except StiltsError as e:
        logger.error(f"STILTS tmatch2 ({matcher}) failed.")
        raise # Re-raise the specific StiltsError

@stilts_retry()
def crossmatch_id(in1, in2, out, id_column_1, id_column_2, join_type='1and2',
                 stilts_cmd_base=None, java_opts=None, tmpdir=None, **kwargs):
    """
    Performs ID-based cross-matching between two catalogs using STILTS tmatch2.
    
    Args:
        in1: Input file for the first catalog
        in2: Input file for the second catalog
        out: Output file for the result
        id_column_1: ID column name in the first catalog
        id_column_2: ID column name in the second catalog
        join_type: Type of join ('1and2', '1or2', 'all1', 'all2', etc.)
        stilts_cmd_base: Base STILTS command
        java_opts: Java options
        tmpdir: Temporary directory
        **kwargs: Additional parameters to pass to tmatch2

    Returns:
        Output file path if successful
    """
    params = {
        "in1": in1,
        "in2": in2,
        "ifmt1": kwargs.get('ifmt1', 'auto'),
        "ifmt2": kwargs.get('ifmt2', 'auto'),
        "matcher": "exact",
        "values1": id_column_1,
        "values2": id_column_2,
        "join": join_type,
        "find": "all",
        "out": out,
        "ofmt": kwargs.get('ofmt', 'auto')
    }
    
    # Add any additional parameters
    for k, v in kwargs.items():
        if k not in ['ifmt1', 'ifmt2', 'find', 'ofmt']:
            params[k] = v
    
    _run_stilts("tmatch2", params, java_opts, tmpdir, stilts_cmd_base)
    
    logger.info(f"STILTS ID cross-match result saved to: {out}")
    return out