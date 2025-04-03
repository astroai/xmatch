import subprocess
import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import pandas as pd
from astropy.table import Table
import functools
import time

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

@stilts_retry()
def crossmatch_sky(in1, in2, out, ra1, dec1, ra2, dec2, radius=1.0, join_type='1and2',
                  stilts_cmd_base=None, java_opts=None, tmpdir=None, **kwargs):
    """
    Performs spatial cross-matching between two catalogs using STILTS tmatch2.
    
    Args:
        in1: Input file for the first catalog
        in2: Input file for the second catalog
        out: Output file for the result
        ra1: RA column name in the first catalog
        dec1: Dec column name in the first catalog
        ra2: RA column name in the second catalog
        dec2: Dec column name in the second catalog
        radius: Match radius in arcseconds
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
        "matcher": kwargs.get('matcher', 'sky'),
        "values1": f"{ra1} {dec1}",
        "values2": f"{ra2} {dec2}",
        "params": str(radius),
        "join": join_type,
        "find": kwargs.get('find', 'best'),
        "out": out,
        "ofmt": kwargs.get('ofmt', 'auto')
    }
    
    # Add any additional parameters
    for k, v in kwargs.items():
        if k not in ['ifmt1', 'ifmt2', 'matcher', 'find', 'ofmt']:
            params[k] = v
    
    _run_stilts("tmatch2", params, java_opts, tmpdir, stilts_cmd_base)
    
    logger.info(f"STILTS crossmatch_sky result saved to: {out}")
    return out

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