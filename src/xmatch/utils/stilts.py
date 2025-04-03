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
        error_message = f"STILTS task '{task}' failed with exit code {e.returncode}."
        if e.stderr:
            error_message += f"\nSTILTS stderr:\n{e.stderr}"
        if e.stdout:
             error_message += f"\nSTILTS stdout:\n{e.stdout}"
        logger.error(error_message)
        raise StiltsError(error_message) from e
    except Exception as e:
        error_message = f"An unexpected error occurred while running STILTS task '{task}': {e}"
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
    vizier_id: str, 
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
            "vizcat": vizier_id,
            "radius": radius,
            "find": kwargs.get('find', 'best'),
            "join": kwargs.get('join', '1and2'),
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

@stilts_retry()
def stilts_tapskymatch(
    catalogue_1_df: pd.DataFrame,
    tap_url: str,
    tap_table: str,
    radius: float,
    ra_column_1: str,
    dec_column_1: str,
    ra_column_2: str,
    dec_column_2: str,
    tap_schema: Optional[str] = None,
    columns_2: Optional[List[str]] = None,
    java_opts: Optional[str] = None,
    tmpdir: Optional[str] = None,
    stilts_cmd_base: Optional[str] = None,
    **kwargs
) -> str:
    """Performs cross-match using STILTS tapskymatch."""
    with tempfile.TemporaryDirectory(prefix='stilts_tap_') as temp_dir:
        input_fits = _prepare_input_table(catalogue_1_df, temp_dir, "catalogue_1_tap.fits")
        output_parquet = str(Path(temp_dir) / "output_tap.parquet")
        
        full_tap_table = f"{tap_schema}.{tap_table}" if tap_schema else tap_table
        
        adql = f"SELECT * FROM {full_tap_table}" 
        if columns_2:
             select_cols = ",".join([ra_column_2, dec_column_2] + columns_2)
             adql = f"SELECT {select_cols} FROM {full_tap_table}"
             
        params = {
            "in": input_fits,
            "ifmt": "fits",
            "ra": ra_column_1,
            "dec": dec_column_1,
            "serviceurl": tap_url,
            "adql": adql,
            "taptable": f"{tap_url} table: {full_tap_table}",
            "tra": ra_column_2,
            "tdec": dec_column_2,
            "radius": radius,
            "find": kwargs.get('find', 'best'),
            "join": kwargs.get('join', '1and2'),
            "ocmd": f'keepcols "*"' if not columns_2 else f'keepcols "* { " ".join(columns_2) }"',
            "out": output_parquet,
            "ofmt": "parquet-snappy"
        }
        params.update({k:v for k,v in kwargs.items() if k not in ['find', 'join']}) 
        
        auth_keys = ['user', 'password']
        for key in auth_keys:
             if key in kwargs:
                 params[key] = kwargs[key]

        _run_stilts("tapskymatch", params, java_opts, tmpdir, stilts_cmd_base)
        
        final_output = tempfile.NamedTemporaryFile(suffix='_tap_result.parquet', delete=False).name
        shutil.copy2(output_parquet, final_output)
        logger.info(f"STILTS tapskymatch result saved to: {final_output}")
        return final_output

@stilts_retry()
def stilts_tmatch2(
    catalogue_1_df: pd.DataFrame,
    radius: float,
    ra_column_1: str,
    dec_column_1: str,
    ra_column_2: str,
    dec_column_2: str,
    catalogue_2_path: Optional[Union[str, Path]] = None,
    catalogue_2_df: Optional[Union[pd.DataFrame, Table]] = None,
    target_tap_url: Optional[str] = None,
    target_tap_table: Optional[str] = None,
    target_tap_schema: Optional[str] = None,
    columns_2: Optional[List[str]] = None,
    id_column_1: Optional[str] = None,
    java_opts: Optional[str] = None,
    tmpdir: Optional[str] = None,
    stilts_cmd_base: Optional[str] = None,
    **kwargs
) -> str:
    """Performs cross-match using STILTS tmatch2 (local file or TAP)."""
    
    if not ((catalogue_2_path is not None) or (catalogue_2_df is not None) or 
            (target_tap_url is not None and target_tap_table is not None)):
        raise StiltsError("tmatch2 requires one of: catalogue_2_path, catalogue_2_df, or target_tap_url+target_tap_table")
    
    if (catalogue_2_path is not None and catalogue_2_df is not None) or \
       ((catalogue_2_path is not None or catalogue_2_df is not None) and 
        (target_tap_url is not None or target_tap_table is not None)):
        raise StiltsError("tmatch2 cannot accept multiple catalogue_2 sources (path, df, or TAP)")
    
    if id_column_1:
        logger.warning("ID matching with STILTS tmatch2 not fully implemented. Performing spatial match.")
         
    with tempfile.TemporaryDirectory(prefix='stilts_tm2_') as temp_dir:
        input1_fits = _prepare_input_table(catalogue_1_df, temp_dir, "catalogue_1_tm2.fits")
        output_parquet = str(Path(temp_dir) / "output_tm2.parquet")
        
        params = {
            "in1": input1_fits,
            "ifmt1": "fits",
            "values1": f"{ra_column_1} {dec_column_1}",
            "matcher": kwargs.get('matcher', 'sky'),
            "params": str(radius),
            "find": kwargs.get('find', 'best'),
            "join": kwargs.get('join', '1and2'),
            "out": output_parquet,
            "ofmt": "parquet-snappy"
        }
        
        if catalogue_2_path is not None:
            catalogue_2_file = Path(catalogue_2_path)
            suffix = catalogue_2_file.suffix.lower()
            if suffix == '.fits': target_fmt = 'fits'
            elif suffix == '.parquet': target_fmt = 'parquet'
            elif suffix == '.csv': target_fmt = 'csv'
            else: 
                logger.warning(f"Cannot determine format for local file {catalogue_2_file}, assuming FITS.")
                target_fmt = 'fits'
            params["in2"] = str(catalogue_2_file)
            params["ifmt2"] = target_fmt
            params["values2"] = f"{ra_column_2} {dec_column_2}"
        
        elif catalogue_2_df is not None:
            catalogue_2_file = Path(temp_dir) / "catalogue_2_input.fits"
            if isinstance(catalogue_2_df, pd.DataFrame):
                Table.from_pandas(catalogue_2_df).write(catalogue_2_file, format='fits', overwrite=True)
            else:
                catalogue_2_df.write(catalogue_2_file, format='fits', overwrite=True)
            params["in2"] = str(catalogue_2_file)
            params["ifmt2"] = 'fits'
            params["values2"] = f"{ra_column_2} {dec_column_2}"
        
        else:
            full_tap_table = f"{target_tap_schema}.{target_tap_table}" if target_tap_schema else target_tap_table
            adql = f"SELECT * FROM {full_tap_table}"
            select_cols_list = [ra_column_2, dec_column_2]
            if columns_2:
                select_cols_list.extend(columns_2)
            select_cols = ",".join(list(set(select_cols_list)))
            adql = f"SELECT {select_cols} FROM {full_tap_table}" 
            
            params["in2"] = target_tap_url
            params["icmd2"] = f'tapquery serviceurl={target_tap_url} adql="{adql}"'
            params["values2"] = f"{ra_column_2} {dec_column_2}"
            
            auth_keys = ['user', 'password']
            for key in auth_keys:
                if key in kwargs:
                    params[key] = kwargs[key]

        if columns_2:
            ocmd_cols = ' '.join([f't2_{col}' for col in columns_2])
            params["ocmd"] = f'keepcols "* {ocmd_cols}"'
        else:
            params["ocmd"] = 'keepcols *'
        
        kwargs.pop('matcher', None)
        kwargs.pop('find', None)
        kwargs.pop('join', None)
        params.update(kwargs)
        
        _run_stilts("tmatch2", params, java_opts, tmpdir, stilts_cmd_base)
        
        final_output = tempfile.NamedTemporaryFile(suffix='_tm2_result.parquet', delete=False).name
        shutil.copy2(output_parquet, final_output)
        logger.info(f"STILTS tmatch2 result saved to: {final_output}")
        return final_output

@stilts_retry()
def stilts_tapquery(
    catalogue_1_df: Optional[pd.DataFrame],
    tap_url: str,
    adql_query: str,
    java_opts: Optional[str] = None,
    tmpdir: Optional[str] = None,
    stilts_cmd_base: Optional[str] = None,
    **kwargs
) -> str:
    """Executes a TAP query using STILTS tapquery."""
    if catalogue_1_df is not None:
         logger.debug("Catalogue 1 DataFrame provided to tapquery, but not directly used by STILTS task.")
         
    with tempfile.TemporaryDirectory(prefix='stilts_tquery_') as temp_dir:
        output_parquet = str(Path(temp_dir) / "output_tquery.parquet")
        
        params = {
            "serviceurl": tap_url,
            "adql": adql_query,
            "out": output_parquet,
            "ofmt": "parquet-snappy"
        }
        auth_keys = ['user', 'password']
        for key in auth_keys:
             if key in kwargs:
                 params[key] = kwargs[key]
                 
        params.update({k:v for k,v in kwargs.items() if k not in auth_keys}) 

        _run_stilts("tapquery", params, java_opts, tmpdir, stilts_cmd_base)
        
        final_output = tempfile.NamedTemporaryFile(suffix='_tquery_result.parquet', delete=False).name
        shutil.copy2(output_parquet, final_output)
        logger.info(f"STILTS tapquery result saved to: {final_output}")
        return final_output