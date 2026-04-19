import functools
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.io.fits.verify import VerifyError
from astropy.table import Table

# Use centralized exceptions
from .exceptions import StiltsError

logger = logging.getLogger(__name__)


def stilts_retry(max_retries=3, delay=5):
    """Decorator to retry STILTS operations on specific transient errors."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except StiltsError as e:
                    last_exception = e
                    error_str_lower = str(e).lower()
                    if any(
                        pattern in error_str_lower
                        for pattern in [
                            "connection reset",
                            "timeout",
                            "temporary failure",
                            "broken pipe",
                            "service unavailable",
                        ]
                    ):
                        retries += 1
                        if retries < max_retries:
                            sleep_time = (delay * (2 ** (retries - 1))) + (
                                np.random.rand() * delay * 0.5
                            )
                            logger.warning(
                                f"STILTS operation failed (Attempt {retries}/{max_retries}), retrying in {sleep_time:.2f}s: {e}"
                            )
                            time.sleep(sleep_time)
                        else:
                            logger.error(
                                f"STILTS operation failed after {max_retries} retries: {e}"
                            )
                            raise
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error during STILTS operation: {e}", exc_info=True)
                    raise StiltsError(f"Unexpected error in STILTS operation: {e}") from e

            if last_exception:
                raise last_exception
            raise StiltsError("STILTS retry logic finished unexpectedly.")

        return wrapper

    return decorator


def _build_stilts_command(
    task: str,
    params: Dict[str, Any],
    java_opts: Optional[str] = None,
    tmpdir: Optional[str] = None,
    stilts_cmd_base: Optional[str] = None,
) -> List[str]:
    """
    Constructs the full STILTS command line arguments.
    Uses stilts_cmd_base if provided, otherwise constructs from components.
    """
    cmd: List[str]
    if stilts_cmd_base:
        logger.debug(f"Using provided stilts_cmd_base: '{stilts_cmd_base}'")
        try:
            cmd = shlex.split(stilts_cmd_base)
            if not cmd:
                raise ValueError("Provided stilts_cmd_base resulted in empty command list.")
        except Exception as e:
            raise StiltsError(
                f"Failed to parse provided stilts_cmd_base '{stilts_cmd_base}': {e}"
            ) from e
    else:
        logger.debug("Constructing STILTS command from java_opts/tmpdir/STILTS_JAR.")
        cmd = ["java"]
        java_options = []
        if java_opts:
            java_options.extend(shlex.split(java_opts))
        if tmpdir:
            java_options.append(f"-Djava.io.tmpdir={tmpdir}")
        cmd.extend(java_options)

        stilts_jar_env = os.getenv("STILTS_JAR")
        stilts_jar_path = None

        if stilts_jar_env:
            if Path(stilts_jar_env).exists():
                stilts_jar_path = stilts_jar_env
                logger.debug(
                    f"Using STILTS JAR from environment variable STILTS_JAR: {stilts_jar_path}"
                )
            else:
                logger.warning(
                    f"STILTS_JAR environment variable set to '{stilts_jar_env}', but file not found."
                )

        if not stilts_jar_path:
            stilts_jar_default = "stilts.jar"
            if shutil.which(stilts_jar_default):
                stilts_jar_path = shutil.which(stilts_jar_default)
                logger.debug(f"Found '{stilts_jar_default}' in PATH: {stilts_jar_path}")
            elif Path(stilts_jar_default).exists():
                stilts_jar_path = stilts_jar_default
                logger.debug(f"Found '{stilts_jar_default}' in current directory.")
            else:
                raise StiltsError(
                    f"Could not find STILTS JAR. Set STILTS_JAR environment variable, "
                    f"ensure '{stilts_jar_default}' is in PATH, or place it in the current directory."
                )

        cmd.extend(["-jar", stilts_jar_path])
        cmd.extend(["-disk"])

    cmd.append(task)

    for key, value in params.items():
        if value is None:
            continue

        if isinstance(value, bool):
            value_str = str(value).lower()
        elif isinstance(value, list):
            value_str = " ".join(map(str, value))
        else:
            value_str = str(value)

        cmd.append(f"{key}={value_str}")

    logger.debug(f"Constructed STILTS command args: {cmd}")
    return cmd


def _run_stilts(
    task: str,
    params: Dict[str, Any],
    java_opts: Optional[str] = None,
    tmpdir: Optional[str] = None,
    stilts_cmd_base: Optional[str] = None,
    _raw_command: Optional[str] = None,
):
    """Runs a STILTS task using subprocess or executes a raw command string."""
    command_args: List[str]
    command_str: str

    if _raw_command:
        logger.warning(f"Executing raw STILTS command provided by user: {_raw_command}")
        try:
            command_args = shlex.split(_raw_command)
            if not command_args:
                raise ValueError("Provided raw STILTS command string is empty or invalid.")
            logger.debug(f"Ignoring intended task '{task}' due to raw command override.")
        except Exception as e:
            raise StiltsError(f"Failed to parse raw STILTS command '{_raw_command}': {e}") from e
        command_str = _raw_command
    else:
        try:
            command_args = _build_stilts_command(task, params, java_opts, tmpdir, stilts_cmd_base)
        except (ValueError, StiltsError) as e:
            raise StiltsError(f"Failed to build STILTS command for task '{task}': {e}") from e
        command_str = " ".join(shlex.quote(arg) for arg in command_args)

    logger.info("Executing STILTS task/command...")
    logger.debug(f"Running command: {command_str}")

    try:
        process = subprocess.run(
            command_args,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
        if logger.isEnabledFor(logging.DEBUG):
            stdout_log = process.stdout.strip() if process.stdout else "(empty)"
            logger.debug(f"STILTS stdout:\n{stdout_log}")

        stderr_log = process.stderr.strip() if process.stderr else "(empty)"
        if stderr_log != "(empty)":
            logger.warning(f"STILTS stderr:\n{stderr_log}")

        logger.info(f"STILTS task/command '{task}' completed successfully.")

    except FileNotFoundError as e:
        missing_cmd = e.filename or command_args[0]
        msg = (
            f"STILTS command failed: '{missing_cmd}' not found. "
            f"Ensure Java/command is installed and accessible via the system PATH, "
            f"or STILTS_JAR environment variable is set correctly."
        )
        logger.error(msg)
        raise StiltsError(msg) from e

    except subprocess.CalledProcessError as e:
        error_message = (
            f"STILTS task '{task}' failed with exit code {e.returncode}."
            f"\nFull command: {command_str}"
            f"\nSTILTS stderr:\n{e.stderr.strip() if e.stderr else '(empty)'}"
        )
        if e.stdout:
            error_message += f"\nSTILTS stdout:\n{e.stdout.strip()}"
        logger.error(error_message)
        raise StiltsError(error_message) from e

    except OSError as e:
        error_message = (
            f"OS error during STILTS execution for task '{task}': {e}. Command: {command_str}"
        )
        logger.error(error_message, exc_info=True)
        raise StiltsError(error_message) from e


def _prepare_input_table(df: pd.DataFrame, temp_dir: str, filename: str = "input.fits") -> str:
    """Writes a DataFrame to a temporary FITS file for STILTS, handling empty input."""
    if not filename.lower().endswith((".fits", ".fit")):
        original_suffix = Path(filename).suffix
        filename = Path(filename).stem + ".fits"
        logger.debug(
            f"Input filename suffix '{original_suffix}' changed to '.fits' for STILTS compatibility."
        )

    temp_file_path = Path(temp_dir) / filename

    if df.empty:
        logger.warning(f"Input DataFrame for '{filename}' is empty. Writing empty FITS file.")
        try:
            if list(df.columns):
                empty_table = Table({col: [] for col in df.columns})
            else:
                empty_table = Table()
            empty_table.write(temp_file_path, format="fits", overwrite=True)
            logger.debug(f"Empty FITS file written to: {temp_file_path}")
            return str(temp_file_path)
        except Exception as e:
            raise StiltsError(
                f"Failed to write empty temporary input FITS file {temp_file_path}: {e}"
            ) from e

    try:
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=["object"]).columns:
            try:
                pd.to_numeric(df_copy[col].dropna())
            except (ValueError, TypeError):
                logger.debug(f"Converting object column '{col}' to string for FITS output.")
                df_copy[col] = df_copy[col].fillna("").astype(str)

        table = Table.from_pandas(df_copy)
        table.write(temp_file_path, format="fits", overwrite=True)
        logger.debug(f"Wrote temporary input table ({len(df)} rows) to: {temp_file_path}")
        return str(temp_file_path)
    except (TypeError, ValueError) as e:
        raise StiltsError(
            f"Error converting DataFrame to FITS table for {temp_file_path}. "
            f"Check DataFrame dtypes (especially 'object'). Error: {e}"
        ) from e
    except VerifyError as e:
        raise StiltsError(f"FITS verification error writing {temp_file_path}: {e}") from e
    except OSError as e:
        raise StiltsError(f"OS error writing temporary FITS file {temp_file_path}: {e}") from e


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
    output_format: str = "parquet",
    find: str = "best",
    **kwargs,
) -> str:
    """
    Performs cross-match using STILTS cdsskymatch against a CDS VizieR table.

    Args:
        catalogue_1_df: DataFrame containing the local catalogue data.
        cds_id: The VizieR identifier for the remote catalogue (e.g., "I/355/gaiadr3").
        radius: Match radius in arcseconds.
        ra_column_1: RA column name in the local DataFrame.
        dec_column_1: Dec column name in the local DataFrame.
        columns_2: Optional list of columns to retrieve from the CDS catalogue.
        java_opts: Java options for STILTS.
        tmpdir: Temporary directory for STILTS.
        stilts_cmd_base: Base STILTS command override.
        output_format: Desired output format ('parquet', 'fits', 'csv').
        find: Match mode ('best', 'all').
        **kwargs: Additional arguments passed to _run_stilts (e.g., _raw_command).

    Returns:
        Path to the temporary output file containing the match results.

    Raises:
        StiltsError: If the operation fails.
    """
    temp_dir_manager = None
    if tmpdir:
        run_temp_dir = tmpdir
        Path(run_temp_dir).mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_manager = tempfile.TemporaryDirectory(prefix="stilts_cds_")
        run_temp_dir = temp_dir_manager.name

    try:
        input_fits = _prepare_input_table(catalogue_1_df, run_temp_dir, "catalogue_1_cds.fits")

        output_suffix = output_format.lower()
        if output_suffix == "parquet":
            ofmt_str = "parquet-snappy"
        elif output_suffix == "fits":
            ofmt_str = "fits-basic"
        elif output_suffix == "csv":
            ofmt_str = "csv-basic"
        else:
            logger.warning(
                f"Unsupported output format '{output_format}' requested. Defaulting to parquet."
            )
            output_suffix = "parquet"
            ofmt_str = "parquet-snappy"

        output_temp_path = str(Path(run_temp_dir) / f"output_cds.{output_suffix}")

        params = {
            "in": input_fits,
            "ifmt": "fits",
            "ra": ra_column_1,
            "dec": dec_column_1,
            "cdstable": cds_id,
            "radius": radius,
            "find": find,
            "out": output_temp_path,
            "ofmt": ofmt_str,
        }

        if columns_2:
            cols_str = " ".join(map(str, columns_2))
            params["ocmd"] = f'keepcols "* {{{cols_str}}}"'
        else:
            params["ocmd"] = 'keepcols "*"'

        _run_stilts(
            "cdsskymatch", params, java_opts, tmpdir, stilts_cmd_base, kwargs.get("_raw_command")
        )

        safe_cds_id = cds_id.replace("/", "-").replace(" ", "_")
        final_output_prefix = f"xmatch_cds_{Path(input_fits).stem}_{safe_cds_id}_"
        final_output_file = tempfile.NamedTemporaryFile(
            prefix=final_output_prefix, suffix=f"_result.{output_suffix}", delete=False
        )
        final_output_path = final_output_file.name
        final_output_file.close()

        try:
            shutil.copy2(output_temp_path, final_output_path)
            logger.info(
                f"STILTS cdsskymatch result saved to persistent temporary file: {final_output_path}"
            )
            return final_output_path
        except OSError as e:
            raise StiltsError(
                f"Failed to copy STILTS result from {output_temp_path} to {final_output_path}: {e}"
            ) from e

    finally:
        if temp_dir_manager:
            try:
                temp_dir_manager.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory {run_temp_dir}: {e}")
