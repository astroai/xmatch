import functools
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas.errors import EmptyDataError
from astropy.io.fits.verify import VerifyError
from astropy.table import Table

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
                    if any(
                        pattern in str(e).lower()
                        for pattern in ["connection reset", "timeout", "temporary failure"]
                    ):
                        retries += 1
                        if retries < max_retries:
                            sleep_time = delay * (2 ** (retries - 1))  # Exponential backoff
                            logger.warning(
                                f"STILTS operation failed, retrying in {sleep_time}s: {e}"
                            )
                            time.sleep(sleep_time)
                        else:
                            logger.error(
                                f"STILTS operation failed after {max_retries} retries: {e}"
                            )
                            raise
                    else:
                        # Not a retriable error
                        raise

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
        # Use the user-provided base command directly
        logger.debug(f"Using provided stilts_cmd_base: '{stilts_cmd_base}'")
        try:
            cmd = stilts_cmd_base.split()
            if not cmd:
                raise ValueError("Provided stilts_cmd_base resulted in empty command list.")
        except Exception as e:
            # Catch potential errors during split (though unlikely for basic strings)
            raise ValueError(f"Error splitting provided stilts_cmd_base '{stilts_cmd_base}': {e}") from e
    else:
        logger.debug("Constructing STILTS command from java_opts/tmpdir/STILTS_JAR.")
        cmd = ["java"]
        java_options = []
        if java_opts:
            # Safely split java options string into list
            java_options.extend(java_opts.split())
        if tmpdir:
            # Add Java system property for temporary directory
            java_options.append(f"-Djava.io.tmpdir={tmpdir}")
        cmd.extend(java_options)

        # Find STILTS JAR: Use environment variable or default name
        stilts_jar = os.getenv("STILTS_JAR", "stilts.jar")
        # Check if the path exists or if it's found in PATH
        if not Path(stilts_jar).exists() and shutil.which(stilts_jar) is None:
            logger.warning(f"STILTS jar '{stilts_jar}' not found directly or in PATH. "
                          f"Attempting to use default 'stilts.jar' in current directory.")
            stilts_jar = "stilts.jar" # Fallback to default name

        cmd.extend(["-jar", stilts_jar])
        # Add standard STILTS options - consider making these configurable?
        cmd.extend(["-verbose", "-disk"])

    cmd.append(task)

    # Add task-specific parameters (key=value)
    for key, value in params.items():
        if value is None:
            continue # Skip parameters with None value

        # Format value based on type for command line
        if isinstance(value, bool):
            # STILTS typically uses name=true/false for boolean params
            value_str = str(value).lower()
        elif isinstance(value, list):
            # Join list items with space (assuming this is standard for STILTS lists)
            value_str = " ".join(map(str, value))
        else:
            # Convert other types (int, float, etc.) to string
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
        # If a raw command is provided, use it directly
        logger.warning(f"Executing raw STILTS command provided by user: {_raw_command}")
        try:
            # Split the raw command string into arguments
            # Use shlex for safer splitting, handling quotes etc.
            import shlex
            command_args = shlex.split(_raw_command)
            if not command_args:
                raise ValueError("Provided raw STILTS command string is empty or invalid.")
            # The 'task' argument is ignored in this case, but log expected vs actual
            logger.debug(f"Ignoring intended task '{task}' due to raw command override.")
        except Exception as e:
            raise StiltsError(f"Failed to parse raw STILTS command '{_raw_command}': {e}") from e
        command_str = _raw_command # Log the original raw command
    else:
        # Build the command normally if no raw command is given
        try:
            # Build the command line arguments list
            command_args = _build_stilts_command(task, params, java_opts, tmpdir, stilts_cmd_base)
        except ValueError as e:
            # Catch error from building command itself
            raise StiltsError(f"Failed to build STILTS command for task '{task}': {e}") from e
        command_str = " ".join(command_args) # Create string representation for logging

    logger.info(f"Executing STILTS task/command...")
    logger.debug(f"Running command: {command_str}") # Log raw or built command string

    try:
        # Execute the command (same execution logic for both raw and built)
        process = subprocess.run(
            command_args,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace"
        )
        # Log stdout only if DEBUG level is enabled to avoid clutter
        if logger.isEnabledFor(logging.DEBUG):
            # Strip trailing newline for cleaner logs
            stdout_log = process.stdout.strip()
            if stdout_log:
                logger.debug(f"STILTS stdout:\n{stdout_log}")
            else:
                logger.debug("STILTS stdout: (empty)")

        # Always log stderr if present, as it might contain important warnings/info
        stderr_log = process.stderr.strip()
        if stderr_log:
            logger.warning(f"STILTS stderr:\n{stderr_log}")

        logger.info(f"STILTS task/command completed successfully.")

    except FileNotFoundError as e:
        # Handle case where Java or STILTS JAR is not found
        missing_cmd = e.filename or command_args[0] # Best guess of the missing executable
        stilts_jar_path = os.getenv('STILTS_JAR', 'stilts.jar')
        msg = (
            f"STILTS command failed: '{missing_cmd}' not found. "
            f"Ensure Java/command is installed and accessible via the system PATH. "
            f"If using Java, ensure STILTS JAR ('{stilts_jar_path}') is accessible."
        )
        logger.error(msg)
        # Raise specific StiltsError wrapping the original FileNotFoundError
        raise StiltsError(msg) from e

    except subprocess.CalledProcessError as e:
        # Handle errors reported by STILTS (non-zero exit code)
        # Construct a detailed error message including command, exit code, stderr
        error_message = (
            f"STILTS task/command failed with exit code {e.returncode}."
            f"\nFull command: {command_str}"
            # Prioritize stderr as it usually contains the core error message from STILTS
            f"\nSTILTS stderr:\n{e.stderr.strip()}"
        )
        # Include stdout only if it might contain useful context (and isn't excessively long?)
        if e.stdout:
            error_message += f"\nSTILTS stdout:\n{e.stdout.strip()}"
        logger.error(error_message)
        # Raise StiltsError wrapping the original CalledProcessError
        raise StiltsError(error_message) from e

    except OSError as e:
        # Catch other potential OS-level errors during subprocess execution (e.g., permissions)
        error_message = f"OS error during STILTS execution: {e}. Command: {command_str}"
        logger.error(error_message, exc_info=True) # Include traceback for OS errors
        raise StiltsError(error_message) from e

    # Removed generic Exception catch - specific errors handled above should cover most cases


def _prepare_input_table(df: pd.DataFrame, temp_dir: str, filename: str = "input.fits") -> str:
    """Writes a DataFrame to a temporary FITS file for STILTS, handling empty input."""
    # Ensure the filename ends with .fits as STILTS relies on this extension
    if not filename.lower().endswith(('.fits', '.fit')):
        original_suffix = Path(filename).suffix
        filename = Path(filename).stem + '.fits'
        logger.debug(
            f"Input filename '{original_suffix}' changed to '{filename}' for STILTS compatibility."
        )

    temp_file_path = Path(temp_dir) / filename

    if df.empty:
        # Handle empty DataFrame: Create an empty FITS file
        # STILTS can generally handle empty tables gracefully.
        logger.warning(f"Input DataFrame for '{filename}' is empty. Writing empty FITS file.")
        try:
            # Create an empty table structure based on original DataFrame columns if possible
            if list(df.columns):
                # Define dtypes? Astropy might infer, but could be explicit for safety
                # Example: dtype=[df[col].dtype for col in df.columns]
                empty_table = Table({col: [] for col in df.columns})
            else:
                 # If DataFrame had no columns, create a truly empty table
                 empty_table = Table()
            empty_table.write(temp_file_path, format="fits", overwrite=True)
            logger.debug(f"Empty FITS file written to: {temp_file_path}")
            return str(temp_file_path)
        except Exception as e:
            # Catch potential errors during empty table write
             raise StiltsError(f"Failed to write empty temporary input FITS file {temp_file_path}: {e}") from e

    # If DataFrame is not empty, proceed with conversion and writing
    try:
        table = Table.from_pandas(df)
        table.write(temp_file_path, format="fits", overwrite=True)
        logger.debug(f"Wrote temporary input table ({len(df)} rows) to: {temp_file_path}")
        return str(temp_file_path)
    except (TypeError, ValueError) as e:
        # Catch errors during pandas->astropy conversion (e.g., unsupported object dtype)
        raise StiltsError(
            f"Error converting DataFrame to FITS table for {temp_file_path}. "
            f"Check DataFrame dtypes (especially 'object'). Error: {e}"
        ) from e
    except VerifyError as e:
        # Catch FITS verification errors (often related to complex headers/data types)
        raise StiltsError(f"FITS verification error writing {temp_file_path}: {e}") from e
    except OSError as e:
        # Catch file system errors (e.g., disk full, permissions)
         raise StiltsError(f"OS error writing temporary FITS file {temp_file_path}: {e}") from e
    # Removed generic Exception catch


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
    **kwargs,
) -> str:
    """Performs cross-match using STILTS cdsskymatch."""
    with tempfile.TemporaryDirectory(prefix="stilts_cds_") as temp_dir:
        # Prepare input table (raises StiltsError on failure)
        input_fits = _prepare_input_table(catalogue_1_df, temp_dir, "catalogue_1_cds.fits")

        # Determine output suffix and STILTS format string
        output_suffix = output_format.lower()
        if output_suffix == "parquet":
            ofmt_str = "parquet-snappy"
        elif output_suffix == "fits":
            ofmt_str = "fits-basic"
        elif output_suffix == "csv":
            ofmt_str = "csv-basic"
        else:
            logger.warning(f"Unsupported output format '{output_format}' requested. Defaulting to parquet.")
            output_suffix = "parquet"
            ofmt_str = "parquet-snappy"

        output_temp_path = str(Path(temp_dir) / f"output_cds.{output_suffix}")

        # Base parameters for cdsskymatch
        params = {
            "in": input_fits,
            "ifmt": "fits", # Input is always prepared as FITS
            "ra": ra_column_1,
            "dec": dec_column_1,
            "cdstable": cds_id,
            "radius": radius,
            "find": kwargs.get("find", "best"), # Default find mode
            "out": output_temp_path,
            "ofmt": ofmt_str,
        }

        # Construct ocmd separately for clarity and correct f-string usage
        if columns_2:
            # Ensure columns are strings, join them
            cols_str = " ".join(map(str, columns_2))
            # Double braces {{ and }} are needed for literal braces inside f-string
            # Keep all input columns (*) and requested CDS columns ({cols_str})
            params["ocmd"] = f'keepcols "* {{{cols_str}}}"'
        else:
            # If no specific columns requested, keep all input columns (*)
            # Note: STILTS might automatically add some default CDS columns here?
            # Check STILTS cdsskymatch documentation if specific behavior is needed.
            params["ocmd"] = 'keepcols "*"'

        # Add other relevant kwargs passed down?
        # Example: processing=sequential/parallel? Needs check in STILTS docs.
        # params.update({k: v for k, v in kwargs.items() if k not in ['find']})

        # Run STILTS (raises StiltsError on failure)
        _run_stilts("cdsskymatch", params, java_opts, tmpdir, stilts_cmd_base, kwargs.get("_raw_command"))

        # Copy result out of temp dir to a more persistent temp file
        # Use a more descriptive prefix/suffix if possible
        # Sanitize cds_id for use in filename
        safe_cds_id = cds_id.replace('/', '-').replace(' ', '_')
        final_output_prefix = f"xmatch_cds_{Path(input_fits).stem}_{safe_cds_id}_"
        final_output = tempfile.NamedTemporaryFile(
            prefix=final_output_prefix,
            suffix=f"_result.{output_suffix}",
            delete=False # Keep the file after closing handle
        ).name

        try:
            # Copy the result file from the temporary task directory
            shutil.copy2(output_temp_path, final_output)
            logger.info(f"STILTS cdsskymatch result saved to: {final_output}")
            return final_output
        except OSError as e:
             # Handle potential errors during file copy
             raise StiltsError(f"Failed to copy STILTS result from {output_temp_path} to {final_output}: {e}") from e


def cdsskymatch(
    in1,
    out,
    ra,
    dec,
    cds_id,
    radius=1.0,
    find="best",
    cdscols=None,
    stilts_cmd_base=None,
    java_opts=None,
    tmpdir=None,
    verbose=True,
):
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
            "cdstable": cds_id,
        }

        # Add optional columns if specified
        if cdscols:
            params["cdscols"] = cdscols

        if verbose:
            logger.info("Running STILTS cdsskymatch")

        # Run the command using existing function
        _run_stilts("cdsskymatch", params, java_opts, tmpdir, stilts_cmd_base, None)

        return True

    except Exception as e:
        logger.error(f"Error in STILTS cdsskymatch: {str(e)}")
        raise StiltsError(f"Error during STILTS cdsskymatch: {str(e)}")


def _get_error_config(
    config: Dict[str, Any], axis: str
) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    """
    Extracts error configuration (column name, units, floor error) for a given axis (ra/dec).
    Prioritizes IVAR columns if available.

    Returns:
        Tuple: (error_col_name, ivar_col_name, floor_error_arcsec, units)
           - error_col_name: Name of the direct error column, or None.
           - ivar_col_name: Name of the inverse variance column, or None.
           - floor_error_arcsec: Default floor error from catalogue config, or global default.
           - units: Units of the error/ivar column (before conversion), or 'arcsec' if only floor is used.
    """
    ivar_col = config.get(f"{axis}_ivar_column")
    err_col = config.get(f"{axis}_err_column")
    units = config.get("pos_err_units", "arcsec")  # Default to arcsec if unspecified
    floor_err = config.get("default_pos_error_arcsec")
    # Consider global default floor if per-catalogue is missing
    if floor_err is None:
        floor_err = config.get("_global_default_floor_error_arcsec")

    if ivar_col:
        logger.debug(f"Using IVAR column '{ivar_col}' for {axis} error.")
        return None, ivar_col, floor_err, units  # Prioritize IVAR
    elif err_col:
        logger.debug(f"Using error column '{err_col}' for {axis} error.")
        return err_col, None, floor_err, units
    elif floor_err is not None:
        logger.debug(f"Using floor error {floor_err} arcsec for {axis} error.")
        # If only floor error is used, the effective unit is arcsec
        return None, None, floor_err, "arcsec"
    else:
        logger.debug(f"No error information found for {axis}.")
        return None, None, None, "arcsec"  # No error info


def _build_error_value_expression(
    err_col_name: Optional[str],
    ivar_col_name: Optional[str],
    floor_error_arcsec: Optional[float],
    units: str,
) -> str:
    """
    Builds a STILTS expression to get the error value in ARCSECONDS.
    Handles unit conversion, IVAR conversion, NULLs, and floor error fallback.
    Assumes floor_error_arcsec is already in arcseconds.
    """

    # Base expression: start with the direct error or IVAR calculation
    base_expr = "null"
    if ivar_col_name:
        # Calculate error from inverse variance: err = 1 / sqrt(ivar)
        # Handle ivar <= 0 to avoid errors/NaNs
        base_expr = f"( {ivar_col_name} > 0 ? 1.0 / sqrt({ivar_col_name}) : null )"
        logger.debug(f"Built IVAR base expression: {base_expr}")
    elif err_col_name:
        base_expr = f"{err_col_name}"
        logger.debug(f"Built error column base expression: {base_expr}")

    # Apply unit conversion to arcseconds if necessary
    unit_scale_factor = 1.0
    if units == "mas":
        unit_scale_factor = 1.0 / 1000.0
        logger.debug(f"Applying mas -> arcsec scaling ({unit_scale_factor})")
    elif units == "deg":
        unit_scale_factor = 3600.0
        logger.debug(f"Applying deg -> arcsec scaling ({unit_scale_factor})")
    elif units != "arcsec":
        logger.warning(f"Unrecognized pos_err_units '{units}'. Assuming arcseconds.")

    scaled_expr = base_expr
    if unit_scale_factor != 1.0 and (ivar_col_name or err_col_name):
        # Only apply scaling if we have a column-based expression
        scaled_expr = f"( {base_expr} * {unit_scale_factor} )"
        logger.debug(f"Applied unit scaling: {scaled_expr}")

    # Fallback to floor error if the scaled expression is null or non-positive
    # Ensure floor error itself is treated as arcseconds
    final_expr = scaled_expr
    if floor_error_arcsec is not None and floor_error_arcsec > 0:
        floor_val_str = str(floor_error_arcsec)
        # Check if base expression exists and is positive after scaling
        valid_check = f"{scaled_expr} != null && {scaled_expr} > 0"
        if ivar_col_name or err_col_name:
            # If we started with columns, use floor as fallback
            final_expr = f"( {valid_check} ? {scaled_expr} : {floor_val_str} )"
        else:
            # If we only had floor error to begin with, just use it
            final_expr = floor_val_str
        logger.debug(f"Applied floor error fallback ({floor_val_str}): {final_expr}")
    elif ivar_col_name or err_col_name:
        # No floor error, but we have columns: ensure result is positive or null
        final_expr = f"( {scaled_expr} != null && {scaled_expr} > 0 ? {scaled_expr} : null )"
        logger.debug(f"Applied null fallback (no floor error): {final_expr}")
    else:
        # No columns and no floor error - result must be null
        final_expr = "null"
        logger.debug("No error columns or floor error provided, final expression is null.")

    return final_expr


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
    ra1: str,
    dec1: str,
    ra2: str,
    dec2: str,
    matcher: str,  # 'sky', 'skyerr', 'skyellipse'
    config1: Dict[str, Any],
    config2: Dict[str, Any],
    max_error: float = 3.0,  # Max separation in units of sigma for error matchers
    radius_arcsec: float = 1.0,  # Fallback radius for 'sky' matcher
    join_type: str = "1and2",
    stilts_cmd_base: Optional[str] = None,
    java_opts: Optional[str] = None,
    tmpdir: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Performs spatial cross-matching between two catalogs using STILTS tmatch2.
    Supports different matchers: sky, skyerr, skyellipse.
    Handles error unit conversions to arcseconds based on config.

    Args:
        in1: Input file path for table 1.
        in2: Input file path for table 2.
        out: Output file path for the result.
        ra1, dec1: RA/Dec column names in table 1.
        ra2, dec2: RA/Dec column names in table 2.
        matcher: The STILTS matcher to use ('sky', 'skyerr', 'skyellipse').
        config1: Configuration dictionary for table 1 (must contain error info if needed).
        config2: Configuration dictionary for table 2 (must contain error info if needed).
        max_error: Maximum separation in units of error (sigma) for skyerr/skyellipse.
        radius_arcsec: Match radius in arcseconds (used only for matcher='sky').
        join_type: Type of join ('1and2', '1or2', 'all1', 'all2', etc.).
        stilts_cmd_base: Base STILTS command.
        java_opts: Java options.
        tmpdir: Temporary directory.
        **kwargs: Additional parameters to pass directly to tmatch2 (e.g., find=best).

    Returns:
        Output file path if successful.
    """
    try:
        # Basic parameter validation
        if matcher not in ["sky", "skyerr", "skyellipse"]:
            raise ValueError(
                f"Unsupported matcher: '{matcher}'. Must be 'sky', 'skyerr', or 'skyellipse'."
            )

        logger.info(f"Preparing STILTS tmatch2 command for matcher='{matcher}'")

        params = {
            "in1": in1,
            "in2": in2,
            "out": out,
            "matcher": matcher,
            "join": join_type,
            "icmd1": f'select "( {ra1} != null && {dec1} != null )"',  # Pre-filter null coords
            "icmd2": f'select "( {ra2} != null && {dec2} != null )"',
            # Default output format, can be overridden by kwargs
            "ofmt": kwargs.get("ofmt", "parquet-snappy"),
            # Default find mode, can be overridden by kwargs
            "find": kwargs.get("find", "best"),
        }

        # Add matcher-specific parameters
        if matcher == "sky":
            params["values1"] = f'"{ra1}" "{dec1}"'
            params["values2"] = f'"{ra2}" "{dec2}"'
            params["params"] = str(radius_arcsec)  # Sky matcher uses radius in arcsec
            logger.debug(f"Using sky matcher with radius: {radius_arcsec} arcsec")

        elif matcher in ["skyerr", "skyellipse"]:
            # Get error configurations
            ra_err_col1, ra_ivar_col1, floor_ra1, units_ra1 = _get_error_config(config1, "ra")
            dec_err_col1, dec_ivar_col1, floor_dec1, units_dec1 = _get_error_config(config1, "dec")
            ra_err_col2, ra_ivar_col2, floor_ra2, units_ra2 = _get_error_config(config2, "ra")
            dec_err_col2, dec_ivar_col2, floor_dec2, units_dec2 = _get_error_config(config2, "dec")

            # Build JEL expressions for error values IN ARCSECONDS
            ra_err_expr1 = _build_error_value_expression(
                ra_err_col1, ra_ivar_col1, floor_ra1, units_ra1
            )
            dec_err_expr1 = _build_error_value_expression(
                dec_err_col1, dec_ivar_col1, floor_dec1, units_dec1
            )
            ra_err_expr2 = _build_error_value_expression(
                ra_err_col2, ra_ivar_col2, floor_ra2, units_ra2
            )
            dec_err_expr2 = _build_error_value_expression(
                dec_err_col2, dec_ivar_col2, floor_dec2, units_dec2
            )

            # --- Skyerr ---
            if matcher == "skyerr":
                # skyerr uses: ra, dec, error (symmetric, in arcsec)
                # We need ONE error expression per catalogue. How to combine ra/dec errors?
                # Option 1: Use the RA error (simplest but maybe inaccurate)
                # Option 2: Use quadrature sum: sqrt(ra_err^2 + dec_err^2) / sqrt(2) ?? No standard way.
                # Option 3: Require specific 'pos_err_column' in config?
                # FOR NOW: Use the RA error expression, log a warning.
                # TODO: Revisit skyerr error combination strategy.
                err_expr1 = ra_err_expr1
                err_expr2 = ra_err_expr2
                logger.warning(
                    "Using RA error expression for symmetric 'skyerr' matcher. Consider using 'skyellipse' or defining a single positional error column if asymmetry is significant."
                )

                # Check if error expressions are valid (not just "null")
                if err_expr1 == "null" or err_expr2 == "null":
                    raise StiltsError(
                        f"Matcher '{matcher}' requires valid error information for both catalogues, but was not found or derived."
                    )

                params["values1"] = f'"{ra1}" "{dec1}" {err_expr1}'
                params["values2"] = f'"{ra2}" "{dec2}" {err_expr2}'
                params["params"] = str(max_error)  # Max separation in sigma
                logger.debug(f"Using skyerr matcher with max_error: {max_error}")
                logger.debug(f"  Values1 Expr: {params['values1']}")
                logger.debug(f"  Values2 Expr: {params['values2']}")

            # --- Skyellipse ---
            elif matcher == "skyellipse":
                # skyellipse uses: ra, dec, err_a, err_b, err_pa (all in arcsec/degrees)
                # Get correlation column info
                corr_col1 = config1.get("corr_column")
                corr_col2 = config2.get("corr_column")
                corr_expr1 = _build_correlation_expression(corr_col1)
                corr_expr2 = _build_correlation_expression(corr_col2)

                # Check if error expressions are valid
                if (
                    ra_err_expr1 == "null"
                    or dec_err_expr1 == "null"
                    or ra_err_expr2 == "null"
                    or dec_err_expr2 == "null"
                ):
                    raise StiltsError(
                        f"Matcher '{matcher}' requires valid RA and Dec error information for both catalogues, but was not found or derived."
                    )

                # STILTS skyellipse uses semi-major (a), semi-minor (b), and position angle (pa)
                # We have ra_err, dec_err, corr. Need to convert.
                # JEL expressions for a, b, pa from sigX, sigY, rho:
                # From https://github.com/astro-informatics/stilts/blob/master/src/java/uk/ac/starlink/ttools/jel/RandomJELRowReader.java#L403
                # double u = (sigX2+sigY2)*0.5;
                # double v = Math.sqrt(Math.pow((sigX2-sigY2)*0.5,2)+(rho*rho*sigX2*sigY2));
                # a = Math.sqrt(u+v);
                # b = Math.sqrt(u-v);
                # pa = Math.toDegrees(0.5*Math.atan2(2*rho*sigX*sigY,(sigX2-sigY2)));

                # Define common terms for JEL expressions (avoids repetition)
                jel_defs1 = (
                    f"sigX1 = ({ra_err_expr1}); sigY1 = ({dec_err_expr1}); rho1 = ({corr_expr1}); "
                    f"sigX2_1 = sigX1*sigX1; sigY2_1 = sigY1*sigY1; "
                    f"u1 = (sigX2_1 + sigY2_1) * 0.5; "
                    f"v1_sq = pow((sigX2_1 - sigY2_1) * 0.5, 2) + (rho1 * rho1 * sigX2_1 * sigY2_1); "
                    f"v1 = (v1_sq > 0 ? sqrt(v1_sq) : 0);"
                )
                jel_defs2 = (
                    f"sigX2 = ({ra_err_expr2}); sigY2 = ({dec_err_expr2}); rho2 = ({corr_expr2}); "
                    f"sigX2_2 = sigX2*sigX2; sigY2_2 = sigY2*sigY2; "
                    f"u2 = (sigX2_2 + sigY2_2) * 0.5; "
                    f"v2_sq = pow((sigX2_2 - sigY2_2) * 0.5, 2) + (rho2 * rho2 * sigX2_2 * sigY2_2); "
                    f"v2 = (v2_sq > 0 ? sqrt(v2_sq) : 0);"
                )

                # Expressions for a, b (arcsec) and pa (degrees)
                a_expr1 = "sqrt(u1 + v1)"
                b_expr1 = "sqrt(u1 - v1 > 0 ? u1 - v1 : 0)"  # Ensure b is not complex
                pa_expr1 = (
                    "0.5 * atan2(2 * rho1 * sigX1 * sigY1, sigX2_1 - sigY2_1)"  # Radians first
                )
                pa_expr1_deg = f"toDegrees({pa_expr1})"  # Convert to degrees for STILTS

                a_expr2 = "sqrt(u2 + v2)"
                b_expr2 = "sqrt(u2 - v2 > 0 ? u2 - v2 : 0)"
                pa_expr2 = "0.5 * atan2(2 * rho2 * sigX2 * sigY2, sigX2_2 - sigY2_2)"
                pa_expr2_deg = f"toDegrees({pa_expr2})"

                # Prepend definitions and append PA calculation to icmd filters
                params[
                    "icmd1"
                ] += f"; {jel_defs1} addcol pa_deg1 ({pa_expr1_deg});"  # Define PA in degrees
                params["icmd2"] += f"; {jel_defs2} addcol pa_deg2 ({pa_expr2_deg});"

                params["values1"] = f'\\"{ra1}\\" \\"{dec1}\\" ({a_expr1}) ({b_expr1}) pa_deg1'
                params["values2"] = f'\\"{ra2}\\" \\"{dec2}\\" ({a_expr2}) ({b_expr2}) pa_deg2'
                logger.debug("Using skyellipse matcher.")
                logger.debug(f"  Values1 Expr: {params['values1']}")
                logger.debug(f"  Values2 Expr: {params['values2']}")

        # Add any other kwargs provided by the user
        # Filter out keys already handled explicitly
        handled_keys = [
            "in1",
            "in2",
            "out",
            "matcher",
            "join",
            "icmd1",
            "icmd2",
            "ofmt",
            "find",
            "values1",
            "values2",
            "params",
        ]
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in handled_keys}
        params.update(extra_kwargs)

        # Run the STILTS command
        _run_stilts("tmatch2", params, java_opts, tmpdir, stilts_cmd_base, kwargs.get("_raw_command"))

        logger.info(f"STILTS tmatch2 sky match ({matcher}) result saved to: {out}")
        return out

    except (StiltsError, ValueError, FileNotFoundError) as e:
        logger.error(f"STILTS tmatch2 sky match failed: {e}", exc_info=True)
        # Re-raise known error types
        raise
    except Exception as e:
        logger.error(f"Unexpected error during STILTS tmatch2 sky match: {e}", exc_info=True)
        raise StiltsError(f"Unexpected error in tmatch2 sky match: {e}") from e


def crossmatch_id(
    in1,
    in2,
    out,
    id_column_1,
    id_column_2,
    join_type="1and2",
    stilts_cmd_base=None,
    java_opts=None,
    tmpdir=None,
    **kwargs,
):
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
        "ifmt1": kwargs.get("ifmt1", "auto"),
        "ifmt2": kwargs.get("ifmt2", "auto"),
        "matcher": "exact",
        "values1": id_column_1,
        "values2": id_column_2,
        "join": join_type,
        "find": "all",
        "out": out,
        "ofmt": kwargs.get("ofmt", "auto"),
    }

    # Add any additional parameters
    for k, v in kwargs.items():
        if k not in ["ifmt1", "ifmt2", "find", "ofmt"]:
            params[k] = v

    _run_stilts("tmatch2", params, java_opts, tmpdir, stilts_cmd_base, kwargs.get("_raw_command"))

    logger.info(f"STILTS ID cross-match result saved to: {out}")
    return out
