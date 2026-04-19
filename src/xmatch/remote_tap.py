import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .exceptions import CrossMatchError, TapError
from .tap import (  # Assuming these are accessible
    execute_tap_query,
    get_tap_service,
)

logger = logging.getLogger(__name__)


def _get_coord_cols_from_config(
    config: Dict[str, Any], prefix: str, params: Dict[str, Any]
) -> tuple[str, str]:
    """Gets RA/Dec column names from parameters or config fallback."""
    # Check params first for overrides (e.g., 'ra_column_1', 'dec_column_1')
    ra_col_param = f"ra_column_{prefix}"
    dec_col_param = f"dec_column_{prefix}"

    ra_col = params.get(ra_col_param, config.get("ra_column"))
    dec_col = params.get(dec_col_param, config.get("dec_column"))

    if not ra_col:
        raise CrossMatchError(
            f"RA column name could not be determined for prefix '{prefix}'. Check config or parameters (e.g., '{ra_col_param}')"
        )
    if not dec_col:
        raise CrossMatchError(
            f"Dec column name could not be determined for prefix '{prefix}'. Check config or parameters (e.g., '{dec_col_param}')"
        )

    return ra_col, dec_col


def download_from_tap(
    config: Dict[str, Any],
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    radius_deg: Optional[float] = None,
    columns_to_download: Optional[List[str]] = None,
    auth_session: Optional[Any] = None,
    maxrec: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Downloads data from a TAP service based on the provided configuration and parameters.

    Args:
        config: Dictionary containing catalogue configuration (must include tap_url/access_url and access_identifier/table_name).
        ra: Right Ascension for cone search center (degrees).
        dec: Declination for cone search center (degrees).
        radius_deg: Radius for cone search (degrees).
        columns_to_download: List of specific columns to select. If None, selects default or all columns.
        auth_session: Authentication session object (e.g., from pyvo).
        maxrec: Maximum number of records to retrieve.
        **kwargs: Additional parameters (ignored for now).

    Returns:
        A pandas DataFrame containing the downloaded data.

    Raises:
        CrossMatchError: If required configuration is missing or download fails.
        TapError: If there's an issue with the TAP query execution.
    """
    cat_name = config.get("_catalogue_name", "unknown_remote")
    tap_url = config.get("tap_url") or config.get("access_url")  # Check for both keys
    table_name = config.get("access_identifier") or config.get("table_name")  # Check for both keys

    if not tap_url or not table_name:
        # Add more detail to the error message
        missing = []
        if not tap_url:
            missing.append("'tap_url' or 'access_url'")
        if not table_name:
            missing.append("'access_identifier' or 'table_name'")
        raise CrossMatchError(f"Missing {' and '.join(missing)} for {cat_name} in config: {config}")

    logger.info(f"Preparing TAP download for {cat_name} from {tap_url} (Table: {table_name})")

    # Get configured coordinate and ID columns using their correct names from config
    ra_col = config.get("ra_column")
    dec_col = config.get("dec_column")
    id_col = config.get("id_column")  # Get ID column name from config
    if not ra_col or not dec_col:
        raise CrossMatchError(f"RA/Dec column names missing in config for {cat_name}")

    essential_cols = {ra_col, dec_col}
    if id_col:
        essential_cols.add(id_col)

    # Determine columns to select
    if columns_to_download:
        # Start with user-requested columns
        # Assume user provided correct source table column names
        cols_to_select = set(columns_to_download)
        logger.debug(f"User requested columns: {columns_to_download}")
        # Ensure essential columns (using config names) are included
        missing_essentials = essential_cols - cols_to_select
        if missing_essentials:
            logger.debug(f"Adding missing essential columns: {missing_essentials}")
            cols_to_select.update(missing_essentials)
        select_cols_str = ", ".join(sorted(list(cols_to_select)))  # Use the combined set
        logger.debug(
            f"Final columns selected based on user request + essentials: {select_cols_str}"
        )
    else:
        # Use default columns from config if available
        default_cols = config.get("default_columns")
        if default_cols:
            cols_to_select = set(default_cols)
            logger.debug(f"Using default columns from config: {default_cols}")
            # Ensure essential columns are included in defaults
            missing_essentials = essential_cols - cols_to_select
            if missing_essentials:
                logger.debug(f"Adding missing essential columns to defaults: {missing_essentials}")
                cols_to_select.update(missing_essentials)
            select_cols_str = ", ".join(sorted(list(cols_to_select)))
            logger.debug(
                f"Final columns selected based on defaults + essentials: {select_cols_str}"
            )
        else:
            # Fallback to selecting all columns if no defaults and no user request
            select_cols_str = "*"
            logger.debug("Selecting all columns (*)")

    # Construct ADQL query with table alias
    table_alias = "t1"
    adql_query = f"SELECT {select_cols_str} FROM {table_name} AS {table_alias}"  # Added AS t1

    # Add spatial constraint if provided
    if ra is not None and dec is not None and radius_deg is not None:
        # Use standard ADQL cone search syntax, but use 1=CONTAINS for compatibility
        # Use alias in POINT function
        where_clause = f"WHERE 1 = CONTAINS(POINT('ICRS', {table_alias}.{ra_col}, {table_alias}.{dec_col}), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))"  # Reverted to 1 = CONTAINS
        adql_query += f" {where_clause}"
        logger.info(f"Applying cone search: RA={ra}, Dec={dec}, Radius={radius_deg} deg")
    elif any(arg is not None for arg in [ra, dec, radius_deg]):
        logger.warning(
            "Partial cone search parameters provided (RA, Dec, Radius). All three are required. Ignoring spatial constraint."
        )

    # Add MAXREC if specified
    if maxrec:
        adql_query = adql_query.replace("SELECT", f"SELECT TOP {maxrec}")
        logger.info(f"Applying record limit: MAXREC={maxrec}")

    logger.debug(f"Executing ADQL query for download: {adql_query}")

    try:
        # Get the TAP service object first, passing auth_session as keyword
        tap_service = get_tap_service(tap_url, auth_session=auth_session)
        if not tap_service:
            # Handle case where get_tap_service might return None or raise error implicitly
            raise TapError(f"Could not establish TAP service connection to {tap_url}")

        # Use the generic execute_tap_query function, passing only required args
        results_table = execute_tap_query(
            tap_service,  # 1st positional: service object
            adql_query,  # 2nd positional: query string
        )
        logger.info(f"Successfully downloaded {len(results_table)} records for {cat_name}.")
        return results_table.to_pandas()
    except TapError as e:
        logger.error(f"TAP query failed during download for {cat_name}: {e}", exc_info=True)
        # Re-raise as CrossMatchError to indicate failure in the crossmatch context
        raise CrossMatchError(f"TAP download failed for {cat_name}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during TAP download for {cat_name}: {e}", exc_info=True)
        raise CrossMatchError(f"Unexpected download error for {cat_name}: {e}") from e


# Placeholder for TAP Upload Match Strategy
def execute_upload_and_tap_match(
    local_df: pd.DataFrame,
    local_config: Dict[str, Any],
    remote_config: Dict[str, Any],
    crossmatch_instance: Any,  # Assuming CrossMatch instance might be needed
    **params,
) -> pd.DataFrame:
    """
    Executes a crossmatch by uploading a local table to a TAP service
    and performing the join remotely. (Placeholder Implementation)
    """
    logger.warning("execute_upload_and_tap_match is a placeholder and not fully implemented.")
    # 1. Get TAP service (with auth)
    # 2. Prepare local table for upload (convert to VOTable?)
    # 3. Generate unique upload table name
    # 4. Build ADQL query using uploaded table and remote table
    # 5. Execute query using execute_tap_query with upload_params
    # 6. Return results
    raise NotImplementedError("execute_upload_and_tap_match needs implementation")
    # return pd.DataFrame() # Example return


# Placeholder for Remote TAP Join Strategy
def execute_remote_join_match(
    config1: Dict[str, Any],
    config2: Dict[str, Any],
    crossmatch_instance: Any,  # Assuming CrossMatch instance might be needed
    **params,
) -> pd.DataFrame:
    """
    Executes a crossmatch by joining two tables directly on the same TAP service.
    (Placeholder Implementation)
    """
    logger.warning("execute_remote_join_match is a placeholder and not fully implemented.")
    # 1. Check if TAP URLs are the same
    # 2. Get TAP service (with auth)
    # 3. Build ADQL query joining the two remote tables (config1['table_name'], config2['table_name'])
    #    - Include spatial join condition (CONTAINS/INTERSECTS)
    #    - Select desired columns from both tables
    # 4. Execute query using execute_tap_query
    # 5. Return results
    raise NotImplementedError("execute_remote_join_match needs implementation")
    # return pd.DataFrame() # Example return
