import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyvo
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
# Specific exceptions
from pandas.errors import EmptyDataError
from pyvo.dal import DALQueryError, DALServiceError, DALFormatError
from requests.exceptions import RequestException, ConnectionError, Timeout

logger = logging.getLogger(__name__)


class TapError(Exception):
    """Base exception for TAP-related errors."""

    pass


# Add connection pooling for TAP services
_tap_service_cache = {}


def get_tap_service(tap_url: str, **kwargs) -> pyvo.dal.TAPService:
    """Get a cached TAP service connection or create a new one."""
    # Create a cache key based on URL and relevant kwargs (like auth)
    cache_key = tap_url
    for k, v in sorted(kwargs.items()):
        # Include auth params in cache key, avoid caching based on other dynamic kwargs
        if k in ["user", "password"]:
            cache_key += f"_{k}_{v}"

    if cache_key not in _tap_service_cache:
        logger.info(f"Establishing new TAP service connection to {tap_url}")
        try:
            # Attempt to create the TAP service connection
            service = pyvo.dal.TAPService(tap_url, **kwargs)
            # Test connection? Some services might require explicit login/check.
            # For now, assume successful instantiation means connection is likely ok.
            _tap_service_cache[cache_key] = service
            logger.debug(f"Successfully created TAP service for {tap_url}")
        except (ConnectionError, Timeout, DALServiceError, RequestException) as e:
            # Catch specific network/service related errors during connection setup
            logger.error(f"Network/Service error connecting to TAP service {tap_url}: {e}")
            raise TapError(f"Failed to connect to TAP service {tap_url}: {e}") from e
        except Exception as e:
            # Catch other unexpected errors during service instantiation
            logger.error(f"Unexpected error instantiating TAP service {tap_url}: {e}", exc_info=True)
            raise TapError(f"Unexpected error connecting to TAP service {tap_url}: {e}") from e
    else:
        logger.debug(f"Using cached TAP service connection for {tap_url}")

    return _tap_service_cache[cache_key]


def tap_crossmatch(
    catalogue_1: Union[str, pd.DataFrame, Table],
    ra: str,
    dec: str,
    radius: float,
    tap_url: Optional[str] = None,
    tap_table: Optional[str] = None,
    columns: Optional[List[str]] = None,
    chunk_size: int = 100000,
    verbose: bool = False,
    retry_delay: int = 60,
    timeout: int = 600,
    input_columns: Optional[List[str]] = None,
    tap_schema: Optional[str] = None,
    output_file: Optional[str] = None,
    id_column_1: Optional[str] = None,
    id_column_2: Optional[str] = None,
    local_table: Optional[str] = None,
    local_table_df: Optional[Union[pd.DataFrame, Table]] = None,
    **kwargs,
) -> str:
    """
    Cross-matches two astronomy catalogues.

    Parameters
    ----------
    catalogue_1 : str or DataFrame or Table
        First catalogue (file path, DataFrame, or Table)
    ra : str
        RA column name in catalogue_1
    dec : str
        Dec column name in catalogue_1
    radius : float
        Cross-match radius in arcseconds
    tap_url : str, optional
        URL of the TAP service (ignored if local_table is provided)
    tap_table : str, optional
        Name of the table on the TAP service
    columns : list of str, optional
        Columns to keep from catalogue_2
    chunk_size : int, optional
        Chunk size for processing
    verbose : bool, optional
        Print progress information
    retry_delay : int, optional
        Seconds to wait before retrying failed queries
    timeout : int, optional
        Timeout for TAP queries in seconds
    input_columns : list of str, optional
        Columns to keep from catalogue_1
    tap_schema : str, optional
        Explicit TAP schema
    output_file : str, optional
        Output file path
    id_column_1 : str, optional
        Name of ID column in catalogue_1 for ID-based matching.
    id_column_2 : str, optional
        Name of ID column in catalogue_2/table for ID-based matching.
        If None, assumes the same name as `id_column_1`.
    local_table : str, optional
        Path to local table file for cross-matching (catalogue_2)
    local_table_df : DataFrame or Table, optional
        DataFrame or Table for catalogue_2 (alternative to local_table)
    **kwargs
        Additional arguments for TAP service

    Returns
    -------
    str
        Output file path
    """
    try:
        # Check for incompatible arguments
        if (local_table or local_table_df) and (tap_url or tap_table):
            logger.warning(
                "Local catalogue_2 provided; ignoring 'tap_url' and 'tap_table' parameters."
            )
            tap_url = None
            tap_table = None

        if not (local_table or local_table_df) and not (tap_url and tap_table):
            raise TapError(
                "Either a local catalogue_2 or both 'tap_url' and 'tap_table' must be provided."
            )

        if local_table and local_table_df:
            logger.warning(
                "Both local_table and local_table_df provided; using local_table file path."
            )

        # Determine effective target ID column name
        join_catalogue_2_id_column = id_column_2 if id_column_2 else id_column_1

        # Create temporary directory
        with tempfile.TemporaryDirectory(prefix="tap_") as temp_dir:
            # Handle input catalogue
            if isinstance(catalogue_1, (str, Path)) and Path(catalogue_1).is_file():
                # Use file directly if it's a parquet file (supported by pandas)
                if Path(catalogue_1).suffix.lower() == ".parquet":
                    input_df = pd.read_parquet(catalogue_1)
                else:
                    # For other formats, we need to read them
                    if Path(catalogue_1).suffix.lower() == ".fits":
                        input_df = Table.read(catalogue_1).to_pandas()
                    elif Path(catalogue_1).suffix.lower() == ".csv":
                        input_df = pd.read_csv(catalogue_1)
                    else:
                        input_df = Table.read(catalogue_1).to_pandas()
            elif isinstance(catalogue_1, pd.DataFrame):
                input_df = catalogue_1
            elif isinstance(catalogue_1, Table):
                input_df = catalogue_1.to_pandas()
            else:
                raise TapError(f"Unsupported catalogue_1 type: {type(catalogue_1)}")

            # Limit input columns if specified
            required_cols = {ra, dec}
            if id_column_1:
                required_cols.add(id_column_1)
            if input_columns:
                cols_to_keep = list(set(input_columns) | required_cols)
                try:
                    input_df = input_df[cols_to_keep]
                except KeyError as e:
                    raise TapError(
                        f"One or more input_columns or required coordinate/ID columns not found: {e}"
                    )
            else:
                # Ensure required columns exist even if input_columns is None
                missing_req = required_cols - set(input_df.columns)
                if missing_req:
                    raise TapError(
                        f"Required coordinate/ID columns missing from input: {missing_req}"
                    )

            # Handle Coordinate Systems (primarily for Gaia)
            # Rely on specific column names rather than table name
            if "RAJ2000" in input_df.columns and "DEJ2000" in input_df.columns:
                logger.info("Using RAJ2000/DEJ2000 columns for coordinates.")
                ra, dec = "RAJ2000", "DEJ2000"
                # Apply Gaia-specific transformations if needed (e.g., proper motion)
                input_df = ensure_j2000_gaia(input_df, ra, dec)
            else:
                # Convert non-Gaia to J2000 ICRS
                input_df = ensure_j2000(input_df, ra, dec)

            # Determine cross-matching strategy & execute
            if id_column_1 and id_column_1 in input_df.columns:
                # --- ID-based matching ---
                if local_table:
                    # Use local file directly
                    logger.info(f"Performing local ID join using file: {local_table}")
                    result_df = perform_local_id_join(
                        input_df,
                        local_table,
                        id_column_1,
                        join_catalogue_2_id_column,
                        columns,
                        verbose,
                    )
                elif local_table_df is not None:
                    # Use provided DataFrame/Table
                    logger.info("Performing local ID join using DataFrame/Table")
                    if isinstance(local_table_df, Table):
                        local_df = local_table_df.to_pandas()
                    else:
                        local_df = local_table_df

                    # Convert perform_local_id_join to use DataFrame directly
                    result_df = perform_local_id_join_df(
                        input_df,
                        local_df,
                        id_column_1,
                        join_catalogue_2_id_column,
                        columns,
                        verbose,
                    )
                else:
                    logger.info(
                        f"Performing TAP ID join on catalogue_1:'{id_column_1}'/catalogue_2:'{join_catalogue_2_id_column}' against {tap_url} / {tap_table}"
                    )
                    try:
                        tap_service = get_tap_service(tap_url, **kwargs)
                    except Exception as e:
                        raise TapError(f"Failed to connect to TAP service {tap_url}: {e}")

                    result_df = perform_tap_id_join(
                        input_df,
                        tap_service,
                        tap_table,
                        id_column_1,
                        join_catalogue_2_id_column,
                        columns,
                        tap_schema,
                        retry_delay,
                        timeout,
                        verbose,
                    )
            else:
                # --- Spatial matching ---
                if local_table:
                    # Use local file directly
                    logger.info(f"Performing local spatial join using file: {local_table}")
                    result_df = perform_local_spatial_join(
                        input_df, local_table, ra, dec, radius, columns, verbose
                    )
                elif local_table_df is not None:
                    # Use provided DataFrame/Table
                    logger.info("Performing local spatial join using DataFrame/Table")
                    if isinstance(local_table_df, Table):
                        local_df = local_table_df.to_pandas()
                    else:
                        local_df = local_table_df

                    # Convert perform_local_spatial_join to use DataFrame directly
                    result_df = perform_local_spatial_join_df(
                        input_df, local_df, ra, dec, radius, columns, verbose
                    )
                else:
                    logger.info(
                        f"Performing TAP spatial join (Radius: {radius} arcsec) against {tap_url} / {tap_table}"
                    )
                    try:
                        tap_service = get_tap_service(tap_url, **kwargs)
                    except Exception as e:
                        raise TapError(f"Failed to connect to TAP service {tap_url}: {e}")

                    result_df = perform_tap_spatial_join(
                        input_df,
                        tap_service,  # Pass the service object
                        tap_table,
                        ra,
                        dec,
                        radius,
                        columns,
                        tap_schema,
                        chunk_size,
                        retry_delay,
                        timeout,
                        verbose,
                    )

            # Generate output file path if not provided
            if output_file is None:
                output_file_path = Path(temp_dir) / "tap_output.parquet"
            else:
                output_file_path = Path(output_file)
                output_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

            # Write result
            logger.info(f"Writing TAP result to {output_file_path}")
            result_df.to_parquet(output_file_path, compression="snappy", index=False)

            return str(output_file_path)

    except Exception as e:
        logger.exception("TAP cross-match failed unexpectedly.")
        raise TapError(f"TAP cross-match failed: {e}")


def ensure_j2000_gaia(df: pd.DataFrame, ra: str, dec: str) -> pd.DataFrame:
    """Convert coordinates to J2000 frame with Gaia-specific handling."""
    try:
        # Check if coordinates are already in J2000
        coords = SkyCoord(ra=df[ra], dec=df[dec], unit=(u.deg, u.deg))

        # Convert to J2000 if needed
        if coords.frame.name != "icrs":
            coords = coords.transform_to("icrs")

        # Apply proper motion correction if available
        if "pmra" in df.columns and "pmdec" in df.columns:
            # Convert proper motion to degrees per year
            pmra = df["pmra"] / 3600.0  # arcsec to degrees
            pmdec = df["pmdec"] / 3600.0

            # Apply proper motion correction
            coords = coords.apply_space_motion(
                pmra=pmra * u.deg / u.yr,
                pmdec=pmdec * u.deg / u.yr,
                radial_velocity=0 * u.km / u.s,
                dt=0 * u.yr,
            )

        df[ra] = coords.ra.deg
        df[dec] = coords.dec.deg

        return df

    except Exception as e:
        raise TapError(f"Error converting coordinates to J2000: {e}")


def perform_local_id_join(
    input_df: pd.DataFrame,
    local_table: str,
    id_column_1: str,
    id_column_2: str,
    columns: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Perform ID-based join with local table using potentially different column names."""
    local_path = Path(local_table)
    if not local_path.is_file():
        raise TapError(f"Local catalogue_2 file not found: {local_path}")

    try:
        # Read local table
        logger.debug(f"Reading local catalogue_2: {local_path}")
        suffix = local_path.suffix.lower()
        if suffix == ".parquet":
            catalogue_2_df = pd.read_parquet(local_path)
        elif suffix == ".csv":
            catalogue_2_df = pd.read_csv(local_path)
        elif suffix == ".fits":
            catalogue_2_df = Table.read(local_path).to_pandas()
        else:
            raise TapError(f"Unsupported local table format: {suffix}")

        # Select columns if specified
        cols_to_keep = {id_column_2}  # Always keep catalogue_2 ID column
        if columns:
            cols_to_keep.update(columns)

        missing_cols = cols_to_keep - set(catalogue_2_df.columns)
        if missing_cols:
            logger.warning(
                f"Columns missing in local catalogue_2 {local_path}: {missing_cols}. They will be ignored."
            )
            cols_to_keep -= missing_cols

        if id_column_2 not in catalogue_2_df.columns:
            raise TapError(f"ID column '{id_column_2}' not found in local catalogue_2 {local_path}")

        catalogue_2_subset = catalogue_2_df[list(cols_to_keep)]

        # Perform join using left_on and right_on
        result_df = input_df.merge(
            catalogue_2_subset,
            left_on=id_column_1,
            right_on=id_column_2,
            how="left",
            suffixes=("", "_catalogue_2"),
        )

        if verbose:
            # Count actual matches (where catalogue_2 columns are not NaN)
            match_count = result_df.filter(regex="_catalogue_2$").notna().any(axis=1).sum()
            logger.info(
                f"Local ID join completed. Found {match_count} matches for {len(input_df)} inputs."
            )

        return result_df

    except Exception as e:
        raise TapError(f"Local ID join failed: {e}")


def perform_local_spatial_join(
    input_df: pd.DataFrame,
    local_table: str,
    ra: str,
    dec: str,
    radius: float,
    columns: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Perform spatial join with local table."""
    local_path = Path(local_table)
    if not local_path.is_file():
        raise TapError(f"Local catalogue_2 file not found: {local_path}")

    try:
        # Read local table
        logger.debug(f"Reading local catalogue_2: {local_path}")
        suffix = local_path.suffix.lower()
        if suffix == ".parquet":
            catalogue_2_df = pd.read_parquet(local_path)
        elif suffix == ".csv":
            catalogue_2_df = pd.read_csv(local_path)
        elif suffix == ".fits":
            catalogue_2_df = Table.read(local_path).to_pandas()
        else:
            raise TapError(f"Unsupported local table format: {suffix}")

        # Select columns if specified
        cols_to_keep = {ra, dec}  # Always keep coordinate columns
        if columns:
            cols_to_keep.update(columns)

        missing_cols = cols_to_keep - set(catalogue_2_df.columns)
        if missing_cols:
            logger.warning(
                f"Columns missing in local catalogue_2 {local_path}: {missing_cols}. They will be ignored."
            )
            cols_to_keep -= missing_cols

        if ra not in catalogue_2_df.columns or dec not in catalogue_2_df.columns:
            raise TapError(
                f"Coordinate columns '{ra}', '{dec}' not found in local catalogue_2 {local_path}"
            )

        catalogue_2_subset = catalogue_2_df[list(cols_to_keep)]

        # Convert to SkyCoord
        # Assume coordinates are already in ICRS/J2000 (input was converted)
        catalogue_1_coords = SkyCoord(
            ra=input_df[ra].values * u.deg, dec=input_df[dec].values * u.deg, frame="icrs"
        )
        catalogue_2_coords = SkyCoord(
            ra=catalogue_2_subset[ra].values * u.deg,
            dec=catalogue_2_subset[dec].values * u.deg,
            frame="icrs",
        )

        # Perform cross-match
        idx, d2d, _ = match_coordinates_sky(catalogue_1_coords, catalogue_2_coords, nthneighbor=1)

        # Filter by radius and prepare results
        max_sep = u.Quantity(radius, u.arcsec)
        match_mask = d2d <= max_sep

        # Create result DataFrame with all original rows
        result_df = input_df.copy()
        result_df["separation_arcsec"] = np.nan  # Initialize separation column
        result_df.loc[match_mask, "separation_arcsec"] = d2d[match_mask].to(u.arcsec).value

        # Add catalogue_2 columns for matched rows
        catalogue_2_cols_to_add = list(cols_to_keep - {ra, dec})
        if catalogue_2_cols_to_add:
            # Add suffix to avoid clashes
            catalogue_2_subset_renamed = catalogue_2_subset.rename(
                columns={c: f"{c}_catalogue_2" for c in catalogue_2_cols_to_add}
            )
            # Get matched catalogue_2 indices
            matched_catalogue_2_indices = idx[match_mask]
            # Align catalogue_2 data using matched indices
            aligned_catalogue_2_data = catalogue_2_subset_renamed.iloc[matched_catalogue_2_indices]
            # Set index of aligned data to match the catalogue_1 index where matches occurred
            aligned_catalogue_2_data.index = result_df.index[match_mask]
            # Join the aligned catalogue_2 data
            result_df = result_df.join(
                aligned_catalogue_2_data[[f"{c}_catalogue_2" for c in catalogue_2_cols_to_add]]
            )
        elif columns:  # If user requested columns but none usable were found
            logger.warning("No usable catalogue_2 columns found to add.")

        if verbose:
            match_count = match_mask.sum()
            logger.info(
                f"Local spatial join completed. Found {match_count} matches for {len(input_df)} inputs within {radius} arcsec."
            )

        return result_df

    except Exception as e:
        raise TapError(f"Local spatial join failed: {e}")


def perform_tap_id_join(
    input_df: pd.DataFrame,
    tap_service: pyvo.dal.TAPService,
    tap_table: str,
    id_column_1: str,
    id_column_2: str,
    columns: Optional[List[str]] = None,
    tap_schema: Optional[str] = None,
    retry_delay: int = 60,
    timeout: int = 600,
    verbose: bool = False,
) -> pd.DataFrame:
    """Perform ID-based join using TAP service with potentially different column names."""
    try:
        # Build column list for SELECT query (use target ID column name)
        cols_to_select = {id_column_2}
        if columns:
            cols_to_select.update(columns)
        col_list = ", ".join(list(cols_to_select))

        # Build table name
        if tap_schema:
            table_name = f"{tap_schema}.{tap_table}"
        else:
            table_name = tap_table

        # Build ID list from catalogue_1
        id_list_str = ", ".join(map(str, input_df[id_column_1].unique()))

        # Build query using catalogue_2 ID column name in WHERE clause
        query = f"""
        SELECT {col_list}
        FROM {table_name}
        WHERE {id_column_2} IN ({id_list_str})
        """

        # Execute query
        tap_result_df = execute_tap_query(tap_service, query, retry_delay, timeout, verbose)

        # Perform left join using potentially different column names
        result_df = input_df.merge(
            tap_result_df,
            left_on=id_column_1,
            right_on=id_column_2,
            how="left",
            suffixes=("", "_catalogue_2"),
        )

        if verbose:
            logging.info(f"TAP ID join completed. Matches: {len(result_df)}")

        return result_df

    except Exception as e:
        raise TapError(f"TAP ID join failed: {e}")


def perform_tap_spatial_join(
    input_df: pd.DataFrame,
    tap_service: pyvo.dal.TAPService,
    tap_table: str,
    ra: str,
    dec: str,
    radius: float,
    columns: Optional[List[str]] = None,
    tap_schema: Optional[str] = None,
    chunk_size: int = 100000,
    retry_delay: int = 60,
    timeout: int = 600,
    verbose: bool = False,
) -> pd.DataFrame:
    """Perform spatial join using TAP service."""
    try:
        # Process in chunks
        chunks = np.array_split(input_df, len(input_df) // chunk_size + 1)
        results = []

        for i, chunk in enumerate(chunks):
            if verbose:
                logging.info(f"Processing chunk {i+1}/{len(chunks)}")

            # Build query for chunk
            query = build_spatial_query(chunk, tap_table, ra, dec, radius, columns, tap_schema)

            # Execute query with retries
            result = execute_tap_query(tap_service, query, retry_delay, timeout, verbose)

            results.append(result)

        # Combine results
        result_df = pd.concat(results, ignore_index=True)

        if verbose:
            logging.info(f"TAP spatial join completed. Matches: {len(result_df)}")

        return result_df

    except Exception as e:
        raise TapError(f"TAP spatial join failed: {e}")


def ensure_j2000(df: pd.DataFrame, ra: str, dec: str) -> pd.DataFrame:
    """Convert coordinates to J2000 frame."""
    try:
        coords = SkyCoord(ra=df[ra], dec=df[dec], unit=(u.deg, u.deg))

        # Convert to J2000 if needed
        if coords.frame.name != "icrs":
            coords = coords.transform_to("icrs")

        df[ra] = coords.ra.deg
        df[dec] = coords.dec.deg

        return df

    except Exception as e:
        raise TapError(f"Error converting coordinates to J2000: {e}")


def get_tap_table_metadata(
    tap_service: pyvo.dal.TAPService,
    tap_table: str,
    tap_schema: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fetches metadata for a specific table from a TAP service."""
    table_meta = None
    if tap_schema:
        # First try with schema
        try:
            table = tap_service.search(f"SELECT * FROM {tap_schema}.{tap_table} WHERE 1=0")
        except Exception as e:  # Be more specific if possible, e.g., pyvo.DALQueryError
            if verbose:
                logger.warning(f"Failed to get metadata for {tap_schema}.{tap_table}: {e}")
            table = None  # Continue to try without schema
        else:
            # Found with schema
            if len(table.fieldnames) > 0:
                table_meta = {"ra": None, "dec": None, "columns": table.fieldnames}
                logger.info(f"Found table '{tap_schema}.{tap_table}' via schema.")
            else:
                logger.warning(f"Query for '{tap_schema}.{tap_table}' returned no columns.")

    # Try without schema if not found or if schema wasn't provided
    if not table_meta:
        schemas_to_try = ["TAP_SCHEMA", "public"]  # Common default schemas
        if tap_schema and tap_schema not in schemas_to_try:  # Add provided schema if different
            schemas_to_try.insert(0, tap_schema)

        for schema in schemas_to_try:
            try:
                table = tap_service.search(f"SELECT * FROM {schema}.{tap_table} WHERE 1=0")
                # Check if the query actually worked and returned columns
                if len(table.fieldnames) > 0:
                    logger.info(f"Found table '{tap_table}' in schema '{schema}'.")
                    table_meta = {"ra": None, "dec": None, "columns": table.fieldnames}
                    break  # Found it
                else:
                    logger.warning(f"Query for '{schema}.{tap_table}' returned no columns.")
            except Exception as e:  # More specific exception if known
                if verbose:
                    logger.debug(f"Did not find table '{tap_table}' in schema '{schema}': {e}")
                continue  # Try next schema
        else:
            # If loop completes without break, table wasn't found in common schemas
            logger.warning(
                f"Could not find table '{tap_table}' in schemas: {schemas_to_try}. Trying without schema."
            )
            # Last attempt: try without schema qualification
            try:
                table = tap_service.search(f"SELECT * FROM {tap_table} WHERE 1=0")
                if len(table.fieldnames) > 0:
                    logger.info(f"Found table '{tap_table}' without explicit schema.")
                    table_meta = {"ra": None, "dec": None, "columns": table.fieldnames}
                else:
                    raise TapError(f"Table '{tap_table}' found but has no columns.")
            except Exception as e:
                logger.error(f"Failed to find table '{tap_table}' or get its metadata: {e}")
                raise TapError(
                    f"Could not find table '{tap_table}' on TAP service or retrieve its metadata."
                ) from e

    # Try to identify RA/Dec columns (example logic)
    if table_meta:
        if "RA" in table_meta["columns"] and "Dec" in table_meta["columns"]:
            logger.info("Found RA and Dec columns in table metadata.")
        elif "RAJ2000" in table_meta["columns"] and "DEJ2000" in table_meta["columns"]:
            logger.info("Found RAJ2000 and DEJ2000 columns in table metadata.")
        else:
            logger.warning("No suitable RA/Dec columns found in table metadata.")

    return table_meta


def build_spatial_query(
    df: pd.DataFrame,
    tap_table: str,
    ra: str,
    dec: str,
    radius: float,
    columns: Optional[List[str]] = None,
    tap_schema: Optional[str] = None,
) -> str:
    """Build spatial query for TAP service."""
    try:
        # Build column list
        if columns:
            col_list = ", ".join(columns)
        else:
            col_list = "*"

        # Build table name
        if tap_schema:
            table_name = f"{tap_schema}.{tap_table}"
        else:
            table_name = tap_table

        # Build coordinate conditions
        conditions = []
        for _, row in df.iterrows():
            conditions.append(
                f"1=CONTAINS(POINT('ICRS', {ra}, {dec}), "
                f"CIRCLE('ICRS', {row[ra]}, {row[dec]}, {radius/3600.0}))"
            )

        # Combine conditions
        where_clause = " OR ".join(conditions)

        # Build query
        query = f"""
        SELECT {col_list}
        FROM {table_name}
        WHERE {where_clause}
        """

        return query

    except Exception as e:
        raise TapError(f"Error building spatial query: {e}")


def execute_tap_query(
    tap_service: pyvo.dal.TAPService,
    query: str,
    maxrec: Optional[int] = None,
    output_format: str = "pandas",
    retry_delay: int = 60,
    max_retries: int = 3,
    timeout: Optional[int] = None, # Use service default if None
    verbose: bool = False,
) -> Union[pd.DataFrame, Table]:
    """Executes a TAP query with retries and error handling."""
    retries = 0
    service_url = tap_service.baseurl # For logging
    logger.info(f"Executing TAP query on {service_url}")
    if verbose or logger.isEnabledFor(logging.DEBUG):
        # Log query only if verbose/debug to avoid very long logs
        logger.debug(f"ADQL Query:\n{query}")

    while retries <= max_retries:
        try:
            # Set timeout for this specific query if provided
            if timeout is not None:
                # PyVO uses session.timeout, modification might not be thread-safe if service is shared
                # Consider creating a temporary service instance or managing sessions carefully if needed.
                # For now, assume setting timeout on the shared service is acceptable.
                # tap_service.session.timeout = timeout
                # Note: As of PyVO 1.4, timeout might not be directly settable per-query this way.
                # It might need to be set during TAPService instantiation.
                # The `run_sync` method itself might accept a timeout.
                logger.debug(f"Attempting query with timeout={timeout}s (if supported by run_sync)")

            # Use run_sync for synchronous execution
            # Potential kwargs: timeout (check PyVO version support)
            results = tap_service.run_sync(
                query,
                maxrec=maxrec,
                language="ADQL",
                # timeout=timeout # Check PyVO docs for run_sync timeout support
            )

            # Convert to desired format
            if output_format.lower() == "pandas":
                logger.info(f"TAP query successful. Converting {len(results)} results to pandas DataFrame.")
                return results.to_table().to_pandas()
            elif output_format.lower() == "astropy":
                logger.info(f"TAP query successful. Returning {len(results)} results as astropy Table.")
                return results.to_table()
            else:
                 # Should not happen if validation occurs earlier, but handle defensively
                 raise ValueError(f"Unsupported output format for TAP query: {output_format}")

        # Specific DAL/Network errors that might warrant a retry
        except (DALQueryError, DALServiceError, ConnectionError, Timeout, RequestException) as e:
            retries += 1
            error_type = type(e).__name__
            logger.warning(f"TAP query failed ({error_type}) on attempt {retries}/{max_retries+1} for {service_url}. Error: {e}")
            if retries > max_retries:
                logger.error(f"TAP query failed after {retries} attempts: {e}")
                raise TapError(f"TAP query failed permanently on {service_url}: {e}") from e
            else:
                sleep_time = retry_delay * (2 ** (retries - 1)) # Exponential backoff
                logger.info(f"Retrying TAP query in {sleep_time} seconds...")
                time.sleep(sleep_time)

        # Handle errors related to result format/conversion after successful query
        except DALFormatError as e:
             logger.error(f"Error processing TAP results from {service_url}: {e}", exc_info=True)
             raise TapError(f"Failed to parse results from TAP service {service_url}: {e}") from e

        # Catch other potential pyvo or general errors during query execution
        except Exception as e:
            logger.error(
                f"Unexpected error during TAP query execution on {service_url}: {e}", exc_info=True
            )
            # Re-raise as TapError to signify the source of the problem
            raise TapError(f"Unexpected error during TAP query on {service_url}: {e}") from e

    # This part should ideally not be reached if retries work or errors are raised
    raise TapError(f"TAP query execution failed unexpectedly after all retries for {service_url}")


def perform_local_id_join_df(
    input_df: pd.DataFrame,
    catalogue_2_df: pd.DataFrame,
    id_column_1: str,
    id_column_2: str,
    columns: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Perform ID-based join with DataFrame (avoids file I/O)."""
    try:
        # Select columns if specified
        cols_to_keep = {id_column_2}  # Always keep catalogue_2 ID column
        if columns:
            cols_to_keep.update(columns)

        missing_cols = cols_to_keep - set(catalogue_2_df.columns)
        if missing_cols:
            logger.warning(
                f"Columns missing in catalogue_2 DataFrame: {missing_cols}. They will be ignored."
            )
            cols_to_keep -= missing_cols

        if id_column_2 not in catalogue_2_df.columns:
            raise TapError(f"ID column '{id_column_2}' not found in catalogue_2 DataFrame")

        catalogue_2_subset = catalogue_2_df[list(cols_to_keep)]

        # Perform join using left_on and right_on
        result_df = input_df.merge(
            catalogue_2_subset,
            left_on=id_column_1,
            right_on=id_column_2,
            how="left",
            suffixes=("", "_catalogue_2"),
        )

        if verbose:
            # Count actual matches (where catalogue_2 columns are not NaN)
            match_count = result_df.filter(regex="_catalogue_2$").notna().any(axis=1).sum()
            logger.info(
                f"Local ID join completed. Found {match_count} matches for {len(input_df)} inputs."
            )

        return result_df

    except Exception as e:
        raise TapError(f"Local ID join with DataFrame failed: {e}")


def perform_local_spatial_join_df(
    input_df: pd.DataFrame,
    catalogue_2_df: pd.DataFrame,
    ra: str,
    dec: str,
    radius: float,
    columns: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Perform spatial join with DataFrame (avoids file I/O)."""
    try:
        # Select columns if specified
        cols_to_keep = {ra, dec}  # Always keep coordinate columns
        if columns:
            cols_to_keep.update(columns)

        missing_cols = cols_to_keep - set(catalogue_2_df.columns)
        if missing_cols:
            logger.warning(
                f"Columns missing in catalogue_2 DataFrame: {missing_cols}. They will be ignored."
            )
            cols_to_keep -= missing_cols

        if ra not in catalogue_2_df.columns or dec not in catalogue_2_df.columns:
            raise TapError(f"Coordinate columns '{ra}', '{dec}' not found in catalogue_2 DataFrame")

        catalogue_2_subset = catalogue_2_df[list(cols_to_keep)]

        # Convert to SkyCoord
        # Assume coordinates are already in ICRS/J2000 (input was converted)
        catalogue_1_coords = SkyCoord(
            ra=input_df[ra].values * u.deg, dec=input_df[dec].values * u.deg, frame="icrs"
        )
        catalogue_2_coords = SkyCoord(
            ra=catalogue_2_subset[ra].values * u.deg,
            dec=catalogue_2_subset[dec].values * u.deg,
            frame="icrs",
        )

        # Perform cross-match
        idx, d2d, _ = match_coordinates_sky(catalogue_1_coords, catalogue_2_coords, nthneighbor=1)

        # Filter by radius and prepare results
        max_sep = u.Quantity(radius, u.arcsec)
        match_mask = d2d <= max_sep

        # Create result DataFrame with all original rows
        result_df = input_df.copy()
        result_df["separation_arcsec"] = np.nan  # Initialize separation column
        result_df.loc[match_mask, "separation_arcsec"] = d2d[match_mask].to(u.arcsec).value

        # Add catalogue_2 columns for matched rows
        catalogue_2_cols_to_add = list(cols_to_keep - {ra, dec})
        if catalogue_2_cols_to_add:
            # Add suffix to avoid clashes
            catalogue_2_subset_renamed = catalogue_2_subset.rename(
                columns={c: f"{c}_catalogue_2" for c in catalogue_2_cols_to_add}
            )
            # Get matched catalogue_2 indices
            matched_catalogue_2_indices = idx[match_mask]
            # Align catalogue_2 data using matched indices
            aligned_catalogue_2_data = catalogue_2_subset_renamed.iloc[matched_catalogue_2_indices]
            # Set index of aligned data to match the catalogue_1 index where matches occurred
            aligned_catalogue_2_data.index = result_df.index[match_mask]
            # Join the aligned catalogue_2 data
            result_df = result_df.join(
                aligned_catalogue_2_data[[f"{c}_catalogue_2" for c in catalogue_2_cols_to_add]]
            )
        elif columns:  # If user requested columns but none usable were found
            logger.warning("No usable catalogue_2 columns found to add.")

        if verbose:
            match_count = match_mask.sum()
            logger.info(
                f"Local spatial join completed. Found {match_count} matches for {len(input_df)} inputs within {radius} arcsec."
            )

        return result_df

    except Exception as e:
        raise TapError(f"Local spatial join with DataFrame failed: {e}")
