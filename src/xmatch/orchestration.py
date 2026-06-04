import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .exceptions import CrossMatchError
from .local_match import execute_local_id_join, execute_local_stilts_match
from .remote_cds import download_from_cds

# Import specific functions needed from other new modules
from .remote_tap import _query_tap_with_constraints, download_from_tap

logger = logging.getLogger(__name__)


def download_and_match(
    config1: Dict[str, Any], config2: Dict[str, Any], crossmatch_instance: Any, **params
) -> pd.DataFrame:
    """Executes the 'download and match locally' strategy."""
    logger.info("Executing download_and_match strategy...")

    is_local1 = config1.get("is_local", False)
    is_local2 = config2.get("is_local", False)
    df1 = None
    df2 = None

    # Handle first catalogue
    if is_local1:
        if "_input_dataframe" in config1:
            df1 = config1["_input_dataframe"]
        elif "_input_path" in config1:
            df1 = crossmatch_instance._load_local_catalogue(config1["_input_path"])
        else:
            raise CrossMatchError("First catalog marked as local but no data source provided")
    else:
        logger.info(f"Downloading first catalogue: {config1.get('_catalogue_name')}")
        try:
            # Pass necessary params like auth, columns etc.
            df1 = download_catalogue(config1, crossmatch_instance=crossmatch_instance, **params)
            if df1 is not None:
                logger.info(f"Downloaded first catalogue ({len(df1)} rows)")
        except Exception as e:
            raise CrossMatchError(f"Failed to download first catalogue: {e}") from e

    # Handle second catalogue
    if is_local2:
        if "_input_dataframe" in config2:
            df2 = config2["_input_dataframe"]
        elif "_input_path" in config2:
            df2 = crossmatch_instance._load_local_catalogue(config2["_input_path"])
        else:
            raise CrossMatchError("Second catalog marked as local but no data source provided")
    else:
        logger.info(f"Downloading second catalogue: {config2.get('_catalogue_name')}")
        try:
            # Pass necessary params like auth, columns etc.
            df2 = download_catalogue(config2, crossmatch_instance=crossmatch_instance, **params)
            if df2 is not None:
                logger.info(f"Downloaded second catalogue ({len(df2)} rows)")
        except Exception as e:
            raise CrossMatchError(f"Failed to download second catalogue: {e}") from e

    if df1 is None or df2 is None:
        raise CrossMatchError("Failed to obtain data for one or both catalogues")
    if df1.empty or df2.empty:
        logger.warning(
            "One or both catalogues are empty after loading/downloading, returning empty result"
        )
        return pd.DataFrame()

    logger.info(f"Performing local match between {len(df1)} and {len(df2)} rows")
    local_params = params.copy()
    local_params["ra1"] = config1.get("ra_column")
    local_params["dec1"] = config1.get("dec_column")
    local_params["ra2"] = config2.get("ra_column")
    local_params["dec2"] = config2.get("dec_column")
    local_params["stilts_cmd_base"] = crossmatch_instance.stilts_cmd_base
    local_params["stilts_java_opts"] = crossmatch_instance.stilts_java_opts
    local_params["stilts_tmpdir"] = crossmatch_instance.stilts_tmpdir

    join_mode = local_params.get("join_mode", "sky")
    if join_mode == "id":
        join_keys = local_params.get("join_keys", {})
        if not join_keys or "cat1" not in join_keys or "cat2" not in join_keys:
            raise CrossMatchError("ID join requested but join_keys missing or incomplete")
        # Call ID join from local_match module
        return execute_local_id_join(df1, df2, join_keys["cat1"], join_keys["cat2"], **local_params)
    else:
        # Call STILTS match from local_match module
        return execute_local_stilts_match(df1, df2, **local_params)


def execute_chunked_match(
    config1: Dict[str, Any], config2: Dict[str, Any], crossmatch_instance: Any, **params
) -> pd.DataFrame:
    """Executes a chunked match strategy for small local catalog against large remote catalog."""
    logger.info("Executing chunked match strategy...")

    # Determine which is local and which is remote
    if config1.get("is_local", False):
        local_config = config1
        remote_config = config2
        is_first_local = True
    else:
        local_config = config2
        remote_config = config1
        is_first_local = False

    # Get local data
    local_df = None
    if "_input_dataframe" in local_config:
        local_df = local_config["_input_dataframe"]
    elif "_input_path" in local_config:
        # Use the instance's method or an imported function
        local_df = crossmatch_instance._load_local_catalogue(local_config["_input_path"])

    if local_df is None or local_df.empty:
        logger.warning("Local catalog is empty. No matching possible.")
        return pd.DataFrame()

    # Get columns and parameters
    local_ra_col = local_config.get("ra_column")
    local_dec_col = local_config.get("dec_column")
    radius_arcsec = params.get("radius_arcsec", 1.0)

    if not local_ra_col or not local_dec_col:
        raise CrossMatchError("Missing RA/Dec column names for local catalog")

    # Verify columns exist
    if local_ra_col not in local_df.columns:
        raise CrossMatchError(f"RA column '{local_ra_col}' not found in local catalog")
    if local_dec_col not in local_df.columns:
        raise CrossMatchError(f"Dec column '{local_dec_col}' not found in local catalog")

    # Check if we're in small catalog mode
    small_catalog_mode = params.get("_small_catalog_mode", False)
    rows = len(local_df)

    logger.info(
        f"Processing {rows} local sources in {'small catalog mode' if small_catalog_mode else 'regular mode'}"
    )

    # Create a copy of params without radius_arcsec to avoid duplicate arguments
    process_params = params.copy()
    if "radius_arcsec" in process_params:
        del process_params["radius_arcsec"]

    if small_catalog_mode:
        # For very small catalogs (≤ 50 rows), process each source individually
        all_results = []

        # Pre-extract columns into a dictionary of numpy arrays
        local_dict = {col: local_df[col].to_numpy() for col in local_df.columns}

        for i in range(rows):
            ra = float(local_dict[local_ra_col][i])
            dec = float(local_dict[local_dec_col][i])

            logger.info(f"Processing source {i + 1}/{rows}: RA={ra}, Dec={dec}")

            # Create a single-row DataFrame for this source
            source_df = pd.DataFrame({local_ra_col: [ra], local_dec_col: [dec]})

            # Add any other columns from the original row
            for col in local_df.columns:
                if col not in source_df.columns:
                    source_df[col] = [local_dict[col][i]]

            # Process this single source - now passing radius_arcsec just once
            result = _process_coordinate_chunk(
                source_df,
                local_ra_col,
                local_dec_col,
                remote_config,
                radius_arcsec,
                is_first_local,
                crossmatch_instance,
                **process_params,
            )

            if not result.empty:
                all_results.append(result)
                logger.info(f"Found {len(result)} matches for source {idx + 1}")
            else:
                logger.info(f"No matches found for source {idx + 1}")

        # Combine all results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        # For larger catalogs, use the existing chunking approach with a reasonable chunk size
        chunk_size = min(100, max(1, rows))  # Default to chunks of 100 for medium-sized catalogs

        if rows <= chunk_size:
            # Process all at once if small enough
            logger.info("Local catalog small enough to process in one chunk")
            return _process_coordinate_chunk(
                local_df,
                local_ra_col,
                local_dec_col,
                remote_config,
                radius_arcsec,
                is_first_local,
                crossmatch_instance,
                **process_params,
            )
        else:
            # Process in chunks using parallel execution
            logger.info(f"Processing {rows} sources in {rows // chunk_size + 1} chunks")

            # Define chunk processing function for parallel execution
            # Need to pass crossmatch_instance or required methods/configs
            def process_chunk_wrapper(chunk, **kwargs):
                # Remove radius_arcsec from kwargs if present (additional safeguard)
                chunk_params = kwargs.copy()
                if "radius_arcsec" in chunk_params:
                    del chunk_params["radius_arcsec"]
                # Extract the crossmatch instance from kwargs if passed this way
                cm_instance = chunk_params.pop("crossmatch_instance", crossmatch_instance)

                return _process_coordinate_chunk(
                    chunk,
                    local_ra_col,
                    local_dec_col,
                    remote_config,
                    radius_arcsec,
                    is_first_local,
                    cm_instance,
                    **chunk_params,
                )

            # Use parallel processing helper (needs access to it)
            # Pass crossmatch_instance within the kwargs for the wrapper
            process_params["crossmatch_instance"] = crossmatch_instance
            return crossmatch_instance._process_in_parallel(
                local_df,
                process_chunk_wrapper,
                n_workers=min(8, (rows // chunk_size) + 1),
                **process_params,
            )


def _process_coordinate_chunk(
    local_chunk: pd.DataFrame,
    ra_col: str,
    dec_col: str,
    remote_config: Dict[str, Any],
    radius_arcsec: float,
    is_first_local: bool,
    crossmatch_instance: Any,
    **params,
) -> pd.DataFrame:
    """Process a chunk of coordinates against remote catalog."""
    if local_chunk.empty:
        return pd.DataFrame()

    small_catalog_mode = params.get("_small_catalog_mode", False) or len(local_chunk) <= 3

    if small_catalog_mode:
        all_results = []
        local_chunk_len = len(local_chunk)
        # Pre-extract columns to dictionary
        local_dict = {col: local_chunk[col].to_numpy() for col in local_chunk.columns}

        for i in range(local_chunk_len):
            ra = float(local_dict[ra_col][i])
            dec = float(local_dict[dec_col][i])
            logger.info(
                f"Querying remote catalog for point: RA={ra:.4f}, Dec={dec:.4f}, Radius={radius_arcsec} arcsec"
            )
            query_params = params.copy()
            query_params.update(
                {"point_query": {"ra": ra, "dec": dec, "radius_arcsec": radius_arcsec}}
            )
            try:
                remote_df = _query_remote_with_constraints(
                    remote_config, crossmatch_instance=crossmatch_instance, **query_params
                )
            except Exception as e:
                logger.error(f"Error querying remote catalog: {e}", exc_info=True)
                raise CrossMatchError(f"Error querying remote catalog: {e}") from e
            if remote_df is None or remote_df.empty:
                logger.info(f"No remote sources found for source at RA={ra:.4f}, Dec={dec:.4f}")
                continue
            logger.info(
                f"Found {len(remote_df)} remote sources near RA={ra:.4f}, Dec={dec:.4f}. Performing match..."
            )
            source_df = pd.DataFrame({ra_col: [ra], dec_col: [dec]})
            for col in local_chunk.columns:
                if col not in source_df.columns:
                    source_df[col] = [local_dict[col][i]]
            remote_ra_col = remote_config.get("ra_column")
            remote_dec_col = remote_config.get("dec_column")
            match_params = {
                "ra1": ra_col if is_first_local else remote_ra_col,
                "dec1": dec_col if is_first_local else remote_dec_col,
                "ra2": remote_ra_col if is_first_local else ra_col,
                "dec2": remote_dec_col if is_first_local else dec_col,
                "radius_arcsec": radius_arcsec,
                "join_type": params.get("join_type", "1and2"),
                "find": params.get("find", "best"),
                # Pass STILTS config from the instance
                "stilts_cmd_base": crossmatch_instance.stilts_cmd_base,
                "stilts_java_opts": crossmatch_instance.stilts_java_opts,
                "stilts_tmpdir": crossmatch_instance.stilts_tmpdir,
            }
            df1 = source_df if is_first_local else remote_df
            df2 = remote_df if is_first_local else source_df
            try:
                # Use local stilts match from local_match module
                result_df = execute_local_stilts_match(df1, df2, **match_params)
                if not result_df.empty:
                    all_results.append(result_df)
                    logger.info(
                        f"Match found {len(result_df)} matches for source at RA={ra:.4f}, Dec={dec:.4f}"
                    )
                else:
                    logger.info(f"No matches found for source at RA={ra:.4f}, Dec={dec:.4f}")
            except Exception as e:
                logger.error(f"Error in STILTS match: {e}", exc_info=True)
                raise CrossMatchError(f"Error in STILTS match: {e}") from e
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        unique_coords = local_chunk.drop_duplicates(subset=[ra_col, dec_col])
        buffer_deg = radius_arcsec / 3600.0 * 1.1
        mean_dec_rad = np.radians(unique_coords[dec_col].mean())
        # Handle potential division by zero or large values at poles
        cos_mean_dec = np.cos(mean_dec_rad)
        if abs(cos_mean_dec) < 1e-6:  # Close to pole
            cos_mean_dec = 1e-6  # Avoid division by zero, use a small value

        min_ra = float(unique_coords[ra_col].min()) - buffer_deg / cos_mean_dec
        max_ra = float(unique_coords[ra_col].max()) + buffer_deg / cos_mean_dec
        min_dec = float(unique_coords[dec_col].min()) - buffer_deg
        max_dec = float(unique_coords[dec_col].max()) + buffer_deg
        min_ra = max(0, min_ra) if min_ra <= 360 else min_ra % 360
        max_ra = min(360, max_ra) if max_ra >= 0 else max_ra % 360
        min_dec = max(-90, min_dec)
        max_dec = min(90, max_dec)
        logger.info(
            f"Querying remote catalog for region: RA=[{min_ra:.4f},{max_ra:.4f}], Dec=[{min_dec:.4f},{max_dec:.4f}]"
        )
        query_params = params.copy()
        query_params.update(
            {
                "spatial_constraint": {
                    "ra_min": min_ra,
                    "ra_max": max_ra,
                    "dec_min": min_dec,
                    "dec_max": max_dec,
                }
            }
        )
        try:
            remote_df = _query_remote_with_constraints(
                remote_config, crossmatch_instance=crossmatch_instance, **query_params
            )
        except Exception as e:
            logger.error(f"Error querying remote catalog: {e}", exc_info=True)
            raise CrossMatchError(f"Error querying remote catalog: {e}") from e
        if remote_df is None or remote_df.empty:
            logger.info("No remote sources found in this region")
            return pd.DataFrame()
        logger.info(f"Found {len(remote_df)} remote sources in region. Performing match...")
        remote_ra_col = remote_config.get("ra_column")
        remote_dec_col = remote_config.get("dec_column")
        match_params = {
            "ra1": ra_col if is_first_local else remote_ra_col,
            "dec1": dec_col if is_first_local else remote_dec_col,
            "ra2": remote_ra_col if is_first_local else ra_col,
            "dec2": remote_dec_col if is_first_local else dec_col,
            "radius_arcsec": radius_arcsec,
            "join_type": params.get("join_type", "1and2"),
            "find": params.get("find", "best"),
            # Pass STILTS config from the instance
            "stilts_cmd_base": crossmatch_instance.stilts_cmd_base,
            "stilts_java_opts": crossmatch_instance.stilts_java_opts,
            "stilts_tmpdir": crossmatch_instance.stilts_tmpdir,
        }
        df1 = local_chunk if is_first_local else remote_df
        df2 = remote_df if is_first_local else local_chunk
        try:
            # Use local stilts match from local_match module
            result_df = execute_local_stilts_match(df1, df2, **match_params)
            logger.info(f"Match completed for chunk, found {len(result_df)} matches")
            return result_df
        except Exception as e:
            logger.error(f"Error in STILTS match: {e}", exc_info=True)
            raise CrossMatchError(f"Error in STILTS match: {e}") from e


def _query_remote_with_constraints(
    remote_config: Dict[str, Any], crossmatch_instance: Any, **params
) -> pd.DataFrame:
    """Query remote catalog with constraints."""
    access_method = remote_config.get("access_method")
    # Pass auth session down
    auth_session = crossmatch_instance.auth_config.get_auth_session(
        remote_config.get("_archive_name")
    )
    params["auth_session"] = auth_session

    if access_method == "tap":
        # Use query function from remote_tap module
        return _query_tap_with_constraints(remote_config, **params)
    elif access_method == "cds_xmatch":
        # This path shouldn't be hit in chunked mode, but handle defensively
        raise CrossMatchError(
            "Cannot use chunked strategy with CDS XMatch catalogs directly via this helper"
        )
    else:
        raise CrossMatchError(f"Unsupported access method '{access_method}' for chunked strategy")


def download_catalogue(config: Dict[str, Any], crossmatch_instance: Any, **params) -> pd.DataFrame:
    """Downloads a catalogue based on its configuration."""
    cat_name = config.get("_catalogue_name", "unnamed")
    access_method = config.get("access_method")
    logger.info(f"Downloading catalogue '{cat_name}' using access method: {access_method}")

    # Pass authentication details if needed
    auth_session = crossmatch_instance.auth_config.get_auth_session(config.get("_archive_name"))
    params["auth_session"] = auth_session  # Add session to params for downstream functions

    # Pass column info correctly
    # Determine if this config is config1 or config2 based on name matching
    is_config1 = config.get("_catalogue_name") == params.get("config1", {}).get("_catalogue_name")
    columns_key = "columns1" if is_config1 else "columns2"
    params["columns_to_download"] = params.get(columns_key)  # Pass specific columns if requested

    if access_method == "tap":
        try:
            # Use download function from remote_tap module
            return download_from_tap(config, **params)
        except Exception as e:
            raise CrossMatchError(f"Failed to download catalogue '{cat_name}' from TAP: {e}") from e
    elif (
        access_method == "cds_xmatch"
    ):  # Assuming download means getting the whole catalog (or region)
        try:
            # Use download function from remote_cds module
            return download_from_cds(config, **params)
        except Exception as e:
            raise CrossMatchError(f"Failed to download catalogue '{cat_name}' from CDS: {e}") from e
    # Add other access methods like local file system if needed, although
    # download_and_match usually handles local files directly.
    # elif access_method == 'file_system':
    #     return crossmatch_instance._load_local_catalogue(config.get('_input_path'))
    else:
        raise CrossMatchError(
            f"Unsupported access method '{access_method}' for downloading catalogue '{cat_name}'"
        )
