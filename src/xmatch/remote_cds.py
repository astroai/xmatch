import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

# Import astroquery conditionally
try:
    from astroquery.vizier import Vizier
    from astroquery.xmatch import XMatch
    from pyvo.dal import DALQueryError # Specific error for CDS XMatch
    from requests.exceptions import ConnectionError, Timeout, RequestException # Network errors
    ASTROQUERY_AVAILABLE = True
except ImportError:
    Vizier = None
    XMatch = None
    DALQueryError = None
    ConnectionError = None
    Timeout = None
    RequestException = None
    ASTROQUERY_AVAILABLE = False
    logging.warning("astroquery not found. CDS functionality will be limited.")


logger = logging.getLogger(__name__)

def download_from_cds(config: Dict[str, Any], **params) -> pd.DataFrame:
    """Downloads a catalogue from CDS VizieR."""
    from .exceptions import CrossMatchError
    if not ASTROQUERY_AVAILABLE or Vizier is None:
        raise CrossMatchError("astroquery.vizier is required for downloading from CDS.")

    catalog_id = config.get('access_identifier')
    if not catalog_id:
        raise CrossMatchError(f"Missing access_identifier for CDS catalogue")

    # Determine columns to fetch
    columns_list = params.get('columns_to_download') or ['*']
    # Vizier might need specific column names like '_RAJ2000', '_DEJ2000'
    # This needs more robust mapping based on config if not using '*'
    vizier_columns = columns_list # Use directly for now

    try:
        v = Vizier(columns=vizier_columns, row_limit=-1) # Fetch all rows matching criteria

        # Add spatial constraint if provided
        ra = params.get('ra')
        dec = params.get('dec')
        radius_arcsec = params.get('radius_arcsec')
        result_table_list = None

        if ra is not None and dec is not None and radius_arcsec is not None and radius_arcsec > 0:
            logger.info(f"Querying CDS Vizier ({catalog_id}) spatially: RA={ra}, Dec={dec}, Radius={radius_arcsec} arcsec")
            center_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            result_table_list = v.query_region(
                center_coord,
                radius=radius_arcsec * u.arcsec,
                catalog=catalog_id
            )
        else:
            # No spatial constraint - download based on other criteria or full catalog (dangerous)
            # Vizier query_object is not ideal for full download.
            # Consider adding a warning or requiring spatial constraints for CDS downloads.
            logger.warning(f"No spatial constraint provided for CDS download of {catalog_id}. This might be slow or incomplete.")
            # Attempting a broad query - adjust as needed
            try:
                 # Query a large region or use query_object if appropriate
                 # This is just an example, might need refinement
                 result_table_list = v.query_region("0 0", radius=180*u.deg, catalog=catalog_id)
                 logger.warning(f"Attempting broad query for {catalog_id}. Results might be very large or limited by Vizier.")
            except Exception as e_broad:
                 logger.error(f"Broad Vizier query failed for {catalog_id}: {e_broad}. Cannot download without constraints.")
                 raise CrossMatchError(f"Cannot download {catalog_id} from CDS without spatial constraints.")


        if not result_table_list:
            logger.warning(f"No results from CDS Vizier query for {catalog_id}")
            return pd.DataFrame()

        # Vizier often returns a list of tables; concatenate if necessary, but usually the first is the main one.
        if len(result_table_list) > 1:
             logger.warning(f"Vizier query returned {len(result_table_list)} tables. Using the first one.")

        # Convert to pandas DataFrame
        result_df = result_table_list[0].to_pandas()
        logger.info(f"Downloaded {len(result_df)} rows from CDS Vizier ({catalog_id})")
        return result_df

    except Exception as e:
        logger.error(f"Failed to download from CDS Vizier ({catalog_id}): {e}", exc_info=True)
        raise CrossMatchError(f"Failed to download from CDS ({catalog_id}): {e}") from e

def execute_cds_xmatch_local_remote(config1: Dict[str, Any], config2: Dict[str, Any], crossmatch_instance: Any, **params) -> pd.DataFrame:
    """Executes a crossmatch using the CDS XMatch service via astroquery."""
    from .exceptions import CrossMatchError
    logger.info("Executing CDS XMatch (local vs remote) strategy...")

    if not ASTROQUERY_AVAILABLE or XMatch is None:
        raise CrossMatchError("astroquery.xmatch is required for CDS XMatch strategy.")

    # Determine which is local and which is remote (CDS)
    if config1.get("access_method") == "cds_xmatch":
        remote_conf = config1
        local_conf = config2
        is_first_remote = True
    elif config2.get("access_method") == "cds_xmatch":
        remote_conf = config2
        local_conf = config1
        is_first_remote = False
    else:
        raise CrossMatchError("Neither input configuration specifies 'cds_xmatch' access method.")

    # Get remote CDS catalogue identifier
    cds_cat_identifier = remote_conf.get("access_identifier")
    if not cds_cat_identifier:
        raise CrossMatchError("Missing 'access_identifier' for CDS catalogue.")

    # Get local data (either DataFrame or path to file)
    local_input = None
    if "_input_dataframe" in local_conf:
        local_input = local_conf["_input_dataframe"]
        logger.info("Using in-memory DataFrame for local input to CDS XMatch.")
        # Astroquery needs RA/Dec columns named specifically or passed explicitly
        ra_col = local_conf.get("ra_column", "ra")
        dec_col = local_conf.get("dec_column", "dec")
        # Ensure columns exist
        if ra_col not in local_input.columns or dec_col not in local_input.columns:
             raise CrossMatchError(f"RA ('{ra_col}') or Dec ('{dec_col}') column not found in local DataFrame.")
        # Convert DataFrame to Astropy Table for astroquery
        try:
            local_table = Table.from_pandas(local_input[[ra_col, dec_col]])
            # Rename columns if necessary for astroquery default expectations
            if ra_col != 'ra': local_table.rename_column(ra_col, 'ra')
            if dec_col != 'dec': local_table.rename_column(dec_col, 'dec')
            local_input = local_table
        except Exception as e:
            raise CrossMatchError(f"Failed to convert local DataFrame to Astropy Table: {e}")

    elif "_input_path" in local_conf:
        local_input = local_conf["_input_path"]
        logger.info(f"Using local file '{local_input}' for input to CDS XMatch.")
        # Need to tell astroquery the RA/Dec column names if not default
        ra_col = local_conf.get("ra_column", "ra")
        dec_col = local_conf.get("dec_column", "dec")
    else:
        raise CrossMatchError("Local input for CDS XMatch requires either a DataFrame or a file path.")

    # Get crossmatch parameters
    radius_arcsec = params.get("radius_arcsec", 1.0)
    # CDS XMatch uses 'distMaxArcsec'
    distMaxArcsec = radius_arcsec

    try:
        logger.info(f"Submitting job to CDS XMatch: Local vs {cds_cat_identifier}, Radius: {distMaxArcsec} arcsec")
        xmatch = XMatch()
        # Perform the crossmatch
        # Need to handle potential differences in column naming for local input
        result_table = xmatch.query(
            cat1=local_input, # Can be Table, file path, or URL
            cat2=f"vizier:{cds_cat_identifier}",
            max_distance=distMaxArcsec * u.arcsec,
            colRA1=ra_col if isinstance(local_input, (str, Path)) else None, # Only specify for files
            colDec1=dec_col if isinstance(local_input, (str, Path)) else None, # Only specify for files
            # Potentially add colRA2, colDec2 if remote config specifies non-default names
        )

        if result_table is None or len(result_table) == 0:
            logger.info("CDS XMatch returned no results.")
            return pd.DataFrame()

        logger.info(f"CDS XMatch successful, received {len(result_table)} matches.")
        # Convert result Astropy Table to pandas DataFrame
        result_df = result_table.to_pandas()

        # --- Important: Reconcile results with original local data ---
        # The result_df from astroquery only contains RA/Dec/Dist and remote columns.
        # We need to merge it back with the *original* local data based on the input RA/Dec.
        # This can be tricky due to potential floating point inaccuracies.

        # Option 1: If local input was DataFrame, merge back
        if isinstance(local_conf.get("_input_dataframe"), pd.DataFrame):
            original_local_df = local_conf["_input_dataframe"]
            local_ra_col_orig = local_conf.get("ra_column", "ra")
            local_dec_col_orig = local_conf.get("dec_column", "dec")

            # Prepare result_df for merge (rename RA/Dec columns from CDS result)
            # CDS result columns are typically named 'ra', 'dec' (from cat1)
            result_df = result_df.rename(columns={'ra': local_ra_col_orig, 'dec': local_dec_col_orig})

            # Perform the merge - use tolerance if needed, or add unique ID before query
            # Simple merge might work if RA/Dec are precise enough
            try:
                # Add suffixes to distinguish columns from cat1 and cat2 if names clash
                # The remote columns are already prefixed by CDS identifier usually
                final_df = pd.merge(
                    original_local_df,
                    result_df,
                    on=[local_ra_col_orig, local_dec_col_orig],
                    how='inner', # Keep only matched rows
                    suffixes=('_local', '_remote') # Suffixes might not be needed if CDS prefixes remote cols
                )
                logger.info(f"Merged CDS results back with local DataFrame, final size: {len(final_df)}")
                return final_df
            except Exception as e:
                logger.error(f"Failed to merge CDS results back with local DataFrame: {e}. Returning raw CDS result.")
                # Fallback: return the raw result, but it lacks original local columns
                return result_df

        # Option 2: If local input was file, return the CDS result directly
        # (User would need to join it themselves later if needed)
        else:
             logger.warning("Local input was a file. Returning raw CDS XMatch result without original local columns.")
             return result_df

    except DALQueryError as e:
        logger.error(f"CDS XMatch query failed (DALQueryError): {e}")
        raise CrossMatchError(f"CDS XMatch query failed: {e}") from e
    except ConnectionError as e:
         logger.error(f"CDS XMatch connection error: {e}")
         raise CrossMatchError(f"CDS XMatch connection error: {e}") from e
    except Timeout as e:
         logger.error(f"CDS XMatch request timed out: {e}")
         raise CrossMatchError(f"CDS XMatch request timed out: {e}") from e
    except RequestException as e:
         logger.error(f"CDS XMatch request failed: {e}")
         raise CrossMatchError(f"CDS XMatch request failed: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during CDS XMatch: {e}", exc_info=True)
        raise CrossMatchError(f"Unexpected CDS XMatch error: {e}") from e
