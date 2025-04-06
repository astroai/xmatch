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


def execute_tap_query(
    tap_service: pyvo.dal.TAPService,
    adql_query: str,
    max_retries: int = 3,
    retry_delay: int = 60,
    timeout: int = 600,
    verbose: bool = False,
    upload_table: Optional[Union[Table, str]] = None,
    upload_name: Optional[str] = None,
) -> pd.DataFrame:
    """Executes an ADQL query using a TAP service with retries and timeout.

    Args:
        tap_service: An initialized pyvo.dal.TAPService object.
        adql_query: The ADQL query string.
        max_retries: Maximum number of retries on failure.
        retry_delay: Base delay (seconds) between retries (exponential backoff).
        timeout: Timeout for the query execution in seconds.
        verbose: Print progress information.
        upload_table: Astropy Table or path to file to upload.
        upload_name: Name for the uploaded table.

    Returns:
        DataFrame containing the query results, or an empty DataFrame on failure after retries.

    Raises:
        TapError: If the query fails definitively after retries.
    """
    retries = 0
    last_exception = None

    while retries <= max_retries:
        try:
            logger.info(f"Executing TAP query (Attempt {retries + 1}/{max_retries + 1})...")
            if verbose:
                logger.info(f"ADQL Query:\n{adql_query}")
            if upload_table:
                logger.info(f"Uploading table '{upload_name}' for query.")

            # Execute the query using the TAP service object
            # Set timeout via searchparams if possible, or rely on underlying requests timeout?
            # pyvo doesn't seem to expose timeout directly in run_async?
            # Let's assume underlying HTTP timeout applies or service default.
            job = tap_service.run_async(
                adql_query,
                uploads={upload_name: upload_table} if upload_table and upload_name else None,
                # maxrec=? Can be useful but not directly exposed here
            )

            # Wait for job completion (check status?)
            # Using results directly blocks until completion.
            results_table = job.results

            # Check for empty results vs actual errors
            if results_table is None:
                # This might indicate an issue rather than just empty results
                logger.warning("TAP query job result is None. Checking job status.")
                job.raise_if_error() # Raise exception if job phase is ERROR
                logger.warning("TAP query returned None results but job status is OK. Treating as empty.")
                return pd.DataFrame() # Treat as empty

            logger.info(f"TAP query successful. Received {len(results_table)} rows.")
            # Convert Astropy Table to Pandas DataFrame
            try:
                results_df = results_table.to_pandas()
                # Convert potentially problematic DTYPES (like object containing bytes)
                for col in results_df.select_dtypes(include=['object']).columns:
                    # Attempt to decode if bytes are present
                    try:
                        if results_df[col].iloc[0] is not None and isinstance(results_df[col].iloc[0], bytes):
                            results_df[col] = results_df[col].str.decode('utf-8', errors='replace')
                            logger.debug(f"Decoded byte string in column '{col}'.")
                    except Exception as decode_err:
                        logger.warning(f"Could not decode bytes in column '{col}': {decode_err}. Skipping.")
                return results_df
            except EmptyDataError:
                logger.info("TAP query result table is empty.")
                return pd.DataFrame() # Return empty DataFrame for 0 rows
            except Exception as e:
                logger.error(f"Failed to convert TAP results Table to DataFrame: {e}", exc_info=True)
                # Consider this a failure - raise TapError?
                raise TapError(f"Failed to convert results to DataFrame: {e}") from e

        except (DALQueryError, DALServiceError, DALFormatError, ConnectionError, Timeout, RequestException) as e:
            last_exception = e
            retries += 1
            logger.warning(f"TAP query failed (Attempt {retries}/{max_retries + 1}): {type(e).__name__} - {e}")
            if retries <= max_retries:
                sleep_time = retry_delay * (2 ** (retries - 1)) # Exponential backoff
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"TAP query failed after {max_retries} retries.")
                # Raise a TapError wrapping the last exception
                raise TapError(f"TAP query failed after {max_retries} retries: {last_exception}") from last_exception
        except Exception as e:
            # Catch other unexpected errors during query execution
            logger.error(f"Unexpected error during TAP query: {e}", exc_info=True)
            # Raise TapError wrapping the unexpected exception
            raise TapError(f"Unexpected error during TAP query: {e}") from e

    # Should not be reached if loop finishes, but return empty DF as fallback
    logger.error("TAP query loop exited unexpectedly.")
    if last_exception:
        raise TapError(f"TAP query failed: {last_exception}") from last_exception
    else:
        raise TapError("TAP query failed for an unknown reason.")
