import logging
import time
from typing import Optional

import pyvo
from astropy.table import Table

# Specific exceptions
# Corrected import: TapService -> TAPService
from pyvo.dal import DALQueryError, DALServiceError, TAPService
from requests.exceptions import ConnectionError, RequestException, Timeout

# Use centralized exceptions
from .exceptions import TapError, TapUploadUnsupportedError

logger = logging.getLogger(__name__)

# Add connection pooling for TAP services
_tap_service_cache = {}


def get_tap_service(tap_url: str, **kwargs) -> pyvo.dal.TAPService:
    """Get a cached TAP service connection or create a new one."""
    # Create a cache key based on URL and relevant kwargs (like auth)
    cache_key_parts = [tap_url]
    auth_session = kwargs.get("auth_session")
    if auth_session:
        # Include relevant auth details if present (e.g., credentials identifier)
        # WARNING: Avoid caching sensitive parts directly. Use a hash or identifier if possible.
        # For simplicity here, we might just use the presence of an auth session.
        # A more robust solution might involve inspecting auth_session details safely.
        cache_key_parts.append(f"auth_present={auth_session is not None}")

    cache_key = tuple(cache_key_parts)

    if cache_key not in _tap_service_cache:
        logger.info(f"Creating new TAP service connection for {tap_url}")
        try:
            # Pass auth session if provided
            service = pyvo.dal.TAPService(tap_url, session=auth_session)
            _tap_service_cache[cache_key] = service
            logger.debug(f"Successfully created and cached TAP service for {tap_url}")
        except (ConnectionError, Timeout, DALServiceError, RequestException) as e:
            logger.error(f"Failed to connect to TAP service {tap_url}: {e}")
            raise TapError(f"Failed to connect to TAP service {tap_url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error creating TAP service for {tap_url}: {e}", exc_info=True)
            raise TapError(f"Unexpected error creating TAP service for {tap_url}: {e}")
    else:
        logger.debug(f"Using cached TAP service connection for {tap_url}")

    return _tap_service_cache[cache_key]


def execute_tap_query(
    tap_service: TAPService,
    query: str,
    maxrec: Optional[int] = None,
    upload_params: Optional[dict] = None,
    retries: int = 3,
    retry_delay: int = 3,
    job_timeout: int = 300,  # Example timeout for waiting
) -> Table:
    """Executes a TAP query, handling async jobs, retries, and detailed error logging."""
    last_exception = None
    job = None  # Initialize job variable

    for attempt in range(1, retries + 1):
        job_detail = ""  # Reset detail for each attempt
        try:
            logger.info(f"Executing TAP query (Attempt {attempt}/{retries})...")
            logger.debug(f"Query: {query}")  # Log the query being sent

            if upload_params:
                raise TapUploadUnsupportedError(
                    "Upload via execute_tap_query not fully implemented here."
                )
            else:
                logger.debug("Using service.submit_job method.")
                job = tap_service.submit_job(
                    query, maxrec=maxrec, language="ADQL"
                )  # Explicitly ADQL
                job.run()  # Start the job

            logger.debug(f"Monitoring TAP job (type: {type(job).__name__}, ID: {job.job_id})...")
            # Wait for job completion or failure
            job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=job_timeout)

            job_phase = job.phase
            logger.debug(f"Job phase after wait: {job_phase}")

            if job_phase == "COMPLETED":
                logger.info("TAP query completed successfully.")
                return job.fetch_result()
            else:
                # --- Enhanced Error Handling ---
                job_detail = f"TAP job failed with phase: {job_phase}"
                error_summary = None  # Initialize error_summary
                try:
                    # Try common attributes first
                    if hasattr(job, "message") and job.message:
                        error_summary = str(job.message)
                    elif (
                        hasattr(job, "error_summary")
                        and job.error_summary
                        and hasattr(job.error_summary, "message")
                        and job.error_summary.message
                    ):
                        error_summary = str(job.error_summary.message)
                    elif (
                        hasattr(job, "parameters")
                        and isinstance(job.parameters, dict)
                        and "error" in job.parameters
                    ):
                        # Check job parameters dictionary
                        error_summary = str(job.parameters["error"])

                    # Attempt standard XML parsing
                    if not error_summary and hasattr(job, "xml"):
                        try:
                            # Look for common error elements/attributes in UWS standard
                            error_node = job.xml.find(
                                ".//{http://www.ivoa.net/xml/UWS/v1.0}message"
                            )
                            if error_node is not None and error_node.text:
                                error_summary = error_node.text
                            else:
                                # Try another common pattern (parameter with id='error')
                                error_param = job.xml.find(
                                    ".//{http://www.ivoa.net/xml/UWS/v1.0}parameter[@id='error']"
                                )
                                if error_param is not None and error_param.text:
                                    error_summary = error_param.text
                        except Exception as xml_parse_err:
                            logger.warning(
                                f"Could not parse job XML for standard error details: {xml_parse_err}"
                            )

                except Exception as detail_err:
                    # Catch errors during standard attribute/parameter checking
                    logger.warning(
                        f"Could not retrieve detailed error message using standard methods: {detail_err}"
                    )

                # --- Log Raw XML if available, regardless of previous success ---
                raw_xml_logged = False
                if hasattr(job, "xml"):
                    logger.warning("Inspecting raw job XML for error details:")
                    try:
                        # Use lxml's tostring for potentially cleaner output if available
                        from lxml import etree

                        xml_string = etree.tostring(job.xml, pretty_print=True, encoding="unicode")
                        logger.warning(f"Raw Job XML:\n{xml_string}")
                        raw_xml_logged = True
                    except ImportError:
                        # Fallback to standard xml.etree
                        try:
                            import xml.etree.ElementTree as ET

                            xml_string = ET.tostring(job.xml, encoding="unicode")
                            logger.warning(f"Raw Job XML:\n{xml_string}")
                            raw_xml_logged = True
                        except Exception as et_xml_log_err:
                            logger.warning(
                                f"Failed to log raw job XML using xml.etree: {et_xml_log_err}"
                            )
                    except Exception as xml_log_err:
                        logger.warning(f"Failed to log raw job XML: {xml_log_err}")
                # --- End Raw XML Logging ---

                # Construct final log message
                if error_summary:
                    job_detail += f". Detail: {error_summary}"
                elif raw_xml_logged:
                    job_detail += ". Detail: See raw XML log above."
                else:
                    job_detail += ". Detail: No detailed error summary could be extracted."

                logger.warning(f"{job_detail} (Attempt {attempt}/{retries})")
                last_exception = TapError(job_detail)
                # --- End Enhanced Error Handling ---

        except DALQueryError as e:
            job_detail = f"DALQueryError during TAP query (Attempt {attempt}/{retries}): {e}"
            logger.warning(job_detail)
            last_exception = TapError(job_detail)
        except TimeoutError:
            job_detail = f"TAP job timed out after {job_timeout}s (Attempt {attempt}/{retries})"
            logger.warning(job_detail)
            last_exception = TapError(job_detail)
            if job:
                job.delete()  # Attempt to clean up timed-out job
        except Exception as e:
            job_detail = f"Unexpected error during TAP query (Attempt {attempt}/{retries}): {e}"
            logger.error(job_detail, exc_info=True)  # Log full traceback for unexpected errors
            last_exception = TapError(job_detail)

        # Wait before retrying if not the last attempt and error occurred
        if last_exception and attempt < retries:
            logger.info(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)
            last_exception = None  # Reset for next attempt unless it's the final one

    # If all retries failed
    logger.error(f"TAP query failed after {retries} attempts.")
    if last_exception:
        raise last_exception
    else:
        raise TapError("TAP query failed after all retries for an unknown reason.")
