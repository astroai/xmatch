"""Minimal TAP service access built on pyvo."""

import logging
from typing import Any, Optional

from .exceptions import TapError

logger = logging.getLogger(__name__)

_service_cache: dict = {}


def get_tap_service(tap_url: str, auth_session: Optional[Any] = None):
    """Return a (cached) pyvo TAPService for *tap_url*."""
    import pyvo

    key = (tap_url, id(auth_session) if auth_session is not None else None)
    if key not in _service_cache:
        try:
            _service_cache[key] = pyvo.dal.TAPService(tap_url, session=auth_session)
            logger.info("Connected to TAP service %s", tap_url)
        except Exception as exc:
            raise TapError(f"Failed to connect to TAP service {tap_url}: {exc}") from exc
    return _service_cache[key]


def execute_tap_query(tap_service, query: str, maxrec: Optional[int] = None):
    """Execute an ADQL query (async) and return the result as an astropy Table."""
    logger.debug("ADQL: %s", query)
    try:
        job = tap_service.submit_job(query, maxrec=maxrec, language="ADQL")
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
        if job.phase != "COMPLETED":
            message = getattr(job, "message", None) or job.phase
            raise TapError(f"TAP job did not complete: {message}")
        return job.fetch_result().to_table()
    except TapError:
        raise
    except Exception as exc:
        raise TapError(f"TAP query failed: {exc}") from exc
