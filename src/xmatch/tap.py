"""Minimal TAP service access built on pyvo."""

import logging
import time
from typing import Any, Callable, Optional

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


# Maximum time we'll keep polling a TAP job before giving up.  This is a
# safety cap (6 hours) for pathological archives, not a feature limit;
# we surface the elapsed time in the progress spinner so users can
# intervene before this triggers.
_POLL_TIMEOUT_SECONDS = 6 * 3600.0


def _poll_job(
    job: Any, progress_cb: Optional[Callable[[str], None]], *, interval: float = 0.5
) -> None:
    """Poll ``job.phase`` on the main thread, surface transitions to the spinner.

    Replaces pyvo's blocking ``job.wait()`` so backend code can report
    state transitions ("queued" → "running") to the user.  Each poll
    sleeps ``interval`` seconds; in practice the foreground progress
    *animation* runs at 10 Hz in a separate thread, so the spinner
    keeps ticking visually while we wait for the server.

    Stops when ``job.phase`` reaches a terminal state, when the safety
    cap (``_POLL_TIMEOUT_SECONDS``) is exceeded, or when the caller
    cancels via ``KeyboardInterrupt``.
    """
    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while job.phase not in ("COMPLETED", "ERROR", "ABORTED"):
        if progress_cb is not None:
            try:
                progress_cb(f"phase: {job.phase.lower()}")
            except Exception:  # noqa: BLE001 — never let a UI bug crash the download
                logger.debug("progress_cb raised; ignoring", exc_info=True)
        if time.monotonic() >= deadline:
            logger.warning(
                "TAP job exceeded %sh polling cap; abandoning.", _POLL_TIMEOUT_SECONDS // 3600
            )
            return
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            raise


def execute_tap_query(
    tap_service,
    query: str,
    maxrec: Optional[int] = None,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Execute an ADQL query (async) and return the result as an astropy Table.

    ``progress_cb`` is invoked with short status strings at three points:
    on submission ("submitting TAP job"), every poll iteration
    ("phase: queued" / "phase: running" / ...), and on completion
    ("fetching N rows").  Pass :data:`None` for silent operation.
    """
    logger.debug("ADQL: %s", query)
    try:
        job = tap_service.submit_job(query, maxrec=maxrec, language="ADQL")
        if progress_cb is not None:
            progress_cb("submitting TAP job")
        job.run()
        _poll_job(job, progress_cb)
        if job.phase != "COMPLETED":
            message = getattr(job, "message", None) or job.phase
            raise TapError(f"TAP job did not complete: {message}")
        result = job.fetch_result().to_table()
        if progress_cb is not None:
            progress_cb(f"fetching {len(result)} rows")
        return result
    except TapError:
        raise
    except Exception as exc:
        raise TapError(f"TAP query failed: {exc}") from exc
