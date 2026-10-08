"""Custom exceptions raised by the xmatcher package.

Every public exception is documented in :mod:`xmatcher.__init__` and inherits
from :class:`CrossMatchError` so callers can catch the full family with a
single ``except CrossMatchError``. Exceptions that are never raised by the
current implementation have been pruned to keep the surface honest.

:class:`InputError` carries an optional ``source`` attribute — the user-supplied
identifier that triggered the failure — so the CLI's error handler can offer
"did you mean?" suggestions without regex-parsing the rendered message.
"""

from __future__ import annotations


class CrossMatchError(Exception):
    """Base class for every public xmatcher exception.

    Subclasses may attach an optional ``source`` attribute — the
    user-supplied identifier that triggered the failure — so the CLI's
    error handler can offer "did you mean?" suggestions.
    """

    source: str | None = None


class ConfigError(CrossMatchError):
    """Raised when the YAML config is missing, malformed, or invalid."""


class InputError(CrossMatchError):
    """Raised when an input file/frame is unreadable or unusable.

    The optional ``source`` attribute (keyword-only) holds the user-supplied
    identifier that triggered the failure, when one is available. The CLI's
    error-handler uses it to render a "did you mean?" suggestion against the
    known catalogue name space without having to regex-parse the message.
    """

    def __init__(self, message: str, *, source: str | None = None) -> None:
        super().__init__(message)
        self.source = source


class TapError(CrossMatchError):
    """Raised when a TAP query (ADQL) or sync/async job fails."""


class StiltsError(CrossMatchError):
    """Raised when the STILTS subprocess fails or is not available."""
