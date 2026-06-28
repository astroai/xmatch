"""Custom exceptions raised by the xmatch package.

Every public exception is documented in ``__init__`` and inherit from
:class:`CrossMatchError` so callers can catch the full family with a single
``except CrossMatchError``. Exceptions that are never raised by the current
implementation have been pruned to keep the surface honest.
"""


class CrossMatchError(Exception):
    """Base class for every public xmatch exception."""


class ConfigError(CrossMatchError):
    """Raised when the YAML config is missing, malformed, or invalid."""


class InputError(CrossMatchError):
    """Raised when an input file/frame is unreadable or unusable."""


class TapError(CrossMatchError):
    """Raised when a TAP query (ADQL) or sync/async job fails."""


class StiltsError(CrossMatchError):
    """Raised when the STILTS subprocess fails or is not available."""
