"""xmatch: cross-match astronomical catalogues (local files, remote archives, HATS)."""

from .crossmatch import CrossMatch
from .exceptions import (
    ConfigError,
    CrossMatchError,
    InputError,
    StiltsError,
    TapError,
    TapUploadUnsupportedError,
)
from .matchers import MatchSpec
from .sources import CatalogueSource

__version__ = "0.2.0"

__all__ = [
    "CrossMatch",
    "CatalogueSource",
    "MatchSpec",
    "CrossMatchError",
    "ConfigError",
    "InputError",
    "TapError",
    "StiltsError",
    "TapUploadUnsupportedError",
    "__version__",
]
