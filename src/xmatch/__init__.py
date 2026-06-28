"""xmatch: cross-match astronomical catalogues (local files, remote archives, HATS)."""

from .crossmatch import CrossMatch
from .exceptions import ConfigError, CrossMatchError, InputError, StiltsError, TapError
from .io_utils import (
    FrameLike,
    astropy_table_to_polars,
    is_hats_dir,
    polars_to_astropy,
    scan_frame,
    write_frame,
)
from .matchers import MatchSpec
from .sources import CatalogueSource

__version__ = "0.2.0"

__all__ = [
    # Main entry points
    "CrossMatch",
    "CatalogueSource",
    "MatchSpec",
    # Exceptions
    "CrossMatchError",
    "ConfigError",
    "InputError",
    "TapError",
    "StiltsError",
    # I/O helpers (Arrow interop)
    "FrameLike",
    "is_hats_dir",
    "scan_frame",
    "write_frame",
    "astropy_table_to_polars",
    "polars_to_astropy",
    # Meta
    "__version__",
]
