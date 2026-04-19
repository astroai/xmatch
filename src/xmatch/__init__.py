"""xmatch: A tool for cross-matching astronomical catalogues using various methods."""

# Import core class
from .crossmatch import CrossMatch

# Import exceptions for easier access
from .exceptions import (
    ConfigError,
    CrossMatchError,
    InputError,
    StiltsError,
    TapError,
    TapUploadUnsupportedError,
)

# Define version
__version__ = "0.1.2"  # Increment version

__all__ = [
    "CrossMatch",
    # Exceptions
    "CrossMatchError",
    "ConfigError",
    "InputError",
    "TapError",
    "StiltsError",
    "TapUploadUnsupportedError",
    # Add other public classes/functions if needed
]
