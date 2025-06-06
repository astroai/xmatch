"""Custom exceptions for the xmatch package."""

class CrossMatchError(Exception):
    """Base exception for general cross-matching errors."""
    pass

class ConfigError(CrossMatchError):
    """Exception related to configuration file loading or validation."""
    pass

class InputError(CrossMatchError):
    """Exception related to invalid input data or file formats."""
    pass

class TapError(CrossMatchError):
    """Exception related to TAP service communication or query errors."""
    pass

class TapUploadUnsupportedError(TapError):
    """Exception raised when TAP upload is requested but not supported."""
    pass

class StiltsError(CrossMatchError):
    """Exception related to errors during STILTS execution."""
    pass

# Ensure AuthError is defined
class AuthError(CrossMatchError):
    """Exception related to authentication errors (e.g., missing credentials)."""
    pass
