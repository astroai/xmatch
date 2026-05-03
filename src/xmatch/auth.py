import functools
import logging
import os
from typing import Any, Dict, Optional

import keyring
import requests  # Assuming requests sessions might be used

logger = logging.getLogger(__name__)

# Define known services for which to attempt loading credentials
# Users should set credentials for these service names using the keyring library
KNOWN_SERVICES = {
    "noao_datalab": {
        "urls": ["https://datalab.noirlab.edu/tap"],
        "description": "NOAO Data Lab TAP service",
    },
    "gaia_archive": {
        "urls": ["https://gea.esac.esa.int/tap-server/tap"],
        "description": "Gaia Archive TAP service",
    },
    "vizier": {
        "urls": ["http://tapvizier.cds.unistra.fr/TAPVizieR/tap"],
        "description": "VizieR TAP service",
    },
    # Add other potential services here
}


# Add a cache decorator to avoid repeated keyring access
def cache_result(func):
    """Cache the result of a function."""
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = str(args) + str(sorted(kwargs.items()))
        if cache_key not in cache:
            cache[cache_key] = func(*args, **kwargs)
        return cache[cache_key]

    return wrapper


# Placeholder for the AuthConfig class
class AuthConfig:
    """Manages authentication configurations for different services."""

    def __init__(self, auth_details: Optional[Dict[str, Any]] = None):
        """
        Initializes AuthConfig.

        Args:
            auth_details: A dictionary where keys are service/archive names
                          and values are authentication details (e.g., requests.Session).
                          If None, attempts to load from keyring or other sources.
        """
        if auth_details is None:
            auth_details = self._load_auth_from_sources()
        self._auth_sessions: Dict[str, Any] = auth_details if auth_details else {}
        logger.info(f"AuthConfig initialized with sessions for: {list(self._auth_sessions.keys())}")

    def _load_auth_from_sources(self) -> Dict[str, Any]:
        """
        Placeholder method to load authentication details from keyring or other secure storage.
        Needs implementation based on how credentials should be stored and retrieved.
        """
        logger.debug("Attempting to load authentication details from sources (keyring, etc.)...")
        loaded_auth = {}
        # Example: Try loading credentials for known services from keyring
        known_services = ["noao_datalab", "gaia_archive", "vizier", "cadc"]  # Add more as needed
        for service_name in known_services:
            try:
                # This is a simplified example. Real implementation needs to handle
                # username/password retrieval and potentially create a requests.Session
                # with appropriate authentication (e.g., HTTPBasicAuth).
                username = keyring.get_password(service_name, "username")
                password = keyring.get_password(service_name, "password")

                if username and password:
                    logger.info(f"Found credentials for service '{service_name}' in keyring.")
                    # Example: Create a requests session with basic auth
                    session = requests.Session()
                    session.auth = requests.auth.HTTPBasicAuth(username, password)
                    loaded_auth[service_name] = session
                    # Clear sensitive variables immediately after use if possible
                    del username
                    del password
                else:
                    logger.debug(
                        f"Credentials for service '{service_name}' not found or incomplete in keyring."
                    )

            except Exception as e:
                logger.debug(f"Error accessing keyring for service '{service_name}': {e}")

        return loaded_auth

    def get_auth_session(self, service_name: str) -> Optional[Any]:
        """
        Retrieves the authentication session/details for a given service name.

        Args:
            service_name: The name of the service/archive (e.g., 'gaia_archive', 'cds').

        Returns:
            The authentication object (e.g., requests.Session) or None if not found.
        """
        session = self._auth_sessions.get(service_name)
        if session:
            logger.debug(f"Retrieved auth session for service '{service_name}'.")
        else:
            logger.debug(f"No pre-configured auth session found for service '{service_name}'.")
        return session

    def add_auth_session(self, service_name: str, session: Any):
        """Adds or updates an authentication session."""
        self._auth_sessions[service_name] = session
        logger.info(f"Added/Updated auth session for service '{service_name}'.")


# Function to load the auth config (called from CrossMatch.__init__)
def load_auth_config() -> AuthConfig:
    """Loads and returns an AuthConfig instance."""
    # In a real scenario, this might load details from a file or environment
    # and pass them to the AuthConfig constructor.
    # For now, it relies on the AuthConfig constructor to load from keyring.
    return AuthConfig()


# Add environment variable support for authentication
def get_credentials_from_env(service_name: str) -> Dict[str, str]:
    """Get credentials from environment variables if available."""
    env_prefix = f"XMATCH_{service_name.upper()}"
    username = os.environ.get(f"{env_prefix}_USER")
    password = os.environ.get(f"{env_prefix}_PASSWORD")

    if username and password:
        logger.info(f"Using credentials from environment variables for {service_name}")
        return {"user": username, "password": password}

    return {}


# Add function to set credentials interactively
def set_credentials_interactive(service_name: str) -> bool:
    """Set credentials interactively for a service."""
    import getpass

    print(f"\nSetting credentials for {service_name}")
    print(
        f"Description: {KNOWN_SERVICES.get(service_name, {}).get('description', 'Custom service')}"
    )

    username = input("Username: ").strip()
    if not username:
        print("Username cannot be empty. Aborting.")
        return False

    password = getpass.getpass("Password: ")
    if not password:
        print("Password cannot be empty. Aborting.")
        return False

    try:
        keyring.set_password(service_name, "username", username)
        keyring.set_password(service_name, "password", password)
        print(f"Credentials for {service_name} saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving credentials: {e}")
        return False


# Example usage (optional, for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    auth_config = load_auth_config()
    gaia_session = auth_config.get_auth_session("gaia_archive")
    print(f"Gaia Session: {gaia_session}")
    # Example of adding a session manually (if needed)
    # custom_session = requests.Session()
    # auth_config.add_auth_session('my_custom_service', custom_session)
