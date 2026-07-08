import logging
import os
import sys
from typing import Any, Dict, Optional

import requests  # HTTP sessions are used for authenticated TAP endpoints.

logger = logging.getLogger(__name__)

# Check keyring availability once at module load.
# Per-service failures are handled at DEBUG level in _load_auth_from_sources().
_HAS_KEYRING = False
try:
    import keyring  # noqa: F401

    _HAS_KEYRING = True
except ImportError:
    pass

# Define known services for which to attempt loading credentials.
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
}


class AuthConfig:
    """Manages authentication configurations for different services."""

    def __init__(self, auth_details: Optional[Dict[str, Any]] = None):
        if auth_details is None:
            auth_details = self._load_auth_from_sources()
        self._auth_sessions: Dict[str, Any] = auth_details if auth_details else {}
        if self._auth_sessions:
            logger.info(
                "AuthConfig initialized with sessions for: %s",
                list(self._auth_sessions.keys()),
            )
        else:
            logger.debug("AuthConfig initialized (no authenticated sessions).")

    def _load_auth_from_sources(self) -> Dict[str, Any]:
        """Load authentication details from environment and keyring."""
        logger.debug("Loading authentication details from environment / keyring...")
        loaded_auth = {}

        # 1) Environment variables (preferred for CLI/CI usage).
        known_services = ["noao_datalab", "gaia_archive", "vizier", "cadc"]
        for service_name in known_services:
            env_prefix = f"XMATCH_{service_name.upper()}"
            username = os.environ.get(f"{env_prefix}_USER")
            password = os.environ.get(f"{env_prefix}_PASSWORD")
            if username and password:
                logger.info("Found credentials for '%s' in environment variables.", service_name)
                session = requests.Session()
                session.auth = requests.auth.HTTPBasicAuth(username, password)
                loaded_auth[service_name] = session

        # 2) Keyring (fallback for desktop users).
        if _HAS_KEYRING:
            import keyring

            for service_name in known_services:
                if service_name in loaded_auth:
                    continue  # env var already provided it
                try:
                    username = keyring.get_password(service_name, "username")
                    password = keyring.get_password(service_name, "password")
                    if username and password:
                        logger.info("Found credentials for '%s' in keyring.", service_name)
                        session = requests.Session()
                        session.auth = requests.auth.HTTPBasicAuth(username, password)
                        loaded_auth[service_name] = session
                    else:
                        logger.debug("No keyring credentials for '%s'.", service_name)
                except Exception:
                    # Expected on headless/CI — no working keyring backend.
                    logger.debug("Keyring unavailable for '%s' (no backend).", service_name)

        return loaded_auth

    def get_auth_session(self, service_name: str) -> Optional[Any]:
        """Return the authenticated session for *service_name*, or None."""
        session = self._auth_sessions.get(service_name)
        if session:
            logger.debug("Retrieved auth session for '%s'.", service_name)
        else:
            logger.debug("No auth session for '%s'.", service_name)
        return session

    def add_auth_session(self, service_name: str, session: Any) -> None:
        """Register a new authenticated session."""
        self._auth_sessions[service_name] = session
        logger.info("Added auth session for '%s'.", service_name)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #


def load_auth_config() -> AuthConfig:
    """Load and return an AuthConfig instance."""
    return AuthConfig()


def get_credentials_from_env(service_name: str) -> Dict[str, str]:
    """Read credentials from XMATCH_<SERVICE>_USER / _PASSWORD env vars."""
    env_prefix = f"XMATCH_{service_name.upper()}"
    username = os.environ.get(f"{env_prefix}_USER")
    password = os.environ.get(f"{env_prefix}_PASSWORD")
    if username and password:
        logger.info("Using credentials from environment for '%s'.", service_name)
        return {"user": username, "password": password}
    return {}


def set_credentials_interactive(service_name: str) -> bool:
    """Set credentials interactively for a service via keyring prompt."""
    import getpass

    if not _HAS_KEYRING:
        print(
            "keyring is not available. Use environment variables instead:\n"
            f"  export XMATCH_{service_name.upper()}_USER=<username>\n"
            f"  export XMATCH_{service_name.upper()}_PASSWORD=<password>",
            file=sys.stderr,
        )
        return False

    import keyring

    print(f"\nSetting credentials for {service_name}")
    desc = KNOWN_SERVICES.get(service_name, {}).get("description", "Custom service")
    print(f"Description: {desc}")

    username = input("Username: ").strip()
    if not username:
        print("Username cannot be empty. Aborting.", file=sys.stderr)
        return False

    password = getpass.getpass("Password: ")
    if not password:
        print("Password cannot be empty. Aborting.", file=sys.stderr)
        return False

    try:
        keyring.set_password(service_name, "username", username)
        keyring.set_password(service_name, "password", password)
        print(f"Credentials for {service_name} saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving credentials: {e}", file=sys.stderr)
        return False
