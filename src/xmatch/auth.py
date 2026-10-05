import logging
import os
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


def load_auth_config() -> AuthConfig:
    """Load and return an AuthConfig instance."""
    return AuthConfig()
