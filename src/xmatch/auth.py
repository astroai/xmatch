import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _make_basic_auth_session(username: str, password: str) -> Any:
    import requests

    session = requests.Session()
    session.auth = requests.auth.HTTPBasicAuth(username, password)
    return session


class AuthConfig:
    """Manages authentication configurations for different services."""

    def __init__(self, auth_details: dict[str, Any] | None = None):
        if auth_details is None:
            auth_details = self._load_auth_from_sources()
        self._auth_sessions: dict[str, Any] = auth_details if auth_details else {}
        if self._auth_sessions:
            logger.info(
                "AuthConfig initialized with sessions for: %s",
                list(self._auth_sessions.keys()),
            )
        else:
            logger.debug("AuthConfig initialized (no authenticated sessions).")

    def _load_auth_from_sources(self) -> dict[str, Any]:
        """Load authentication details from environment variables."""
        logger.debug("Loading authentication details from environment...")
        loaded_auth = {}

        known_services = [
            "noao_datalab",
            "cds",
            "esa_gaia",
            "gaia_archive",
            "vizier",
            "cadc",
        ]
        for service_name in known_services:
            env_prefix = f"XMATCH_{service_name.upper()}"
            username = os.environ.get(f"{env_prefix}_USER")
            password = os.environ.get(f"{env_prefix}_PASSWORD")
            if username and password:
                logger.info("Found credentials for '%s' in environment variables.", service_name)
                loaded_auth[service_name] = _make_basic_auth_session(username, password)

        return loaded_auth

    def get_auth_session(self, service_name: str) -> Any | None:
        """Return the authenticated session for *service_name*, or None."""
        session = self._auth_sessions.get(service_name)
        if session is None:
            legacy_name = {"cds": "vizier", "esa_gaia": "gaia_archive"}.get(service_name)
            if legacy_name is not None:
                session = self._auth_sessions.get(legacy_name)
        if session:
            logger.debug("Retrieved auth session for '%s'.", service_name)
        else:
            logger.debug("No auth session for '%s'.", service_name)
        return session


def load_auth_config() -> AuthConfig:
    """Load and return an AuthConfig instance."""
    return AuthConfig()
