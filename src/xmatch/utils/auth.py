import keyring
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Define known services for which to attempt loading credentials
# Users should set credentials for these service names using the keyring library
KNOWN_SERVICES = {
    'noao_datalab': {
        'urls': ['https://datalab.noirlab.edu/tap'],
        'description': 'NOAO Data Lab TAP service'
    },
    'gaia_archive': {
        'urls': ['https://gea.esac.esa.int/tap-server/tap'],
        'description': 'Gaia Archive TAP service'
    },
    'vizier': {
        'urls': ['http://tapvizier.cds.unistra.fr/TAPVizieR/tap'],
        'description': 'VizieR TAP service'
    }
    # Add other potential services here
}

def load_auth_config() -> Dict[str, Dict[str, str]]:
    """
    Loads authentication credentials securely using the keyring library
    for predefined services.

    Returns:
        A dictionary where keys are service names (e.g., 'noao_datalab')
        and values are dictionaries containing 'user' and 'password'.
        Returns an empty dictionary if no credentials are found or
        if keyring is unavailable.
    """
    auth_config = {}
    try:
        for service_name, service_info in KNOWN_SERVICES.items():
            username = keyring.get_password(service_name, 'username')
            password = keyring.get_password(service_name, 'password')

            if username is not None and password is not None:
                logger.info(f"Found credentials for service: {service_name}")
                # Use 'user' key as this is common for STILTS/TAP parameters
                auth_config[service_name] = {
                    'user': username,
                    'password': password,
                    'urls': service_info.get('urls', [])
                }
            else:
                logger.debug(f"Credentials for service '{service_name}' not found in keyring.")

    except keyring.errors.NoKeyringError:
        logger.warning(
            "No keyring backend found. Cannot load credentials securely. "
            "Please install a keyring backend (e.g., 'keyrings.cryptfile', 'keyrings.osx') "
            "and configure credentials using the 'keyring' command-line tool."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while accessing the keyring: {e}", exc_info=True)

    return auth_config