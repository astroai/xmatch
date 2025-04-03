import keyring
import logging
import functools
import os
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

@cache_result
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
            # Try environment variables first
            env_creds = get_credentials_from_env(service_name)
            if env_creds:
                env_creds['urls'] = service_info.get('urls', [])
                auth_config[service_name] = env_creds
                continue
                
            # Fall back to keyring
            username = keyring.get_password(service_name, 'username')
            password = keyring.get_password(service_name, 'password')

            if username is not None and password is not None:
                logger.info(f"Found credentials for service: {service_name}")
                auth_config[service_name] = {
                    'user': username,
                    'password': password,
                    'urls': service_info.get('urls', [])
                }
            else:
                logger.debug(f"Credentials for service '{service_name}' not found.")

    except keyring.errors.NoKeyringError:
        logger.warning(
            "No keyring backend found. Cannot load credentials securely. "
            "Please install a keyring backend or use environment variables."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while accessing the keyring: {e}", exc_info=True)

    return auth_config

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
    print(f"Description: {KNOWN_SERVICES.get(service_name, {}).get('description', 'Custom service')}")
    
    username = input("Username: ").strip()
    if not username:
        print("Username cannot be empty. Aborting.")
        return False
    
    password = getpass.getpass("Password: ")
    if not password:
        print("Password cannot be empty. Aborting.")
        return False
    
    try:
        keyring.set_password(service_name, 'username', username)
        keyring.set_password(service_name, 'password', password)
        print(f"Credentials for {service_name} saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving credentials: {e}")
        return False