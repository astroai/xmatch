import logging
import multiprocessing
import time  # For timing operations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import yaml
from astropy.table import Table

from . import auth
from .astro_utils import (
    find_coord_columns,
    get_dataframe_extent,
)
from .exceptions import ConfigError, CrossMatchError, InputError
from .local_match import execute_local_stilts  # Import local execution
from .remote_cds import execute_cds_xmatch_local_remote  # Import CDS execution
from .remote_tap import (  # Import TAP execution
    download_from_tap,
    execute_remote_join_match,
    execute_upload_and_tap_match,
)
from .tap import TapUploadUnsupportedError, get_tap_service

logger = logging.getLogger(__name__)


# --- Constants ---
def _find_default_config_path():
    """Find the default configuration file, using multiple strategies."""
    logger.debug("Searching for default configuration file...")

    # Strategy 1: Try importlib.resources (best for installed packages)
    try:
        # Use files API if available (Python 3.9+)
        from importlib.resources import files

        config_path = files("xmatch") / "xmatch.yaml"
        if config_path.is_file():  # Check if it's a file
            logger.debug(f"Found config via importlib.resources (files): {config_path}")
            return config_path
    except (
        ImportError,
        TypeError,
        FileNotFoundError,
    ):  # Catch errors if files API not available or path invalid
        try:
            # Fallback for older Python versions or if files API fails
            import importlib.resources as pkg_resources

            with pkg_resources.path("xmatch", "xmatch.yaml") as p:
                if p.exists():
                    logger.debug(f"Found config via pkg_resources.path: {p}")
                    return p
        except Exception as e_pkg:
            logger.debug(f"importlib.resources approach failed: {e_pkg}")

    # Strategy 2: Look relative to this script for development mode
    try:
        script_dir = Path(__file__).parent.resolve()
        local_config = script_dir / "xmatch.yaml"
        if local_config.exists():
            logger.debug(f"Found config relative to script: {local_config}")
            return local_config
    except Exception as e_script:
        logger.debug(f"Script-relative approach failed: {e_script}")

    # Strategy 3: Check current working directory
    try:
        cwd_config = Path.cwd() / "xmatch.yaml"
        if cwd_config.exists():
            logger.debug(f"Found config in current working directory: {cwd_config}")
            return cwd_config
    except Exception as e_cwd:
        logger.debug(f"Current directory approach failed: {e_cwd}")

    # Strategy 4: Check system config locations (e.g., ~/.config/xmatch/xmatch.yaml)
    try:
        home_dir = Path.home()
        config_locations = [
            home_dir / ".config" / "xmatch" / "xmatch.yaml",
            home_dir / ".xmatch" / "xmatch.yaml",
        ]

        for loc in config_locations:
            if loc.exists():
                logger.debug(f"Found config in user config directory: {loc}")
                return loc
    except Exception as e_user:
        logger.debug(f"User config directory approach failed: {e_user}")

    # No config found - will need to be provided explicitly
    logger.warning("Could not find default configuration file 'xmatch.yaml'")
    return None


DEFAULT_CONFIG_PATH = _find_default_config_path()

SUPPORTED_INPUT_FORMATS = [".parquet", ".fits", ".csv"]


class CrossMatch:
    """Handles cross-matching of astronomical catalogues."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initializes the CrossMatch object with enhanced configuration loading.

        Args:
            config_file: Path to the YAML configuration file.
                        If None, uses the default packaged config found by _find_default_config_path().
            **kwargs: Additional configuration overrides (e.g., java_opts, chunk_size).
        """
        # Determine config file path
        if config_file is None:
            if DEFAULT_CONFIG_PATH is None:
                raise ConfigError(
                    "Default configuration file 'xmatch.yaml' could not be found. "
                    "Ensure the package is installed correctly or provide an explicit --config path."
                )
            self.config_file = DEFAULT_CONFIG_PATH
            logger.info(f"Using default configuration file: {self.config_file}")
        else:
            self.config_file = Path(config_file)
            if not self.config_file.exists():
                raise ConfigError(f"Specified configuration file not found: {self.config_file}")
            logger.info(f"Using specified configuration file: {self.config_file}")

        self.config = self._load_config()  # Loads the entire YAML

        # --- Load structured configuration ---
        self.archives_config = self.config.get("archives", {})
        self.catalogues_config = self.config.get("catalogues", {})
        self.aliases_config = self.config.get("catalogue_aliases", {})
        self.methods_config = self.config.get("crossmatch_methods", {})
        self.stilts_config = self.config.get("stilts_config", {})
        self.global_chunking_config = self.config.get("crossmatch", {}).get("chunking", {})
        # --- End configuration loading ---

        # Apply kwargs overrides to specific settings
        # STILTS settings
        self.stilts_cmd_base = kwargs.get(
            "stilts_cmd_base", self.stilts_config.get("stilts_cmd_base")
        )
        self.stilts_java_opts = kwargs.get("java_opts", self.stilts_config.get("java_opts"))
        self.stilts_tmpdir = kwargs.get("tmpdir", self.stilts_config.get("tmpdir"))
        self.global_floor_error = self.stilts_config.get(
            "default_floor_error_arcsec", 0.01
        )  # Global floor error

        # Chunking/Parallelism settings
        self.chunk_size = kwargs.get(
            "chunk_size", self.global_chunking_config.get("chunk_size", 100000)
        )
        self.n_workers = kwargs.get(
            "n_workers", multiprocessing.cpu_count()
        )  # Default to CPU count

        self.auth_config = auth.load_auth_config()  # Load credentials securely

        self._catalogue_config_cache = {}  # Cache for resolved catalogue configs
        self._local_file_cache = {}  # Cache for loaded local files

        self._validate_config()
        logger.info("CrossMatch initialized.")
        if kwargs:
            logger.info(f"Applied config overrides: {kwargs}")
        if self.stilts_cmd_base:
            logger.info(f"Using STILTS base command: '{self.stilts_cmd_base}'")
        else:
            logger.info("Using STILTS command constructed from Java path and STILTS_JAR.")

    def _load_config(self) -> Dict[str, Any]:
        """Loads the YAML configuration file."""
        try:
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    raise CrossMatchError("Configuration file is not a valid YAML dictionary.")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_file}")
            raise CrossMatchError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file {self.config_file}: {e}")
            raise CrossMatchError(f"Error parsing configuration file {self.config_file}: {e}")
        except IOError as e:
            logger.error(f"IO error loading configuration file {self.config_file}: {e}")
            raise CrossMatchError(f"IO error loading config {self.config_file}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error loading config {self.config_file}: {e}", exc_info=True)
            raise CrossMatchError(f"Unexpected error loading config {self.config_file}: {e}") from e

    def _validate_config(self):
        """Validates the loaded configuration."""
        if not isinstance(self.config, dict):
            raise ConfigError("Configuration must be a dictionary.")

        required_top_level = ["archives", "catalogues"]
        for key in required_top_level:
            if key not in self.config:
                raise ConfigError(f"Missing required top-level key in config: '{key}'")
            if not isinstance(self.config[key], dict):
                raise ConfigError(f"Top-level key '{key}' must be a dictionary.")

        # Validate Archives Structure
        for archive_name, archive_config in self.archives_config.items():
            if not isinstance(archive_config, dict):
                raise ConfigError(f"Archive '{archive_name}' config must be a dictionary.")
            # Check for at least one service definition
            has_service = any(k.endswith("_service") for k in archive_config)
            if not has_service and not archive_config.get(
                "description"
            ):  # Allow description-only entries?
                logger.warning(
                    f"Archive '{archive_name}' has no defined services (e.g., tap_service)."
                )
            # Validate individual services
            for service_id, service_config in archive_config.items():
                if isinstance(service_config, dict) and not service_id.startswith(
                    ("_", "description", "service_priority", "has_", "crossmatch_")
                ):
                    if "access_method" not in service_config:
                        logger.warning(
                            f"Service '{service_id}' in archive '{archive_name}' is missing 'access_method'."
                        )
                    # Add more checks? e.g., access_url for TAP?

        # Validate Catalogues Structure
        for cat_name, cat_config in self.catalogues_config.items():
            if not isinstance(cat_config, dict):
                raise ConfigError(f"Catalogue '{cat_name}' config must be a dictionary.")
            # Required keys for any catalogue
            required_cat_keys = [
                "archive",
                "service_id",
                "access_identifier",
                "ra_column",
                "dec_column",
            ]
            missing_keys = [key for key in required_cat_keys if key not in cat_config]
            if missing_keys:
                raise ConfigError(
                    f"Catalogue '{cat_name}' is missing required keys: {missing_keys}."
                )

            # Check if archive and service_id exist and are valid
            archive_name = cat_config["archive"]
            service_id = cat_config["service_id"]
            if archive_name not in self.archives_config:
                raise ConfigError(
                    f"Archive '{archive_name}' (for catalogue '{cat_name}') not found in 'archives'."
                )
            if service_id not in self.archives_config[archive_name]:
                raise ConfigError(
                    f"Service '{service_id}' (for catalogue '{cat_name}') not found in archive '{archive_name}'."
                )
            if not isinstance(self.archives_config[archive_name][service_id], dict):
                raise ConfigError(
                    f"Service '{service_id}' in archive '{archive_name}' must be a dictionary."
                )
            # Check if coordinate columns are strings
            if not isinstance(cat_config["ra_column"], str) or not isinstance(
                cat_config["dec_column"], str
            ):
                raise ConfigError(
                    f"RA/Dec column names for catalogue '{cat_name}' must be strings."
                )
            # Check default columns if present
            if "default_columns" in cat_config and not isinstance(
                cat_config["default_columns"], list
            ):
                raise ConfigError(f"'default_columns' for catalogue '{cat_name}' must be a list.")

        # Validate Aliases
        for alias, target_cat in self.aliases_config.items():
            if not isinstance(alias, str) or not isinstance(target_cat, str):
                raise ConfigError(
                    f"Catalogue alias '{alias}' and target '{target_cat}' must be strings."
                )
            if target_cat not in self.catalogues_config:
                raise ConfigError(
                    f"Catalogue alias '{alias}' points to non-existent catalogue '{target_cat}'."
                )

        logger.debug("Configuration validation successful.")

    def get_catalogue_config(self, catalogue_name: str) -> Dict[str, Any]:
        """Retrieves the fully resolved configuration for a specific catalogue.

        Merges the catalogue-specific settings with the settings from its
        designated archive service.

        Args:
            catalogue_name: The name of the catalogue (lowercase).

        Returns:
            A dictionary containing the merged configuration.

        Raises:
            ConfigError: If the catalogue, its archive, or its service is not found
                       or improperly configured.
        """
        catalogue_name = catalogue_name.lower()
        logger.debug(f"Resolving configuration for catalogue: {catalogue_name}")

        if "catalogues" not in self.config or catalogue_name not in self.config["catalogues"]:
            logger.error(f"Catalogue '{catalogue_name}' not found in configuration.")
            raise CrossMatchError(f"Catalogue '{catalogue_name}' not found in configuration.")

        cat_config = self.config["catalogues"][catalogue_name]

        archive_name = cat_config.get("archive")
        service_id = cat_config.get("service_id")

        if not archive_name or not service_id:
            raise CrossMatchError(
                f"Catalogue '{catalogue_name}' is missing 'archive' or 'service_id'."
            )

        if "archives" not in self.config or archive_name not in self.config["archives"]:
            logger.error(f"Archive '{archive_name}' (for catalogue '{catalogue_name}') not found.")
            raise CrossMatchError(
                f"Archive '{archive_name}' (for catalogue '{catalogue_name}') not found."
            )

        archive_config = self.config["archives"][archive_name]

        if service_id not in archive_config or not isinstance(archive_config[service_id], dict):
            logger.error(
                f"Service '{service_id}' not found or invalid in archive '{archive_name}' (for catalogue '{catalogue_name}')."
            )
            raise CrossMatchError(
                f"Service '{service_id}' not found or invalid in archive '{archive_name}' (for catalogue '{catalogue_name}')."
            )

        service_config = archive_config[service_id]

        # Merge configurations: Start with service config, override with catalogue config
        # This ensures catalogue specifics take precedence over service defaults.
        resolved_config = service_config.copy()
        resolved_config.update(cat_config)  # Catalogue settings override service settings

        # Inject names for reference
        resolved_config["_catalogue_name"] = catalogue_name
        resolved_config["_archive_name"] = archive_name
        # Keep service_id for potential use
        resolved_config["_service_id"] = service_id

        # Add archive-level description if catalogue lacks one?
        if "description" not in resolved_config and "description" in archive_config:
            resolved_config["description"] = (
                archive_config["description"] + f" ({resolved_config.get('access_identifier', '')})"
            )

        logger.debug(f"Resolved config for {catalogue_name}: {resolved_config}")
        return resolved_config

    def _resolve_input_config(
        self, catalogue_input: Union[str, pd.DataFrame], params: Dict[str, Any], prefix: str
    ) -> Dict[str, Any]:
        """
        Resolves the configuration for a given input (local file/DataFrame or remote catalogue name).

        Args:
            catalogue_input: The input identifier (path, name, or DataFrame).
            params: Dictionary of crossmatch parameters containing potential overrides.
            prefix: Identifier prefix ('1' or '2') for parameter keys.

        Returns:
            A dictionary containing the resolved configuration for the input.

        Raises:
            InputError: If the input cannot be resolved or is invalid.
            ConfigError: If a remote catalogue configuration is invalid.
        """
        logger.debug(f"Resolving config for input (prefix {prefix}): {type(catalogue_input)}")
        config = {}

        # Parameter keys for overrides
        ra_col_key = f"ra_column_{prefix}"
        dec_col_key = f"dec_column_{prefix}"
        epoch_col_key = f"epoch_column_{prefix}"
        pm_ra_col_key = f"pm_ra_column_{prefix}"
        pm_dec_col_key = f"pm_dec_column_{prefix}"
        floor_err_key = f"floor_error_arcsec_{prefix}"

        if isinstance(catalogue_input, pd.DataFrame):
            logger.info(f"Input {prefix} is a DataFrame.")
            config["is_local"] = True
            config["_input_dataframe"] = catalogue_input
            config["_input_type"] = "dataframe"
            # Try to auto-detect coordinate columns if not provided
            ra_col, dec_col = find_coord_columns(catalogue_input)
            config["ra_column"] = params.get(ra_col_key, ra_col)
            config["dec_column"] = params.get(dec_col_key, dec_col)
            # Add other relevant local config defaults or inferred values
            config["access_identifier"] = f"local_dataframe_{prefix}"
            config["_catalogue_name"] = f"local_dataframe_{prefix}"  # Internal name

        elif isinstance(catalogue_input, (str, Path)):
            input_path_str = str(catalogue_input)
            input_path = Path(input_path_str)
            logger.debug(f"Input {prefix} is a string/path: '{input_path_str}'")

            # Check if it's a known catalogue name/alias first
            resolved_name = self.aliases_config.get(input_path_str.lower(), input_path_str.lower())
            if resolved_name in self.catalogues_config:
                logger.info(
                    f"Input {prefix} ('{input_path_str}') resolved as remote catalogue: {resolved_name}"
                )
                config = self.get_catalogue_config(resolved_name)  # Fetch remote config
                config["is_local"] = False
                config["_input_type"] = "remote_catalogue"
            # Check if it's a file path
            elif input_path.exists() and input_path.is_file():
                logger.info(f"Input {prefix} ('{input_path_str}') resolved as local file.")
                config["is_local"] = True
                config["_input_path"] = input_path
                config["_input_type"] = "local_file"
                # Determine format
                suffix = input_path.suffix.lower()
                if suffix not in SUPPORTED_INPUT_FORMATS:
                    raise InputError(
                        f"Unsupported file format for input {prefix}: '{suffix}'. Supported: {SUPPORTED_INPUT_FORMATS}"
                    )
                config["format"] = suffix.lstrip(".")  # e.g., 'csv', 'parquet', 'fits'

                # Load a small sample or header to detect columns? Or require params?
                # For now, require RA/Dec params for local files unless we add auto-detection
                if not params.get(ra_col_key) or not params.get(dec_col_key):
                    # Try loading and detecting if not provided
                    try:
                        temp_df = self._load_local_catalogue(
                            input_path, nrows=5
                        )  # Load small sample
                        ra_col, dec_col = find_coord_columns(temp_df)
                        config["ra_column"] = params.get(ra_col_key, ra_col)
                        config["dec_column"] = params.get(dec_col_key, dec_col)
                        logger.info(
                            f"Auto-detected columns for local file {prefix}: RA='{config['ra_column']}', Dec='{config['dec_column']}'"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not auto-detect columns for local file {prefix}: {e}. Please provide --ra_column_{prefix} and --dec_column_{prefix}."
                        )
                        # Raise error if still missing after attempt
                        if not params.get(ra_col_key) or not params.get(dec_col_key):
                            raise InputError(
                                f"RA/Dec columns must be provided for local file input {prefix} (e.g., --{ra_col_key}, --{dec_col_key})"
                            )

                config["ra_column"] = params.get(
                    ra_col_key, config.get("ra_column")
                )  # Apply override if exists
                config["dec_column"] = params.get(
                    dec_col_key, config.get("dec_column")
                )  # Apply override if exists
                config["access_identifier"] = str(input_path.resolve())
                config["_catalogue_name"] = input_path.stem  # Use filename stem as name

            else:
                raise InputError(
                    f"Input '{input_path_str}' for catalogue {prefix} is not a valid file path, DataFrame, or known catalogue name."
                )
        else:
            raise InputError(
                f"Unsupported input type for catalogue {prefix}: {type(catalogue_input)}. Must be str, Path, or DataFrame."
            )

        # Apply common parameter overrides AFTER initial config setup
        config["ra_column"] = params.get(ra_col_key, config.get("ra_column"))
        config["dec_column"] = params.get(dec_col_key, config.get("dec_column"))
        config["epoch_column"] = params.get(epoch_col_key, config.get("epoch_column"))  # Optional
        config["pm_ra_column"] = params.get(pm_ra_col_key, config.get("pm_ra_column"))  # Optional
        config["pm_dec_column"] = params.get(
            pm_dec_col_key, config.get("pm_dec_column")
        )  # Optional
        config["floor_error_arcsec"] = params.get(
            floor_err_key, config.get("floor_error_arcsec", self.global_floor_error)
        )  # Use specific, then catalogue default, then global default

        # Validate essential columns are present in the final config
        if not config.get("ra_column") or not config.get("dec_column"):
            raise ConfigError(
                f"Could not determine RA/Dec columns for input {prefix}. Provide --{ra_col_key} and --{dec_col_key}."
            )

        logger.debug(f"Resolved config for input {prefix}: {config}")
        return config

    def _load_local_catalogue(self, file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        """Loads a local catalogue file into a pandas DataFrame."""
        file_path_str = str(file_path)
        logger.info(
            f"Loading local catalogue: {file_path_str}"
            + (f" (reading first {nrows} rows)" if nrows else "")
        )

        # Check cache first
        cache_key = (file_path_str, nrows)
        if cache_key in self._local_file_cache:
            logger.debug(f"Returning cached DataFrame for {file_path_str} (nrows={nrows})")
            return self._local_file_cache[cache_key]

        suffix = file_path.suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(file_path, nrows=nrows)
            elif suffix == ".parquet":
                # pandas read_parquet doesn't directly support nrows, read full then slice
                if nrows:
                    # This might be inefficient for large files if only header needed
                    # Consider pyarrow for more efficient partial reads if needed
                    df_full = pd.read_parquet(file_path)
                    df = df_full.head(nrows)
                else:
                    df = pd.read_parquet(file_path)
            elif suffix == ".fits":
                # Astropy Table read, then convert
                # FITS can have multiple HDUs, assume first table HDU
                try:
                    table = Table.read(file_path, hdu=1)  # Try HDU 1 first (common for tables)
                except Exception:
                    try:
                        logger.debug("Failed to read HDU 1, trying HDU 0...")
                        table = Table.read(file_path, hdu=0)  # Try HDU 0 as fallback
                    except Exception as fits_err:
                        raise InputError(
                            f"Could not find a readable table HDU in FITS file {file_path}: {fits_err}"
                        )

                if nrows:
                    df = table[:nrows].to_pandas()
                else:
                    df = table.to_pandas()
            else:
                # Should have been caught by _resolve_input_config, but double-check
                raise InputError(f"Unsupported file format: {suffix}")

            logger.info(f"Successfully loaded {len(df)} rows from {file_path_str}")
            # Cache the result only if the full file was read (nrows is None)
            if nrows is None:
                self._local_file_cache[cache_key] = df
            return df
        except FileNotFoundError:
            logger.error(f"Local file not found: {file_path_str}")
            raise InputError(f"Local file not found: {file_path_str}")
        except Exception as e:
            logger.error(f"Failed to load local file {file_path_str}: {e}", exc_info=True)
            raise InputError(f"Failed to load local file {file_path_str}: {e}")

    def _determine_crossmatch_strategy(
        self, config1: Dict[str, Any], config2: Dict[str, Any], **params
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Determines the optimal crossmatch strategy based on input types and configurations.

        Args:
            config1: Resolved configuration for the first catalogue.
            config2: Resolved configuration for the second catalogue.
            **params: Additional crossmatch parameters.

        Returns:
            A tuple containing:
                - The name of the selected strategy (str).
                - A dictionary of parameters potentially modified or added by the strategy logic (dict).
        """
        logger.info("Determining crossmatch strategy...")
        strategy_params = params.copy()  # Start with incoming params
        strategy_name = "unknown"  # Default

        is_local1 = config1.get("is_local", False)
        is_local2 = config2.get("is_local", False)
        access_method1 = config1.get("access_method")
        access_method2 = config2.get("access_method")
        archive1 = config1.get("_archive_name")
        archive2 = config2.get("_archive_name")
        service1 = config1.get("_service_id")
        service2 = config2.get("_service_id")

        logger.debug(
            f"Input 1: local={is_local1}, access={access_method1}, archive={archive1}, service={service1}"
        )
        logger.debug(
            f"Input 2: local={is_local2}, access={access_method2}, archive={archive2}, service={service2}"
        )

        # --- Strategy Logic ---

        # 1. Both Local: Always use local STILTS
        if is_local1 and is_local2:
            strategy_name = "local_stilts"
            logger.info("Strategy: Both inputs are local -> local_stilts")

        # 2. One Local, One Remote
        elif is_local1 != is_local2:  # XOR condition
            local_config = config1 if is_local1 else config2
            remote_config = config2 if is_local1 else config1
            local_prefix = "1" if is_local1 else "2"
            remote_prefix = "2" if is_local1 else "1"
            remote_access = remote_config.get("access_method")
            remote_archive = remote_config.get("_archive_name")
            remote_config.get("_service_id")
            remote_cat_name = remote_config.get("_catalogue_name")

            logger.info(
                f"Strategy: One local ({local_prefix}), one remote ({remote_prefix}, {remote_cat_name}, method={remote_access})"
            )

            # 2a. Remote is CDS XMatch Service
            if remote_access == "cds_xmatch":
                # Check if CDS supports remote table name directly
                if remote_config.get("access_identifier"):  # e.g., "vizier:I/355/gaiadr3"
                    strategy_name = "cds_xmatch_local_remote"
                    logger.info("Strategy: Local vs CDS XMatch -> cds_xmatch_local_remote")
                else:
                    logger.warning(
                        "CDS XMatch selected, but remote catalogue 'access_identifier' missing. Falling back."
                    )
                    # Fallback: Download remote via TAP (if possible) and match locally
                    if remote_config.get("tap_url"):  # Check if TAP info is available as fallback
                        remote_config["access_method"] = "tap"  # Temporarily override for download
                        logger.warning("Falling back to download_and_match (using TAP).")
                        strategy_name = "download_and_match"
                        strategy_params["_catalogue_to_download"] = remote_prefix
                        # Calculate extent of local file for download region
                        extent = self._get_local_file_extent(local_config, params, local_prefix)
                        if extent:
                            strategy_params.update(extent)  # Add ra, dec, radius_deg
                        else:
                            logger.warning(
                                "Could not determine local file extent for download. Download may be very large or fail."
                            )
                    else:
                        raise ConfigError(
                            f"Cannot execute CDS XMatch for {remote_cat_name} (missing identifier) and no TAP fallback available."
                        )

            # 2b. Remote is TAP Service
            elif remote_access == "tap":
                tap_service_url = remote_config.get("tap_url")
                # Need auth session for the *remote* archive
                remote_auth = self.auth_config.get_auth_session(remote_archive)
                tap_service = get_tap_service(tap_service_url, auth_session=remote_auth)

                # Check TAP capabilities (UPLOAD capability)
                can_upload = False
                try:
                    # Check for UPLOAD table - this is the standard way
                    upload_tables = [t for t in tap_service.tables if t.type == "UPLOAD"]
                    if upload_tables:
                        can_upload = True
                        logger.info(f"Remote TAP service ({remote_cat_name}) supports UPLOAD.")
                    else:
                        # Some services might advertise upload capability differently
                        # Check capabilities endpoint (less reliable parsing needed)
                        # For now, rely on UPLOAD table presence
                        logger.info(
                            f"Remote TAP service ({remote_cat_name}) does not explicitly list UPLOAD tables."
                        )
                        # Heuristic: Check if it's a known service that supports uploads (e.g., CADC, GAIA)
                        known_upload_services = ["gaia_archive", "cadc"]  # Example
                        if remote_archive in known_upload_services:
                            logger.warning(
                                f"Assuming TAP service {remote_archive} supports uploads based on known services list."
                            )
                            can_upload = True  # Tentatively assume yes

                except Exception as e:
                    logger.warning(
                        f"Could not reliably determine TAP upload capability for {remote_cat_name}: {e}. Assuming no upload support."
                    )
                    can_upload = False

                if can_upload:
                    strategy_name = "upload_and_tap_match"
                    logger.info("Strategy: Local vs TAP (Upload supported) -> upload_and_tap_match")
                else:
                    # Fallback: Download remote and match locally
                    strategy_name = "download_and_match"
                    strategy_params["_catalogue_to_download"] = remote_prefix
                    logger.info("Strategy: Local vs TAP (No Upload) -> download_and_match")
                    # Calculate extent of local file for download region
                    extent = self._get_local_file_extent(local_config, params, local_prefix)
                    if extent:
                        strategy_params.update(extent)  # Add ra, dec, radius_deg
                    else:
                        logger.warning(
                            "Could not determine local file extent for download. Download may be very large or fail."
                        )

            # 2c. Other Remote Access Methods (Add more as needed)
            else:
                logger.warning(
                    f"Remote access method '{remote_access}' for {remote_cat_name} not directly supported for local/remote match. Falling back."
                )
                # Fallback: Try download and match if TAP info exists
                if remote_config.get("tap_url"):
                    remote_config["access_method"] = "tap"  # Temporarily override for download
                    logger.warning("Falling back to download_and_match (using TAP).")
                    strategy_name = "download_and_match"
                    strategy_params["_catalogue_to_download"] = remote_prefix
                    extent = self._get_local_file_extent(local_config, params, local_prefix)
                    if extent:
                        strategy_params.update(extent)
                    else:
                        logger.warning("Could not determine local file extent for download.")
                else:
                    raise CrossMatchError(
                        f"Unsupported remote access method '{remote_access}' for catalogue {remote_cat_name} and no TAP fallback."
                    )

        # 3. Both Remote
        elif not is_local1 and not is_local2:
            logger.info("Strategy: Both inputs are remote.")
            # 3a. Both are CDS XMatch Service (unlikely to be efficient?)
            if access_method1 == "cds_xmatch" and access_method2 == "cds_xmatch":
                # CDS XMatch service typically matches an uploaded table against ONE remote catalogue.
                # Matching two remote CDS catalogues directly via the service isn't standard.
                logger.warning("Strategy: Both remote CDS XMatch. This is unusual. Falling back.")
                # Fallback: Download one (or both?) and match locally? Or use TAP?
                # Simplest fallback: Download both via TAP (if possible) and match locally.
                if config1.get("tap_url") and config2.get("tap_url"):
                    config1["access_method"] = "tap"
                    config2["access_method"] = "tap"
                    logger.warning("Falling back to download_and_match (using TAP for both).")
                    strategy_name = "download_and_match"
                    strategy_params["_catalogue_to_download"] = "both"
                    # Need a region for download - use user params or default?
                    if not ("ra" in params and "dec" in params and "radius_deg" in params):
                        logger.warning(
                            "No region specified for remote-remote download. Downloads might be very large or fail."
                        )
                        # Could potentially try to get full sky if service allows? Risky.
                else:
                    raise CrossMatchError(
                        "Cannot match two remote CDS catalogues directly, and no TAP fallback available for download."
                    )

            # 3b. Both are TAP Services
            elif access_method1 == "tap" and access_method2 == "tap":
                # Check if they are on the SAME TAP service
                if config1.get("tap_url") == config2.get("tap_url"):
                    strategy_name = "remote_join"
                    logger.info("Strategy: Both remote TAP, same service -> remote_join")
                else:
                    # Different TAP services - need to download at least one
                    logger.info("Strategy: Both remote TAP, different services.")
                    # Heuristic: Download the smaller catalogue? Or the one less common?
                    # Simple approach: Download catalogue 2, match locally with catalogue 1 (downloaded on demand)
                    # This becomes download_and_match, downloading #2 first.
                    strategy_name = "download_and_match"
                    strategy_params["_catalogue_to_download"] = (
                        "both"  # Need to download both eventually
                    )
                    logger.info("Falling back to download_and_match (downloading both).")
                    if not ("ra" in params and "dec" in params and "radius_deg" in params):
                        logger.warning(
                            "No region specified for remote-remote download. Downloads might be very large or fail."
                        )

            # 3c. Mixed Remote (TAP vs CDS)
            elif access_method1 == "tap" and access_method2 == "cds_xmatch":
                logger.warning("Strategy: Remote TAP vs Remote CDS. Falling back.")
                # Fallback: Download both (via TAP if possible) and match locally
                if config1.get("tap_url") and config2.get("tap_url"):
                    config1["access_method"] = "tap"
                    config2["access_method"] = "tap"
                    logger.warning("Falling back to download_and_match (using TAP for both).")
                    strategy_name = "download_and_match"
                    strategy_params["_catalogue_to_download"] = "both"
                    if not ("ra" in params and "dec" in params and "radius_deg" in params):
                        logger.warning("No region specified for remote-remote download.")
                else:
                    raise CrossMatchError(
                        "Cannot match remote TAP vs CDS, and no TAP fallback available for download."
                    )

            elif access_method1 == "cds_xmatch" and access_method2 == "tap":
                # Symmetric to above
                logger.warning("Strategy: Remote CDS vs Remote TAP. Falling back.")
                if config1.get("tap_url") and config2.get("tap_url"):
                    config1["access_method"] = "tap"
                    config2["access_method"] = "tap"
                    logger.warning("Falling back to download_and_match (using TAP for both).")
                    strategy_name = "download_and_match"
                    strategy_params["_catalogue_to_download"] = "both"
                    if not ("ra" in params and "dec" in params and "radius_deg" in params):
                        logger.warning("No region specified for remote-remote download.")
                else:
                    raise CrossMatchError(
                        "Cannot match remote CDS vs TAP, and no TAP fallback available for download."
                    )

            # 3d. Other Remote Combinations
            else:
                raise CrossMatchError(
                    f"Unsupported combination of remote access methods: {access_method1} and {access_method2}"
                )

        # --- Final Check ---
        if strategy_name == "unknown":
            # Default fallback if no specific strategy matched (should ideally not happen)
            logger.warning(
                "Could not determine a specific strategy. Defaulting to 'download_and_match'."
            )
            strategy_name = "download_and_match"
            # Determine which needs download based on original types
            if is_local1 and not is_local2:
                strategy_params["_catalogue_to_download"] = "2"
            elif not is_local1 and is_local2:
                strategy_params["_catalogue_to_download"] = "1"
            else:
                strategy_params["_catalogue_to_download"] = (
                    "both"  # Both remote or both local (though local/local handled above)
                )
            # Add extent calculation if one is local
            if is_local1 != is_local2:
                local_c = config1 if is_local1 else config2
                local_pfx = "1" if is_local1 else "2"
                extent = self._get_local_file_extent(local_c, params, local_pfx)
                if extent:
                    strategy_params.update(extent)

        logger.info(f"Selected strategy: {strategy_name}")
        logger.debug(f"Final strategy parameters: {strategy_params}")
        return strategy_name, strategy_params

    def crossmatch(
        self,
        catalogue_1_input: Union[str, pd.DataFrame],
        catalogue_2_input: Union[str, pd.DataFrame],
        output_file: Optional[Union[str, Path]] = None,
        **params,
    ) -> Optional[pd.DataFrame]:
        """
        Performs the crossmatch operation between two catalogues.

        Args:
            catalogue_1_input: Path/name of the first catalogue or DataFrame.
            catalogue_2_input: Path/name of the second catalogue or DataFrame.
            output_file: Path to save the output results. If None, returns DataFrame.
            **params: Additional crossmatch parameters (radius, columns, join_type, etc.).

        Returns:
            Optional[pd.DataFrame]: The crossmatch result as a DataFrame if output_file is None.
                                    Returns None if output_file is specified.

        Raises:
            CrossMatchError: If the crossmatch fails for any reason.
        """
        start_time = time.time()
        logger.info("Starting crossmatch process...")

        # --- 1. Resolve Configurations ---
        try:
            config1 = self._resolve_input_config(catalogue_1_input, params, prefix="1")
            config2 = self._resolve_input_config(catalogue_2_input, params, prefix="2")
            logger.debug(f"Config 1 resolved: {config1}")
            logger.debug(f"Config 2 resolved: {config2}")
        except Exception as e:
            logger.error(f"Failed to resolve input configurations: {e}", exc_info=True)
            raise CrossMatchError(f"Configuration error: {e}") from e

        # --- 2. Determine Strategy ---
        try:
            strategy_params = params.copy()  # Pass relevant params down
            strategy_name, determined_strategy_params = self._determine_crossmatch_strategy(
                config1, config2, **strategy_params
            )
            strategy_params.update(
                determined_strategy_params
            )  # Add params determined by strategy logic
            logger.info(f"Selected strategy: {strategy_name}")
        except Exception as e:
            logger.error(f"Failed to determine crossmatch strategy: {e}", exc_info=True)
            raise CrossMatchError(f"Strategy determination failed: {e}") from e

        # --- 3. Execute Strategy ---
        result_df = None
        try:
            # Inject resolved configs into params for execution functions
            strategy_params["_config1"] = config1
            strategy_params["_config2"] = config2

            if strategy_name == "local_stilts":
                result_df = execute_local_stilts(config1, config2, self, **strategy_params)
            elif strategy_name == "cds_xmatch_local_remote":
                result_df = execute_cds_xmatch_local_remote(
                    config1, config2, self, **strategy_params
                )
            elif strategy_name == "upload_and_tap_match":
                try:
                    result_df = execute_upload_and_tap_match(
                        config1, config2, self, **strategy_params
                    )
                except TapUploadUnsupportedError as upload_err:
                    logger.warning(
                        f"TAP upload failed or not supported: {upload_err}. Falling back to 'download_and_match' strategy."
                    )
                    # Fallback strategy: Download remote, match locally
                    strategy_name = (
                        "download_and_match"  # Update strategy name for logging/consistency
                    )
                    # Determine which catalogue needs downloading based on original config
                    strategy_params["_catalogue_to_download"] = (
                        "2" if config1.get("is_local") else "1"
                    )
                    result_df = self._execute_download_and_match(
                        config1, config2, self, **strategy_params
                    )
                except Exception as tap_err:
                    # Catch other errors during TAP upload/match
                    logger.error(f"Error during 'upload_and_tap_match': {tap_err}", exc_info=True)
                    raise CrossMatchError(f"TAP match failed: {tap_err}") from tap_err
            elif strategy_name == "remote_join":
                result_df = execute_remote_join_match(config1, config2, self, **strategy_params)
            elif strategy_name == "download_and_match":
                result_df = self._execute_download_and_match(
                    config1, config2, self, **strategy_params
                )
            else:
                raise CrossMatchError(f"Unknown or unsupported strategy: {strategy_name}")

            if result_df is None:
                logger.warning(f"Strategy '{strategy_name}' did not return a DataFrame.")
                result_df = pd.DataFrame()  # Ensure result_df is a DataFrame

        except Exception as e:
            logger.error(
                f"Crossmatch execution failed using strategy '{strategy_name}': {e}", exc_info=True
            )
            raise CrossMatchError(f"Execution failed: {e}") from e

        # --- 4. Handle Output ---
        if output_file:
            logger.info(f"Saving {len(result_df)} results to {output_file}")
            self._save_output(result_df, output_file)
            end_time = time.time()
            logger.info(f"Crossmatch completed in {end_time - start_time:.2f} seconds.")
            return None  # Indicate success but no DataFrame returned
        else:
            logger.info(f"Crossmatch finished, returning {len(result_df)} results as DataFrame.")
            end_time = time.time()
            logger.info(f"Crossmatch completed in {end_time - start_time:.2f} seconds.")
            return result_df

    def _get_local_file_extent(
        self, local_config: Dict[str, Any], params: Dict[str, Any], prefix: str
    ) -> Optional[Dict[str, float]]:
        """Calculates the approximate sky coverage of a local file."""
        logger.info(f"Calculating extent for local file (prefix {prefix})...")
        df = None
        if "_input_dataframe" in local_config:
            df = local_config["_input_dataframe"]
        elif "_input_path" in local_config:
            # Avoid reloading if already loaded during config resolution
            # This might require caching the loaded DataFrame in _resolve_input_config
            # For now, reload - less efficient but safer.
            try:
                df = self._load_local_catalogue(local_config["_input_path"])
            except Exception as e:
                logger.error(
                    f"Failed to load local file {local_config['_input_path']} to get extent: {e}"
                )
                return None  # Cannot determine extent
        else:
            logger.warning("Cannot determine extent: No DataFrame or path found in local config.")
            return None

        if df is None or df.empty:
            logger.warning("Cannot determine extent: Local DataFrame is empty or failed to load.")
            return None

        # Get RA/Dec columns (already resolved in local_config)
        ra_col = local_config.get("ra_column")
        dec_col = local_config.get("dec_column")

        if not ra_col or not dec_col:
            logger.error("Cannot determine extent: RA/Dec columns not resolved for local file.")
            # Attempt auto-detection again? Or rely on initial resolution.
            # For now, fail if not present in config.
            return None

        if ra_col not in df.columns or dec_col not in df.columns:
            logger.error(
                f"Cannot determine extent: Resolved RA ('{ra_col}') or Dec ('{dec_col}') columns not found in DataFrame."
            )
            return None

        try:
            # Use the utility function
            extent_result = get_dataframe_extent(df, ra_col, dec_col)
            # Check if the function returned a valid result before unpacking
            if extent_result is None:
                logger.error(
                    "Failed to calculate extent for local file (get_dataframe_extent returned None)."
                )
                return None

            # Correctly unpack the dictionary using keys
            center_ra = extent_result["ra_center_deg"]
            center_dec = extent_result["dec_center_deg"]
            radius_deg = extent_result["radius_deg"]

            logger.info(
                f"Local file extent: RA={center_ra:.4f}, Dec={center_dec:.4f}, Radius={radius_deg:.4f} deg"
            )
            # Add a buffer based on the crossmatch radius
            match_radius_arcsec = params.get("radius_arcsec", 1.0)
            buffer_deg = match_radius_arcsec / 3600.0  # Convert match radius to degrees
            # Ensure buffer isn't excessively large if radius_deg is tiny
            # Use max(radius_deg, some_min_radius) + buffer? Or just add buffer? Adding seems fine.
            total_radius_deg = radius_deg + buffer_deg
            logger.info(
                f"Using download radius: {total_radius_deg:.4f} deg (extent radius {radius_deg:.4f} deg + match radius buffer {buffer_deg:.4f} deg)"
            )

            # Return the center and the *total* radius needed for download
            return {"ra": center_ra, "dec": center_dec, "radius_deg": total_radius_deg}

        except Exception as e:
            logger.error(f"Error calculating extent for local file: {e}", exc_info=True)
            return None

    def _execute_download_and_match(
        self, config1: Dict[str, Any], config2: Dict[str, Any], crossmatch_instance: Any, **params
    ) -> pd.DataFrame:
        """Executes the download-and-match strategy."""
        logger.info("Executing download and match strategy...")

        cat_to_download = params.get(
            "_catalogue_to_download", "both"
        )  # Default to both if not specified
        df1, df2 = None, None
        # Extract region params determined by strategy selection (if any)
        region_params = {k: v for k, v in params.items() if k in ["ra", "dec", "radius_deg"]}

        # --- Load/Download Catalogue 1 ---
        if config1.get("is_local"):
            if "_input_dataframe" in config1:
                df1 = config1["_input_dataframe"]
            elif "_input_path" in config1:
                df1 = self._load_local_catalogue(config1["_input_path"])
        elif cat_to_download in ["1", "both"]:
            logger.info(f"Downloading remote catalogue 1: {config1.get('_catalogue_name')}")
            # Pass necessary params for download (columns, auth)
            download_params = self._prepare_download_params(config1, params, prefix="1")
            # Combine specific download params with general region params
            all_download_params = {**download_params, **region_params}
            df1 = self._download_catalogue(config1, **all_download_params)
        else:  # Remote but not downloading (shouldn't happen with this strategy?)
            raise CrossMatchError(
                "Invalid state in download_and_match: Remote cat 1 not marked for download."
            )

        # --- Load/Download Catalogue 2 ---
        if config2.get("is_local"):
            if "_input_dataframe" in config2:
                df2 = config2["_input_dataframe"]
            elif "_input_path" in config2:
                df2 = self._load_local_catalogue(config2["_input_path"])
        elif cat_to_download in ["2", "both"]:
            logger.info(f"Downloading remote catalogue 2: {config2.get('_catalogue_name')}")
            download_params = self._prepare_download_params(config2, params, prefix="2")
            # Combine specific download params with general region params
            all_download_params = {**download_params, **region_params}
            df2 = self._download_catalogue(config2, **all_download_params)
        else:  # Remote but not downloading
            raise CrossMatchError(
                "Invalid state in download_and_match: Remote cat 2 not marked for download."
            )

        # Check if dataframes were loaded/downloaded
        if df1 is None or df2 is None:
            raise CrossMatchError("Failed to load or download data for one or both catalogues.")
        if df1.empty or df2.empty:
            logger.warning(
                "One or both catalogues are empty after loading/downloading. No matches possible."
            )
            return pd.DataFrame()

        # --- Perform Local Match ---
        logger.info("Performing local match on downloaded/loaded data...")
        # Update configs to mark them as effectively local for the execution step
        config1_local = config1.copy()
        config2_local = config2.copy()
        config1_local["is_local"] = True
        config2_local["is_local"] = True
        config1_local["_input_dataframe"] = df1  # Pass loaded dataframes
        config2_local["_input_dataframe"] = df2

        # Use the existing local execution function
        return execute_local_stilts(config1_local, config2_local, self, **params)

    def _prepare_download_params(
        self, config: Dict[str, Any], params: Dict[str, Any], prefix: str
    ) -> Dict[str, Any]:
        """Prepares parameters specifically for downloading a remote catalogue."""
        download_params = {}
        # Columns to download
        cols_key = f"columns_{prefix}"
        download_params["columns_to_download"] = params.get(cols_key) or config.get(
            "default_columns"
        )

        # Spatial region (RA, Dec, Radius) - these are handled separately now
        # by extracting from the main 'params' dict in the calling function.

        # Authentication
        archive_name = config.get("_archive_name")
        if archive_name:
            # Corrected line: Use dict.get() instead of method call
            download_params["auth_session"] = self.auth_config.get_auth_session(archive_name)

        # Add other relevant params? e.g., row limits?
        # download_params['maxrec'] = params.get('download_maxrec')

        return download_params

    def _download_catalogue(self, config: Dict[str, Any], **params) -> pd.DataFrame:
        """Downloads data for a single remote catalogue based on its config."""
        access_method = config.get("access_method")
        cat_name = config.get("_catalogue_name", "remote")
        logger.debug(
            f"Downloading {cat_name} via {access_method} with params: {params}"
        )  # Log params

        try:
            if access_method == "tap":
                logger.debug(
                    f"Config passed to download_from_tap for {cat_name}: {config}"
                )  # Add this line
                # Ensure download_from_tap accepts ra, dec, radius_deg etc. from **params
                return download_from_tap(config, **params)
            elif (
                access_method == "cds_xmatch"
            ):  # Can we download via CDS service? Assume Vizier TAP for now.
                # Check if remote_cds.download_from_cds exists and handles params
                try:
                    from .remote_cds import download_from_cds

                    # Need to map params if download_from_cds expects different names
                    return download_from_cds(config, **params)
                except ImportError:
                    logger.warning(
                        "remote_cds.download_from_cds not found. Cannot download via cds_xmatch method."
                    )
                    # Fallback? Or error? For now, error.
                    raise CrossMatchError("Download via 'cds_xmatch' method not implemented/found.")

            # Add other access methods (e.g., http download) if needed
            else:
                raise CrossMatchError(
                    f"Unsupported access method '{access_method}' for downloading '{cat_name}'"
                )
        except Exception as e:
            logger.error(f"Failed to download catalogue '{cat_name}': {e}", exc_info=True)
            raise CrossMatchError(f"Download failed for '{cat_name}': {e}") from e

    def _save_output(self, df: pd.DataFrame, output_file: Union[str, Path]):
        """Saves the output DataFrame to the specified file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
            if output_path.suffix == ".csv":
                df.to_csv(output_path, index=False)
            elif output_path.suffix == ".parquet":
                df.to_parquet(output_path, index=False)
            elif output_path.suffix == ".fits":
                # Convert object columns to string before saving to FITS
                object_cols = list(df.select_dtypes(include=["object", "string"]).columns)
                if object_cols:
                    for col in object_cols:
                        # Check if column contains non-numeric types that FITS might struggle with
                        if not pd.api.types.is_numeric_dtype(df[col].dropna()):
                            logger.debug(
                                f"Converting object column '{col}' to string for FITS output."
                            )
                            df[col] = df[col].astype(str)
                table = Table.from_pandas(df)
                table.write(output_path, overwrite=True)
            else:
                raise CrossMatchError(f"Unsupported output format: {output_path.suffix}")
            logger.info(f"Output saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output to {output_file}: {e}", exc_info=True)
            raise CrossMatchError(f"Failed to save output: {e}") from e
