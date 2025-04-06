import logging
import multiprocessing
import time  # For timing operations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import tempfile

import astropy.table  # Import the module itself for explicit referencing
import numpy as np
import pandas as pd
import yaml
from astropy.table import Table
from pyvo.dal import TAPService, DALQueryError, DALServiceError # Correct pyvo imports
from requests.exceptions import ConnectionError, Timeout, RequestException # For network errors
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

from astroquery.xmatch import XMatch # Import CDS XMatch

from . import auth
from .stilts import StiltsError, crossmatch_sky, _run_stilts # Import the crossmatch function and _run_stilts
from .tap import TapError, get_tap_service, execute_tap_query
from .astro_utils import propagate_coordinates_to_epoch # Import the new function

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_CONFIG_PATH = Path(__file__).parent / "catalogues.yaml"
SUPPORTED_INPUT_FORMATS = [".parquet", ".fits", ".csv"]


class CrossMatchError(Exception):
    """Custom exception for cross-matching errors."""

    pass


class CrossMatch:
    """Handles cross-matching of astronomical catalogues."""

    def __init__(self, config_file: Union[str, Path] = DEFAULT_CONFIG_PATH, **kwargs):
        """
        Initializes the CrossMatch object with enhanced configuration loading.

        Args:
            config_file: Path to the YAML configuration file.
            **kwargs: Additional configuration overrides (e.g., java_opts, chunk_size).
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()  # Loads the entire YAML

        # --- Load new structured configuration ---
        self.archives_config = self.config.get("archives", {})
        self.catalogues_config = self.config.get("catalogues", {})
        self.methods_config = self.config.get("crossmatch_methods", {})
        self.stilts_config = self.config.get("stilts_config", {})
        # --- End new configuration loading ---

        # Apply kwargs overrides to specific settings (e.g., STILTS path, Java opts)
        self.stilts_cmd_base = kwargs.get(
            "stilts_cmd_base", self.stilts_config.get("stilts_cmd_base")
        )
        self.stilts_java_opts = kwargs.get("java_opts", self.stilts_config.get("java_opts"))
        self.stilts_tmpdir = kwargs.get("tmpdir", self.stilts_config.get("tmpdir"))
        # Default chunk size for potential parallel processing (can be overridden)
        self.chunk_size = kwargs.get("chunk_size", 100000)

        self.auth_config = auth.load_auth_config()  # Load credentials securely

        self._catalogue_config_cache = {}  # Cache for resolved catalogue configs

        self._validate_config()
        logger.info(f"CrossMatch initialized with config: {self.config_file}")
        if kwargs:
            logger.info(f"Applied config overrides: {kwargs}")
        if self.stilts_cmd_base:
            logger.info(f"Using STILTS base command: '{self.stilts_cmd_base}'")

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
            raise CrossMatchError("Configuration must be a dictionary.")

        required_top_level = ["archives", "catalogues"]
        for key in required_top_level:
            if key not in self.config:
                raise CrossMatchError(f"Missing required top-level key in config: '{key}'")
            if not isinstance(self.config[key], dict):
                raise CrossMatchError(f"Top-level key '{key}' must be a dictionary.")

        # Validate Archives Structure (New Nested Structure)
        for archive_name, archive_config in self.config["archives"].items():
            if not isinstance(archive_config, dict):
                raise CrossMatchError(f"Archive '{archive_name}' config must be a dictionary.")
            if not any(k.endswith("_service") for k in archive_config):
                logger.warning(
                    f"Archive '{archive_name}' has no defined services (e.g., tap_service, xmatch_service). Is this intended?"
                )
            for service_id, service_config in archive_config.items():
                # Simple check, could be expanded (e.g., check for access_url/method in service)
                if isinstance(service_config, dict) and not service_id.startswith(
                    ("_", "description", "service_priority", "has_", "crossmatch_")
                ):
                    if "access_method" not in service_config:
                        logger.warning(
                            f"Service '{service_id}' in archive '{archive_name}' might be missing 'access_method'."
                        )

        # Validate Catalogues Structure
        for cat_name, cat_config in self.config["catalogues"].items():
            if not isinstance(cat_config, dict):
                raise CrossMatchError(f"Catalogue '{cat_name}' config must be a dictionary.")
            required_cat_keys = [
                "archive",
                "service_id",
                "access_identifier",
                "ra_column",
                "dec_column",
            ]
            for key in required_cat_keys:
                if key not in cat_config:
                    raise CrossMatchError(
                        f"Missing required key '{key}' in catalogue '{cat_name}'."
                    )
            # Check if archive and service_id exist
            archive_name = cat_config["archive"]
            service_id = cat_config["service_id"]
            if archive_name not in self.config["archives"]:
                raise CrossMatchError(
                    f"Archive '{archive_name}' referenced by catalogue '{cat_name}' not found in 'archives' section."
                )
            if service_id not in self.config["archives"][archive_name]:
                raise CrossMatchError(
                    f"Service '{service_id}' referenced by catalogue '{cat_name}' not found in archive '{archive_name}'."
                )
            if not isinstance(self.config["archives"][archive_name][service_id], dict):
                raise CrossMatchError(
                    f"Service '{service_id}' in archive '{archive_name}' (referenced by '{cat_name}') must be a dictionary."
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

    def _get_config_for_input(
        self, cat_input: Union[str, Path, pd.DataFrame, Table]
    ) -> Dict[str, Any]:
        """Gets the appropriate config, handling names, paths, or DataFrames.

        Returns a dictionary representing the configuration, which might be
        a resolved config from YAML or a temporary config for local files/
        DataFrames.

        It also stores the loaded DataFrame in the config dict under
        '_input_dataframe' if the input was a DataFrame or Table.
        """
        if isinstance(cat_input, pd.DataFrame):
            df = cat_input.copy()
            return {
                "_catalogue_name": "input_dataframe",
                "_input_dataframe": df,
                "description": f"Input DataFrame (shape {df.shape})",
                "archive": None,
                "service_type": "local_file", # Treat DF as local
                "access_method": "file_system", # Treat DF as local
                "is_local": True,
                "ra_column": "ra",  # Assume default column names
                "dec_column": "dec",
                "default_pos_error_arcsec": 0.1,
            }
        elif isinstance(cat_input, astropy.table.Table):
            try:
                df = cat_input.to_pandas()
                return {
                    "_catalogue_name": "input_table",
                    "_input_dataframe": df,
                    "description": f"Input Astropy Table (shape {df.shape})",
                    "archive": None,
                    "service_type": "local_file",
                    "access_method": "file_system",
                    "is_local": True,
                    "ra_column": "ra",
                    "dec_column": "dec",
                    "default_pos_error_arcsec": 0.1,
                }
            except Exception as e:
                raise CrossMatchError(f"Failed to convert input Astropy Table to DataFrame: {e}") from e
        elif isinstance(cat_input, (str, Path)):
            path_or_name = str(cat_input)
            try:
                path = Path(path_or_name)
                if path.is_file():
                    logger.info(f"Input '{path.name}' recognized as a local file.")
                    # Create a temporary config for the local file
                    return {
                        "_catalogue_name": path.stem,
                        "_input_path": str(path),
                        "description": f"Local file: {path.name}",
                        "archive": None,
                        "service_type": "local_file",
                        "access_method": "file_system",
                        "estimated_size": "unknown",  # Could estimate from file size
                        "is_local": True,
                        "ra_column": "ra",  # Assume default column names
                        "dec_column": "dec",
                        "default_pos_error_arcsec": 0.1,
                    }
            except OSError as e:
                 # Handle cases where the string might be too long for a path or invalid
                 logger.debug(f"Input '{path_or_name}' not a valid file path: {e}")
                 # Continue to check if it's a catalogue name

            # If not a file, assume it's a configured catalogue name
            if path_or_name.lower() in self.catalogues_config:
                return self.get_catalogue_config(path_or_name.lower())
            else:
                raise CrossMatchError(
                    f"Input '{path_or_name}' is not a valid file path or configured catalogue name."
                )
        else:
            raise TypeError(f"Unsupported input type for catalogue: {type(cat_input)}")

    def _load_local_catalogue(self, catalogue_path: Union[str, Path]) -> pd.DataFrame:
        """Reads local catalogue data from file into a pandas DataFrame."""
        path = Path(catalogue_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

        logger.info(f"Reading catalogue from file: {path}")
        try:
            # Determine file type and read accordingly
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            elif path.suffix.lower() == ".csv":
                # TODO: Add options for CSV parsing (sep, header, etc.)?
                return pd.read_csv(path)
            elif path.suffix.lower() in [".fits", ".fit"]:
                try:
                    table = Table.read(path)
                    return table.to_pandas()
                except ImportError:
                    logger.error("Reading FITS requires 'astropy' library.")
                    raise CrossMatchError("Reading FITS requires 'astropy' library.")
                except Exception as e:
                    logger.error(f"Error reading FITS file {path}: {e}")
                    raise CrossMatchError(f"Error reading FITS file {path}: {e}") from e
            else:
                raise CrossMatchError(
                    f"Unsupported file type: {path.suffix}. Use .parquet, .csv, or .fits"
                )
        except pd.errors.EmptyDataError:
             logger.warning(f"Local catalogue file is empty: {path}")
             return pd.DataFrame() # Return empty DataFrame for empty files
        except MemoryError as e:
            logger.error(f"Memory error reading local file {path}. File might be too large.")
            raise CrossMatchError(f"Memory error reading {path}") from e
        except Exception as e:
            logger.error(f"Error reading catalogue file {path}: {e}", exc_info=True)
            raise CrossMatchError(f"Error reading catalogue file {path}: {e}") from e

    def _write_catalogue(self, df: pd.DataFrame, output_path: Union[str, Path]):
        """Writes a DataFrame to a specified output file path (default Parquet)."""
        output_path = Path(output_path)
        logger.info(f"Writing output to: {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # TODO: Support other formats like FITS, CSV based on output filename or arg
            df.to_parquet(output_path, compression="snappy", index=False)
        except MemoryError as e:
             logger.error(f"Memory error writing output file {output_path}. DataFrame might be too large.")
             raise CrossMatchError(f"Memory error writing {output_path}") from e
        except IOError as e:
             logger.error(f"IO error writing output file {output_path}: {e}")
             raise CrossMatchError(f"IO error writing {output_path}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error writing output file {output_path}: {e}", exc_info=True)
            raise CrossMatchError(f"Unexpected error writing output file {output_path}: {e}") from e

    def _process_in_parallel(
        self,
        catalogue_df: pd.DataFrame,
        process_func: Callable,  # The function to apply to chunks
        n_workers: Optional[int] = None,  # Number of parallel workers
        **kwargs,
    ) -> pd.DataFrame:
        """Processes a DataFrame in parallel chunks using ProcessPoolExecutor."""
        # Determine optimal chunk count based on CPU cores if not specified
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 default
        if n_workers <= 0:  # Allow sequential processing if 0 or negative workers specified
            n_workers = 1

        total_rows = len(catalogue_df)
        if total_rows == 0:
            logger.warning("Empty DataFrame provided to parallel processing")
            return pd.DataFrame()

        # Split dataframe into chunks ensuring all rows are included
        # Ensure at least one chunk, even if dataframe is smaller than chunk_size
        # Adjust n_workers if fewer chunks than workers
        actual_workers = min(n_workers, total_rows)  # Cannot have more workers than rows
        if actual_workers <= 0:
            actual_workers = 1

        # Use numpy split for potentially more even distribution
        chunks = np.array_split(catalogue_df, actual_workers)
        logger.info(
            f"Processing {total_rows} rows in {len(chunks)} chunks using {actual_workers} workers"
        )

        # If very small dataset or only 1 worker, process sequentially
        if total_rows < 1000 or actual_workers == 1:
            logger.info("Small dataset or 1 worker requested, processing sequentially")
            try:
                # Pass 'chunk' kwarg even for sequential for consistency?
                # Or just pass df directly?
                return process_func(catalogue_df, **kwargs)
            except Exception as e:
                logger.error(f"Sequential processing failed: {e}", exc_info=True)
                raise CrossMatchError(f"Sequential processing failed: {e}") from e # Wrap in CrossMatchError

        results = []
        failed_chunks = []

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            # Submit function with the chunk and any other necessary args from kwargs
            futures = {
                executor.submit(process_func, chunk=chunk, **kwargs): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, pd.DataFrame):
                        results.append(result)
                    else:
                        logger.warning(
                            f"Chunk {chunk_idx} returned non-DataFrame result (type: {type(result)}). Skipping."
                        )
                        # Optionally store non-dataframe results if needed
                except Exception as e: # Catch errors from the child process
                    logger.error(f"Error processing chunk {chunk_idx}: {e}", exc_info=True)
                    failed_chunks.append(chunk_idx)

        # Raise error if any chunks failed
        if failed_chunks:
            raise CrossMatchError(
                f"Processing failed for {len(failed_chunks)} chunks: {failed_chunks}"
            )

        # Combine results
        if not results:
            logger.warning("Parallel processing yielded no valid results.")
            return pd.DataFrame()  # Return empty DataFrame consistent with input type

        logger.info(f"Successfully processed {len(results)} chunks.")
        try:
            combined_df = pd.concat(results, ignore_index=True)
            logger.info(f"Combined results into DataFrame with {len(combined_df)} rows.")
            return combined_df
        except MemoryError as e:
            logger.error("Memory error combining parallel processing results.")
            raise CrossMatchError("Memory error combining parallel results") from e
        except Exception as e:
            logger.error(f"Unexpected error combining parallel processing results: {e}", exc_info=True)
            raise CrossMatchError(f"Unexpected error combining parallel results: {e}") from e

    # -----------------------------------------------------------
    # New Core Crossmatch Orchestration Logic
    # -----------------------------------------------------------

    def crossmatch(
        self,
        catalogue_1_input: Union[str, Path, pd.DataFrame, Table],
        catalogue_2_input: Union[str, Path, pd.DataFrame, Table],
        output_file: Optional[Union[str, Path]] = None,
        method: Optional[str] = None,  # Optional: Suggest a method (e.g., 'stilts_skyerr')
        join_on_ids: Optional[Dict[str, str]] = None, # NEW: {cat1: colA, cat2: colB}
        **kwargs,
    ) -> Union[pd.DataFrame, None]:
        """Performs cross-matching between two catalogues.

        Args:
            catalogue_1_input: Name, path, DataFrame, or Table for the first catalogue.
            catalogue_2_input: Name, path, DataFrame, or Table for the second catalogue.
            output_file: Optional path to save the result (default: Parquet). If None, returns DataFrame.
            method: Optional hint for the cross-matching method.
            join_on_ids: Optional dictionary to perform an ID-based join instead of spatial.
                         Must contain keys 'cat1' and 'cat2' mapping to the join column names
                         in the respective catalogues (e.g., {'cat1': 'source_id', 'cat2': 'gaia_id'}).
                         If provided, 'radius_arcsec' and 'matcher' related to spatial joins are ignored for remote joins.
            **kwargs: Additional parameters passed to the underlying execution methods
                      (e.g., radius_arcsec, find, join_type, columns1, columns2, ra, dec).

        Returns:
            A pandas DataFrame with the crossmatch results if output_file is None, otherwise None.

        Raises:
            CrossMatchError: If configuration is invalid, inputs are problematic,
                             or the cross-match execution fails.
            ValueError: If join_on_ids is provided in an invalid format.
        """
        start_time = time.time()
        logger.info("--- Crossmatch initiated ---")
        logger.info(f"Input 1: {catalogue_1_input}")
        logger.info(f"Input 2: {catalogue_2_input}")
        if output_file:
            logger.info(f"Output file: {output_file}")
        if method:
            logger.info(f"Suggested method: {method}")
        if join_on_ids:
            logger.info(f"Requested ID Join on: {join_on_ids}")
        logger.info(f"Other parameters: {kwargs}")


        try:
            # 1. Resolve input configurations
            # _get_config_for_input also handles loading DF/Table inputs
            config1 = self._get_config_for_input(catalogue_1_input)
            config2 = self._get_config_for_input(catalogue_2_input)
            cat_name1 = config1.get("_catalogue_name", "Input1")
            cat_name2 = config2.get("_catalogue_name", "Input2")
            logger.debug(f"Resolved Config 1 ({cat_name1}): {config1}")
            logger.debug(f"Resolved Config 2 ({cat_name2}): {config2}")

            # 2. Prepare parameters for strategy determination and execution
            exec_params = kwargs.copy()
            if join_on_ids:
                if not isinstance(join_on_ids, dict) or 'cat1' not in join_on_ids or 'cat2' not in join_on_ids:
                     raise ValueError("join_on_ids dict must have keys 'cat1' and 'cat2' mapping to column names.")
                exec_params['join_mode'] = 'id'
                exec_params['join_keys'] = join_on_ids # Pass {'cat1': 'col_name_1', 'cat2': 'col_name_2'}
            else:
                # Default to spatial join if join_on_ids is not provided
                exec_params['join_mode'] = 'sky'
                # Ensure radius is present if doing a sky join later
                if 'radius_arcsec' not in exec_params:
                    exec_params['radius_arcsec'] = 1.0 # Default radius if not specified
                    logger.info("No radius_arcsec provided for spatial join, defaulting to 1.0 arcsec.")


            # 3. Determine the optimal cross-matching strategy
            # Pass potentially updated exec_params
            strategy, resolved_params = self._determine_crossmatch_strategy(
                config1, config2, **exec_params
            )
            logger.info(f"Selected strategy: {strategy}")
            logger.debug(f"Resolved execution parameters: {resolved_params}")


            # 4. Execute the chosen strategy
            result_df = None
            if strategy == "local_stilts":
                result_df = self._execute_local_stilts(config1, config2, **resolved_params)
            elif strategy == "remote_join":
                result_df = self._execute_remote_join_match(config1, config2, **resolved_params)
            elif strategy == "download_and_match":
                result_df = self._execute_download_and_match(config1, config2, **resolved_params)
            elif strategy == "remote_spatial_chunked_match":
                result_df = self._execute_remote_spatial_chunked_match(config1, config2, **resolved_params)
            elif strategy == "chunked_match": # Local chunked match
                result_df = self._execute_chunked_match(config1, config2, **resolved_params)
            elif strategy == "cds_xmatch_local_remote":
                 result_df = self._execute_cds_xmatch_local_remote(config1, config2, **resolved_params)
            else:
                raise CrossMatchError(f"Unsupported cross-matching strategy: {strategy}")

            # 5. Handle results (write to file or return DataFrame)
            if result_df is None:
                logger.warning("Crossmatch execution returned no results (None DataFrame).")
                # Decide whether to write empty file or just return None/empty DF
                if output_file:
                     logger.warning(f"Writing empty file to {output_file} as no results were returned.")
                     # Create empty file or write empty DF? Writing empty DF is safer.
                     pd.DataFrame().to_parquet(output_file, index=False)
                     return None # Indicate file written
                else:
                     return pd.DataFrame() # Return empty DataFrame


            logger.info(f"Crossmatch completed, resulted in {len(result_df)} rows.")
            if output_file:
                self._write_catalogue(result_df, output_file)
                end_time = time.time()
                logger.info(f"Result saved to {output_file}. Total time: {end_time - start_time:.2f} seconds.")
                return None  # Indicate file was written
            else:
                end_time = time.time()
                logger.info(f"Returning result DataFrame. Total time: {end_time - start_time:.2f} seconds.")
                return result_df

        except (CrossMatchError, StiltsError, TapError, ValueError, TypeError, KeyError, FileNotFoundError, MemoryError, ImportError) as e:
            logger.error(f"Crossmatch failed: {e}", exc_info=True)
            raise  # Re-raise known/specific errors
        except Exception as e:
            # Catch-all for truly unexpected errors
            logger.error(f"An unexpected error occurred during crossmatch: {e}", exc_info=True)
            raise CrossMatchError(f"Unexpected crossmatch error: {e}") from e

    def _determine_crossmatch_strategy(
        self, config1: Dict[str, Any], config2: Dict[str, Any], **params
    ) -> Tuple[str, Dict[str, Any]]:
        """Determines the best cross-matching strategy based on input types and configs."""
        logger.info("Determining crossmatch strategy...")
        cat_name1 = config1.get("_catalogue_name", "Input1")
        cat_name2 = config2.get("_catalogue_name", "Input2")

        # Default values
        requires_local_stilts = False
        catalogue_to_download = None # Which catalogue to download ('1', '2', 'both', None)

        # --- Analyze Input Types and Capabilities ---
        is_local1 = config1.get("service_type") == "local_file"
        is_local2 = config2.get("service_type") == "local_file"
        access_method1 = config1.get("access_method")
        access_method2 = config2.get("access_method")
        archive1 = config1.get("_archive_name")
        archive2 = config2.get("_archive_name")

        # Track which remote catalogue might need downloading
        strategy = "unknown"

        if is_local1 and is_local2:
            logger.info("Strategy: Both inputs are local. Using local STILTS.")
            strategy = "local_stilts"
            requires_local_stilts = True
        elif is_local1 ^ is_local2:  # One local, one remote
            local_conf = config1 if is_local1 else config2
            remote_conf = config2 if is_local1 else config1
            local_cat_name = local_conf.get("_catalogue_name")
            remote_cat_name = remote_conf.get("_catalogue_name")
            logger.info(
                f"Strategy: One local ({local_cat_name}), one remote ({remote_cat_name})."
            )

            remote_access_method = remote_conf.get("access_method")
            remote_archive_name = remote_conf.get("_archive_name")

            # Check if remote uses CDS XMatch service
            if remote_access_method == "cds_xmatch":
                logger.info("Remote catalogue uses CDS XMatch service. Selecting CDS XMatch strategy.")
                strategy = "cds_xmatch_local_remote"
                requires_local_stilts = False # Handled by astroquery
            # Check if remote TAP supports upload (preferred remote strategy if available)
            # elif "table_upload" in remote_conf.get("_service_config", {}).get("crossmatch_input_methods", []):
            #     logger.info("Remote TAP supports table upload. Strategy: Upload local table and remote join/match.")
            #     # This requires implementing the upload and remote execution logic
            #     strategy = "upload_and_remote_match" # Placeholder for future strategy
            #     requires_local_stilts = False
            #     logger.warning("Strategy 'upload_and_remote_match' not yet fully implemented.")
            #     # Fallback for now:
            #     logger.warning("Falling back to download & local STILTS.")
            #     strategy = "download_and_match"
            #     catalogue_to_download = remote_cat_name
            #     requires_local_stilts = True
            else:
                 # Default: Download the remote catalogue and match locally
                 logger.info(f"Remote catalogue ({remote_cat_name}) does not use CDS XMatch or support preferred remote methods. Defaulting to download & local STILTS.")
                 strategy = "download_and_match"
                 catalogue_to_download = remote_cat_name
                 requires_local_stilts = True

        # --- Both Remote ---
        elif not is_local1 and not is_local2:
            logger.info(f"Strategy: Both inputs are remote ({archive1} vs {archive2}).")

            # Check if they are on the same TAP service and support JOIN
            can_remote_join = False
            on_same_tap_service = False # Add flag for same service
            if archive1 == archive2 and archive1 is not None:
                access_url1 = config1.get("access_url")
                access_url2 = config2.get("access_url")
                if access_url1 and access_url1 == access_url2:
                    on_same_tap_service = True # Mark that they share the service
                    adql_features = config1.get("adql_features", [])
                    if "table_join" in adql_features:
                        can_remote_join = True
                        logger.info(f"Both catalogues on same TAP service ({access_url1}) supporting table_join.")
                    else:
                        logger.info(f"Both catalogues on same TAP service, but service does not advertise 'table_join' support.")
                else:
                    logger.info("Catalogues share an archive name but have different service access URLs.")
            else:
                logger.info("Catalogues are on different archives.")


            # Decide strategy based on join capability and requested join mode
            join_mode = params.get("join_mode", "sky") # Get join mode from params
            has_spatial_constraints = all(k in params for k in ["ra", "dec", "radius_arcsec"])

            # --- Strategy Selection Logic ---
            if on_same_tap_service and join_mode == "id":
                if can_remote_join:
                     # ID join requested and possible remotely via ADQL JOIN
                    logger.info("Selecting remote ADQL ID JOIN strategy.")
                    strategy = "remote_join"
                    requires_local_stilts = False
                    catalogue_to_download = None
                    if "join_keys" not in params:
                        logger.error("ID join mode selected for remote join, but 'join_keys' missing in params.")
                        logger.warning("Falling back to download & local match due to missing join_keys for remote ID join.")
                        strategy = "download_and_match"
                        requires_local_stilts = True
                        catalogue_to_download = "both"
                else:
                     # Cannot do remote ID JOIN, must download both
                     logger.warning(f"Remote ID join requested but not possible on this TAP service (can_remote_join={can_remote_join}). Falling back to download & local match.")
                     strategy = "download_and_match"
                     requires_local_stilts = True
                     catalogue_to_download = "both"

            elif on_same_tap_service and join_mode == "sky":
                # Spatial join requested on same TAP service
                if has_spatial_constraints:
                    # Spatial constraints provided - use the NEW remote chunked strategy
                    logger.info("Spatial constraints provided for same-TAP match. Selecting NEW Remote TAP Chunked Spatial JOIN strategy.")
                    strategy = "remote_tap_chunked_spatial_join"
                    requires_local_stilts = False # The join happens remotely per chunk
                    catalogue_to_download = None
                elif can_remote_join:
                    # No spatial constraints, but service supports JOIN - use direct ADQL JOIN
                    logger.info("No spatial constraints, but same TAP service supports JOIN. Selecting remote ADQL spatial JOIN strategy.")
                    strategy = "remote_join"
                    requires_local_stilts = False
                    catalogue_to_download = None
                    if "radius_arcsec" not in params or params["radius_arcsec"] <= 0:
                        logger.warning("Radius missing or invalid for remote spatial join. Falling back to download & local match.")
                        strategy = "download_and_match"
                        requires_local_stilts = True
                        catalogue_to_download = config2.get("_catalogue_name")
                else:
                    # No constraints, cannot JOIN remotely - fallback to download
                    logger.warning("No spatial constraints and remote TAP JOIN not supported. Falling back to download & local match.")
                    strategy = "download_and_match"
                    requires_local_stilts = True
                    catalogue_to_download = config2.get("_catalogue_name")

            elif not on_same_tap_service and join_mode == "id":
                 # ID join requested but on different services - must download both
                 logger.warning(f"Remote ID join requested but catalogues are on different services. Falling back to download & local match.")
                 strategy = "download_and_match"
                 requires_local_stilts = True
                 catalogue_to_download = "both"

            else: # Different services, spatial join OR same service but unrecognized mode?
                 # Default to downloading and matching locally
                 logger.warning(f"Cannot perform remote join (different services or unsupported mode: {join_mode}). Defaulting to download & local match.")
                 strategy = "download_and_match"
                 requires_local_stilts = True
                 # Decide which to download based on join mode
                 if join_mode == "id":
                     catalogue_to_download = "both"
                 else:
                     # For sky join, just download one (cat 2 by default)
                     catalogue_to_download = config2.get("_catalogue_name")


            # TODO: Handle other remote strategies like cds_xmatch if relevant here?

        else: # Both local case handled earlier
            raise CrossMatchError("Internal Error: Could not determine strategy based on input types.")


        # --- Final Checks and Refinements ---
        if strategy == "unknown":
             logger.error("Failed to determine a valid crossmatch strategy.")
             raise CrossMatchError("Could not determine crossmatch strategy.")

        # Check if STILTS is required but not configured/found
        if requires_local_stilts:
             # Check for STILTS command path early? Or let _run_stilts handle it?
             # Letting _run_stilts handle it provides a clearer error location.
             logger.info("Selected strategy requires local STILTS execution.")
             pass


        # Return strategy and potentially modified params
        params["_requires_local_stilts"] = requires_local_stilts
        params["_catalogue_to_download"] = catalogue_to_download # Track which one to download if needed by caller
        logger.info(f"Final strategy: {strategy}")
        return strategy, params

    # -----------------------------------------------------------
    # Placeholder Execution Methods
    # -----------------------------------------------------------

    def _execute_local_stilts(
        self, config1: Dict[str, Any], config2: Dict[str, Any], **params
    ) -> pd.DataFrame:
        """Executes a crossmatch using local STILTS (tmatch2 or tmatch1).

        Handles epoch propagation for sky joins if necessary.
        Handles both sky-based and ID-based joins.
        Reads input from files or DataFrames.
        Writes temporary files for STILTS and reads the output.
        """
        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")
        join_mode = params.get("join_mode", "sky") # 'sky' or 'id'

        if join_mode == 'sky':
            matcher = params.get("matcher") # Get matcher determined by strategy (sky, skyerr, etc.)
            if not matcher:
                 # Auto-detect best matcher based on available error columns in *both* configs
                 has_err1 = all(col and col in config1 for col in [config1.get("ra_err_column"), config1.get("dec_err_column")])
                 has_err2 = all(col and col in config2 for col in [config2.get("ra_err_column"), config2.get("dec_err_column")])
                 has_corr1 = config1.get("corr_column") and config1.get("corr_column") in config1
                 has_corr2 = config2.get("corr_column") and config2.get("corr_column") in config2

                 if has_err1 and has_err2:
                     if has_corr1 and has_corr2:
                         matcher = "skyellipse"
                         logger.info("Auto-selected matcher: skyellipse (found RA/Dec errors and correlation in both catalogues)")
                     else:
                         matcher = "skyerr"
                         logger.info("Auto-selected matcher: skyerr (found RA/Dec errors in both catalogues)")
                 else:
                     matcher = "sky"
                     logger.info("Auto-selected matcher: sky (required error columns not found in both catalogues)")
                 params["matcher"] = matcher # Store auto-detected matcher

            logger.info(f"Executing local STILTS sky match ({matcher}): {cat_name1} vs {cat_name2}")
            # --- Epoch Propagation (Sky Joins Only) ---
            epoch1 = config1.get("epoch")
            epoch2 = config2.get("epoch")
            target_epoch = epoch1 # Default to epoch1 if epoch2 is missing or they are the same
            if epoch1 and epoch2 and abs(epoch1 - epoch2) > 0.01: # Use a small tolerance
                # Basic logic: propagate the one with proper motion to the epoch of the other
                # Assumes config has necessary pm columns if propagation is feasible
                pm_cols1 = config1.get("pm_ra_column") and config1.get("pm_dec_column")
                pm_cols2 = config2.get("pm_ra_column") and config2.get("pm_dec_column")

                if pm_cols1 and not pm_cols2: target_epoch = epoch2 # Propagate 1 -> 2
                elif pm_cols2 and not pm_cols1: target_epoch = epoch1 # Propagate 2 -> 1
                elif pm_cols1 and pm_cols2: # Both have PM? Propagate earlier epoch to later?
                    target_epoch = max(epoch1, epoch2)
                else: # Neither has PM, no propagation despite epoch difference
                    target_epoch = None

                if target_epoch:
                    logger.info(f"Epochs differ ({epoch1} vs {epoch2}), setting target epoch for propagation: {target_epoch}")
                else:
                    logger.info(f"Epochs differ but PM columns missing, skipping propagation.")
            elif epoch1: target_epoch = epoch1
            elif epoch2: target_epoch = epoch2
            else: target_epoch = None # No propagation needed if epochs unknown/same

            if target_epoch is None:
                 logger.info("No epoch propagation needed/possible for local STILTS match.")
        else: # join_mode == 'id'
            logger.info(f"Executing local STILTS ID match: {cat_name1} vs {cat_name2}")
            matcher = None # Matcher not used for ID joins
            target_epoch = None # Epoch propagation not used for ID joins

        with tempfile.TemporaryDirectory(prefix="stilts_match_") as temp_dir:
            # --- Prepare Input 1 (Handles potential epoch propagation) ---
            input1_path = None
            config1_local = config1.copy() # Use local copies for potential modifications
            if "_input_dataframe" in config1_local:
                df1 = config1_local["_input_dataframe"]
                if join_mode == 'sky' and target_epoch and config1_local.get("epoch") and config1_local.get("epoch") != target_epoch:
                    df1 = self._apply_epoch_propagation(config1_local, df1, target_epoch)
                input1_path = self._prepare_stilts_input(df1, temp_dir, "input1.parquet")
                if "ra_propagated" in df1.columns: config1_local['ra_column'] = 'ra_propagated'
                if "dec_propagated" in df1.columns: config1_local['dec_column'] = 'dec_propagated'
            elif "_input_path" in config1_local:
                input1_path_orig = config1_local["_input_path"]
                input1_path = input1_path_orig
                if join_mode == 'sky' and target_epoch and config1_local.get("epoch") and config1_local.get("epoch") != target_epoch:
                    logger.info(f"Loading {cat_name1} from {input1_path_orig} for epoch propagation...")
                    df1 = self._load_local_catalogue(input1_path_orig)
                    df1 = self._apply_epoch_propagation(config1_local, df1, target_epoch)
                    input1_path = self._prepare_stilts_input(df1, temp_dir, "input1_propagated.parquet")
                    if "ra_propagated" in df1.columns: config1_local['ra_column'] = 'ra_propagated'
                    if "dec_propagated" in df1.columns: config1_local['dec_column'] = 'dec_propagated'
                    logger.info(f"Applied epoch propagation to {cat_name1} from file.")
            else:
                raise CrossMatchError(f"Could not find input data for {cat_name1}")

            # --- Prepare Input 2 (Handles potential epoch propagation) ---
            input2_path = None
            config2_local = config2.copy()
            if "_input_dataframe" in config2_local:
                df2 = config2_local["_input_dataframe"]
                if join_mode == 'sky' and target_epoch and config2_local.get("epoch") and config2_local.get("epoch") != target_epoch:
                    df2 = self._apply_epoch_propagation(config2_local, df2, target_epoch)
                input2_path = self._prepare_stilts_input(df2, temp_dir, "input2.parquet")
                if "ra_propagated" in df2.columns: config2_local['ra_column'] = 'ra_propagated'
                if "dec_propagated" in df2.columns: config2_local['dec_column'] = 'dec_propagated'
            elif "_input_path" in config2_local:
                input2_path_orig = config2_local["_input_path"]
                input2_path = input2_path_orig
                if join_mode == 'sky' and target_epoch and config2_local.get("epoch") and config2_local.get("epoch") != target_epoch:
                    logger.info(f"Loading {cat_name2} from {input2_path_orig} for epoch propagation...")
                    df2 = self._load_local_catalogue(input2_path_orig)
                    df2 = self._apply_epoch_propagation(config2_local, df2, target_epoch)
                    input2_path = self._prepare_stilts_input(df2, temp_dir, "input2_propagated.parquet")
                    if "ra_propagated" in df2.columns: config2_local['ra_column'] = 'ra_propagated'
                    if "dec_propagated" in df2.columns: config2_local['dec_column'] = 'dec_propagated'
                    logger.info(f"Applied epoch propagation to {cat_name2} from file.")
            else:
                raise CrossMatchError(f"Could not find input data for {cat_name2}")

            # --- Define Output ---
            output_path = str(Path(temp_dir) / "output.parquet")

            # --- Execute STILTS ---
            try:
                # Prepare common STILTS parameters
                stilts_base_params = {
                    "in1": input1_path,
                    "in2": input2_path,
                    "out": output_path,
                    "ofmt": "parquet-snappy", # Use efficient parquet output
                    "join": params.get("join", "1and2"), # STILTS join parameter
                    "find": params.get("find", "best"),
                    # Pass STILTS config overrides if present
                    "stilts_cmd_base": self.stilts_cmd_base,
                    "java_opts": self.stilts_java_opts,
                    "tmpdir": self.stilts_tmpdir,
                    # Pass any other relevant kwargs from the main call?
                    **params.get("stilts_options", {}) # Allow passing specific STILTS params
                }
                stilts_params = {k: v for k, v in stilts_base_params.items() if v is not None}

                stilts_cmd = ""

                if join_mode == "sky":
                    # --- Sky Join Parameters (tmatch2) ---
                    stilts_cmd = "tmatch2"
                    sky_params = {
                        "matcher": matcher,
                        "ra1": config1_local["ra_column"], # Use potentially updated column name
                        "dec1": config1_local["dec_column"],
                        "ra2": config2_local["ra_column"],
                        "dec2": config2_local["dec_column"],
                    }
                    # Add error columns if needed by matcher
                    if matcher in ["skyerr", "skyellipse"]:
                        sky_params["error1"] = config1_local.get("default_pos_error_arcsec") # Default error
                        sky_params["error2"] = config2_local.get("default_pos_error_arcsec")
                        if config1_local.get("ra_err_column") and config1_local.get("dec_err_column"):
                             # Prefer specific error columns if available
                             sky_params["error1"] = f"hypot({config1_local['ra_err_column']}, {config1_local['dec_err_column']})" # TODO: Check units!
                        if config2_local.get("ra_err_column") and config2_local.get("dec_err_column"):
                             sky_params["error2"] = f"hypot({config2_local['ra_err_column']}, {config2_local['dec_err_column']})"
                    # Add radius for 'sky' matcher, max_error for error matchers
                    if matcher == 'sky':
                        radius = params.get("radius_arcsec", 1.0)
                        sky_params["params"] = radius # tmatch2 'params' is radius for sky
                    else: # skyerr, skyellipse
                        max_error = params.get("max_error", 5.0) # Default sigma separation
                        sky_params["params"] = max_error # tmatch2 'params' is max error

                    stilts_params.update(sky_params)
                    logger.info(f"Calling STILTS tmatch2 with sky params: {sky_params}")

                elif join_mode == "id":
                    # --- ID Join Parameters (tmatch1) ---
                    stilts_cmd = "tmatch1" # Use tmatch1 for single-column value matching
                    join_keys = params.get("join_keys")
                    if not join_keys or 'cat1' not in join_keys or 'cat2' not in join_keys:
                         raise ValueError("Missing or invalid 'join_keys' for local ID join.")
                    id_params = {
                        "values1": join_keys['cat1'],
                        "values2": join_keys['cat2'],
                        # 'matcher' for tmatch1 is usually 'exact' or similar, but defaults work
                    }
                    stilts_params.update(id_params)
                    logger.info(f"Calling STILTS tmatch1 with ID params: {id_params}")

                else:
                    raise ValueError(f"Invalid join_mode '{join_mode}' for local STILTS.")

                # Execute STILTS using the generic runner
                _run_stilts(stilts_cmd, stilts_params)

                # --- Read Result ---
                logger.info(f"STILTS {stilts_cmd} completed. Reading result from {output_path}")
                result_df = pd.read_parquet(output_path)
                logger.info(f"Successfully read {len(result_df)} rows from STILTS output.")
                return result_df

            except StiltsError as e:
                logger.error(f"STILTS execution failed: {e}", exc_info=True)
                raise
            except FileNotFoundError as e:
                logger.error(f"Input/Output file not found during STILTS execution: {e}")
                raise CrossMatchError(f"File not found: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error during STILTS execution: {e}", exc_info=True)
                raise CrossMatchError(f"Unexpected STILTS error: {e}") from e

    def _prepare_stilts_input(self, df: pd.DataFrame, temp_dir: str, filename: str) -> str:
        """Writes a DataFrame to a temporary file (Parquet) for STILTS."""
        temp_file_path = Path(temp_dir) / filename
        logger.debug(f"Preparing STILTS input file: {temp_file_path}")
        try:
            # Ensure columns expected by STILTS (RA/Dec/Errors) exist
            # Validation should happen before this point
            df.to_parquet(temp_file_path, index=False)
            logger.debug(f"Wrote temporary input table ({len(df)} rows) to: {temp_file_path}")
            return str(temp_file_path)
        except MemoryError as e:
            logger.error(f"Memory error writing temporary input file {temp_file_path}. DataFrame might be too large.")
            raise CrossMatchError(f"Memory error writing temporary file {temp_file_path}") from e
        except IOError as e:
             logger.error(f"IO error writing temporary input file {temp_file_path}: {e}")
             raise CrossMatchError(f"IO error writing temporary file {temp_file_path}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to write temporary input file {temp_file_path}: {e}", exc_info=True)
            raise CrossMatchError(f"Unexpected error writing temporary file {temp_file_path}: {e}") from e

    def _apply_epoch_propagation(
        self, config: Dict[str, Any], df: pd.DataFrame, target_epoch: float
    ) -> pd.DataFrame:
        """Applies epoch propagation if configuration and data allow."""
        ra_col = config.get("ra_column")
        dec_col = config.get("dec_column")
        pm_ra_col = config.get("pm_ra_column")
        pm_dec_col = config.get("pm_dec_column")
        epoch_col = config.get("epoch_column") # The column containing the source epoch
        current_epoch = config.get("epoch") # The reference epoch from YAML

        if not all([ra_col, dec_col, pm_ra_col, pm_dec_col, epoch_col, current_epoch]):
            logger.warning(
                f"Catalogue {config.get('_catalogue_name')} lacks required columns/config "
                f"(ra/dec/pm_ra/pm_dec/epoch_col/epoch) for propagation. Using original coordinates."
            )
            # Add ra_propagated/dec_propagated columns with original values
            df_out = df.copy()
            if ra_col and "ra_propagated" not in df_out.columns:
                 df_out["ra_propagated"] = df_out[ra_col]
            if dec_col and "dec_propagated" not in df_out.columns:
                 df_out["dec_propagated"] = df_out[dec_col]
            return df_out

        # Add the fixed epoch from YAML as a column if epoch_col doesn't exist in df
        if epoch_col not in df.columns:
            logger.debug(f"Adding fixed epoch column '{epoch_col}' = {current_epoch} for propagation.")
            df[epoch_col] = current_epoch

        logger.info(
            f"Applying epoch propagation for {config.get('_catalogue_name')} to target epoch {target_epoch}"
        )
        try:
            propagated_df = propagate_coordinates_to_epoch(
                df,
                ra_col=ra_col,
                dec_col=dec_col,
                pm_ra_col=pm_ra_col,
                pm_dec_col=pm_dec_col,
                epoch_col=epoch_col, # Use the column name specified in YAML
                target_epoch=target_epoch,
            )
            return propagated_df
        except (KeyError, ValueError, TypeError) as e:
             logger.error(f"Error applying epoch propagation: {e}", exc_info=True)
             raise CrossMatchError(f"Error during epoch propagation: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during epoch propagation: {e}", exc_info=True)
            raise CrossMatchError(f"Unexpected error during epoch propagation: {e}") from e

    def _execute_remote_join_match(
        self, config1: Dict[str, Any], config2: Dict[str, Any], **params
    ) -> pd.DataFrame:
        """Executes a JOIN query on a remote TAP service when both catalogues reside there
           and the service supports ADQL JOIN. Handles both spatial and ID-based joins.
        """
        join_mode = params.get("join_mode", "sky") # Default to sky join if not specified
        logger.info(f"Executing Remote TAP JOIN strategy (Mode: {join_mode})...")

        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")

        # --- Get Service and Table Info ---
        # Assumes _determine_strategy confirmed they are on the same joinable service
        service_config = config1 # Configs should be the same service
        tap_url = service_config.get("access_url")
        if not tap_url:
            raise CrossMatchError(f"Missing access_url in service config for remote join.")

        table1 = config1.get("access_identifier")
        table2 = config2.get("access_identifier")
        if not table1 or not table2:
             raise CrossMatchError("Missing access_identifier for one or both catalogues in remote join.")


        # --- Define Table Aliases ---
        alias1 = "t1"
        alias2 = "t2"

        # --- Column Selection ---
        # Get default columns or essential columns if defaults aren't specified
        cols1_req_raw = params.get("columns1") or config1.get("default_columns") or [config1.get("ra_column"), config1.get("dec_column")]
        cols2_req_raw = params.get("columns2") or config2.get("default_columns") or [config2.get("ra_column"), config2.get("dec_column")]

        # Ensure essential columns are always included
        essential_cols1 = {config1.get(c) for c in ["ra_column", "dec_column", "ra_err_column", "dec_err_column", "pm_ra_column", "pm_dec_column", "epoch_column"] if config1.get(c)}
        essential_cols2 = {config2.get(c) for c in ["ra_column", "dec_column", "ra_err_column", "dec_err_column", "pm_ra_column", "pm_dec_column", "epoch_column"] if config2.get(c)}

        cols1_req = list(set(cols1_req_raw) | essential_cols1)
        cols2_req = list(set(cols2_req_raw) | essential_cols2)

        # Ensure required join columns are selected if doing ID join
        if join_mode == 'id':
            join_keys = params.get('join_keys', {})
            id_col1 = join_keys.get('cat1')
            id_col2 = join_keys.get('cat2')
            if id_col1 and id_col1 not in cols1_req: cols1_req.append(id_col1)
            if id_col2 and id_col2 not in cols2_req: cols2_req.append(id_col2)

        # Remove None values from column lists
        cols1_req = [c for c in cols1_req if c is not None]
        cols2_req = [c for c in cols2_req if c is not None]

        # Format column selection with aliases to avoid name clashes
        cols1_select = [f'{alias1}."{c}" AS {alias1}_{c}' for c in set(cols1_req)] # Use set to deduplicate
        cols2_select = [f'{alias2}."{c}" AS {alias2}_{c}' for c in set(cols2_req)]

        # Handle SELECT * case if requested (though aliasing is safer)
        if "*" in cols1_req_raw: cols1_select = [f"{alias1}.*"]
        if "*" in cols2_req_raw: cols2_select = [f"{alias2}.*"]

        select_clause = ", ".join(cols1_select + cols2_select)
        if not select_clause:
            # Default to selecting all if specific columns failed resolution?
            logger.warning("Could not determine columns to SELECT, defaulting to SELECT *.")
            select_clause = f"{alias1}.*, {alias2}.*"

        # --- Epoch Propagation Check (Only relevant for Sky Joins) ---
        propagate_gaia_epoch = False
        target_epoch = None
        if join_mode == 'sky':
            epoch1 = config1.get("epoch")
            epoch2 = config2.get("epoch")
            is_gaia_service = config1.get("_archive_name") == "esa_gaia" # Assumes same archive

            if epoch1 and epoch2 and abs(epoch1 - epoch2) > 0.1: # Check if epochs differ significantly
                logger.warning(f"Epochs differ ({epoch1} vs {epoch2}) for remote TAP JOIN.")
                if is_gaia_service:
                    # Check if necessary columns for EPOCH_PROP are likely available
                    gaia_cols_present1 = all(config1.get(c) for c in ["ra_column", "dec_column", "pm_ra_column", "pm_dec_column", "epoch_column", "parallax_column"])
                    if gaia_cols_present1:
                        propagate_gaia_epoch = True
                        target_epoch = epoch2 # Target epoch is epoch of table 2
                        logger.info(f"Attempting ADQL EPOCH_PROP for {cat_name1} to target epoch {target_epoch} (Gaia TAP detected).")
                    else:
                         logger.warning("Gaia TAP detected, but catalogue 1 missing required columns (pmra, pmdec, parallax, ref_epoch) for EPOCH_PROP.")
                else:
                    logger.warning("Epochs differ, but service is not Gaia TAP or required columns missing. Cannot apply epoch propagation in ADQL query.")
            else:
                logger.info("Epochs are the same or not specified for both catalogues, no ADQL epoch propagation needed.")


        # --- Construct ADQL JOIN ON Clause ---
        adql_join_type = params.get("join_type", "INNER").upper() # e.g., INNER, LEFT, RIGHT
        join_on_clause = ""

        if join_mode == "sky":
            # --- Spatial Join ---
            radius_arcsec = params.get("radius_arcsec", 1.0) # Use resolved radius
            if radius_arcsec <= 0:
                raise ValueError("Search radius must be positive for sky join.")
            radius_deg = radius_arcsec / 3600.0

            # Get RA/Dec columns from config (already fetched above)
            ra_col1 = config1.get("ra_column")
            dec_col1 = config1.get("dec_column")
            ra_col2 = config2.get("ra_column")
            dec_col2 = config2.get("dec_column")
            if not ra_col1 or not dec_col1 or not ra_col2 or not dec_col2:
                 raise CrossMatchError("Missing RA/Dec column configuration for spatial join.")

            ra_expr1 = f'{alias1}."{ra_col1}"'
            dec_expr1 = f'{alias1}."{dec_col1}"'

            # Handle epoch propagation complexity (as before) - disable for now
            if propagate_gaia_epoch:
                 logger.warning("ADQL EPOCH_PROP usage within JOIN ON clause is complex/non-standard. Falling back to using original coordinates.")
                 propagate_gaia_epoch = False # Disable propagation due to complexity

            # Spatial join clause using DISTANCE
            join_on_clause = f'DISTANCE({ra_expr1}, {dec_expr1}, {alias2}."{ra_col2}", {alias2}."{dec_col2}") <= {radius_deg}'
            logger.info(f"Constructing ADQL spatial JOIN with radius {radius_arcsec} arcsec ({radius_deg} deg).")

        elif join_mode == "id":
            # --- ID-based Join ---
            join_keys = params.get("join_keys")
            if not join_keys or not isinstance(join_keys, dict) or 'cat1' not in join_keys or 'cat2' not in join_keys:
                 raise ValueError("Missing or invalid 'join_keys' dictionary in params for ID join. Expected {'cat1': 'col_name_1', 'cat2': 'col_name_2'}.")
            id_col1 = join_keys['cat1']
            id_col2 = join_keys['cat2']
            # ID join clause (ensure proper quoting for identifiers)
            join_on_clause = f'{alias1}."{id_col1}" = {alias2}."{id_col2}"'
            logger.info(f"Constructing ADQL ID JOIN on {alias1}.\"{id_col1}\" = {alias2}.\"{id_col2}\"")
            # Cannot do epoch propagation for ID joins
            if propagate_gaia_epoch:
                logger.warning("Epoch propagation is not applicable for ID-based joins. Ignoring.")
                propagate_gaia_epoch = False

        else:
            raise ValueError(f"Unsupported join_mode: '{join_mode}'. Must be 'sky' or 'id'.")

        # --- Final ADQL Query ---
        adql_query = f"""
        SELECT {select_clause}
        FROM {table1} AS {alias1}
        {adql_join_type} JOIN {table2} AS {alias2}
        ON {join_on_clause}
        """

        logger.info(f"Executing ADQL JOIN on {tap_url}:\n{adql_query}")

        # --- Execute Query ---
        try:
            # Get authenticated TAP service if needed
            auth_info = self.auth_config.get(archive1) # Use archive name
            tap_kwargs = {}
            if auth_info:
                tap_kwargs["user"] = auth_info.get("user")
                tap_kwargs["password"] = auth_info.get("password")

            tap_service = get_tap_service(tap_url, **tap_kwargs)

            # Execute query using tap.py function (handles retries etc.)
            # TODO: Make retry_delay and timeout configurable via params?
            results_df = execute_tap_query(
                 tap_service, adql_query, retry_delay=60, timeout=600, verbose=True
            )


            logger.info(f"ADQL JOIN completed, received {len(results_df)} rows.")
            if results_df is None or results_df.empty:
                logger.warning(f"ADQL JOIN returned 0 rows for {cat_name1} x {cat_name2}.")
                return pd.DataFrame()  # Return empty DataFrame

            return results_df

        except (TapError, ConnectionError, Timeout) as e:
            logger.error(f"Network or TAP error during remote JOIN for {cat_name1} x {cat_name2}: {e}", exc_info=True)
            raise # Re-raise TAP/network errors
        except Exception as e:
            # Catch potential ADQL syntax errors etc.
            logger.error(
                f"Unexpected error executing ADQL JOIN for {cat_name1} x {cat_name2}: {e}", exc_info=True
            )
            raise CrossMatchError(f"Failed TAP JOIN for {cat_name1} x {cat_name2}: {e}") from e

    def _execute_download_and_match(
        self, config1: Dict[str, Any], config2: Dict[str, Any], **params
    ) -> pd.DataFrame:
        """Downloads remote catalogue(s) and then performs a local match."""
        logger.info("Executing Download & Local Match strategy...")
        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")
        config1_local = config1.copy()  # Create copies to modify for local execution
        config2_local = config2.copy()

        is_remote1 = config1.get("access_method") != "file_system"
        is_remote2 = config2.get("access_method") != "file_system"

        try:
            # Extract potential spatial constraints from params for download filtering
            cone_params = None
            box_params = None # Add box params possibility
            if params.get("join_mode") == "sky" and all(k in params for k in ["ra", "dec", "radius_arcsec"]):
                cone_params = {
                    "ra": params["ra"],
                    "dec": params["dec"],
                    "radius_deg": params["radius_arcsec"] / 3600.0
                }
                logger.info(f"Found spatial constraints (cone) in params for download filtering: {cone_params}")
            # TODO: Add logic to extract box_params if needed

            # --- Download Remote Catalogues (if needed) ---
            if is_remote1:
                logger.info(f"Downloading remote catalogue 1: {cat_name1}")
                # Pass spatial constraints if available
                remote_df1 = self._fetch_remote_catalogue(
                    config1_local,
                    columns=params.get("columns1"),
                    cone_params=cone_params,
                    box_params=box_params # Pass box params too
                )
                if remote_df1 is None or remote_df1.empty:
                    logger.warning(
                        f"Failed to download or received empty data for remote catalogue: {cat_name1}. Result will be empty."
                    )
                    return pd.DataFrame()
                # Update config to point to the downloaded DataFrame
                config1_local["_input_dataframe"] = remote_df1
                config1_local["access_method"] = "file_system"  # Mark as local now
                config1_local.pop("_input_path", None)  # Remove original path if any
                logger.info(f"Successfully downloaded {len(remote_df1)} rows for {cat_name1}.")

            if is_remote2:
                logger.info(f"Downloading remote catalogue 2: {cat_name2}")
                # Pass spatial constraints if available
                remote_df2 = self._fetch_remote_catalogue(
                    config2_local,
                    columns=params.get("columns2"),
                    cone_params=cone_params,
                    box_params=box_params
                )
                if remote_df2 is None or remote_df2.empty:
                     logger.warning(
                        f"Failed to download or received empty data for remote catalogue: {cat_name2}. Result will be empty."
                    )
                     return pd.DataFrame()
                config2_local["_input_dataframe"] = remote_df2
                config2_local["access_method"] = "file_system"
                config2_local.pop("_input_path", None)
                logger.info(f"Successfully downloaded {len(remote_df2)} rows for {cat_name2}.")

            # --- Execute Local Match ---
            # Now that required data is local (as DataFrames or files), run local STILTS
            logger.info("Proceeding with local STILTS match on downloaded/local data.")
            # _execute_local_stilts handles DataFrames, file paths, and propagation
            result_df = self._execute_local_stilts(config1_local, config2_local, **params)

            logger.info("Download & Local Match completed.")
            return result_df

        except (CrossMatchError, TapError, StiltsError, ConnectionError, Timeout, RequestException, FileNotFoundError, ValueError, MemoryError) as e:
            logger.error(f"Download & Local Match failed: {e}", exc_info=True)
            # Re-raise specific known errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Download & Local Match: {e}", exc_info=True)
            raise CrossMatchError(f"Unexpected error in _execute_download_and_match: {e}") from e

    def _execute_remote_spatial_chunked_match(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        nside: int = 32, # Default HEALPix nside for chunking
        **params,
    ) -> pd.DataFrame:
        """Executes remote-remote match by chunking the sky area using HEALPix.

        Downloads data for each catalogue within each spatial chunk (pixel bounding box)
        and performs a local STILTS match on the chunks.

        Requires 'ra', 'dec', 'radius_arcsec' in params to define the area.
        """
        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")
        logger.info(
            f"Executing Remote Spatial Chunked Match ({cat_name1} vs {cat_name2}) using HEALPix nside={nside}"
        )

        # --- Validate and Extract Parameters ---
        if not all(k in params for k in ["ra", "dec", "radius_arcsec"]):
            raise ValueError(
                "Missing required parameters 'ra', 'dec', 'radius_arcsec' for spatial chunking."
            )
        try:
            center_coord = SkyCoord(ra=params["ra"] * u.deg, dec=params["dec"] * u.deg, frame="icrs")
            search_radius = params["radius_arcsec"] * u.arcsec
        except (ValueError, TypeError, u.UnitConversionError) as e:
            raise CrossMatchError(f"Invalid spatial parameters (ra/dec/radius): {e}") from e
        # Add a buffer to search radius to account for pixel boundaries? Maybe half pixel size?
        # Simplified for now.

        # --- Determine HEALPix Pixels ---
        try:
            hp = HEALPix(nside=nside, order="nested", frame="icrs")
            pixels = hp.cone_search_skycoord(center_coord, search_radius)
            logger.info(f"Identified {len(pixels)} HEALPix pixels (nside={nside}) covering the search area.")
            if len(pixels) == 0:
                 logger.warning("No HEALPix pixels found in the search cone. Returning empty result.")
                 return pd.DataFrame()
        except ImportError:
            logger.error("'astropy-healpix' library is required for chunked matching.")
            raise CrossMatchError(
                "'astropy-healpix' library is required for remote_spatial_chunked_match strategy. Please install it."
            )
        except Exception as e:
            logger.error(f"Error during HEALPix pixel determination: {e}", exc_info=True)
            raise CrossMatchError(f"Error during HEALPix pixel determination: {e}") from e

        # --- Process Chunks ---       
        all_results = []
        for i, pix_id in enumerate(pixels):
            logger.info(f"Processing chunk {i+1}/{len(pixels)} (Pixel ID: {pix_id})...")
            try:
                # Get pixel boundaries and approximate with bounding box
                corners = hp.boundaries_skycoord([pix_id])[0]
                # Simple bounding box (handle RA wrap)
                ra_corners = corners.ra.wrap_at(180 * u.deg).deg
                dec_corners = corners.dec.deg
                box_params = {
                    "ra_min": np.min(ra_corners),
                    "ra_max": np.max(ra_corners),
                    "dec_min": np.min(dec_corners),
                    "dec_max": np.max(dec_corners),
                }
                # Convert back to 0-360 range if needed? ADQL BOX might handle wrap?
                # Let's assume ADQL BOX handles wrap/center correctly for now.
                # TODO: Verify ADQL BOX behavior with different services.
                logger.debug(f"Pixel {pix_id} Bounding Box: {box_params}")

                # Fetch chunk for catalogue 1
                logger.debug(f"Fetching chunk {i+1} for {cat_name1}...")
                df1_chunk = self._fetch_remote_catalogue(
                    config1, columns=params.get("columns1"), box_params=box_params
                )
                if df1_chunk is None or df1_chunk.empty:
                    logger.info(f"Skipping chunk {i+1}: No data found for {cat_name1} in this area.")
                    continue
                logger.debug(f"Fetched {len(df1_chunk)} rows for {cat_name1} chunk {i+1}.")

                # Fetch chunk for catalogue 2
                logger.debug(f"Fetching chunk {i+1} for {cat_name2}...")
                df2_chunk = self._fetch_remote_catalogue(
                    config2, columns=params.get("columns2"), box_params=box_params
                )
                if df2_chunk is None or df2_chunk.empty:
                    logger.info(f"Skipping chunk {i+1}: No data found for {cat_name2} in this area.")
                    continue
                logger.debug(f"Fetched {len(df2_chunk)} rows for {cat_name2} chunk {i+1}.")

                # Match the chunks locally
                logger.info(f"Matching chunk {i+1} ({len(df1_chunk)} vs {len(df2_chunk)} rows)..." )
                # Create temporary configs pointing to the dataframes
                config1_chunk = {**config1, "_input_dataframe": df1_chunk, "access_method": "file_system"}
                config2_chunk = {**config2, "_input_dataframe": df2_chunk, "access_method": "file_system"}
                config1_chunk.pop("_input_path", None)
                config2_chunk.pop("_input_path", None)

                # Execute local stilts match on the chunk data
                chunk_result_df = self._execute_local_stilts(config1_chunk, config2_chunk, **params)

                if chunk_result_df is not None and not chunk_result_df.empty:
                    logger.info(f"Chunk {i+1} matched {len(chunk_result_df)} pairs.")
                    all_results.append(chunk_result_df)
                else:
                     logger.info(f"Chunk {i+1} yielded no matches.")

            except (CrossMatchError, TapError, StiltsError, ConnectionError, Timeout, RequestException) as e:
                logger.warning(
                    f"Failed to process chunk {i+1} (Pixel {pix_id}): {e}. Skipping chunk.", exc_info=True # Log stack trace for warnings too
                )
                continue # Skip to the next chunk
            except Exception as e:
                logger.error(
                    f"Unexpected error processing chunk {i+1} (Pixel {pix_id}): {e}", exc_info=True
                )
                continue # Skip to the next chunk

        # --- Combine Results ---       
        if not all_results:
            logger.warning("Spatial chunked match resulted in no matches across all chunks.")
            return pd.DataFrame()
        else:
            logger.info(f"Concatenating results from {len(all_results)} chunks...")
            try:
                final_df = pd.concat(all_results, ignore_index=True)
                # TODO: Add deduplication step? Matches near chunk borders might be duplicated.
                logger.info(f"Spatial chunked match completed. Total pairs found: {len(final_df)}")
                return final_df
            except MemoryError as e:
                logger.error("Memory error combining chunked match results.")
                raise CrossMatchError("Memory error combining chunked match results") from e
            except Exception as e:
                logger.error(f"Error combining chunked match results: {e}", exc_info=True)
                raise CrossMatchError(f"Error combining chunked match results: {e}") from e

    def _execute_chunked_match(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        chunk_rows: int = 1_000_000,  # Default chunk size
        chunk_input_index: int = 1,  # Which input to chunk (1 or 2)
        **params,
    ) -> pd.DataFrame:
        """Executes a local crossmatch by chunking one of the inputs.

        Args:
            config1: Resolved configuration for the first input (local file or DataFrame).
            config2: Resolved configuration for the second input (local file or DataFrame).
            chunk_rows: The number of rows to process per chunk.
            chunk_input_index: Which input to chunk (1 or 2). Currently simple selection,
                               could be determined dynamically based on size in the future.
            params: Additional parameters passed to _execute_local_stilts for each chunk
                    (e.g., radius_arcsec, matcher, join_type).

        Returns:
            A pandas DataFrame containing the concatenated results from all chunks.

        Raises:
            CrossMatchError: If inputs are not local, chunking fails, or a sub-match fails.
            ValueError: If chunk_input_index is invalid.
        """
        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")
        logger.info(
            f"Executing Chunked Local Match strategy between: {cat_name1} and {cat_name2}, chunking input {chunk_input_index}"
        )

        # --- Validate Inputs are Local ---        
        if (
            config1.get("access_method") != "file_system"
            or config2.get("access_method") != "file_system"
        ):
            raise CrossMatchError(
                "Chunked match requires both inputs to be local (file or DataFrame)."
            )

        # --- Determine which config to chunk and which is static ---
        if chunk_input_index == 1:
            chunk_config = config1
            static_config = config2
            chunk_cat_name = cat_name1
            static_cat_name = cat_name2
        elif chunk_input_index == 2:
            chunk_config = config2
            static_config = config1
            chunk_cat_name = cat_name2
            static_cat_name = cat_name1
        else:
            raise ValueError("chunk_input_index must be 1 or 2")

        chunk_input_path = chunk_config.get("_input_path")
        chunk_input_df = chunk_config.get("_input_dataframe")
        
        # The static input needs to be loaded fully into memory if it's a file
        static_input_df = None
        if static_config.get("_input_dataframe") is not None:
            static_input_df = static_config.get("_input_dataframe")
        elif static_config.get("_input_path") is not None:
            logger.info(f"Loading static input catalogue '{static_cat_name}' into memory...")
            try:
                static_input_df = self._load_local_catalogue(static_config.get("_input_path"))
                static_config["_input_dataframe"] = static_input_df # Store for reuse by stilts
                static_config.pop("_input_path", None)
            except (FileNotFoundError, MemoryError, CrossMatchError) as e:
                 logger.error(f"Failed to load static catalogue '{static_cat_name}': {e}")
                 raise
        else:
             raise CrossMatchError(f"Static input catalogue '{static_cat_name}' has no path or DataFrame.")
        
        if static_input_df is None or static_input_df.empty:
            logger.warning(f"Static input catalogue '{static_cat_name}' is empty. Returning empty result.")
            return pd.DataFrame()
        logger.info(f"Static catalogue '{static_cat_name}' ({len(static_input_df)} rows) ready.")

        all_results = []
        chunk_num = 0

        try:
            # --- Handle Chunking from DataFrame ---            
            if chunk_input_df is not None:
                logger.info(
                    f"Chunking input DataFrame '{chunk_cat_name}' ({len(chunk_input_df)} rows) in chunks of {chunk_rows}..."
                )
                for i in range(0, len(chunk_input_df), chunk_rows):
                    chunk_num += 1
                    chunk_df = chunk_input_df.iloc[i : i + chunk_rows]
                    logger.info(
                        f"Processing chunk {chunk_num} ({len(chunk_df)} rows) for '{chunk_cat_name}' vs '{static_cat_name}'"
                    )

                    # Prepare configs for this chunk's match
                    current_chunk_config = chunk_config.copy()
                    current_chunk_config["_input_dataframe"] = chunk_df
                    current_chunk_config.pop("_input_path", None)

                    # Determine argument order for _execute_local_stilts
                    if chunk_input_index == 1:
                        conf1, conf2 = current_chunk_config, static_config
                    else:
                        conf1, conf2 = static_config, current_chunk_config
                    try:
                        result_chunk = self._execute_local_stilts(conf1, conf2, **params)
                        if result_chunk is not None and not result_chunk.empty:
                            all_results.append(result_chunk)
                            logger.debug(f"Chunk {chunk_num} match yielded {len(result_chunk)} results.")
                        else:
                            logger.debug(f"Chunk {chunk_num} yielded no matches.")
                    except (StiltsError, CrossMatchError) as e:
                         logger.warning(f"Failed to match chunk {chunk_num}: {e}. Skipping.", exc_info=True)
                         continue # Continue to next chunk

            # --- Handle Chunking from File ---            
            elif chunk_input_path:
                input_path = Path(chunk_input_path)
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file for chunking not found: {input_path}")
                
                logger.info(
                    f"Chunking input file '{chunk_cat_name}' ({input_path}) in chunks of {chunk_rows}..."
                )
                file_format = input_path.suffix.lower()

                # Use appropriate pandas reader with chunksize                
                reader = None
                if file_format == ".csv":
                    # TODO: Detect/handle CSV dialect, headers, comments etc.
                    reader = pd.read_csv(input_path, chunksize=chunk_rows, low_memory=False)
                elif file_format == ".parquet":
                    # Actual chunking for parquet might require pyarrow.dataset or iterative reading
                    logger.warning(
                        "Direct file chunking for Parquet not fully implemented; reading full file then slicing. This might consume significant memory."
                    )
                    # For now, read whole and then treat as DataFrame chunking - INEFFICIENT!
                    full_df = self._load_local_catalogue(input_path) # Use existing loader
                    # Re-run the DataFrame chunking logic (could refactor this)
                    for i in range(0, len(full_df), chunk_rows):
                        chunk_num += 1
                        chunk_df = full_df.iloc[i : i + chunk_rows]
                        logger.info(
                            f"Processing chunk {chunk_num} ({len(chunk_df)} rows) for '{chunk_cat_name}' vs '{static_cat_name}'"
                        )
                        current_chunk_config = chunk_config.copy()
                        current_chunk_config["_input_dataframe"] = chunk_df
                        current_chunk_config.pop("_input_path", None)
                        if chunk_input_index == 1:
                            conf1, conf2 = current_chunk_config, static_config
                        else:
                            conf1, conf2 = static_config, current_chunk_config
                        try:
                            result_chunk = self._execute_local_stilts(conf1, conf2, **params)
                            if result_chunk is not None and not result_chunk.empty:
                                all_results.append(result_chunk)
                                logger.debug(
                                    f"Chunk {chunk_num} match yielded {len(result_chunk)} results."
                                )
                            else:
                                logger.debug(f"Chunk {chunk_num} yielded no matches.")
                        except (StiltsError, CrossMatchError) as e:
                            logger.warning(f"Failed to match chunk {chunk_num}: {e}. Skipping.", exc_info=True)
                            continue
                    reader = None # Prevent entering the reader loop below
                elif file_format == ".fits":
                    # TODO: Handle FITS chunking - Astropy tables might need specific handling
                    logger.warning("Chunking directly from FITS files not yet implemented.")
                    # Read full file as fallback? Defeats purpose of chunking...
                    full_df = self._load_local_catalogue(input_path)
                    logger.warning(f"Read full FITS file ({len(full_df)} rows) for chunking fallback.")
                    for i in range(0, len(full_df), chunk_rows):
                        chunk_num += 1
                        chunk_df = full_df.iloc[i : i + chunk_rows]
                        logger.info(
                            f"Processing chunk {chunk_num} ({len(chunk_df)} rows) for '{chunk_cat_name}' vs '{static_cat_name}'"
                        )
                        current_chunk_config = chunk_config.copy()
                        current_chunk_config["_input_dataframe"] = chunk_df
                        current_chunk_config.pop("_input_path", None)
                        if chunk_input_index == 1:
                            conf1, conf2 = current_chunk_config, static_config
                        else:
                            conf1, conf2 = static_config, current_chunk_config
                        try:
                            result_chunk = self._execute_local_stilts(conf1, conf2, **params)
                            if result_chunk is not None and not result_chunk.empty:
                                all_results.append(result_chunk)
                                logger.debug(
                                    f"Chunk {chunk_num} match yielded {len(result_chunk)} results."
                                )
                            else:
                                logger.debug(f"Chunk {chunk_num} yielded no matches.")
                        except (StiltsError, CrossMatchError) as e:
                            logger.warning(f"Failed to match chunk {chunk_num}: {e}. Skipping.", exc_info=True)
                            continue
                    reader = None
                    # raise NotImplementedError("Chunking from FITS files needs implementation.")
                else:
                    raise CrossMatchError(f"Unsupported file format for chunking: {file_format}")

                # Process chunks from CSV reader (only if reader was created)
                if reader:
                    for chunk_df in reader:
                        chunk_num += 1
                        logger.info(
                            f"Processing chunk {chunk_num} ({len(chunk_df)} rows) for '{chunk_cat_name}' vs '{static_cat_name}'"
                        )
                        current_chunk_config = chunk_config.copy()
                        current_chunk_config["_input_dataframe"] = chunk_df
                        current_chunk_config.pop("_input_path", None)

                        if chunk_input_index == 1:
                            conf1, conf2 = current_chunk_config, static_config
                        else:
                            conf1, conf2 = static_config, current_chunk_config
                        
                        try:
                            result_chunk = self._execute_local_stilts(conf1, conf2, **params)
                            if result_chunk is not None and not result_chunk.empty:
                                all_results.append(result_chunk)
                                logger.debug(
                                    f"Chunk {chunk_num} match yielded {len(result_chunk)} results."
                                )
                            else:
                                logger.debug(f"Chunk {chunk_num} yielded no matches.")
                        except (StiltsError, CrossMatchError) as e:
                            logger.warning(f"Failed to match chunk {chunk_num}: {e}. Skipping.", exc_info=True)
                            continue

            else:
                raise CrossMatchError(
                    f"Cannot chunk input {chunk_input_index} ('{chunk_cat_name}'): No DataFrame or file path provided in config."
                )

            # --- Combine Results ---
            if not all_results:
                logger.warning("Chunked match completed, but produced no results.")
                return pd.DataFrame() # Return empty DataFrame

            logger.info(f"Concatenating results from {chunk_num} chunks...")
            final_result_df = pd.concat(all_results, ignore_index=True)
            logger.info(f"Chunked match completed. Final result has {len(final_result_df)} rows.")
            return final_result_df

        except (StiltsError, FileNotFoundError, ValueError, MemoryError) as e:
            logger.error(f"Error during chunked match execution: {e}", exc_info=True)
            raise CrossMatchError(f"Chunked match failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during chunked match: {e}", exc_info=True)
            raise CrossMatchError(f"Unexpected error in chunked match: {e}") from e

    def _fetch_remote_catalogue(
        self,
        config: Dict[str, Any],
        columns: Optional[List[str]] = None,
        query_constraints: Optional[str] = None, # e.g., "WHERE phot_g_mean_mag < 18"
        cone_params: Optional[Dict[str, float]] = None, # {ra, dec, radius_deg}
        box_params: Optional[Dict[str, float]] = None, # {ra_min, ra_max, dec_min, dec_max}
        # TODO: Add support for polygon constraints later?
    ) -> pd.DataFrame:
        """Fetches a remote catalogue based on its configuration.

        Currently supports TAP.
        Applies epoch propagation if needed based on target_epoch.

        Args:
            config: Resolved configuration for the remote catalogue.
            columns: List of specific columns to fetch (overrides default_columns).
            query_constraints: Additional ADQL WHERE clause constraints.
            cone_params: Parameters for a cone search (ra, dec, radius_deg).
            box_params: Parameters for a box search (ra_min, ra_max, dec_min, dec_max).

        Returns:
            DataFrame containing the fetched catalogue data.

        Raises:
            CrossMatchError: If fetching fails or the access method is unsupported.
            TapError: For TAP-specific issues.
        """
        cat_name = config.get("_catalogue_name", "remote_catalogue")
        access_method = config.get("access_method")
        logger.info(f"Fetching remote catalogue: {cat_name} via {access_method}")

        if access_method == "tap":
            tap_url = config.get("access_url")
            table_name = config.get("access_identifier")
            if not tap_url or not table_name:
                raise CrossMatchError(
                    f"TAP configuration missing 'access_url' or 'access_identifier' for {cat_name}"
                )

            # Determine columns to fetch
            cols_to_fetch_raw = columns or config.get("default_columns")
            if not cols_to_fetch_raw:
                # Fetch all columns if none specified (use with caution)
                logger.warning(
                    f"No columns specified for {cat_name}, fetching all columns (*). This might be slow."
                )
                select_cols = "*"
            else:
                # Ensure essential columns (RA, Dec, errors, PM, epoch if available) are included
                essential_cols = {
                    config.get("ra_column"),
                    config.get("dec_column"),
                    config.get("ra_err_column"),
                    config.get("dec_err_column"),
                    config.get("corr_column"),
                    config.get("pm_ra_column"), # Need to add pm_ra/pm_dec to YAML where applicable
                    config.get("pm_dec_column"),
                    config.get("epoch_column"), # Need to add epoch_column to YAML where applicable (e.g., for Gaia)
                }
                # Filter out None values from essential_cols
                essential_cols = {c for c in essential_cols if c is not None}
                # Use set union to combine requested and essential columns
                all_cols_set = set(cols_to_fetch_raw) | essential_cols
                select_cols = ", ".join(f'"{c}"' for c in sorted(list(all_cols_set)))

            # Build ADQL query
            adql_query = f"SELECT {select_cols} FROM {table_name}"
            where_clauses = []
            if query_constraints:
                # Assume query_constraints already starts with WHERE or AND/OR if needed
                # Basic check to add WHERE if not present
                # This logic is tricky, better to require constraints to be valid standalone predicates
                where_clauses.append(f"({query_constraints})") # Wrap user constraints

            # Add spatial constraints
            ra_col = config.get("ra_column")
            dec_col = config.get("dec_column")
            if not ra_col or not dec_col:
                 logger.warning(f"RA/Dec columns not defined for {cat_name}, cannot apply spatial constraints.")
            else:
                if cone_params:
                    if all(k in cone_params for k in ["ra", "dec", "radius_deg"]):
                        clause = (
                            f"CONTAINS(POINT('ICRS', \"{ra_col}\", \"{dec_col}\"), "
                            f"CIRCLE('ICRS', {cone_params['ra']}, {cone_params['dec']}, {cone_params['radius_deg']})) = 1"
                        )
                        where_clauses.append(clause)
                    else:
                        logger.warning("Cone search parameters incomplete, skipping cone constraint.")
                elif box_params: # Prioritize cone if both provided?
                     if all(k in box_params for k in ["ra_min", "ra_max", "dec_min", "dec_max"]):
                        # ADQL BOX function: BOX('ICRS', ra_cen, dec_cen, width, height)
                        # Need to calculate center and width/height from min/max
                        # Handle RA wrap-around carefully!
                        # Simplified: Assume no wrap for now. TODO: Add wrap-around logic.
                        ra_cen = (box_params['ra_min'] + box_params['ra_max']) / 2
                        dec_cen = (box_params['dec_min'] + box_params['dec_max']) / 2
                        width = box_params['ra_max'] - box_params['ra_min']
                        height = box_params['dec_max'] - box_params['dec_min']
                        if width < 0 or height < 0:
                            logger.warning("Invalid box dimensions (max < min?), skipping box constraint.")
                        else:
                            clause = (
                                f"CONTAINS(POINT('ICRS', \"{ra_col}\", \"{dec_col}\"), "
                                f"BOX('ICRS', {ra_cen}, {dec_cen}, {width}, {height})) = 1"
                            )
                            where_clauses.append(clause)
                     else:
                        logger.warning("Box search parameters incomplete, skipping box constraint.")

            # Combine WHERE clauses
            if where_clauses:
                adql_query += " WHERE " + " AND ".join(where_clauses)

            logger.info(f"Executing ADQL query on {tap_url}:\n{adql_query}")

            try:
                # Get authenticated TAP service if needed
                auth_info = self.auth_config.get(config.get("_archive_name"))
                tap_kwargs = {}
                if auth_info:
                    tap_kwargs["user"] = auth_info.get("user")
                    tap_kwargs["password"] = auth_info.get("password")
                    # TODO: Add other auth methods (e.g., token) if needed by TAP services

                tap_service = get_tap_service(tap_url, **tap_kwargs)

                # Execute query using tap.py function
                # Fetch relevant params for execute_tap_query
                exec_params = {k: v for k, v in params.items() if k in ['retry_delay', 'timeout', 'max_retries', 'verbose']}
                result_df = execute_tap_query(
                    tap_service, adql_query, **exec_params
                )
                logger.info(f"Successfully fetched {len(result_df)} rows for {cat_name}.")
                return result_df

            except (TapError, ConnectionError, Timeout) as e:
                logger.error(f"Failed to fetch {cat_name} from {tap_url}: {e}", exc_info=True)
                raise  # Re-raise TAP/network errors
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching {cat_name} from {tap_url}: {e}", exc_info=True
                )
                raise CrossMatchError(
                    f"Unexpected error fetching remote catalogue {cat_name}: {e}"
                ) from e

        elif access_method == "cds_xmatch":
            # TODO: Implement fetching/querying via CDS X-Match service if needed
            # This is usually used for crossmatching directly, not fetching full tables.
            raise NotImplementedError("Fetching via cds_xmatch service is not implemented.")
        else:
            raise CrossMatchError(
                f"Unsupported access method '{access_method}' for remote catalogue {cat_name}"
            )

    def _execute_cds_xmatch_local_remote(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        **params,
    ) -> pd.DataFrame:
        """Executes a crossmatch using the CDS XMatch service (astroquery).

        Assumes one input is local (file or DataFrame) and the other is a remote
        catalogue configured with access_method: cds_xmatch.

        Args:
            config1: Resolved configuration for the first input.
            config2: Resolved configuration for the second input.
            params: Dictionary of execution parameters, must include 'radius_arcsec'.

        Returns:
            A pandas DataFrame with the crossmatch results.

        Raises:
            CrossMatchError: If inputs are not one local/one CDS, astroquery fails,
                             or required parameters are missing.
            ImportError: If astroquery is not installed.
        """
        logger.info("Executing CDS XMatch (Local vs Remote) strategy...")

        # --- Determine Local and Remote Configs ---
        if config1.get("access_method") == "file_system":
            local_config = config1
            remote_config = config2
        elif config2.get("access_method") == "file_system":
            local_config = config2
            remote_config = config1
        else:
            raise CrossMatchError("CDS XMatch strategy requires one local and one remote input.")

        if remote_config.get("access_method") != "cds_xmatch":
            raise CrossMatchError(
                f"Remote input {remote_config.get('_catalogue_name')} is not configured for CDS XMatch."
            )

        local_cat_name = local_config.get("_catalogue_name", "local_input")
        remote_cat_name = remote_config.get("_catalogue_name", "remote_cds_input")
        logger.info(f"Matching local '{local_cat_name}' against remote CDS '{remote_cat_name}'")

        # --- Validate Parameters ---
        if "radius_arcsec" not in params:
            raise ValueError("Missing required parameter 'radius_arcsec' for CDS XMatch.")
        radius_arcsec = params["radius_arcsec"]

        # --- Prepare Local Input Data ---       
        local_table = None
        local_input_df = local_config.get("_input_dataframe")
        local_input_path = local_config.get("_input_path")

        if local_input_df is not None:
            logger.debug(f"Using DataFrame for local input '{local_cat_name}'")
            # Convert DataFrame to Astropy Table for astroquery
            try:
                local_table = Table.from_pandas(local_input_df)
            except Exception as e:
                raise CrossMatchError(f"Failed to convert local DataFrame to Astropy Table: {e}") from e
        elif local_input_path is not None:
            logger.debug(f"Using file path '{local_input_path}' for local input '{local_cat_name}'")
            # Astroquery XMatch can often take file paths directly
            local_table = str(local_input_path)
            # Ensure file actually exists before passing path
            if not Path(local_table).is_file():
                 raise FileNotFoundError(f"Local input file for CDS XMatch not found: {local_table}")
        else:
            raise CrossMatchError(f"Local input '{local_cat_name}' has no DataFrame or file path.")

        # --- Get Remote Catalogue Identifier ---
        # Use the 'access_identifier' which should be the VizieR table name for CDS
        vizier_table_id = remote_config.get("access_identifier")
        if not vizier_table_id:
            raise CrossMatchError(f"Remote CDS config '{remote_cat_name}' missing 'access_identifier'.")

        # --- Execute CDS XMatch ---       
        try:
            logger.info(
                f"Submitting job to CDS XMatch: {local_cat_name} vs {vizier_table_id} (Radius: {radius_arcsec} arcsec)"
            )
            xmatch = XMatch()
            # Need RA/Dec column names from the *local* table
            ra_col_local = local_config.get("ra_column", "ra") # Default to ra/dec if not in temp config
            dec_col_local = local_config.get("dec_column", "dec")

            # Use xmatch_local for file paths or astropy Tables
            result_table = xmatch.query_async(
                cat1=local_table,
                cat2=f"vizier:{vizier_table_id}",
                max_distance=radius_arcsec * u.arcsec,
                colRA1=ra_col_local,
                colDec1=dec_col_local,
            )

            if result_table is None:
                logger.warning("CDS XMatch query returned no results.")
                return pd.DataFrame()
            
            logger.info(f"CDS XMatch successful, received {len(result_table)} matches.")
            # Convert result Astropy Table to DataFrame
            return result_table.to_pandas()

        except ImportError:
             logger.error("'astroquery' library is required for CDS XMatch.")
             raise CrossMatchError(
                 "'astroquery' library is required for cds_xmatch_local_remote strategy. Please install it."
             )
        except FileNotFoundError as e:
             # Catch if a local file path was provided but not found by astroquery
             logger.error(f"Local input file not found by astroquery: {e}")
             raise CrossMatchError(f"Local input file not found by astroquery: {local_table}") from e
        except Exception as e:
            # Catch potential astroquery errors (connection, query failure, etc.)
            logger.error(f"CDS XMatch query failed: {e}", exc_info=True)
            raise CrossMatchError(f"Error during CDS XMatch query: {e}") from e

    def _apply_epoch_propagation(
        self, config: Dict[str, Any], df: pd.DataFrame, target_epoch: float
    ) -> pd.DataFrame:
        """Applies epoch propagation if configuration and data allow."""
        ra_col = config.get("ra_column")
        dec_col = config.get("dec_column")
        pm_ra_col = config.get("pm_ra_column")
        pm_dec_col = config.get("pm_dec_column")
        epoch_col = config.get("epoch_column") # The column containing the source epoch
        current_epoch = config.get("epoch") # The reference epoch from YAML

        if not all([ra_col, dec_col, pm_ra_col, pm_dec_col, epoch_col, current_epoch]):
            logger.warning(
                f"Catalogue {config.get('_catalogue_name')} lacks required columns/config "
                f"(ra/dec/pm_ra/pm_dec/epoch_col/epoch) for propagation. Using original coordinates."
            )
            # Add ra_propagated/dec_propagated columns with original values
            df_out = df.copy()
            if ra_col and "ra_propagated" not in df_out.columns:
                 df_out["ra_propagated"] = df_out[ra_col]
            if dec_col and "dec_propagated" not in df_out.columns:
                 df_out["dec_propagated"] = df_out[dec_col]
            return df_out

        # Add the fixed epoch from YAML as a column if epoch_col doesn't exist in df
        if epoch_col not in df.columns:
            logger.debug(f"Adding fixed epoch column '{epoch_col}' = {current_epoch} for propagation.")
            df[epoch_col] = current_epoch

        logger.info(
            f"Applying epoch propagation for {config.get('_catalogue_name')} to target epoch {target_epoch}"
        )
        try:
            propagated_df = propagate_coordinates_to_epoch(
                df,
                ra_col=ra_col,
                dec_col=dec_col,
                pm_ra_col=pm_ra_col,
                pm_dec_col=pm_dec_col,
                epoch_col=epoch_col, # Use the column name specified in YAML
                target_epoch=target_epoch,
            )
            return propagated_df
        except (KeyError, ValueError, TypeError) as e:
             logger.error(f"Error applying epoch propagation: {e}", exc_info=True)
             raise CrossMatchError(f"Error during epoch propagation: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during epoch propagation: {e}", exc_info=True)
            raise CrossMatchError(f"Unexpected error during epoch propagation: {e}") from e

    def _execute_remote_tap_chunked_spatial_join(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        nside: int = 32, # Default HEALPix nside for chunking
        **params,
    ) -> pd.DataFrame:
        """Executes remote-remote spatial match by chunking the sky area using HEALPix.

        Performs a spatial JOIN query on the remote TAP service for each chunk
        and downloads the results.

        Requires 'ra', 'dec', 'radius_arcsec' in params to define the area.
        Assumes both catalogues are on the same TAP service.
        """
        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")
        logger.info(
            f"Executing Remote TAP Chunked Spatial JOIN ({cat_name1} vs {cat_name2}) using HEALPix nside={nside}"
        )

        # --- Validate and Extract Parameters ---
        if not all(k in params for k in ["ra", "dec", "radius_arcsec"]):
            raise ValueError(
                "Missing required parameters 'ra', 'dec', 'radius_arcsec' for spatial chunking."
            )
        try:
            center_coord = SkyCoord(ra=params["ra"] * u.deg, dec=params["dec"] * u.deg, frame="icrs")
            search_radius = params["radius_arcsec"] * u.arcsec
            radius_deg = search_radius.to(u.deg).value # Radius for JOIN ON clause
        except (ValueError, TypeError, u.UnitConversionError) as e:
            raise CrossMatchError(f"Invalid spatial parameters (ra/dec/radius): {e}") from e

        # --- Get Service and Table Info ---
        service_config = config1 # Assumes same service
        tap_url = service_config.get("access_url")
        if not tap_url:
            raise CrossMatchError("Missing access_url in service config for remote chunked join.")

        table1 = config1.get("access_identifier")
        table2 = config2.get("access_identifier")
        if not table1 or not table2:
             raise CrossMatchError("Missing access_identifier for one or both catalogues in remote chunked join.")

        # --- Determine HEALPix Pixels ---
        try:
            hp = HEALPix(nside=nside, order="nested", frame="icrs")
            # Add buffer to cone search radius? Maybe 1/sqrt(NPIX)*60 arcmin? Safer to search slightly larger area.
            buffer = hp.pixel_resolution.to(u.arcsec) * 1.5 # Example buffer
            logger.debug(f"Using search radius {search_radius.arcsec:.2f} + buffer {buffer.arcsec:.2f} arcsec for pixel identification.")
            pixels = hp.cone_search_skycoord(center_coord, search_radius + buffer)
            logger.info(f"Identified {len(pixels)} HEALPix pixels (nside={nside}) covering the search area + buffer.")
            if len(pixels) == 0:
                 logger.warning("No HEALPix pixels found in the search cone. Returning empty result.")
                 return pd.DataFrame()
        except ImportError:
            logger.error("'astropy-healpix' library is required for chunked matching.")
            raise CrossMatchError(
                "'astropy-healpix' library is required for this strategy. Please install it."
            )
        except Exception as e:
            logger.error(f"Error during HEALPix pixel determination: {e}", exc_info=True)
            raise CrossMatchError(f"Error during HEALPix pixel determination: {e}") from e

        # --- Define Aliases and Column Selection (similar to _execute_remote_join_match) ---
        alias1 = "t1"
        alias2 = "t2"
        cols1_req_raw = params.get("columns1") or config1.get("default_columns")
        cols2_req_raw = params.get("columns2") or config2.get("default_columns")
        essential_cols1 = {config1.get(c) for c in ["ra_column", "dec_column"] if config1.get(c)}
        essential_cols2 = {config2.get(c) for c in ["ra_column", "dec_column"] if config2.get(c)}
        cols1_req = list((set(cols1_req_raw) if cols1_req_raw else set()) | essential_cols1)
        cols2_req = list((set(cols2_req_raw) if cols2_req_raw else set()) | essential_cols2)
        cols1_req = [c for c in cols1_req if c is not None]
        cols2_req = [c for c in cols2_req if c is not None]
        cols1_select = [f'{alias1}."{c}" AS {alias1}_{c}' for c in set(cols1_req)]
        cols2_select = [f'{alias2}."{c}" AS {alias2}_{c}' for c in set(cols2_req)]
        if "*" in (cols1_req_raw or []): cols1_select = [f"{alias1}.*"]
        if "*" in (cols2_req_raw or []): cols2_select = [f"{alias2}.*"]
        select_clause = ", ".join(cols1_select + cols2_select)
        if not select_clause: select_clause = f"{alias1}.*, {alias2}.*"

        # --- Get RA/Dec Column Names ---
        ra_col1 = config1.get("ra_column")
        dec_col1 = config1.get("dec_column")
        ra_col2 = config2.get("ra_column")
        dec_col2 = config2.get("dec_column")
        if not ra_col1 or not dec_col1 or not ra_col2 or not dec_col2:
            raise CrossMatchError("Missing RA/Dec column configuration for remote chunked spatial join.")

        # --- Process Chunks ---       
        all_results = []
        tap_service = None # Initialize TAP service connection

        for i, pix_id in enumerate(pixels):
            logger.info(f"Processing chunk {i+1}/{len(pixels)} (Pixel ID: {pix_id})...")
            try:
                # Get pixel boundaries and approximate with bounding box
                corners = hp.boundaries_skycoord([pix_id])[0]
                ra_corners = corners.ra.wrap_at(180 * u.deg).deg
                dec_corners = corners.dec.deg
                ra_min, ra_max = np.min(ra_corners), np.max(ra_corners)
                dec_min, dec_max = np.min(dec_corners), np.max(dec_corners)
                # ADQL BOX: center and width/height
                # TODO: Handle RA wrap-around for BOX definition more robustly if needed
                ra_cen = (ra_min + ra_max) / 2
                dec_cen = (dec_min + dec_max) / 2
                width = ra_max - ra_min
                height = dec_max - dec_min
                # Ensure width is positive, handle wrap near RA=0/360 if min>max after wrap
                if width < 0: width += 360
                if width <= 0 or height <= 0: # Skip degenerate pixels
                    logger.warning(f"Skipping degenerate pixel {pix_id} (width={width}, height={height}).")
                    continue
                
                adql_box = f"BOX('ICRS', {ra_cen}, {dec_cen}, {width}, {height})"
                logger.debug(f"Pixel {pix_id} ADQL Box: {adql_box}")

                # --- Construct ADQL Query for the Chunk ---
                # Join ON distance, WHERE both points are contained in the pixel box
                adql_query = f"""
                SELECT {select_clause}
                FROM {table1} AS {alias1}
                INNER JOIN {table2} AS {alias2}
                ON DISTANCE({alias1}."{ra_col1}", {alias1}."{dec_col1}", {alias2}."{ra_col2}", {alias2}."{dec_col2}") <= {radius_deg}
                WHERE CONTAINS(POINT('ICRS', {alias1}."{ra_col1}", {alias1}."{dec_col1}"), {adql_box}) = 1
                  AND CONTAINS(POINT('ICRS', {alias2}."{ra_col2}", {alias2}."{dec_col2}"), {adql_box}) = 1
                """
                # Note: Some TAP services might prefer INTERSECTS(REGION, BOX)
                # Note: Applying spatial constraint in WHERE after JOIN might be slow on some systems.
                # Optimizations depend on the specific TAP service.

                logger.debug(f"Chunk {i+1} ADQL Query:\n{adql_query}")

                # --- Execute Query for the Chunk ---
                if tap_service is None:
                    # Get authenticated TAP service on first chunk
                    auth_info = self.auth_config.get(config1.get("_archive_name"))
                    tap_kwargs = {}
                    if auth_info:
                        if "user" in auth_info and "password" in auth_info:
                            tap_kwargs["user"] = auth_info.get("user")
                            tap_kwargs["password"] = auth_info.get("password")
                    tap_service = get_tap_service(tap_url, **tap_kwargs)

                # Fetch relevant params for execute_tap_query
                exec_params = {k: v for k, v in params.items() if k in ['retry_delay', 'timeout', 'max_retries', 'verbose']}
                chunk_result_df = execute_tap_query(
                    tap_service, adql_query, **exec_params
                )

                if chunk_result_df is not None and not chunk_result_df.empty:
                    logger.info(f"Chunk {i+1} TAP JOIN yielded {len(chunk_result_df)} pairs.")
                    all_results.append(chunk_result_df)
                else:
                     logger.info(f"Chunk {i+1} yielded no results from TAP JOIN.")

            except (TapError, ConnectionError, Timeout, RequestException) as e:
                logger.warning(
                    f"TAP query failed for chunk {i+1} (Pixel {pix_id}): {e}. Skipping chunk.", exc_info=True
                )
                continue # Skip to the next chunk
            except Exception as e:
                logger.error(
                    f"Unexpected error processing chunk {i+1} (Pixel {pix_id}): {e}", exc_info=True
                )
                continue # Skip to the next chunk

        # --- Combine Results ---       
        if not all_results:
            logger.warning("Remote TAP Chunked Spatial Join resulted in no matches across all chunks.")
            return pd.DataFrame()
        else:
            logger.info(f"Concatenating results from {len(all_results)} chunks...")
            try:
                final_df = pd.concat(all_results, ignore_index=True)
                # Optional: Deduplicate based on primary keys if overlaps might cause issues?
                # Depends on how precise the BOX/CONTAINS is vs the JOIN radius.
                # E.g., final_df = final_df.drop_duplicates(subset=[f'{alias1}_ID_COL', f'{alias2}_ID_COL'])
                logger.info(f"Remote TAP Chunked Spatial Join completed. Total pairs found: {len(final_df)}")
                return final_df
            except MemoryError as e:
                logger.error("Memory error combining chunked TAP JOIN results.")
                raise CrossMatchError("Memory error combining chunked TAP JOIN results") from e
            except Exception as e:
                logger.error(f"Error combining chunked TAP JOIN results: {e}", exc_info=True)
                raise CrossMatchError(f"Error combining chunked TAP JOIN results: {e}") from e
