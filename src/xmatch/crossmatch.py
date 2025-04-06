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
from .astro_utils import apply_epoch_propagation # Import the new function

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
        """Writes a DataFrame to a specified output file path.

        Supports Parquet (.parquet), FITS (.fits, .fit), and CSV (.csv).
        Determines format based on file extension.
        """
        output_path = Path(output_path)
        output_format = output_path.suffix.lower()
        logger.info(f"Writing output ({output_format}) to: {output_path}")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_format == ".parquet":
                df.to_parquet(output_path, compression="snappy", index=False)
            elif output_format in [".fits", ".fit"]:
                try:
                    # Convert DataFrame to Astropy Table for FITS writing
                    table = Table.from_pandas(df)
                    table.write(output_path, format="fits", overwrite=True)
                except ImportError:
                    logger.error("Writing FITS requires 'astropy' library.")
                    raise CrossMatchError("Writing FITS requires 'astropy' library.")
                except Exception as e:
                    logger.error(f"Error writing FITS file {output_path}: {e}")
                    raise CrossMatchError(f"Error writing FITS file {output_path}: {e}") from e
            elif output_format == ".csv":
                # TODO: Add options for CSV writing (sep, header, quoting)?
                df.to_csv(output_path, index=False)
            else:
                # Default to Parquet if extension is unknown/unsupported?
                logger.warning(f"Unsupported output file extension '{output_format}'. Defaulting to Parquet.")
                # Change extension for the actual write
                output_path_parquet = output_path.with_suffix(".parquet")
                logger.warning(f"Actual output file will be: {output_path_parquet}")
                df.to_parquet(output_path_parquet, compression="snappy", index=False)

            logger.info(f"Successfully wrote {len(df)} rows to {output_path}" + (f" (as {output_path_parquet})" if output_format not in [".parquet", ".fits", ".fit", ".csv"] else ""))
        except MemoryError as e:
             logger.error(f"Memory error writing output file {output_path}. DataFrame might be too large.")

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

        Handles epoch propagation, sky/ID joins, and temporary file management.
        Uses helper methods for cleaner logic.
        """
        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")
        join_mode = params.get("join_mode", "sky") # 'sky' or 'id'
        matcher = params.get("matcher") # User-provided matcher hint?
        target_epoch = None

        if join_mode == 'sky':
            # Determine matcher if not provided
            if not matcher:
                 matcher = self._determine_stilts_matcher(config1, config2)
                 params["matcher"] = matcher # Store resolved matcher in params
            logger.info(f"Executing local STILTS sky match ({matcher}): {cat_name1} vs {cat_name2}")

            # Determine target epoch for propagation
            epoch1 = config1.get("epoch")
            epoch2 = config2.get("epoch")
            if epoch1 and epoch2 and abs(epoch1 - epoch2) > 0.01:
                can_propagate1 = all(config1.get(k) for k in ["pm_ra_column", "pm_dec_column", "epoch_column", "epoch"])
                can_propagate2 = all(config2.get(k) for k in ["pm_ra_column", "pm_dec_column", "epoch_column", "epoch"])
                if can_propagate1 and not can_propagate2: target_epoch = epoch2
                elif can_propagate2 and not can_propagate1: target_epoch = epoch1
                elif can_propagate1 and can_propagate2: target_epoch = max(epoch1, epoch2)
                else: target_epoch = None
                if target_epoch: logger.info(f"Target epoch for propagation: {target_epoch}")
                else: logger.info("Epochs differ but propagation not possible/needed.")
            elif epoch1: target_epoch = epoch1 # Use epoch1 if only it exists
            elif epoch2: target_epoch = epoch2 # Use epoch2 if only it exists

            if target_epoch is None:
                 logger.info("No epoch propagation needed/possible for local STILTS match.")
        else: # join_mode == 'id'
            logger.info(f"Executing local STILTS ID match: {cat_name1} vs {cat_name2}")
            matcher = None # Matcher not used for ID joins
            target_epoch = None # Epoch propagation not used for ID joins

        with tempfile.TemporaryDirectory(prefix="stilts_match_") as temp_dir:
            try:
                # --- Prepare Inputs ---
                input1_path, config1_updated = self._prepare_local_stilts_input(
                    config1, temp_dir, "input1", target_epoch, join_mode, cat_name1
                )
                input2_path, config2_updated = self._prepare_local_stilts_input(
                    config2, temp_dir, "input2", target_epoch, join_mode, cat_name2
                )

                # --- Define Output ---
                output_filename = "output.parquet"
                output_path = str(Path(temp_dir) / output_filename)

                # --- Build STILTS Parameters ---
                stilts_cmd, stilts_task_params = self._build_stilts_match_params(
                    config1_updated, config2_updated, params, matcher
                )

                # Combine with common STILTS parameters
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
                    "tmpdir": self.stilts_tmpdir or temp_dir, # Use context temp_dir if specific one not set
                    # Allow passing specific STILTS params
                    **params.get("stilts_options", {})
                }
                stilts_params = {k: v for k, v in stilts_base_params.items() if v is not None}
                stilts_params.update(stilts_task_params) # Add task-specific params

                # --- Execute STILTS ---
                _run_stilts(stilts_cmd, stilts_params)

                # --- Read Result ---
                logger.info(f"STILTS {stilts_cmd} completed. Reading result from {output_path}")
                # Check if output file exists and is not empty before reading
                if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                     logger.warning(f"STILTS output file {output_filename} is missing or empty.")
                     return pd.DataFrame() # Return empty DataFrame

                result_df = pd.read_parquet(output_path)
                logger.info(f"Successfully read {len(result_df)} rows from STILTS output.")
                return result_df

            except StiltsError as e:
                logger.error(f"STILTS execution failed: {e}", exc_info=True)
                raise # Re-raise StiltsError
            except FileNotFoundError as e:
                logger.error(f"Input/Output file not found during STILTS execution: {e}")
                raise CrossMatchError(f"File not found: {e}") from e
            except ValueError as e:
                 # Catch specific ValueErrors from param building etc.
                 logger.error(f"Parameter or configuration error during local STILTS: {e}", exc_info=True)
                 raise CrossMatchError(f"Parameter error: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error during local STILTS execution: {e}", exc_info=True)
                raise CrossMatchError(f"Unexpected local STILTS error: {e}") from e

    def _prepare_local_stilts_input(
        self,
        config: Dict[str, Any],
        temp_dir: str,
        input_suffix: str,
        target_epoch: Optional[float],
        join_mode: str,
        catalogue_name: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Prepares a single input for local STILTS.

        Loads data (if path), applies epoch propagation (if needed),
        saves to a temporary file, and returns the file path and potentially
        updated configuration (with propagated RA/Dec columns).

        Args:
            config: The input catalogue configuration.
            temp_dir: The temporary directory path.
            input_suffix: Suffix for the temporary file (e.g., 'input1.parquet').
            target_epoch: The target epoch for propagation (if applicable).
            join_mode: 'sky' or 'id'.
            catalogue_name: Name for logging.

        Returns:
            Tuple: (path_to_temp_file, updated_config_dict)

        Raises:
            CrossMatchError: If data loading, propagation, or writing fails.
        """
        config_local = config.copy()
        input_df = None
        propagated = False
        error_col_added = None # Track name of added error column

        # --- Load or get DataFrame ---
        if "_input_dataframe" in config_local:
            input_df = config_local["_input_dataframe"].copy() # Copy to avoid modifying original DF
        elif "_input_path" in config_local:
            input_path_orig = config_local["_input_path"]
            logger.debug(f"Loading {catalogue_name} from {input_path_orig} for STILTS prep...")
            input_df = self._load_local_catalogue(input_path_orig)
        else:
            raise CrossMatchError(f"Could not find input data (DataFrame or path) for {catalogue_name}")

        if input_df is None or input_df.empty:
            logger.warning(f"Input {catalogue_name} is empty. Creating empty temp file.")
            input_df = pd.DataFrame()
        else:
            # --- Apply Epoch Propagation (if needed) ---
            if join_mode == 'sky' and target_epoch and config_local.get("epoch") and config_local.get("epoch") != target_epoch:
                try:
                    input_df = apply_epoch_propagation(input_df, target_epoch, config_local, catalogue_name)
                    propagated = True
                    # Update config with propagated column names if needed
                    if "ra_propagated" in input_df.columns: config_local['ra_column'] = 'ra_propagated'
                    if "dec_propagated" in input_df.columns: config_local['dec_column'] = 'dec_propagated'
                except ValueError as e:
                    raise CrossMatchError(f"Epoch propagation failed for {catalogue_name}: {e}") from e

            # --- Calculate STILTS Position Error Column (if needed) ---
            ra_err_col = config_local.get("ra_err_column")
            dec_err_col = config_local.get("dec_err_column")
            if ra_err_col and dec_err_col:
                if ra_err_col in input_df.columns and dec_err_col in input_df.columns:
                    # Get units, default to arcsec with warning
                    ra_unit_str = config_local.get("ra_err_unit")
                    dec_unit_str = config_local.get("dec_err_unit")
                    if not ra_unit_str:
                        logger.warning(f"Unit key 'ra_err_unit' missing for {catalogue_name}, assuming arcsec for column '{ra_err_col}'.")
                        ra_unit_str = 'arcsec'
                    if not dec_unit_str:
                        logger.warning(f"Unit key 'dec_err_unit' missing for {catalogue_name}, assuming arcsec for column '{dec_err_col}'.")
                        dec_unit_str = 'arcsec'

                    try:
                        logger.debug(f"Calculating combined position error for {catalogue_name} from '{ra_err_col}' [{ra_unit_str}] and '{dec_err_col}' [{dec_unit_str}]")
                        ra_err = input_df[ra_err_col].values * u.Unit(ra_unit_str)
                        dec_err = input_df[dec_err_col].values * u.Unit(dec_unit_str)
                        # Calculate hypotenuse, ensuring result is in degrees for STILTS
                        pos_err_deg = np.hypot(ra_err, dec_err).to(u.deg).value
                        error_col_added = "_stilts_pos_error_deg" # Define standard name
                        input_df[error_col_added] = pos_err_deg
                        config_local["_stilts_pos_error_col"] = error_col_added # Store name in config
                        logger.debug(f"Added temporary column '{error_col_added}' with error in degrees.")
                    except (u.UnitConversionError, ValueError, TypeError, KeyError) as e:
                        logger.error(
                            f"Failed to calculate/convert position error for {catalogue_name} from columns "
                            f"'{ra_err_col}', '{dec_err_col}' with units '{ra_unit_str}', '{dec_unit_str}': {e}"
                        )
                        # Don't raise error, STILTS call will use default error later
                        error_col_added = None # Ensure we don't try to use it
                    except Exception as e:
                        logger.error(f"Unexpected error calculating position error: {e}", exc_info=True)
                        error_col_added = None
                else:
                    logger.warning(f"Configured error columns '{ra_err_col}' or '{dec_err_col}' not found in DataFrame for {catalogue_name}.")
            else:
                logger.debug(f"No RA/Dec error columns configured for {catalogue_name}.")

        # --- Write DataFrame to temporary file ---
        # Use parquet as it's generally efficient for STILTS
        filename = f"{catalogue_name}_{input_suffix}.parquet"
        # Ensure the temporary error column is included if it was added
        temp_file_path = self._prepare_stilts_input_file(input_df, temp_dir, filename)

        return temp_file_path, config_local

    def _build_stilts_match_params(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        params: Dict[str, Any],
        matcher: Optional[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Builds the task-specific parameters for STILTS tmatch1 or tmatch2."""
        join_mode = params.get("join_mode", "sky")
        stilts_task_params = {}
        stilts_cmd = ""

        if join_mode == "sky":
            stilts_cmd = "tmatch2"
            if not matcher:
                 raise ValueError("Matcher parameter is required for sky join.")

            sky_params = {
                "matcher": matcher,
                "ra1": config1["ra_column"], # Use potentially updated column name
                "dec1": config1["dec_column"],
                "ra2": config2["ra_column"],
                "dec2": config2["dec_column"],
            }
            # Add error columns if needed by matcher
            if matcher in ["skyerr", "skyellipse"]:
                # Check if pre-calculated error column exists, otherwise use default
                err_col1 = config1.get("_stilts_pos_error_col")
                if err_col1 and err_col1 in config1: # Check config AND df (implicitly via creation)
                    sky_params["error1"] = err_col1
                    logger.debug(f"Using pre-calculated error column '{err_col1}' for catalogue 1")
                else:
                    default_err_arcsec1 = config1.get("default_pos_error_arcsec", 0.1)
                    sky_params["error1"] = default_err_arcsec1 / 3600.0 # Convert default to degrees
                    logger.debug(f"Using default pos error {default_err_arcsec1} arcsec (converted to {sky_params['error1']} deg) for catalogue 1")

                err_col2 = config2.get("_stilts_pos_error_col")
                if err_col2 and err_col2 in config2:
                    sky_params["error2"] = err_col2
                    logger.debug(f"Using pre-calculated error column '{err_col2}' for catalogue 2")
                else:
                    default_err_arcsec2 = config2.get("default_pos_error_arcsec", 0.1)
                    sky_params["error2"] = default_err_arcsec2 / 3600.0 # Convert default to degrees
                    logger.debug(f"Using default pos error {default_err_arcsec2} arcsec (converted to {sky_params['error2']} deg) for catalogue 2")

                # Add correlation if matcher is skyellipse
                if matcher == "skyellipse":
                    corr1_col = config1.get("corr_column")
                    corr2_col = config2.get("corr_column")
                    if corr1_col and corr2_col:
                        sky_params["corr1"] = corr1_col
                        sky_params["corr2"] = corr2_col
                    else:
                        logger.warning(f"Matcher is skyellipse but correlation column missing for one/both inputs. STILTS might default/fail.")
                        # STILTS might default corr to 0 if column arg is just the name
                        if corr1_col: sky_params["corr1"] = corr1_col
                        if corr2_col: sky_params["corr2"] = corr2_col

            # Add radius for 'sky' matcher, max_error for error matchers
            if matcher == 'sky':
                radius = params.get("radius_arcsec", 1.0)
                sky_params["params"] = radius # tmatch2 'params' is radius for sky
            else: # skyerr, skyellipse
                max_error = params.get("max_error", 5.0) # Default sigma separation
                sky_params["params"] = max_error # tmatch2 'params' is max error

            stilts_task_params.update(sky_params)
            logger.info(f"STILTS tmatch2 parameters prepared: {sky_params}")

        elif join_mode == "id":
            stilts_cmd = "tmatch1" # Use tmatch1 for single-column value matching
            join_keys = params.get("join_keys")
            if not join_keys or 'cat1' not in join_keys or 'cat2' not in join_keys:
                    raise ValueError("Missing or invalid 'join_keys' for local ID join.")
            id_params = {
                "values1": join_keys['cat1'],
                "values2": join_keys['cat2'],
                # 'matcher' for tmatch1 is usually 'exact' or similar, but defaults work
            }
            stilts_task_params.update(id_params)
            logger.info(f"STILTS tmatch1 parameters prepared: {id_params}")

        else:
            raise ValueError(f"Invalid join_mode '{join_mode}' for local STILTS.")

        return stilts_cmd, stilts_task_params

    def _determine_stilts_matcher(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> str:
        """Auto-detects the best STILTS sky matcher based on available error columns."""
        # Check for necessary error columns in both configs
        has_err1 = all(config1.get(k) for k in ["ra_err_column", "dec_err_column"])
        has_err2 = all(config2.get(k) for k in ["ra_err_column", "dec_err_column"])
        has_corr1 = bool(config1.get("corr_column")) # Just check existence
        has_corr2 = bool(config2.get("corr_column"))

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
        return matcher

    def _prepare_stilts_input_file(self, df: pd.DataFrame, temp_dir: str, filename: str) -> str:
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

    def _prepare_remote_join_columns(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        alias1: str,
        alias2: str,
        params: Dict[str, Any],
    ) -> str:
        """Prepares the SELECT clause for a remote ADQL JOIN query."""
        join_mode = params.get("join_mode", "sky")

        # Get default columns or essential columns if defaults aren't specified
        cols1_req_raw = params.get("columns1") or config1.get("default_columns")
        cols2_req_raw = params.get("columns2") or config2.get("default_columns")

        # Ensure essential columns are always included
        essential_cols1 = {config1.get(c) for c in ["ra_column", "dec_column", "ra_err_column", "dec_err_column", "pm_ra_column", "pm_dec_column", "epoch_column"] if config1.get(c)}
        essential_cols2 = {config2.get(c) for c in ["ra_column", "dec_column", "ra_err_column", "dec_err_column", "pm_ra_column", "pm_dec_column", "epoch_column"] if config2.get(c)}

        # Ensure required join columns are selected if doing ID join
        if join_mode == 'id':
            join_keys = params.get('join_keys', {})
            id_col1 = join_keys.get('cat1')
            id_col2 = join_keys.get('cat2')
            if id_col1: essential_cols1.add(id_col1)
            if id_col2: essential_cols2.add(id_col2)

        # Combine requested and essential, remove None
        cols1_req = list((set(cols1_req_raw) if cols1_req_raw else set()) | essential_cols1)
        cols2_req = list((set(cols2_req_raw) if cols2_req_raw else set()) | essential_cols2)
        cols1_req = [c for c in cols1_req if c is not None]
        cols2_req = [c for c in cols2_req if c is not None]

        # Format column selection with aliases to avoid name clashes
        cols1_select = [f'{alias1}.\"{c}\" AS {alias1}_{c}' for c in set(cols1_req)] # Use set to deduplicate
        cols2_select = [f'{alias2}.\"{c}\" AS {alias2}_{c}' for c in set(cols2_req)]

        # Handle SELECT * case if requested (though aliasing is safer)
        # Note: SELECT * might override careful aliasing.
        select_items = []
        if "*" in (cols1_req_raw or []): select_items.append(f"{alias1}.*")
        else: select_items.extend(cols1_select)
        if "*" in (cols2_req_raw or []): select_items.append(f"{alias2}.*")
        else: select_items.extend(cols2_select)

        select_clause = ", ".join(select_items)
        if not select_clause:
            # Default to selecting all if specific columns failed resolution?
            logger.warning("Could not determine columns to SELECT, defaulting to SELECT *.")
            select_clause = f"{alias1}.*, {alias2}.*"

        return select_clause

    def _build_remote_adql_join_clause(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        alias1: str,
        alias2: str,
        params: Dict[str, Any],
    ) -> str:
        """Builds the ADQL JOIN ON clause based on join mode and parameters."""
        join_mode = params.get("join_mode", "sky")
        join_on_clause = ""

        if join_mode == "sky":
            radius_arcsec = params.get("radius_arcsec", 1.0)
            if radius_arcsec <= 0:
                raise ValueError("Search radius must be positive for sky join.")
            radius_deg = radius_arcsec / 3600.0

    # --- Chunked Execution Helpers ---

    def _setup_healpix_chunking(
        self,
        nside: int,
        center_coord: SkyCoord,
        search_radius: u.Quantity
    ) -> Tuple[HEALPix, np.ndarray]:
        """Initializes HEALPix and finds pixels within the search cone."""
        try:
            hp = HEALPix(nside=nside, order="nested", frame="icrs")
            # Add buffer to cone search radius for robust pixel coverage
            buffer = hp.pixel_resolution.to(u.arcsec) * 1.5
            logger.debug(f"Using search radius {search_radius.arcsec:.2f} + buffer {buffer.arcsec:.2f} arcsec for pixel identification.")
            pixels = hp.cone_search_skycoord(center_coord, search_radius + buffer)
            logger.info(f"Identified {len(pixels)} HEALPix pixels (nside={nside}) covering the search area + buffer.")
            if len(pixels) == 0:
                 logger.warning("No HEALPix pixels found in the search cone.")
                 # Return empty pixel array, caller should handle
            return hp, pixels
        except ImportError:
            logger.error("'astropy-healpix' library is required for chunked matching.")
            raise CrossMatchError(
                "'astropy-healpix' library is required for this strategy. Please install it."
            )
        except Exception as e:
            logger.error(f"Error during HEALPix pixel determination: {e}", exc_info=True)
            raise CrossMatchError(f"Error during HEALPix pixel determination: {e}") from e

    def _get_healpix_adql_box(self, hp: HEALPix, pix_id: int) -> Optional[str]:
        """Calculates the ADQL BOX string for a given HEALPix pixel."""
        try:
            corners = hp.boundaries_skycoord([pix_id])[0]
            # Use degrees directly for ADQL
            ra_corners = corners.ra.wrap_at(180 * u.deg).deg
            dec_corners = corners.dec.deg
            ra_min, ra_max = np.min(ra_corners), np.max(ra_corners)
            dec_min, dec_max = np.min(dec_corners), np.max(dec_corners)

            # ADQL BOX: center and width/height
            # Handle RA wrap-around: If min > max after wrap_at(180), it crosses 0/360.
            # Center calculation needs care. Width is simpler: max - min + 360 if wrapped.
            ra_cen = (ra_min + ra_max) / 2.0
            if ra_min > ra_max: # Wrapped around 180
                ra_cen = (ra_min + ra_max + 360) / 2.0
                # Normalize center back to [0, 360) or (-180, 180)? ADQL usually takes 0-360.
                ra_cen = ra_cen % 360

            dec_cen = (dec_min + dec_max) / 2.0
            width = ra_max - ra_min
            if width < 0: width += 360 # Adjust width for wrap
            height = dec_max - dec_min

            # Check for degenerate pixels (zero width/height)
            if width <= 1e-9 or height <= 1e-9: # Use small tolerance for float issues
                logger.warning(f"Skipping degenerate pixel {pix_id} (width={width:.2e}, height={height:.2e}).")
                return None

            # Ensure positive width/height for ADQL BOX
            width = max(width, 1e-9)
            height = max(height, 1e-9)

            adql_box = f"BOX('ICRS', {ra_cen}, {dec_cen}, {width}, {height})"
            logger.debug(f"Pixel {pix_id} ADQL Box: {adql_box}")
            return adql_box
        except Exception as e:
            logger.error(f"Error calculating ADQL BOX for pixel {pix_id}: {e}", exc_info=True)
            # Let caller skip this pixel
            return None

    def _build_chunked_tap_join_adql(
        self,
        select_clause: str,
        join_on_clause: str,
        table1: str, table2: str,
        alias1: str, alias2: str,
        adql_box_clause: str,
        ra_col1: str, dec_col1: str,
        ra_col2: str, dec_col2: str,
        join_type: str
    ) -> str:
        """Constructs the full ADQL query for a single TAP spatial join chunk."""
        # WHERE clause requires both points to be within the ADQL box
        where_clause = (
            f"CONTAINS(POINT('ICRS', {alias1}.\"{ra_col1}\", {alias1}.\"{dec_col1}\"), {adql_box_clause}) = 1"
            f" AND CONTAINS(POINT('ICRS', {alias2}.\"{ra_col2}\", {alias2}.\"{dec_col2}\"), {adql_box_clause}) = 1"
        )
        # Note: Some TAP services might prefer INTERSECTS(REGION, BOX)
        # Note: Applying spatial constraint in WHERE after JOIN might be slow on some systems.

        adql_query = f"""
        SELECT {select_clause}
        FROM {table1} AS {alias1}
        {join_type} JOIN {table2} AS {alias2}
        ON {join_on_clause}
        WHERE {where_clause}
        """
        return adql_query

    # --- Main Execution Methods ---

    def _execute_download_and_match(
        self, config1: Dict[str, Any], config2: Dict[str, Any], **params
    ) -> pd.DataFrame:
        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")
        logger.info(f"Executing download and match strategy for {cat_name1} and {cat_name2}")

        try:
            # Implement download and match logic here
            # This is a placeholder and should be replaced with the actual implementation
            # For now, we'll return an empty DataFrame
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching remote catalogue {cat_name1}: {e}")
            raise CrossMatchError(f"Unexpected error fetching remote catalogue {cat_name1}: {e}") from e

    def _execute_remote_tap_chunked_spatial_join(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        nside: int = 32, # Default HEALPix nside for chunking
        **params,
    ) -> pd.DataFrame:
        """Executes remote-remote spatial match using chunked TAP queries."""
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
        except (ValueError, TypeError, u.UnitConversionError) as e:
            raise CrossMatchError(f"Invalid spatial parameters (ra/dec/radius): {e}") from e

        # --- Get Service and Table Info ---
        service_config = config1 # Assumes same service
        tap_url = service_config.get("access_url")
        if not tap_url:
            raise CrossMatchError("Missing access_url in service config for remote chunked join.")
        archive_name = config1.get("_archive_name") # For auth
        table1 = config1.get("access_identifier")
        table2 = config2.get("access_identifier")
        if not table1 or not table2:
             raise CrossMatchError("Missing access_identifier for one or both catalogues.")

        # --- Determine HEALPix Pixels ---
        hp, pixels = self._setup_healpix_chunking(nside, center_coord, search_radius)
        if len(pixels) == 0:
            return pd.DataFrame()

        # --- Prepare Common Query Parts ---
        alias1 = "t1"
        alias2 = "t2"
        try:
            select_clause = self._prepare_remote_join_columns(config1, config2, alias1, alias2, params)
            # We specifically need the spatial join ON clause here
            spatial_join_params = {**params, "join_mode": "sky"} # Force sky mode
            join_on_clause = self._build_remote_adql_join_clause(config1, config2, alias1, alias2, spatial_join_params)
            adql_join_type = params.get("join_type", "INNER").upper()
        except (ValueError, CrossMatchError) as e:
            logger.error(f"Failed to prepare common ADQL query parts: {e}", exc_info=True)
            raise

        # Required column names for the WHERE clause builder
        ra_col1 = config1.get("ra_column")
        dec_col1 = config1.get("dec_column")
        ra_col2 = config2.get("ra_column")
        dec_col2 = config2.get("dec_column")
        if not ra_col1 or not dec_col1 or not ra_col2 or not dec_col2:
            raise CrossMatchError("Missing RA/Dec column config needed for chunked WHERE clause.")

        # --- Process Chunks ---       
        all_results = []
        # Note: TAP connection is established per query in _execute_tap_join_query helper

        for i, pix_id in enumerate(pixels):
            logger.info(f"Processing chunk {i+1}/{len(pixels)} (Pixel ID: {pix_id})...")
            try:
                # Get ADQL BOX string for the current pixel
                adql_box = self._get_healpix_adql_box(hp, pix_id)
                if adql_box is None: # Skip if box calculation failed or pixel degenerate
                    continue

                # --- Construct ADQL Query for the Chunk ---
                adql_query = self._build_chunked_tap_join_adql(
                    select_clause, join_on_clause,
                    table1, table2, alias1, alias2,
                    adql_box,
                    ra_col1, dec_col1, ra_col2, dec_col2,
                    adql_join_type
                )
                logger.debug(f"Chunk {i+1} ADQL Query:\n{adql_query}")

                # --- Execute Query for the Chunk ---
                # Pass original params dict for execution settings (timeout, retries)
                chunk_result_df = self._execute_tap_join_query(
                    tap_url, adql_query, archive_name, **params
                )

                if chunk_result_df is not None and not chunk_result_df.empty:
                    logger.info(f"Chunk {i+1} TAP JOIN yielded {len(chunk_result_df)} pairs.")
                    all_results.append(chunk_result_df)
                else:
                     logger.info(f"Chunk {i+1} yielded no results from TAP JOIN.")

            except (TapError, ConnectionError, Timeout, RequestException) as e:
                # Log TAP/network errors for the chunk but continue processing others
                logger.warning(
                    f"TAP query failed for chunk {i+1} (Pixel {pix_id}): {e}. Skipping chunk.", exc_info=True
                )
                continue
            except (ValueError, CrossMatchError) as e:
                 # Log other config/logic errors for the chunk but continue
                 logger.warning(
                    f"Error processing chunk {i+1} (Pixel {pix_id}): {e}. Skipping chunk.", exc_info=True
                 )
                 continue
            except Exception as e:
                # Log unexpected errors but continue
                logger.error(
                    f"Unexpected error processing chunk {i+1} (Pixel {pix_id}): {e}", exc_info=True
                )
                continue

        # --- Combine Results ---       
        if not all_results:
            logger.warning("Remote TAP Chunked Spatial Join resulted in no matches across all chunks.")
            return pd.DataFrame()
        else:
            logger.info(f"Concatenating results from {len(all_results)} successful chunks...")
            try:
                final_df = pd.concat(all_results, ignore_index=True)
                # Optional: Deduplicate based on primary keys if overlaps might cause issues?
                logger.info(f"Remote TAP Chunked Spatial Join completed. Total pairs found: {len(final_df)}")
                # TODO: Add deduplication if necessary
                # Example: if 't1_id' in final_df.columns and 't2_id' in final_df.columns:
                #     final_df = final_df.drop_duplicates(subset=['t1_id', 't2_id'], keep='first')
                #     logger.info(f"DataFrame deduplicated, final count: {len(final_df)}")
                return final_df
            except MemoryError as e:
                logger.error("Memory error combining chunked TAP JOIN results.")
                raise CrossMatchError("Memory error combining chunked TAP JOIN results") from e
            except Exception as e:
                logger.error(f"Error combining chunked TAP JOIN results: {e}", exc_info=True)
                raise CrossMatchError(f"Error combining chunked TAP JOIN results: {e}") from e

    # -----------------------------------------------------------
    # Catalogue Creation Helpers
    # -----------------------------------------------------------

    def _connect_tap_for_creation(self, archive_name: str, service_id: str) -> Tuple[TAPService, Dict[str, Any]]:
        """Validates config and connects to the TAP service for catalogue creation."""
        logger.debug(f"Validating archive '{archive_name}' and service '{service_id}' for TAP connection.")
        if archive_name not in self.archives_config:
            raise CrossMatchError(f"Archive '{archive_name}' not found in configuration.")
        archive_conf = self.archives_config[archive_name]
        if service_id not in archive_conf:
            raise CrossMatchError(f"Service ID '{service_id}' not found in archive '{archive_name}'.")
        service_conf = archive_conf[service_id]
        if service_conf.get("access_method") != "tap":
            raise CrossMatchError(f"Service '{service_id}' in archive '{archive_name}' is not a TAP service.")
        tap_url = service_conf.get("access_url")
        if not tap_url:
            raise CrossMatchError(f"TAP Service '{service_id}' in archive '{archive_name}' has no access_url.")

        try:
            logger.info(f"Connecting to TAP service at {tap_url}...")
            auth_info = self.auth_config.get(archive_name)
            tap_kwargs = {}
            if auth_info:
                # Assuming user/password auth
                tap_kwargs["user"] = auth_info.get("user")
                tap_kwargs["password"] = auth_info.get("password")
            tap_service = get_tap_service(tap_url, **tap_kwargs)
            return tap_service, service_conf # Return service and its config
        except (TapError, ConnectionError, Timeout, RequestException) as e:
            logger.error(f"Failed to connect to TAP service for archive '{archive_name}': {e}", exc_info=True)
            raise CrossMatchError(f"Failed to connect to TAP service '{tap_url}': {e}") from e

    def _query_tap_schema_table(self, tap_service: TAPService, table_name: str) -> Optional[str]:
        """Queries TAP_SCHEMA.tables for the table description."""
        try:
            table_query = f"SELECT description FROM TAP_SCHEMA.tables WHERE table_name = '{table_name}'"
            logger.info(f"Querying TAP_SCHEMA.tables for description: {table_query}")
            desc_result = execute_tap_query(tap_service, table_query, max_retries=1)
            if not desc_result.empty and 'description' in desc_result.columns and desc_result.iloc[0]['description']:
                description = desc_result.iloc[0]['description']
                logger.info(f"Found table description: {description}")
                return description
            else:
                logger.warning(f"Could not find description for table '{table_name}' in TAP_SCHEMA.tables.")
                return None
        except (TapError, Exception) as e:
            logger.warning(f"Failed to query TAP_SCHEMA.tables for description: {e}.")
            return None

    def _query_tap_schema_columns(self, tap_service: TAPService, table_name: str) -> Dict[str, Dict[str, Any]]:
        """Queries TAP_SCHEMA.columns for metadata."""
        columns_metadata = {}
        try:
            cols_query = (
                f"SELECT column_name, ucd, unit, datatype, description, principal "
                f"FROM TAP_SCHEMA.columns WHERE table_name = '{table_name}'"
            )
            logger.info(f"Querying TAP_SCHEMA.columns for metadata: {cols_query}")
            cols_result = execute_tap_query(tap_service, cols_query, max_retries=1)
            if not cols_result.empty:
                logger.info(f"Found {len(cols_result)} columns for table '{table_name}'.")
                for _, row in cols_result.iterrows():
                    # Clean up potential whitespace in keys/values?
                    metadata = {k.strip(): v.strip() if isinstance(v, str) else v for k, v in row.to_dict().items()}
                    columns_metadata[metadata['column_name']] = metadata
                return columns_metadata
            else:
                logger.warning(f"Could not retrieve column metadata for table '{table_name}' from TAP_SCHEMA.columns.")
                return {}
        except (TapError, Exception) as e:
            logger.error(f"Failed to query TAP_SCHEMA.columns: {e}", exc_info=True)
            # Raise error here, as column metadata is essential
            raise CrossMatchError(f"Failed to query TAP_SCHEMA.columns for '{table_name}': {e}") from e

    def _detect_standard_columns(self, columns_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Detects standard columns (RA, Dec, etc.) using UCDs and common names."""
        detected_cols = {}
        if not columns_metadata:
             logger.warning("No column metadata provided, cannot detect standard columns.")
             return detected_cols

        # Define UCDs and common names maps (as used previously)
        ucd_map = {
            "ra_column": ["pos.eq.ra", "POS_EQ_RA_MAIN"],
            "dec_column": ["pos.eq.dec", "POS_EQ_DEC_MAIN"],
            "pm_ra_column": ["pos.pm;pos.eq.ra", "stat.mean;pos.pm;pos.eq.ra"], # Added mean
            "pm_dec_column": ["pos.pm;pos.eq.dec", "stat.mean;pos.pm;pos.eq.dec"], # Added mean
            "epoch_column": ["time.epoch"],
            "ra_err_column": ["stat.error;pos.eq.ra"],
            "dec_err_column": ["stat.error;pos.eq.dec"],
            "corr_column": ["stat.correlation;pos.eq.ra;pos.eq.dec"],
            "id_column": ["meta.id", "meta.id;meta.main"],
            "parallax_column": ["pos.parallax", "pos.parallax;stat.mean"], # Added mean
        }
        common_name_map = {
            "ra_column": ["ra", "ra_icrs", "raj2000", "ra_deg"],
            "dec_column": ["dec", "dec_icrs", "dej2000", "de_icrs", "dec_deg"],
            "pm_ra_column": ["pmra", "pm_ra"],
            "pm_dec_column": ["pmdec", "pm_dec"],
            "epoch_column": ["epoch", "ref_epoch"],
            "ra_err_column": ["ra_error", "e_ra", "sigra", "err_maj"],
            "dec_err_column": ["dec_error", "e_dec", "sigdec", "err_min"],
            "corr_column": ["ra_dec_corr", "corr"],
            "id_column": ["id", "source_id", "objid"],
             "parallax_column": ["plx", "parallax"],
        }

        logger.info("Attempting to auto-detect standard columns...")
        # Prioritize UCD matching
        for standard_name, ucds in ucd_map.items():
            found_col = None
            best_match_strength = 0 # 0: no match, 1: secondary ucd, 2: primary ucd
            for col_name, meta in columns_metadata.items():
                ucd_str = str(meta.get('ucd', '')).lower()
                if not ucd_str: continue

                current_match_strength = 0
                if ucds[0].lower() in ucd_str: current_match_strength = 2 # Primary match
                elif len(ucds) > 1 and ucds[1].lower() in ucd_str: current_match_strength = 1 # Secondary

                if current_match_strength > best_match_strength:
                    best_match_strength = current_match_strength
                    found_col = col_name

            if found_col:
                 log_msg = f"  Detected {standard_name}: '{found_col}' (using UCD '{ucds[0] if best_match_strength == 2 else ucds[1]}')\""
                 if best_match_strength == 1: log_msg += " [secondary UCD match]"
                 logger.info(log_msg)
                 detected_cols[standard_name] = found_col

        # Fallback to common names if UCD didn't find it
        for standard_name, names in common_name_map.items():
            if standard_name not in detected_cols:
                 found_col = None
                 for col_name in columns_metadata.keys():
                     # Simple substring match might be too broad, use exact match for common names
                     if col_name.lower() in names:
                        found_col = col_name
                        break
                 if found_col:
                     logger.info(f"  Detected {standard_name}: '{found_col}' (using common name)")
                     detected_cols[standard_name] = found_col

        # Basic validation: Check if RA/Dec were found
        if "ra_column" not in detected_cols or "dec_column" not in detected_cols:
            logger.error("Failed to automatically detect RA and/or Dec columns.")
            # Don't raise error here, let caller decide

        return detected_cols

    def _build_new_catalogue_entry(
        self,
        table_description: str,
        archive_name: str,
        service_id: str,
        access_identifier: str,
        detected_columns: Dict[str, str],
    ) -> Dict[str, Any]:
        """Constructs the dictionary for the new catalogue entry."""
        new_entry = {
            "description": table_description,
            "archive": archive_name,
            "service_id": service_id,
            "access_identifier": access_identifier,
            "table_name": access_identifier, # Assume same for TAP
            "estimated_size": 'unknown', # Cannot easily estimate size
            "default_pos_error_arcsec": 0.1, # Sensible default
        }
        # Add detected columns to the entry
        new_entry.update(detected_columns)

        # Add default columns list (use detected ID, RA, Dec if available)
        default_cols_set = {
            detected_columns.get("id_column"),
            detected_columns.get("ra_column"),
            detected_columns.get("dec_column"),
        }
        new_entry["default_columns"] = sorted([c for c in default_cols_set if c is not None])
        return new_entry

    def _add_entry_to_config_file(
        self, new_catalogue_name: str, new_entry: Dict[str, Any]
    ) -> None:
        """Loads the config, adds the new entry, and saves the file."""
        try:
            logger.info(f"Attempting to update config file: {self.config_file}")
            # Load the whole config again to ensure we have the latest
            # Use internal _load_config to handle potential errors during load
            full_config = self._load_config()
            if 'catalogues' not in full_config:
                full_config['catalogues'] = {}
            elif not isinstance(full_config['catalogues'], dict):
                 logger.error("Invalid config format: 'catalogues' is not a dictionary. Cannot add entry.")
                 raise CrossMatchError("Config format error: 'catalogues' section invalid.")

            # Check again in case config changed between initial check and write
            if new_catalogue_name in full_config['catalogues']:
                 logger.error(f"Catalogue entry '{new_catalogue_name}' already exists (race condition?). Aborting write.")
                 raise CrossMatchError(f"Catalogue '{new_catalogue_name}' exists (race condition?).")

            # Add the new entry
            full_config['catalogues'][new_catalogue_name] = new_entry

            # Write back to the file
            with open(self.config_file, 'w') as f:
                 yaml.dump(full_config, f, default_flow_style=False, sort_keys=False, indent=2)
            logger.info(f"Successfully added '{new_catalogue_name}' entry to {self.config_file}")

        except (IOError, yaml.YAMLError) as e:
            logger.error(f"Failed to write updated configuration to {self.config_file}: {e}", exc_info=True)
            # Re-raise as CrossMatchError
            raise CrossMatchError(f"Failed to write config file {self.config_file}: {e}") from e
        except CrossMatchError: # Catch specific error from _load_config or race condition
             raise
        except Exception as e:
             logger.error(f"Unexpected error updating config file: {e}", exc_info=True)
             raise CrossMatchError(f"Unexpected error updating config file: {e}") from e

    # -----------------------------------------------------------
    # Catalogue Creation Method
    # -----------------------------------------------------------
    def create_catalogue_entry(
        self,
        new_catalogue_name: str,
        archive_name: str,
        access_identifier: str,
        service_id: str = "tap_service",
        description_override: Optional[str] = None,
    ) -> bool:
        """Creates a new catalogue entry in the config file by querying TAP_SCHEMA."""
        new_catalogue_name = new_catalogue_name.lower()
        logger.info(
            f"Attempting to create config entry '{new_catalogue_name}' for table "
            f"'{access_identifier}' in archive '{archive_name}' (service: {service_id})"
        )

        # --- Validate Input Name ---
        if new_catalogue_name in self.catalogues_config:
            logger.error(f"Catalogue entry '{new_catalogue_name}' already exists in config file.")
            return False

        try:
            # --- Connect to TAP Service ---
            tap_service, _ = self._connect_tap_for_creation(archive_name, service_id)

            # --- Get Table Description ---
            table_description = description_override
            if not table_description:
                table_description = self._query_tap_schema_table(tap_service, access_identifier)
                if not table_description:
                    logger.warning(f"Using default description for '{new_catalogue_name}'.")
                    table_description = f"Table {access_identifier} from {archive_name}"

            # --- Get Column Metadata ---
            columns_metadata = self._query_tap_schema_columns(tap_service, access_identifier)
            if not columns_metadata:
                 # Error already raised by helper if query failed, but check if empty dict returned
                 logger.error(f"No column metadata retrieved for '{access_identifier}'. Cannot proceed.")
                 return False

            # --- Auto-detect Essential Columns ---
            detected_columns = self._detect_standard_columns(columns_metadata)

            # --- Validate Essential Columns (RA/Dec) ---
            if "ra_column" not in detected_columns or "dec_column" not in detected_columns:
                logger.error("Failed to automatically detect required RA and/or Dec columns. Cannot create entry.")
                logger.error("Available columns detected:")
                for col_name, meta in columns_metadata.items():
                     ucd = meta.get('ucd', 'N/A')
                     desc = meta.get('description', 'N/A')
                     logger.error(f"  - {col_name} (UCD: {ucd}, Desc: {desc})")
                return False

            # --- Build New Entry Dictionary ---
            new_entry = self._build_new_catalogue_entry(
                table_description,
                archive_name,
                service_id,
                access_identifier,
                detected_columns,
            )

            # --- Update YAML File ---
            self._add_entry_to_config_file(new_catalogue_name, new_entry)

            # --- Reload internal config state ---           
            logger.info("Reloading configuration after adding new entry...")
            self.__init__(self.config_file)
            logger.info("Configuration reloaded.")

            return True

        except CrossMatchError as e:
            # Catch errors raised by helpers or validation
            logger.error(f"Failed to create catalogue entry '{new_catalogue_name}': {e}", exc_info=True)
            return False
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error creating catalogue entry '{new_catalogue_name}': {e}", exc_info=True)
            return False

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
            chunk_input_index: Which input to chunk (1 or 2).
            params: Additional parameters passed to _execute_local_stilts for each chunk.

        Returns:
            A pandas DataFrame containing the concatenated results from all chunks.
        """
        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")
        logger.info(
            f"Executing Chunked Local Match strategy: {cat_name1} vs {cat_name2}, chunking input {chunk_input_index}"
        )

        # --- Validate Inputs are Local ---       
        if config1.get("access_method") != "file_system" or config2.get("access_method") != "file_system":
            raise CrossMatchError("Chunked match requires both inputs to be local (file or DataFrame).")

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

        # --- Load Static Catalogue into Memory ---
        static_input_df = None
        if static_config.get("_input_dataframe") is not None:
            static_input_df = static_config.get("_input_dataframe")
        elif static_config.get("_input_path") is not None:
            logger.info(f"Loading static input catalogue '{static_cat_name}' into memory...")
            try:
                static_input_df = self._load_local_catalogue(static_config.get("_input_path"))
                # Store in config only if successfully loaded
                static_config["_input_dataframe"] = static_input_df
                static_config.pop("_input_path", None)
            except (FileNotFoundError, MemoryError, CrossMatchError) as e:
                 logger.error(f"Failed to load static catalogue '{static_cat_name}': {e}")
                 raise # Re-raise critical errors
        else:
             raise CrossMatchError(f"Static input catalogue '{static_cat_name}' has no path or DataFrame.")

        if static_input_df is None or static_input_df.empty:
            logger.warning(f"Static input catalogue '{static_cat_name}' is empty. Returning empty result.")
            return pd.DataFrame()
        logger.info(f"Static catalogue '{static_cat_name}' ({len(static_input_df)} rows) ready.")

        # --- Process Chunked Input ---       
        all_results = []
        chunk_num = 0
        chunk_input_path = chunk_config.get("_input_path")
        chunk_input_df = chunk_config.get("_input_dataframe")
        iterator = None
        file_format = None

        try:
            # --- Setup Iterator/Source for Chunking ---
            if chunk_input_df is not None:
                logger.info(f"Chunking input DataFrame '{chunk_cat_name}' ({len(chunk_input_df)} rows) in chunks of {chunk_rows}...")
                # Use iloc for DataFrame chunking later
                total_rows = len(chunk_input_df)
            elif chunk_input_path:
                input_path = Path(chunk_input_path)
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file for chunking not found: {input_path}")

                file_format = input_path.suffix.lower()
                logger.info(f"Preparing to chunk input file '{chunk_cat_name}' ({input_path}, format: {file_format}) in chunks of {chunk_rows}...")

                if file_format == ".csv":
                    # Use pandas chunked reader for CSV
                    try:
                        iterator = pd.read_csv(input_path, chunksize=chunk_rows, low_memory=False)
                    except Exception as e:
                         raise CrossMatchError(f"Failed to create CSV reader for {input_path}: {e}") from e
                elif file_format == ".parquet":
                    # Use pyarrow iterator for Parquet
                    try:
                        import pyarrow.parquet as pq
                        parquet_file = pq.ParquetFile(input_path)
                        iterator = parquet_file.iter_batches(batch_size=chunk_rows)
                        logger.debug(f"Using pyarrow.iter_batches for Parquet file: {input_path}")
                    except ImportError:
                        logger.error("'pyarrow' library required for efficient Parquet chunking.")
                        raise CrossMatchError("'pyarrow' is required for Parquet chunking. Please install it.")
                    except Exception as e:
                         raise CrossMatchError(f"Failed to create Parquet reader for {input_path}: {e}") from e
                elif file_format in [".fits", ".fit"]:
                    # FITS chunking is less direct
                    logger.warning("Attempting FITS chunking via memory mapping; may still use significant memory depending on access patterns.")
                    try:
                        from astropy.io import fits
                        # Open with memmap=True
                        hdul = fits.open(input_path, memmap=True)
                        # Find the first table HDU
                        table_hdu = None
                        for hdu in hdul:
                             if isinstance(hdu, (fits.TableHDU, fits.BinTableHDU)):
                                 table_hdu = hdu
                                 break
                        if table_hdu is None:
                            raise CrossMatchError(f"No table HDU found in FITS file: {input_path}")
                        total_rows = table_hdu.header['NAXIS2']
                        # Store HDUList and index for iterative slicing
                        iterator = (hdul, table_hdu, total_rows) # Tuple indicates FITS mode
                        logger.debug(f"Prepared FITS file for chunked access: {input_path} ({total_rows} rows)")
                    except ImportError:
                        logger.error("'astropy' library required for FITS reading.")
                        raise CrossMatchError("'astropy' is required for FITS reading.")
                    except Exception as e:
                         hdul.close() # Ensure file handle is closed on error
                         raise CrossMatchError(f"Failed to open/prepare FITS file {input_path} for chunking: {e}") from e
                else:
                    raise CrossMatchError(f"Unsupported file format for chunking: {file_format}")
            else:
                raise CrossMatchError(
                    f"Cannot chunk input {chunk_input_index} ('{chunk_cat_name}'): No DataFrame or file path provided."
                )

            # --- Process Chunks ---           
            if chunk_input_df is not None:
                 # Iterate over DataFrame chunks
                 for i in range(0, total_rows, chunk_rows):
                     chunk_num += 1
                     current_chunk_df = chunk_input_df.iloc[i : i + chunk_rows]
                     if current_chunk_df.empty:
                         continue # Skip empty chunks

                     logger.info(f"Processing DataFrame chunk {chunk_num} ({len(current_chunk_df)} rows)...")
                     # (Logic to match chunk is below the iterator handling)
                     result_chunk = self._match_chunk_against_static(
                         chunk_df=current_chunk_df,
                         chunk_config=chunk_config,
                         static_config=static_config,
                         chunk_input_index=chunk_input_index,
                         params=params
                     )
                     if result_chunk is not None: all_results.append(result_chunk)

            elif iterator is not None:
                 # Iterate over file chunks (CSV, Parquet, FITS)
                 if isinstance(iterator, tuple): # FITS mode
                     hdul, table_hdu, total_rows = iterator
                     try:
                         for i in range(0, total_rows, chunk_rows):
                             chunk_num += 1
                             start_row = i
                             end_row = min(i + chunk_rows, total_rows)
                             logger.info(f"Processing FITS chunk {chunk_num} (Rows {start_row}-{end_row-1})...")
                             # Read the slice
                             try:
                                 table_slice = Table(table_hdu.data[start_row:end_row])
                                 current_chunk_df = table_slice.to_pandas()
                             except Exception as e:
                                 logger.warning(f"Error reading/converting FITS chunk {chunk_num}: {e}. Skipping chunk.", exc_info=True)
                                 continue # Skip to next chunk

                             if current_chunk_df.empty:
                                 continue
                             logger.debug(f"Read {len(current_chunk_df)} rows for FITS chunk {chunk_num}.")

                             result_chunk = self._match_chunk_against_static(
                                 chunk_df=current_chunk_df,
                                 chunk_config=chunk_config,
                                 static_config=static_config,
                                 chunk_input_index=chunk_input_index,
                                 params=params
                             )
                             if result_chunk is not None: all_results.append(result_chunk)
                     finally:
                         # Ensure FITS file handle is closed regardless of loop errors
                         logger.debug(f"Closing FITS file handle for {chunk_input_path}")
                         hdul.close()

                 elif file_format == ".parquet": # PyArrow iterator
                     for batch in iterator:
                         chunk_num += 1
                         logger.info(f"Processing Parquet batch {chunk_num} ({len(batch)} rows)...")
                         result_chunk = self._match_chunk_against_static(
                             chunk_df=batch,
                             chunk_config=chunk_config,
                             static_config=static_config,
                             chunk_input_index=chunk_input_index,
                             params=params
                         )
                         if result_chunk is not None: all_results.append(result_chunk)

        except (ValueError, CrossMatchError) as e:
            logger.error(f"Error processing chunked match: {e}", exc_info=True)
            raise CrossMatchError(f"Error processing chunked match: {e}") from e

        # --- Combine Results ---       
        if not all_results:
            logger.warning("No matches found in any chunk.")
            return pd.DataFrame()
        else:
            logger.info(f"Concatenating results from {len(all_results)} successful chunks...")
            try:
                final_df = pd.concat(all_results, ignore_index=True)
                logger.info(f"Final DataFrame shape: {final_df.shape}")
                return final_df
            except MemoryError as e:
                logger.error("Memory error combining chunked match results.")
                raise CrossMatchError("Memory error combining chunked match results") from e
            except Exception as e:
                logger.error(f"Error combining chunked match results: {e}", exc_info=True)
                raise CrossMatchError(f"Error combining chunked match results: {e}") from e

    def _match_chunk_against_static(
        self,
        chunk_df: pd.DataFrame,
        chunk_config: Dict[str, Any],
        static_config: Dict[str, Any],
        chunk_input_index: int,
        params: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Matches a single chunk against the static catalogue."""
        # Implement matching logic here
        # This is a placeholder and should be replaced with the actual implementation
        # For now, we'll return an empty DataFrame
        return pd.DataFrame()

    # -----------------------------------------------------------
    # CDS XMatch Local-Remote Strategy
    # -----------------------------------------------------------

    def _execute_cds_xmatch_local_remote(
        self, config1: Dict[str, Any], config2: Dict[str, Any], **params
    ) -> pd.DataFrame:
        cat_name1 = config1.get("_catalogue_name", "input1")
        cat_name2 = config2.get("_catalogue_name", "input2")
        logger.info(f"Executing CDS XMatch Local-Remote strategy for {cat_name1} and {cat_name2}")

        try:
            # Implement CDS XMatch Local-Remote logic here
            # This is a placeholder and should be replaced with the actual implementation
            # For now, we'll return an empty DataFrame
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error executing CDS XMatch Local-Remote strategy: {e}")
            raise CrossMatchError(f"Unexpected error executing CDS XMatch Local-Remote strategy: {e}") from e
