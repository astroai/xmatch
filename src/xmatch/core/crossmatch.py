import os
import yaml
import logging
from typing import Optional, Dict, List, Union, Any, Tuple, Callable
from pathlib import Path
import pandas as pd
from astropy.table import Table
import pyvo as vo
from astroquery.xmatch import XMatch
from astroquery.vizier import Vizier
import tempfile
import shutil
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from joblib import Parallel, delayed

from ..utils import stilts, tap, auth # Assuming auth utility exists
from ..utils.stilts import StiltsError
from ..utils.tap import TapError

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "catalogues.yaml"
SUPPORTED_INPUT_FORMATS = [".parquet", ".fits", ".csv"]

class CrossMatchError(Exception):
    """Custom exception for cross-matching errors."""
    pass

class CrossMatch:
    """Handles cross-matching of astronomical catalogues."""

    def __init__(self, config_file: Union[str, Path] = DEFAULT_CONFIG_PATH, **kwargs):
        """
        Initializes the CrossMatch object.

        Args:
            config_file: Path to the YAML configuration file.
            **kwargs: Additional configuration overrides (e.g., java_opts, chunk_size).
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.catalogues_config = self.config.get('catalogues', {})
        self.methods_config = self.config.get('crossmatch_methods', {})
        self.stilts_config = self.config.get('stilts_config', {})
        
        # Apply kwargs overrides to config settings
        # Prioritize CLI kwargs over config for these specific settings
        self.stilts_cmd_base = kwargs.get('stilts_cmd_base', self.stilts_config.get('stilts_cmd_base'))
        self.stilts_java_opts = kwargs.get('java_opts', self.stilts_config.get('java_opts'))
        self.stilts_tmpdir = kwargs.get('tmpdir', self.stilts_config.get('tmpdir'))
        self.chunk_size = kwargs.get('chunk_size', 100000) # Default chunk size
        
        self.auth_config = auth.load_auth_config() # Load credentials securely

        self._catalogue_config_cache = {}  # Add caching for catalogue configs

        self._validate_config()
        logger.info(f"CrossMatch initialized with config: {self.config_file}")
        if kwargs:
             logger.info(f"Applied config overrides: {kwargs}")
        if self.stilts_cmd_base:
             logger.info(f"Using STILTS base command from config/kwargs: '{self.stilts_cmd_base}'")

    def _load_config(self) -> Dict[str, Any]:
        """Loads the YAML configuration file."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                     raise CrossMatchError("Configuration file is not a valid YAML dictionary.")
                return config
        except FileNotFoundError:
            raise CrossMatchError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise CrossMatchError(f"Error parsing configuration file {self.config_file}: {e}")
        except Exception as e:
             raise CrossMatchError(f"Unexpected error loading config {self.config_file}: {e}")

    def _validate_config(self):
        """Validates the loaded configuration."""
        if not self.catalogues_config:
            logger.warning("Configuration file missing or empty 'catalogues' section.")
        if not self.methods_config:
             logger.warning("Configuration file missing or empty 'crossmatch_methods' section.")
        # Add more specific validation rules as needed (e.g., check required keys for methods)

    def get_catalogue_config(self, catalogue_name: str) -> Dict[str, Any]:
        """Retrieves configuration for a specific catalogue with caching."""
        if catalogue_name in self._catalogue_config_cache:
            return self._catalogue_config_cache[catalogue_name]
            
        if catalogue_name not in self.catalogues_config:
            raise CrossMatchError(f"Catalogue '{catalogue_name}' not found in configuration file: {self.config_file}")
            
        config = self.catalogues_config[catalogue_name]
        self._catalogue_config_cache[catalogue_name] = config
        return config

    def _read_catalogue(self, catalogue_path_or_df: Union[str, Path, pd.DataFrame, Table]) -> pd.DataFrame:
        """Reads a catalogue into a pandas DataFrame, handling various input types."""
        logger.debug(f"Reading catalogue: {type(catalogue_path_or_df)}")
        if isinstance(catalogue_path_or_df, pd.DataFrame):
            return catalogue_path_or_df.copy()
        if isinstance(catalogue_path_or_df, Table):
            logger.debug("Converting astropy Table to pandas DataFrame")
            df = catalogue_path_or_df.to_pandas()
            # Convert byte strings (often from FITS) and object columns to standard strings
            for col in df.select_dtypes(include=['object', 'bytes']).columns:
                try:
                     # Decode bytes if possible, otherwise convert to string
                     if df[col].iloc[0] is not None and isinstance(df[col].iloc[0], bytes):
                           df[col] = df[col].str.decode('utf-8', errors='ignore')
                     # Convert remaining objects safely to strings
                     df[col] = df[col].astype(str)
                except Exception as e:
                     logger.warning(f"Could not safely convert column '{col}' to string: {e}")
            # Handle masked values (convert to NaN)
            # This is implicitly handled by to_pandas() now, but good to be aware of.
            return df

        catalogue_path = Path(catalogue_path_or_df)
        if not catalogue_path.is_file():
            raise FileNotFoundError(f"Input catalogue file not found: {catalogue_path}")

        suffix = catalogue_path.suffix.lower()
        logger.info(f"Reading file: {catalogue_path} (format: {suffix})")
        try:
            if suffix == '.parquet':
                return pd.read_parquet(catalogue_path)
            elif suffix == '.fits':
                with warnings.catch_warnings(): # Suppress FITSFixedWarning etc.
                     warnings.simplefilter("ignore")
                     table = Table.read(catalogue_path)
                return self._read_catalogue(table) # Reuse Table conversion logic
            elif suffix == '.csv':
                return pd.read_csv(catalogue_path)
            else:
                try:
                     logger.warning(f"Attempting generic read for {catalogue_path} with astropy")
                     table = Table.read(catalogue_path)
                     return self._read_catalogue(table)
                except Exception as e_astropy:
                     raise CrossMatchError(f"Unsupported input file format: {suffix}. Error: {e_astropy}")
        except Exception as e:
            raise CrossMatchError(f"Error reading catalogue file {catalogue_path}: {e}")

    def _write_catalogue(self, df: pd.DataFrame, output_file: Union[str, Path]):
        """Writes a pandas DataFrame to a parquet file."""
        output_path = Path(output_file)
        logger.info(f"Writing output to: {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, compression='snappy', index=False)
        except Exception as e:
            raise CrossMatchError(f"Error writing output file {output_path}: {e}")

    def _process_in_parallel(self, 
                           catalogue_df: pd.DataFrame, 
                           process_func: Callable,
                           n_chunks: int = 4, 
                           **kwargs) -> pd.DataFrame:
        """Process large catalogues in parallel chunks and combine results."""
        # Determine optimal chunk count based on CPU cores if not specified
        if n_chunks <= 0:
            n_chunks = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid excessive memory usage
        
        total_rows = len(catalogue_df)
        if total_rows == 0:
            logger.warning("Empty DataFrame provided to parallel processing")
            return pd.DataFrame()
            
        # Split dataframe into chunks ensuring all rows are included
        chunk_size = max(1, (total_rows + n_chunks - 1) // n_chunks)  # Ceiling division
        chunks = [catalogue_df.iloc[i:min(i+chunk_size, total_rows)] for i in range(0, total_rows, chunk_size)]
        logger.info(f"Processing {total_rows} rows in {len(chunks)} chunks of ~{chunk_size} rows each")
        
        # If very small dataset, process sequentially
        if total_rows < 1000:
            logger.info(f"Small dataset detected ({total_rows} rows), processing sequentially")
            return process_func(catalogue_df, **kwargs)
        
        results = []
        failed_chunks = []
        
        with ProcessPoolExecutor(max_workers=n_chunks) as executor:
            futures = {executor.submit(process_func, chunk, **kwargs): i 
                      for i, chunk in enumerate(chunks)}
            
            # Collect results as they complete
            for future in futures:
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        results.append(result)
                    else:
                        logger.warning(f"Chunk {chunk_idx} returned empty or invalid result")
                except Exception as e:
                    logger.error(f"Error in chunk {chunk_idx}: {e}")
                    failed_chunks.append(chunk_idx)
            
        # If all chunks failed, raise error
        if failed_chunks and len(failed_chunks) == len(chunks):
            raise CrossMatchError(f"All {len(chunks)} parallel chunks failed processing")
        
        # If some chunks failed but others succeeded, log warning
        if failed_chunks:
            logger.warning(f"{len(failed_chunks)} out of {len(chunks)} chunks failed processing")
        
        # Combine results if we got any
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            raise CrossMatchError("No valid results returned from parallel processing")

    def _process_in_parallel_joblib(self, 
                           catalogue_df: pd.DataFrame, 
                           process_func: Callable,
                           n_chunks: int = 4, 
                           backend: str = 'multiprocessing',
                           timeout: Optional[int] = None,
                           **kwargs) -> pd.DataFrame:
        """Process large catalogues in parallel chunks using joblib.
        
        Args:
            catalogue_df: Input DataFrame to process
            process_func: Function to apply to each chunk 
            n_chunks: Number of chunks to split the data into
            backend: Joblib parallel backend ('multiprocessing', 'threading', 'loky')
            timeout: Optional timeout per job in seconds
            **kwargs: Additional arguments passed to process_func
        
        Returns:
            Combined DataFrame with results from all chunks
        """
        if n_chunks <= 0:
            n_chunks = min(multiprocessing.cpu_count(), 8)  # Cap at reasonable value
            
        total_rows = len(catalogue_df)
        if total_rows == 0:
            logger.warning("Empty DataFrame provided to parallel processing")
            return pd.DataFrame()
            
        # If very small dataset, process sequentially
        if total_rows < 1000:
            logger.info(f"Small dataset detected ({total_rows} rows), processing sequentially")
            return process_func(catalogue_df, **kwargs)

        # Split dataframe into chunks ensuring all rows are included
        chunk_size = max(1, (total_rows + n_chunks - 1) // n_chunks)  # Ceiling division
        chunks = [catalogue_df.iloc[i:min(i+chunk_size, total_rows)] for i in range(0, total_rows, chunk_size)]
        logger.info(f"Processing {total_rows} rows in {len(chunks)} chunks of ~{chunk_size} rows each using joblib ({backend} backend)")

        try:
            parallel_args = {
                'n_jobs': n_chunks,
                'backend': backend,
                'verbose': 10,  # Increased verbosity for debugging
            }
            
            if timeout:
                parallel_args['timeout'] = timeout
                
            results = Parallel(**parallel_args)(
                delayed(process_func)(chunk, **kwargs) for chunk in chunks
            )

            # Filter out any non-DataFrame results or errors
            valid_results = [res for res in results if isinstance(res, pd.DataFrame) and not res.empty]
            
            if not valid_results:
                if results and all(result is None for result in results):
                    logger.warning("All chunks returned None - check if process_func is returning properly")
                raise CrossMatchError(f"No valid DataFrames returned from parallel processing ({len(results)} chunks processed)")
                
            logger.info(f"Successfully processed {len(valid_results)} out of {len(chunks)} chunks")
            return pd.concat(valid_results, ignore_index=True)

        except Exception as e:
            logger.error(f"Error in parallel processing with joblib: {str(e)}")
            # Try sequential processing as fallback for debugging
            logger.warning("Attempting sequential processing as fallback...")
            try:
                return process_func(catalogue_df, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Sequential fallback also failed: {str(fallback_error)}")
                raise CrossMatchError(f"Both parallel and sequential processing failed: {str(e)}")

    def crossmatch(
        self,
        catalogue_1: Union[str, Path, pd.DataFrame, Table],
        catalogue_2: Union[str, Path, pd.DataFrame, Table],
        output_file: Union[str, Path],
        method: Optional[str] = None,
        radius_arcsec: Optional[float] = None,
        id_column_1: Optional[str] = None, 
        id_column_2: Optional[str] = None, 
        columns_1: Optional[List[str]] = None,
        columns_2: Optional[List[str]] = None,
        swap_catalogues: bool = False,
        parallel: bool = False,
        n_chunks: int = 4,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **kwargs
    ) -> str:
        """
        Performs the cross-match operation, handling direct and chained joins.
        
        Args:
            catalogue_1: The first catalogue (file path, DataFrame, Table or name from config)
            catalogue_2: The second catalogue (file path, DataFrame, Table or name from config)
            output_file: Path where the cross-matched result will be saved
            method: Cross-matching method to use (overrides config)
            radius_arcsec: Match radius in arcseconds (for spatial matches)
            id_column_1: Column in catalogue_1 to use for ID-based matching
            id_column_2: Column in catalogue_2 to use for ID-based matching
            columns_1: List of columns to keep from catalogue_1
            columns_2: List of columns to retrieve from catalogue_2
            swap_catalogues: Whether to swap the order of catalogues (making cat1=cat2 and cat2=cat1)
            parallel: Whether to use parallel processing for large catalogues
            n_chunks: Number of chunks for parallel processing
            progress_callback: Optional callback function to report progress (message, percent)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Path to the output file with cross-matched results
        """
        def report_progress(message, percent=None):
            if progress_callback:
                progress_callback(message, percent or 0.0)
            logger.info(message)
        
        report_progress("Starting cross-match process", 0.0)
        
        # Swap catalogues if requested
        if swap_catalogues:
            report_progress("Swapping catalogues as requested", 0.1)
            catalogue_1, catalogue_2 = catalogue_2, catalogue_1
            id_column_1, id_column_2 = id_column_2, id_column_1
            columns_1, columns_2 = columns_2, columns_1
        
        try:
            # Read catalogue_1
            initial_catalogue_1_df = self._read_catalogue(catalogue_1)
            report_progress("Catalogue 1 loaded successfully", 0.2)

            # Apply columns_1 filter if specified
            current_catalogue_1_df = initial_catalogue_1_df
            if columns_1:
                 try:
                     cols_to_keep = set(columns_1)
                     if id_column_1: cols_to_keep.add(id_column_1)
                     current_catalogue_1_df = initial_catalogue_1_df[list(cols_to_keep)]
                     report_progress(f"Limited initial catalogue_1 columns to: {list(cols_to_keep)}", 0.3)
                 except KeyError as e:
                     raise CrossMatchError(f"Column not found in catalogue_1 while applying columns_1: {e}")
            elif id_column_1 and id_column_1 not in current_catalogue_1_df.columns:
                 raise CrossMatchError(f"Specified id_column_1 '{id_column_1}' not found in catalogue_1.")

            # Determine if catalogue_2 is a DataFrame, Table, local file or a named catalog in config
            catalogue_2_config = None
            is_local_catalogue = False
            
            if isinstance(catalogue_2, (pd.DataFrame, Table)):
                # If catalogue_2 is already a DataFrame or Table, use it directly
                report_progress("Using provided DataFrame/Table as catalogue_2", 0.4)
                catalogue_2_df = self._read_catalogue(catalogue_2)
                catalogue_2 = catalogue_2_df  # Use the DataFrame directly
            elif isinstance(catalogue_2, (str, Path)):
                # Check if it's a local file
                if Path(catalogue_2).is_file():
                    report_progress(f"Using local file as catalogue_2: {catalogue_2}", 0.5)
                    is_local_catalogue = True
                else:
                    try:
                        # Check if it's a named catalog in config
                        catalogue_2_config = self.get_catalogue_config(catalogue_2)
                        report_progress(f"Using configured catalog as catalogue_2: {catalogue_2}", 0.6)
                    except CrossMatchError:
                        # If not in config and not a local file, raise error
                        if not Path(catalogue_2).exists():
                            raise CrossMatchError(f"catalogue_2 '{catalogue_2}' is neither a configured catalog name nor an existing file")
                        # Otherwise, assume it's a local file with a path issue
                        logger.warning(f"catalogue_2 path '{catalogue_2}' not found as a file or catalog name, but file exists. Proceeding with caution.")
                        is_local_catalogue = True
            else:
                raise CrossMatchError(f"Unsupported type for catalogue_2: {type(catalogue_2)}")

            # Check for intermediate join configuration
            intermediate_join_config = None
            intermediate_catalogue_name = None
            intermediate_catalogue_1_col = None
            intermediate_catalogue_2_col = None
            final_catalogue_2_join_col = None
            
            if catalogue_2_config and 'intermediate_join' in catalogue_2_config:
                intermediate_join_config = catalogue_2_config['intermediate_join']
                intermediate_catalogue_name = intermediate_join_config.get('via_catalogue')
                intermediate_catalogue_1_col = intermediate_join_config.get('intermediate_catalogue_1_column')
                intermediate_catalogue_2_col = intermediate_join_config.get('intermediate_catalogue_2_column')
                final_catalogue_2_join_col = intermediate_join_config.get('final_catalogue_2_join_column')
                
                # Validate intermediate join config
                if not (intermediate_catalogue_name and intermediate_catalogue_1_col and 
                       intermediate_catalogue_2_col and final_catalogue_2_join_col):
                    raise CrossMatchError("Incomplete intermediate_join configuration. Required: via_catalogue, intermediate_catalogue_1_column, intermediate_catalogue_2_column, final_catalogue_2_join_column")
                
                # Check if intermediate catalog exists in config
                try:
                    self.get_catalogue_config(intermediate_catalogue_name)
                    report_progress(f"Using intermediate catalog for join: {intermediate_catalogue_name}", 0.7)
                except CrossMatchError:
                    raise CrossMatchError(f"Intermediate catalog '{intermediate_catalogue_name}' not found in configuration")

            # Prepare STILTS and other configuration parameters
            final_kwargs_for_steps = {} 
            final_kwargs_for_steps.update(self.stilts_config) 
            final_kwargs_for_steps.update(kwargs) 
            if self.stilts_cmd_base: final_kwargs_for_steps['stilts_cmd_base'] = self.stilts_cmd_base
            if self.stilts_java_opts: final_kwargs_for_steps['java_opts'] = self.stilts_java_opts
            if self.stilts_tmpdir: final_kwargs_for_steps['tmpdir'] = self.stilts_tmpdir
            final_kwargs_for_steps['chunk_size'] = self.chunk_size
            final_kwargs_for_steps.setdefault('timeout', 600) 
            final_kwargs_for_steps.setdefault('retry_delay', 60) 

            # Perform the join(s)
            if intermediate_join_config:
                report_progress(f"Starting chained join through intermediate catalog {intermediate_catalogue_name}", 0.8)
                
                # Step 1: Join catalogue_1 with intermediate catalog
                step1_df = self._perform_single_join(
                    catalogue_1_df=current_catalogue_1_df,
                    catalogue_2=intermediate_catalogue_name,
                    method_hint='tap',
                    id_column_1_override=id_column_1, 
                    id_column_2_override=intermediate_catalogue_1_col, 
                    columns_2_override=[intermediate_catalogue_2_col],
                    final_kwargs=final_kwargs_for_steps
                )
                
                # Step 2: Join intermediate result with final catalog
                final_df = self._perform_single_join(
                    catalogue_1_df=step1_df,
                    catalogue_2=catalogue_2,
                    method_hint=catalogue_2_config.get('best_method', 'tap'), 
                    id_column_1_override=intermediate_catalogue_2_col, 
                    id_column_2_override=final_catalogue_2_join_col, 
                    columns_2_override=columns_2,
                    final_kwargs=final_kwargs_for_steps 
                )
                result_df = final_df

            else:
                report_progress(f"Performing direct join: Catalogue 1 -> '{catalogue_2}'", 0.9)
                result_df = self._perform_single_join(
                    catalogue_1_df=current_catalogue_1_df,
                    catalogue_2=catalogue_2,
                    method_hint=method,
                    id_column_1_override=id_column_1,
                    id_column_2_override=id_column_2, 
                    columns_2_override=columns_2,
                    radius_override=radius_arcsec,
                    final_kwargs=final_kwargs_for_steps
                )

            if result_df is None or not isinstance(result_df, pd.DataFrame):
                 raise CrossMatchError("Cross-match process did not return a valid DataFrame.")

            self._write_catalogue(result_df, output_file)
            report_progress("Cross-match completed successfully", 1.0)
            return str(output_file)

        except (CrossMatchError, StiltsError, TapError, FileNotFoundError) as e:
            logger.error(f"Cross-match process failed: {e}")
            raise e 
        except Exception as e:
            logger.exception("Cross-match process failed unexpectedly.")
            raise CrossMatchError(f"Cross-match failed: {e}") from e
            
    def _find_coord_cols(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Attempt to automatically find RA and Dec columns."""
        ra_patterns = ['ra', 'raj2000', 'rightascension']
        dec_patterns = ['dec', 'dej2000', 'declination']
        
        ra_col, dec_col = None, None
        df_cols_lower = {col.lower(): col for col in df.columns}

        for pattern in ra_patterns:
            if pattern in df_cols_lower:
                ra_col = df_cols_lower[pattern]
                break
        for pattern in dec_patterns:
            if pattern in df_cols_lower:
                dec_col = df_cols_lower[pattern]
                break
                
        if not ra_col or not dec_col:
            raise CrossMatchError(f"Could not automatically determine RA/Dec columns from patterns {ra_patterns}/{dec_patterns} in catalogue_1 columns: {list(df.columns)}. Please specify via configuration or arguments.")
            
        logger.info(f"Using catalogue_1 coordinate columns: RA='{ra_col}', Dec='{dec_col}'")
        return ra_col, dec_col