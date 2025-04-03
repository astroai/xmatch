import os
import yaml
import logging
from typing import Optional, Dict, List, Union, Any, Tuple
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
        """Retrieves configuration for a specific catalogue."""
        if catalogue_name not in self.catalogues_config:
            raise CrossMatchError(f"Catalogue '{catalogue_name}' not found in configuration file: {self.config_file}")
        return self.catalogues_config[catalogue_name]

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

    def _determine_best_method(self, catalogue_2_config: Optional[Dict[str, Any]],
                               is_local_catalogue: bool, id_column_1: Optional[str]) -> str:
        """Determines the best cross-matching method based on configuration and input."""
        if is_local_catalogue:
             if id_column_1:
                 logger.info("Local catalogue_2 with ID column: defaulting to local ID join (astropy_xmatch/pandas).")
                 if self.methods_config.get('astropy_xmatch', {}).get('enabled', False):
                     return 'astropy_xmatch'
                 else:
                     raise CrossMatchError("Local ID join requires 'astropy_xmatch' method to be enabled.")
             else:
                 logger.info("Local catalogue_2 without ID column: defaulting to local spatial join (stilts_tmatch2 or astropy_xmatch).")
                 if self.methods_config.get('stilts_tmatch2', {}).get('enabled', False):
                      return 'stilts_tmatch2'
                 elif self.methods_config.get('astropy_xmatch', {}).get('enabled', False):
                      return 'astropy_xmatch'
                 else:
                     raise CrossMatchError("No enabled local spatial cross-match method found (checked stilts_tmatch2, astropy_xmatch).")
        elif catalogue_2_config:
            best_method = catalogue_2_config.get('best_method')
            if best_method and self.methods_config.get(best_method, {}).get('enabled', True):
                logger.info(f"Using configured best method for catalogue_2: {best_method}")
                return best_method
            else:
                 logger.warning(f"Configured best_method '{best_method}' not found, not enabled, or uses old naming. Attempting fallback.")
                 if id_column_1 and self.methods_config.get('tap',{}).get('enabled', True):
                      logger.info("Fallback: Using 'tap' method for ID join.")
                      return 'tap'
                 if catalogue_2_config.get('tap_url') and self.methods_config.get('stilts_tapskymatch', {}).get('enabled', True):
                      logger.info("Fallback: Using 'stilts_tapskymatch' for TAP spatial join.")
                      return 'stilts_tapskymatch'
                 if catalogue_2_config.get('vizier_id') and self.methods_config.get('stilts_cdsskymatch', {}).get('enabled', True):
                      logger.info("Fallback: Using 'stilts_cdsskymatch' for Vizier spatial join.")
                      return 'stilts_cdsskymatch'
                 if self.methods_config.get('astropy_xmatch', {}).get('enabled', False):
                      logger.warning("Fallback: No suitable remote method found, falling back to 'astropy_xmatch' (requires local catalogue_2 data or download capability). This might fail if catalogue_2 is remote.")
                      return 'astropy_xmatch'
        
        logger.warning("Could not determine a preferred method. Defaulting to 'astropy_xmatch' if enabled.")
        if self.methods_config.get('astropy_xmatch', {}).get('enabled', False):
             return 'astropy_xmatch'
        else:
             raise CrossMatchError("Cannot determine cross-match method. No configuration or fallback method available/enabled.")

    def _perform_single_join(
        self,
        catalogue_1_df: pd.DataFrame,
        catalogue_2: Union[str, Path, pd.DataFrame, Table],
        method_hint: Optional[str] = None, 
        id_column_1_override: Optional[str] = None, 
        id_column_2_override: Optional[str] = None, 
        columns_2_override: Optional[List[str]] = None, 
        radius_override: Optional[float] = None,
        **kwargs
        ) -> pd.DataFrame:
        
        logger.info(f"--- Starting Single Join Step --- Catalogue 2: '{catalogue_2}'")
        
        # Handle catalogue_2 consistently regardless of type
        if isinstance(catalogue_2, (pd.DataFrame, Table)):
            # If catalogue_2 is already a DataFrame or Table, use it directly
            logger.info(f"Using provided DataFrame/Table as catalogue_2")
            is_local_catalogue = True
            catalogue_2_config = None
            effective_catalogue_2 = catalogue_2  # Pass the DataFrame/Table directly, don't read/write
        else:
            # String or Path
            is_local_catalogue = Path(catalogue_2).is_file()
            
            if is_local_catalogue:
                catalogue_2_path = Path(catalogue_2)
                effective_catalogue_2 = str(catalogue_2_path)  # Pass the file path directly
                catalogue_2_config = None
                logger.info(f"Catalogue 2 '{catalogue_2}' is a local file path.")
            else:
                # It's a catalog name in config
                catalogue_2_config = self.get_catalogue_config(catalogue_2)
                effective_catalogue_2 = catalogue_2
                logger.info(f"Catalogue 2 '{catalogue_2}' is a configured catalog name.")
        
        final_columns_2 = columns_2_override
        if not final_columns_2 and catalogue_2_config and 'default_columns' in catalogue_2_config:
            final_columns_2 = catalogue_2_config['default_columns']
            logger.info(f"Using default columns from catalog config: {final_columns_2}")

        method = method_hint 
        if not method:
            method = self._determine_best_method(catalogue_2_config, is_local_catalogue, id_column_1_override)
        
        logger.info(f"Selected method for this join step: {method}")
        
        current_method_config = self.methods_config.get(method, {})
        if not current_method_config.get('enabled', True):
             raise CrossMatchError(f"Method '{method}' selected for join step is disabled.")
        
        final_kwargs_for_step = kwargs.get('final_kwargs', {})
        
        common_params = {
            'catalogue_1_df': catalogue_1_df,
            'catalogue_2': effective_catalogue_2, 
            'catalogue_2_config': catalogue_2_config, 
            'is_local_catalogue': is_local_catalogue,
            'radius_arcsec': radius_override if radius_override is not None else catalogue_2_config.get('radius_arcsec', current_method_config.get('default_radius_arcsec', 1.0)),
            'id_column_1': id_column_1_override,
            'id_column_2': id_column_2_override,
            'ra_column_1': None,
            'dec_column_1': None,
            'ra_column_2': None,
            'dec_column_2': None,
            'columns_1': None,
            'columns_2': final_columns_2,
            'kwargs': final_kwargs_for_step
        }
        if not common_params['id_column_1']:
             common_params['ra_column_1'], common_params['dec_column_1'] = self._find_coord_cols(catalogue_1_df)
             common_params['ra_column_2'] = catalogue_2_config.get('ra_column', common_params['ra_column_1']) if catalogue_2_config else common_params['ra_column_1']
             common_params['dec_column_2'] = catalogue_2_config.get('dec_column', common_params['dec_column_1']) if catalogue_2_config else common_params['dec_column_1']

        radius_arcsec = common_params['radius_arcsec'] 
        if common_params['id_column_1']:
             logger.info(f"Performing ID-based join (Catalogue 1 Col: '{common_params['id_column_1']}', Catalogue 2 Col: '{common_params['id_column_2']}')")
        else:
             if common_params['ra_column_1'] and common_params['dec_column_1']:
                 logger.info(f"Performing spatial join (Radius: {radius_arcsec:.2f} arcsec, Catalogue 1 Coords: '{common_params['ra_column_1']}', '{common_params['dec_column_1']}')")
             else:
                 logger.warning("Attempting spatial join, but could not determine coordinate columns.")

        if common_params['columns_2']:
            logger.info(f"Requesting catalogue 2 columns: {common_params['columns_2']}")
        else:
             logger.info(f"Requesting default catalogue 2 columns (method-dependent).")

        logger.info(f"Executing join step using method: '{method}'")
        result_df = None
        try: 
            if method == 'astropy_xmatch': 
                result_df = self._crossmatch_astropy(**common_params)
            elif method == 'local_id': 
                 logger.warning("Method 'local_id' called directly, prefer 'astropy_xmatch' for local ID joins.")
                 result_df = self._crossmatch_local_id(**common_params)
            elif method in ['stilts_cdsskymatch', 'stilts_tapskymatch', 'stilts_tmatch2', 'stilts_tapquery']: 
                 result_df = self._crossmatch_stilts(method=method, **common_params)
            elif method == 'tap': 
                 result_df = self._crossmatch_tap(**common_params)
            elif method == 'astroquery_cdsxmatch': 
                 result_df = self._crossmatch_cdsxmatch(**common_params)
            else:
                raise CrossMatchError(f"Unsupported or unimplemented method selected: '{method}'")

            if result_df is None or not isinstance(result_df, pd.DataFrame):
                 raise CrossMatchError(f"Join step method '{method}' did not return a valid DataFrame.")
            
            logger.info(f"--- Completed Single Join Step --- Method: {method}, Result Shape: {result_df.shape}")
            return result_df
            
        except (StiltsError, TapError, FileNotFoundError, CrossMatchError) as e:
            logger.error(f"Error during single join step against '{effective_catalogue_2}' using method '{method}': {e}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error during single join step against '{effective_catalogue_2}' using method '{method}'.")
            raise CrossMatchError(f"Unexpected error in join step method '{method}': {e}") from e

    def _crossmatch_stilts(self, method, catalogue_1_df, catalogue_2, catalogue_2_config,
                         is_local_catalogue, radius_arcsec, id_column_1, id_column_2, ra_column_1, dec_column_1,
                         ra_column_2, dec_column_2, columns_1, columns_2, kwargs):
        """
        Perform cross-matching using STILTS (Starlink Tables Infrastructure Library Tool Set).
        
        Parameters
        ----------
        method : str
            The specific STILTS method to use (e.g., 'stilts_cdsskymatch').
        catalogue_1_df : pandas.DataFrame
            DataFrame containing the first catalogue data.
        catalogue_2 : Union[str, Path, pd.DataFrame, Table]
            The second catalogue (file path, DataFrame, Table or name from config)
        catalogue_2_config : dict or None
            Configuration for catalogue_2 if it's a named catalog
        is_local_catalogue : bool
            Whether catalogue_2 is a local file/DataFrame/Table
        Other parameters are passed through from _perform_single_join
                
        Returns
        -------
        pandas.DataFrame
            The result of the cross-matching operation.
        """
        logger.info(f"Performing STILTS cross-match using method: {method}")
        
        # Extract STILTS configuration parameters
        stilts_cmd_base = kwargs.get('stilts_cmd_base', self.stilts_cmd_base)
        java_opts = kwargs.get('java_opts', self.stilts_java_opts)
        tmpdir = kwargs.get('tmpdir', self.stilts_tmpdir)
        
        # Common keyword arguments for all STILTS methods
        stilts_kwargs = {
            'java_opts': java_opts,
            'tmpdir': tmpdir,
            'stilts_cmd_base': stilts_cmd_base
        }
        
        # Add any additional method-specific kwargs
        for k, v in kwargs.items():
            if k not in stilts_kwargs:
                stilts_kwargs[k] = v
        
        result_file = None
        
        if method == 'stilts_cdsskymatch':
            if not catalogue_2_config or 'vizier_id' not in catalogue_2_config:
                raise CrossMatchError(f"Method {method} requires 'vizier_id' in catalogue_2 config")
            
            vizier_id = catalogue_2_config['vizier_id']
            result_file = stilts.stilts_cdsskymatch(
                catalogue_1_df=catalogue_1_df,
                vizier_id=vizier_id,
                radius=radius_arcsec,
                ra_column_1=ra_column_1,
                dec_column_1=dec_column_1,
                columns_2=columns_2,
                **stilts_kwargs
            )
        
        elif method == 'stilts_tapskymatch':
            if not catalogue_2_config or 'tap_url' not in catalogue_2_config or 'tap_table' not in catalogue_2_config:
                raise CrossMatchError(f"Method {method} requires 'tap_url' and 'tap_table' in catalogue_2 config")
            
            tap_url = catalogue_2_config['tap_url']
            tap_table = catalogue_2_config['tap_table']
            tap_schema = catalogue_2_config.get('tap_schema')
            
            # Determine whether auth is required for this TAP service
            service_name = None
            for known_service, config in self.auth_config.items():
                if tap_url in config.get('urls', []) or known_service in tap_url.lower():
                    service_name = known_service
                    break
            
            if service_name and service_name in self.auth_config:
                # Add auth parameters from secure storage
                for auth_key, auth_value in self.auth_config[service_name].items():
                    stilts_kwargs[auth_key] = auth_value
            
            result_file = stilts.stilts_tapskymatch(
                catalogue_1_df=catalogue_1_df,
                tap_url=tap_url,
                tap_table=tap_table,
                radius=radius_arcsec,
                ra_column_1=ra_column_1,
                dec_column_1=dec_column_1,
                ra_column_2=ra_column_2,
                dec_column_2=dec_column_2,
                tap_schema=tap_schema,
                columns_2=columns_2,
                **stilts_kwargs
            )
        
        elif method == 'stilts_tmatch2':
            if is_local_catalogue:
                # For local file path, use it directly
                if isinstance(catalogue_2, (str, Path)) and Path(catalogue_2).is_file():
                    # Direct file path - most efficient
                    result_file = stilts.stilts_tmatch2(
                        catalogue_1_df=catalogue_1_df,
                        radius=radius_arcsec,
                        ra_column_1=ra_column_1,
                        dec_column_1=dec_column_1,
                        ra_column_2=ra_column_2,
                        dec_column_2=dec_column_2,
                        catalogue_2_path=catalogue_2,  # Pass the file path directly
                        columns_2=columns_2,
                        id_column_1=id_column_1,
                        **stilts_kwargs
                    )
                # For DataFrame/Table, we need to convert (no way around it for STILTS)
                elif isinstance(catalogue_2, (pd.DataFrame, Table)):
                    # Need conversion since STILTS requires a file
                    result_file = stilts.stilts_tmatch2(
                        catalogue_1_df=catalogue_1_df,
                        radius=radius_arcsec,
                        ra_column_1=ra_column_1,
                        dec_column_1=dec_column_1,
                        ra_column_2=ra_column_2,
                        dec_column_2=dec_column_2,
                        catalogue_2_df=catalogue_2,  # Pass DataFrame/Table directly and let stilts handle conversion
                        columns_2=columns_2,
                        id_column_1=id_column_1,
                        **stilts_kwargs
                    )
            else:
                # TAP-based remote crossmatch
                if not catalogue_2_config or 'tap_url' not in catalogue_2_config or 'tap_table' not in catalogue_2_config:
                    raise CrossMatchError(f"Method {method} with remote target requires 'tap_url' and 'tap_table' in catalogue_2 config")
                
                tap_url = catalogue_2_config['tap_url']
                tap_table = catalogue_2_config['tap_table']
                tap_schema = catalogue_2_config.get('tap_schema')
                
                # Add auth if available
                service_name = None
                for known_service, config in self.auth_config.items():
                    if tap_url in config.get('urls', []) or known_service in tap_url.lower():
                        service_name = known_service
                        break
                
                if service_name and service_name in self.auth_config:
                    for auth_key, auth_value in self.auth_config[service_name].items():
                        stilts_kwargs[auth_key] = auth_value
                
                result_file = stilts.stilts_tmatch2(
                    catalogue_1_df=catalogue_1_df,
                    radius=radius_arcsec,
                    ra_column_1=ra_column_1,
                    dec_column_1=dec_column_1,
                    ra_column_2=ra_column_2,
                    dec_column_2=dec_column_2,
                    target_tap_url=tap_url,
                    target_tap_table=tap_table,
                    target_tap_schema=tap_schema,
                    columns_2=columns_2,
                    id_column_1=id_column_1,
                    **stilts_kwargs
                )
        
        elif method == 'stilts_tapquery':
            if not catalogue_2_config or 'tap_url' not in catalogue_2_config:
                raise CrossMatchError(f"Method {method} requires 'tap_url' in catalogue_2 config")
            
            tap_url = catalogue_2_config['tap_url']
            
            # Build ADQL query
            if id_column_1 and id_column_1 in catalogue_1_df.columns:
                # ID-based join
                if not id_column_2:
                    id_column_2 = catalogue_2_config.get('id_column', id_column_1)
                
                # Get unique IDs
                unique_ids = catalogue_1_df[id_column_1].unique()
                if len(unique_ids) > 1000:
                    logger.warning(f"Large number of IDs ({len(unique_ids)}) for TAP query. This might fail.")
                
                # Format IDs for SQL
                id_list = ", ".join([f"'{id}'" if isinstance(id, str) else str(id) for id in unique_ids])
                
                # Build columns list
                if columns_2:
                    cols_str = ", ".join([id_column_2] + columns_2)
                else:
                    cols_str = "*"
                
                # Build table name
                table_name = catalogue_2_config['tap_table']
                if catalogue_2_config.get('tap_schema'):
                    table_name = f"{catalogue_2_config['tap_schema']}.{table_name}"
                
                adql_query = f"SELECT {cols_str} FROM {table_name} WHERE {id_column_2} IN ({id_list})"
            else:
                # Spatial join using ADQL's CONTAINS function
                # Build columns list
                if columns_2:
                    cols_str = ", ".join([ra_column_2, dec_column_2] + columns_2)
                else:
                    cols_str = "*"
                
                # Build table name
                table_name = catalogue_2_config['tap_table']
                if catalogue_2_config.get('tap_schema'):
                    table_name = f"{catalogue_2_config['tap_schema']}.{table_name}"
                
                # Build spatial constraints
                constraints = []
                for _, row in catalogue_1_df.iterrows():
                    constraints.append(
                        f"CONTAINS(POINT('ICRS', {ra_column_2}, {dec_column_2}), "
                        f"CIRCLE('ICRS', {row[ra_column_1]}, {row[dec_column_1]}, {radius_arcsec/3600.0}))"
                    )
                
                where_clause = " OR ".join(constraints)
                adql_query = f"SELECT {cols_str} FROM {table_name} WHERE {where_clause}"
            
            logger.debug(f"ADQL Query: {adql_query}")
            
            # Add auth if available
            service_name = None
            for known_service, config in self.auth_config.items():
                if tap_url in config.get('urls', []) or known_service in tap_url.lower():
                    service_name = known_service
                    break
            
            if service_name and service_name in self.auth_config:
                for auth_key, auth_value in self.auth_config[service_name].items():
                    stilts_kwargs[auth_key] = auth_value
            
            result_file = stilts.stilts_tapquery(
                catalogue_1_df=catalogue_1_df,
                tap_url=tap_url,
                adql_query=adql_query,
                **stilts_kwargs
            )
        
        else:
            raise CrossMatchError(f"Unsupported STILTS method: {method}")
        
        # Read the result file
        if result_file and Path(result_file).exists():
            try:
                result_df = pd.read_parquet(result_file)
                # Clean up the temporary file
                Path(result_file).unlink(missing_ok=True)
                return result_df
            except Exception as e:
                raise CrossMatchError(f"Failed to read STILTS result file: {e}")
        else:
            raise CrossMatchError(f"STILTS method {method} did not produce a valid result file")

    def _crossmatch_astropy(self, catalogue_1_df, catalogue_2, catalogue_2_config, 
                           is_local_catalogue, radius_arcsec, id_column_1, id_column_2, ra_column_1, 
                           dec_column_1, ra_column_2, dec_column_2, columns_1, columns_2, kwargs):
        """Perform cross-matching using astropy's matching functions."""
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        from astropy import units as u
        
        logger.info(f"Performing cross-match using astropy")
        
        # Handle ID-based join if specified
        if id_column_1 and id_column_1 in catalogue_1_df.columns:
            if not is_local_catalogue:
                raise CrossMatchError("ID-based join with astropy requires a local catalogue_2 file or DataFrame")
            
            logger.info(f"Using ID-based join on columns {id_column_1}/{id_column_2}")
            
            # Get catalogue_2 as DataFrame (only read from file if necessary)
            if isinstance(catalogue_2, (pd.DataFrame, Table)):
                catalogue_2_df = catalogue_2 if isinstance(catalogue_2, pd.DataFrame) else catalogue_2.to_pandas()
            else:
                # Only read the file if it's a file path
                catalogue_2_df = self._read_catalogue(catalogue_2)
            
            # Ensure ID column exists in catalogue_2
            effective_id_column_2 = id_column_2 if id_column_2 else id_column_1
            if effective_id_column_2 not in catalogue_2_df.columns:
                raise CrossMatchError(f"ID column '{effective_id_column_2}' not found in catalogue_2")
            
            # Perform merge
            merged_df = pd.merge(
                catalogue_1_df,
                catalogue_2_df,
                left_on=id_column_1,
                right_on=effective_id_column_2,
                how='left',
                suffixes=('', '_catalogue_2')
            )
            
            # Filter columns if specified
            if columns_2:
                # Keep only specified columns from catalogue_2
                all_cols = list(catalogue_1_df.columns)
                for col in columns_2:
                    if col in catalogue_2_df.columns:
                        all_cols.append(f"{col}_catalogue_2" if col in catalogue_1_df.columns else col)
                
                merged_df = merged_df[all_cols]
            
            logger.info(f"ID-based join completed with {len(merged_df)} entries")
            return merged_df
        
        # Spatial matching
        # First convert catalogue_1 coordinates to SkyCoord
        coords1 = SkyCoord(
            ra=catalogue_1_df[ra_column_1].values * u.deg,
            dec=catalogue_1_df[dec_column_1].values * u.deg,
            frame='icrs'
        )
        
        # Read and process catalogue_2
        if is_local_catalogue:
            # Get catalogue_2 as DataFrame (only read from file if necessary)
            if isinstance(catalogue_2, (pd.DataFrame, Table)):
                catalogue_2_df = catalogue_2 if isinstance(catalogue_2, pd.DataFrame) else catalogue_2.to_pandas()
            else:
                catalogue_2_df = self._read_catalogue(catalogue_2)
            
            # Verify coordinate columns
            effective_ra_col_2 = ra_column_2 if ra_column_2 else ra_column_1
            effective_dec_col_2 = dec_column_2 if dec_column_2 else dec_column_1
            
            if effective_ra_col_2 not in catalogue_2_df.columns or effective_dec_col_2 not in catalogue_2_df.columns:
                raise CrossMatchError(f"Coordinate columns '{effective_ra_col_2}'/'{effective_dec_col_2}' not found in catalogue_2")
            
            # Convert catalogue_2 coordinates to SkyCoord
            coords2 = SkyCoord(
                ra=catalogue_2_df[effective_ra_col_2].values * u.deg,
                dec=catalogue_2_df[effective_dec_col_2].values * u.deg,
                frame='icrs'
            )
            
            # Perform the cross-match
            idx, d2d, _ = match_coordinates_sky(coords1, coords2)
            
            # Filter matches by radius
            max_sep = radius_arcsec * u.arcsec
            matches = d2d < max_sep
            match_indices = idx[matches]
            
            # Create result DataFrame
            result_df = catalogue_1_df.copy()
            
            # Add a separation column
            result_df['separation_arcsec'] = float('nan')
            result_df.loc[matches, 'separation_arcsec'] = d2d[matches].to(u.arcsec).value
            
            # Add catalogue_2 data for matches
            if columns_2:
                cols_to_add = [c for c in columns_2 if c in catalogue_2_df.columns]
            else:
                cols_to_add = [c for c in catalogue_2_df.columns if c not in result_df.columns]
            
            for col in cols_to_add:
                col_name = f"{col}_catalogue_2" if col in result_df.columns else col
                result_df[col_name] = float('nan')
                for i, match_idx in enumerate(match_indices):
                    matched_row_idx = np.where(matches)[0][i]
                    result_df.loc[matched_row_idx, col_name] = catalogue_2_df.loc[match_idx, col]
            
            logger.info(f"Spatial match completed. Found {sum(matches)} matches within {radius_arcsec} arcsec")
            return result_df
        
        else:
            # Remote catalogue - not feasible with pure astropy, need to download first
            raise CrossMatchError("Astropy spatial matching with remote catalogues requires download. Please use 'stilts_tapskymatch' or 'stilts_cdsskymatch' instead.")

    def _crossmatch_tap(self, catalogue_1_df, catalogue_2, catalogue_2_config, 
                      is_local_catalogue, radius_arcsec, id_column_1, id_column_2, ra_column_1, 
                      dec_column_1, ra_column_2, dec_column_2, columns_1, columns_2, kwargs):
        """Perform cross-matching using the TAP protocol with PyVO/pandas."""
        
        logger.info(f"Performing cross-match using TAP")
        
        # Set up common parameters
        tap_params = {
            'catalogue_1': catalogue_1_df,  # Pass DataFrame directly
            'chunk_size': kwargs.get('chunk_size', self.chunk_size),
            'verbose': kwargs.get('verbose', logger.level <= logging.INFO),
            'retry_delay': kwargs.get('retry_delay', 60),
            'timeout': kwargs.get('timeout', 600)
        }
        
        # Determine if we're doing an ID or spatial join
        is_id_join = id_column_1 and id_column_1 in catalogue_1_df.columns
        
        if is_id_join:
            logger.info(f"Using ID-based TAP join on columns {id_column_1}/{id_column_2}")
            tap_params['id_column_1'] = id_column_1
            tap_params['id_column_2'] = id_column_2 if id_column_2 else catalogue_2_config.get('id_column', id_column_1)
        else:
            logger.info(f"Using spatial TAP join (radius: {radius_arcsec} arcsec)")
            tap_params['ra'] = ra_column_1
            tap_params['dec'] = dec_column_1
            tap_params['radius'] = radius_arcsec
        
        # Handle local vs remote catalogue
        if is_local_catalogue:
            if isinstance(catalogue_2, (str, Path)) and Path(catalogue_2).is_file():
                # Direct file path - most efficient
                logger.info(f"Using local file path for catalogue_2: {catalogue_2}")
                tap_params['local_table'] = str(catalogue_2)  # Pass file path directly
            elif isinstance(catalogue_2, (pd.DataFrame, Table)):
                # DataFrame/Table needs to be handled directly by TAP module
                logger.info(f"Using DataFrame/Table for catalogue_2")
                tap_params['local_table_df'] = catalogue_2  # Modified param name for clarity
        else:
            # Check for required TAP configuration
            if not catalogue_2_config or 'tap_url' not in catalogue_2_config or 'tap_table' not in catalogue_2_config:
                raise CrossMatchError(f"TAP method requires 'tap_url' and 'tap_table' in catalogue_2 config")
            
            logger.info(f"Using remote TAP service for catalogue_2: {catalogue_2_config['tap_url']}")
            tap_params['tap_url'] = catalogue_2_config['tap_url']
            tap_params['tap_table'] = catalogue_2_config['tap_table']
            tap_params['tap_schema'] = catalogue_2_config.get('tap_schema')
            
            # If coordinates haven't been provided for a spatial join
            if not is_id_join:
                if not ra_column_2:
                    tap_params['ra_column_2'] = catalogue_2_config.get('ra_column')
                if not dec_column_2:
                    tap_params['dec_column_2'] = catalogue_2_config.get('dec_column')
        
        # Add columns if specified
        if columns_2:
            tap_params['columns'] = columns_2
        elif catalogue_2_config and 'default_columns' in catalogue_2_config:
            tap_params['columns'] = catalogue_2_config['default_columns']
        
        # Add authentication if available
        if not is_local_catalogue:
            service_name = None
            tap_url = catalogue_2_config['tap_url']
            
            for known_service, config in self.auth_config.items():
                if tap_url in config.get('urls', []) or known_service in tap_url.lower():
                    service_name = known_service
                    break
            
            if service_name and service_name in self.auth_config:
                logger.info(f"Using authentication for TAP service: {service_name}")
                for auth_key, auth_value in self.auth_config[service_name].items():
                    tap_params[auth_key] = auth_value
        
        # Execute the TAP crossmatch
        try:
            result_file = tap.tap_crossmatch(**tap_params)
            
            if result_file and Path(result_file).exists():
                result_df = pd.read_parquet(result_file)
                # Clean up the temporary file
                Path(result_file).unlink(missing_ok=True)
                logger.info(f"TAP cross-match completed with {len(result_df)} entries")
                return result_df
            else:
                raise CrossMatchError(f"TAP cross-match did not produce a valid result file")
        
        except TapError as e:
            raise CrossMatchError(f"TAP cross-match failed: {e}")

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
            **kwargs: Additional method-specific parameters
            
        Returns:
            Path to the output file with cross-matched results
        """
        
        logger.info(f"===== Starting Cross-Match Process =====")
        
        # Swap catalogues if requested
        if swap_catalogues:
            logger.info("Swapping catalogues as requested")
            catalogue_1, catalogue_2 = catalogue_2, catalogue_1
            id_column_1, id_column_2 = id_column_2, id_column_1
            columns_1, columns_2 = columns_2, columns_1
        
        try:
            # Read catalogue_1
            initial_catalogue_1_df = self._read_catalogue(catalogue_1)

            # Apply columns_1 filter if specified
            current_catalogue_1_df = initial_catalogue_1_df
            if columns_1:
                 try:
                     cols_to_keep = set(columns_1)
                     if id_column_1: cols_to_keep.add(id_column_1)
                     current_catalogue_1_df = initial_catalogue_1_df[list(cols_to_keep)]
                     logger.info(f"Limited initial catalogue_1 columns to: {list(cols_to_keep)}")
                 except KeyError as e:
                     raise CrossMatchError(f"Column not found in catalogue_1 while applying columns_1: {e}")
            elif id_column_1 and id_column_1 not in current_catalogue_1_df.columns:
                 raise CrossMatchError(f"Specified id_column_1 '{id_column_1}' not found in catalogue_1.")

            # Determine if catalogue_2 is a DataFrame, Table, local file or a named catalog in config
            catalogue_2_config = None
            is_local_catalogue = False
            
            if isinstance(catalogue_2, (pd.DataFrame, Table)):
                # If catalogue_2 is already a DataFrame or Table, use it directly
                logger.info(f"Using provided DataFrame/Table as catalogue_2")
                catalogue_2_df = self._read_catalogue(catalogue_2)
                catalogue_2 = catalogue_2_df  # Use the DataFrame directly
            elif isinstance(catalogue_2, (str, Path)):
                # Check if it's a local file
                if Path(catalogue_2).is_file():
                    logger.info(f"Using local file as catalogue_2: {catalogue_2}")
                    is_local_catalogue = True
                else:
                    try:
                        # Check if it's a named catalog in config
                        catalogue_2_config = self.get_catalogue_config(catalogue_2)
                        logger.info(f"Using configured catalog as catalogue_2: {catalogue_2}")
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
                    logger.info(f"Using intermediate catalog for join: {intermediate_catalogue_name}")
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
                logger.info(f"Starting chained join through intermediate catalog {intermediate_catalogue_name}")
                
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
                logger.info(f"Performing direct join: Catalogue 1 -> '{catalogue_2}'")
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
            logger.info(f"Cross-match completed successfully. Output: {output_file}")
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