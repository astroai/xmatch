#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import yaml
import json

from .core.crossmatch import CrossMatch, CrossMatchError
from .utils.tap import TapError
from .utils.stilts import StiltsError

def setup_logging(level=logging.INFO):
    """Configure logging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)
    # Suppress overly verbose logs from dependencies if necessary
    logging.getLogger("pyvo").setLevel(logging.WARNING)

def parse_kwargs(kwargs_list):
    """Parse key=value pairs into a dictionary."""
    kwargs_dict = {}
    if kwargs_list:
        for item in kwargs_list:
            try:
                key, value = item.split('=', 1)
                # Attempt to convert value to number if possible
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass # Keep as string
                kwargs_dict[key.strip()] = value
            except ValueError:
                logging.warning(f"Ignoring invalid kwarg: '{item}'. Expected format: key=value")
    return kwargs_dict

def list_catalogues(config_path: str) -> int:
    """List all available catalogues from the config file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logging.error(f"Config file not found: {config_path}")
            return 1
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        if 'catalogues' not in config:
            logging.error("No catalogues section found in config file")
            return 1
            
        print("\nAvailable Catalogues:\n")
        print(f"{'Catalogue Name':<25} {'Description':<50}")
        print("-" * 75)
        
        for name, details in config['catalogues'].items():
            description = details.get('description', 'No description available')
            print(f"{name:<25} {description:<50}")
            
        return 0
    except Exception as e:
        logging.error(f"Error listing catalogues: {e}")
        return 1

def parse_args() -> argparse.Namespace:
    """Parse command line arguments with improved help and examples."""
    parser = argparse.ArgumentParser(
        description="Cross-match astronomical catalogues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cross-match Gaia DR3 with GALEX
  xmatch GAIA_ESA_TAP GALEX_CDS_XMATCH -o gaia_galex.parquet

  # Cross-match with custom radius and specific columns
  xmatch GAIA_ESA_TAP GALEX_CDS_XMATCH -r 3.0 -c "objid,FUVmag,NUVmag" -o gaia_galex.parquet

  # Perform ID-based cross-match
  xmatch gaiadr3_data.fits galex_data.fits --id-column-1 source_id --id-column-2 gaia_source_id

  # Use specific method and view detailed logging
  xmatch GAIA_ESA_TAP UKIDSS_NOAO_TAP -m stilts_tapskymatch --log-level DEBUG

  # List available catalogues
  xmatch --list-catalogues
"""
    )
    
    # Create a mutually exclusive group for main operation vs. listing catalogues
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--list-catalogues",
        action="store_true",
        help="List all available catalogues defined in the config file and exit."
    )
    
    group.add_argument(
        "catalogue_1",
        help="First catalogue: path to a local file (.parquet, .fits, .csv) or name of a catalogue defined in the config file.",
        nargs="?",  # Make it optional when using --list-catalogues
    )
    
    parser.add_argument(
        "catalogue_2",
        help="Second catalogue: path to a local file (.parquet, .fits, .csv) or name of a catalogue defined in the config file.",
        nargs="?"  # Make it optional when using --list-catalogues
    )
    
    parser.add_argument(
        "-o", "--output",
        default="crossmatch_output.parquet",
        help="Path for the output cross-matched file."
    )
    
    parser.add_argument(
        "-m", "--method",
        help="Cross-matching method to use (overrides config best_method). See config file for available methods."
    )
    
    parser.add_argument(
        "-r", "--radius",
        type=float,
        help="Cross-match radius in arcseconds (overrides config default)."
    )
    
    parser.add_argument(
        "-c", "--columns-2",
        help="Comma-separated list of columns to retrieve from catalogue 2 (e.g., 'ra,dec,gmag,rmag')."
    )
    
    parser.add_argument(
        "--id-column-1",
        help="Column name in catalogue 1 for ID-based exact matching (if applicable)."
    )
    
    parser.add_argument(
        "--id-column-2",
        help="Column name in catalogue 2 for ID-based exact matching (if applicable)."
    )
    
    parser.add_argument(
        "--columns-1",
        help="Comma-separated list of columns to keep from catalogue 1."
    )
    
    parser.add_argument(
        "--config",
        default="src/xmatch/config/catalogues.yaml",
        help="Path to the configuration file."
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size for processing (TAP/local joins)."
    )
    
    parser.add_argument(
        "--kwargs",
        nargs='*',
        help="Additional key=value arguments for the cross-match method (e.g., tap_timeout=300 java_opts='-Xmx8g')."
    )
    
    parser.add_argument(
        "--swap-catalogues",
        action="store_true",
        help="Swap the order of catalogues (use catalogue_2 as catalogue_1 and vice versa)."
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing for large catalogues where possible."
    )
    
    parser.add_argument(
        "--save-config",
        metavar="CONFIG_NAME",
        help="Save the current command configuration for future use."
    )
    
    parser.add_argument(
        "--load-config",
        metavar="CONFIG_NAME",
        help="Load a previously saved command configuration."
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for operations like TAP queries."
    )
    
    return parser.parse_args()

def save_config(args: argparse.Namespace, config_name: str):
    """Save current command arguments to a configuration file."""
    config_dir = Path.home() / ".xmatch"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / f"{config_name}.json"
    
    # Convert args to dictionary, excluding None values
    config = {k: v for k, v in vars(args).items() if v is not None}
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Configuration saved to {config_file}")

def load_config(config_name: str) -> Dict[str, Any]:
    """Load a saved command configuration."""
    config_file = Path.home() / ".xmatch" / f"{config_name}.json"
    
    if not config_file.exists():
        logging.error(f"Configuration file not found: {config_file}")
        return {}
    
    with open(config_file, 'r') as f:
        return json.load(f)

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Skip validation for --list-catalogues
    if args.list_catalogues:
        return
        
    # Check if catalogue 1 is provided
    if not args.catalogue_1:
        logging.error("Catalogue 1 not provided")
        sys.exit(1)
    
    # For catalogue_1, we should check if it's a file before requiring it exists
    if Path(args.catalogue_1).suffix in ['.parquet', '.fits', '.csv'] and not Path(args.catalogue_1).is_file():
        logging.error(f"Catalogue 1 file not found: {args.catalogue_1}")
        sys.exit(1)
    
    # Validate output directory if specified
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

def parse_columns(columns_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated columns string into a list."""
    if not columns_str:
        return None
    return [col.strip() for col in columns_str.split(',')]

def main() -> int:
    """Main CLI entry point with enhanced configuration handling."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set up logging
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        setup_logging(level=log_level)
        
        # Handle config loading if specified
        if args.load_config:
            loaded_config = load_config(args.load_config)
            if loaded_config:
                # Update args with loaded config, but command line args take precedence
                for k, v in loaded_config.items():
                    if not hasattr(args, k) or getattr(args, k) is None:
                        setattr(args, k, v)
                logging.info(f"Loaded configuration: {args.load_config}")
        
        # Handle config saving if specified
        if args.save_config:
            save_config(args, args.save_config)
            if args.catalogue_1 is None:
                # If just saving config without running a command
                return 0
        
        # Handle --list-catalogues flag
        if args.list_catalogues:
            return list_catalogues(args.config)
        
        # Parse kwargs
        method_kwargs = parse_kwargs(args.kwargs)
        
        # Validate arguments
        validate_args(args)
        
        # Parse columns
        catalogue_2_columns = parse_columns(args.columns_2)
        catalogue_1_columns = parse_columns(args.columns_1)
        
        logging.info("Initializing CrossMatch...")
        # Pass config path and any overriding kwargs from CLI
        config_kwargs = method_kwargs.copy() # Start with CLI kwargs
        if args.chunk_size:
             config_kwargs['chunk_size'] = args.chunk_size # Add chunk_size if specified

        cm = CrossMatch(config_file=args.config, **config_kwargs)
        
        logging.info(f"Starting cross-match: '{args.catalogue_1}' vs '{args.catalogue_2}'")
        
        # Prepare arguments for the crossmatch method
        crossmatch_params = {
            "catalogue_1": args.catalogue_1,
            "catalogue_2": args.catalogue_2,
            "output_file": str(Path(args.output)),
            "swap_catalogues": args.swap_catalogues if hasattr(args, 'swap_catalogues') else False
        }
        if args.method:
            crossmatch_params["method"] = args.method
        if args.radius:
            crossmatch_params["radius_arcsec"] = args.radius
        if catalogue_2_columns:
            crossmatch_params["columns_2"] = catalogue_2_columns
        if args.id_column_1:
            crossmatch_params["id_column_1"] = args.id_column_1
        if args.id_column_2:
            crossmatch_params["id_column_2"] = args.id_column_2
        if catalogue_1_columns:
             crossmatch_params["columns_1"] = catalogue_1_columns
             
        # Add remaining method-specific kwargs
        crossmatch_params["kwargs"] = method_kwargs 
        
        # Pass new parameters to crossmatch
        crossmatch_params["parallel"] = args.parallel if hasattr(args, 'parallel') else False
        crossmatch_params["timeout"] = args.timeout
        
        # Perform the cross-match
        result_path = cm.crossmatch(**crossmatch_params)
        
        logging.info(f"Cross-match completed successfully. Output saved to: {result_path}")

        return 0
        
    except (CrossMatchError, TapError, StiltsError) as e:
        logging.error(f"Cross-matching failed: {e}")
        if 'args' in locals() and hasattr(args, 'catalogue_1') and hasattr(args, 'catalogue_2'):
            logging.error(f"Failed catalogues: catalogue_1='{args.catalogue_1}', catalogue_2='{args.catalogue_2}'")
        return 1
    except FileNotFoundError as e:
         logging.error(f"File not found during cross-match: {e}")
         if 'args' in locals() and hasattr(args, 'catalogue_1') and hasattr(args, 'catalogue_2'):
            logging.error(f"Failed catalogues: catalogue_1='{args.catalogue_1}', catalogue_2='{args.catalogue_2}'")
         return 1
    except KeyboardInterrupt:
        logging.warning("Cross-match interrupted by user.")
        if 'args' in locals() and hasattr(args, 'catalogue_1') and hasattr(args, 'catalogue_2'):
            logging.warning(f"Interrupted catalogues: catalogue_1='{args.catalogue_1}', catalogue_2='{args.catalogue_2}'")
        return 1
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}") # Log full traceback for unexpected errors
        if 'args' in locals() and hasattr(args, 'catalogue_1') and hasattr(args, 'catalogue_2'):
            logging.error(f"Failed catalogues: catalogue_1='{args.catalogue_1}', catalogue_2='{args.catalogue_2}'")
        return 1

if __name__ == "__main__":
    sys.exit(main())