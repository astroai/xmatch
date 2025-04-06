import argparse
import sys
import logging

from .crossmatch import CrossMatch, CrossMatchError
from pathlib import Path # Added for Path handling

# Basic logging configuration (can be refined later)
# Configure logging using standard setup for consistency
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__) # Use module name for logger

def parse_columns(column_str):
    """Helper function to parse comma-separated column names."""
    if not column_str:
        # Return None instead of empty list, potentially easier to handle in crossmatch method
        return None
    # Handle potential extra spaces and empty elements
    cols = [col.strip() for col in column_str.split(',') if col.strip()]
    return cols if cols else None

# Added: Helper to parse ID join argument
def parse_id_join(join_str):
    """Parses --join-on-ids COL1:COL2 format."""
    if not join_str:
        return None
    parts = join_str.split(':')
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise argparse.ArgumentTypeError(
            "Invalid format for --join-on-ids. Expected 'COLUMN_NAME_1:COLUMN_NAME_2'."
        )
    return {'cat1': parts[0].strip(), 'cat2': parts[1].strip()}

# Removed load_config, list_catalogues, list_archives, describe_catalogue, describe_archive

# --- Helper to print dictionary nicely ---
def print_dict_details(data: dict, indent: str = "  "):
    """Prints dictionary key-value pairs with indentation."""
    if not isinstance(data, dict):
        print(f"{indent}Invalid data format (expected dictionary).")
        return
    # Define a preferred order for common keys if desired
    keys_order = ['description', 'format', 'path', 'ra_col', 'dec_col', 'required_columns', 'type', 'service_url', 'table_name', 'access_method', 'access_identifier', 'epoch']
    printed_keys = set()

    for key in keys_order:
        if key in data:
            print(f"{indent}{key}: {data[key]}")
            printed_keys.add(key)

    # Print any remaining keys (sorted alphabetically for consistency)
    remaining_keys = sorted(data.keys() - printed_keys)
    for key in remaining_keys:
         print(f"{indent}{key}: {data[key]}")


def main():
    # Define epilog separately for clarity
    epilog_text = ( # Enclose multi-line string in parentheses
        "Examples:\n"
        "  # Spatial match local files, save specific columns, override cat2 RA/Dec cols\n"
        "  xmatch cat1.fits cat2.csv --radius 5 --columns-1 ra,dec,mag_g --columns-2 ID,RA,DEC --ra2-col RA --dec2-col DEC -o matched.csv\n\n"
        "  # List configured catalogues\n"
        "  xmatch --list-catalogues --config my_config.yaml\n\n"
        "  # Describe a specific catalogue\n"
        "  xmatch --describe-catalogue gaia_dr3\n\n"
        "  # Match a local catalogue against a configured one, suggesting STILTS sky match\n"
        "  xmatch local_cat gaia_dr3 --radius 2 --method stilts_sky -o gaia_local_match.parquet\n\n"
        "  # Join two configured catalogues based on ID columns\n"
        "  xmatch catalogue_a catalogue_b --join-on-ids source_id:original_ext_source_id -o id_joined.fits"
    )

    parser = argparse.ArgumentParser(
        description="xmatch: Cross-match astronomical catalogues using the xmatch library.",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    ) # Correctly closed ArgumentParser call

    # --- Informational arguments ---
    info_group = parser.add_argument_group('Informational Commands')
    info_group.add_argument(
        '--list-catalogues',
        action='store_true',
        help='List available catalogues defined in the config file.'
    )
    info_group.add_argument(
        '--list-archives',
        action='store_true',
        help='List available archives structures defined in the config file.'
    )
    info_group.add_argument(
        '--describe-catalogue',
        metavar='CATALOGUE_ENTRY',
        help='Show details about a specific catalogue entry from the config file.'
    )
    info_group.add_argument(
        '--describe-archive',
        metavar='ARCHIVE_ENTRY',
        help='Show details about a specific archive structure entry from the config file.'
    )

    # --- Catalogue Creation (New Mode) ---
    create_group = parser.add_argument_group('Catalogue Creation')
    create_group.add_argument(
        '--create-catalogue-entry',
        metavar='NEW_CAT_NAME',
        help='Automatically create a new catalogue entry in the config file by querying TAP_SCHEMA.'
    )
    create_group.add_argument(
        '--archive',
        metavar='ARCHIVE_NAME',
        help='Specify the archive (must be defined in config) containing the table for --create-catalogue-entry.'
    )
    create_group.add_argument(
        '--access-id',
        metavar='TABLE_NAME',
        help='Specify the actual table name/access identifier on the remote service for --create-catalogue-entry.'
    )
    create_group.add_argument(
        '--service-id',
        metavar='SERVICE_ID',
        default='tap_service',
        help='Specify the service ID within the archive (default: tap_service) for --create-catalogue-entry.'
    )
    create_group.add_argument(
        '--description',
        metavar='\"Description Text\"',
        help='Optionally provide a description for the new catalogue entry, overriding TAP_SCHEMA discovery.'
    )

    # --- Configuration ---
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config',
        metavar='PATH',
        type=str,
        default=None,
        help='Path to the xmatch configuration file (e.g., xmatch.yaml). '
             'If not provided, defaults are checked (see documentation).'
    )

    # --- Cross-match parameters ---
    xmatch_group = parser.add_argument_group('Cross-match Parameters')
    xmatch_group.add_argument(
        '--columns-1',
        metavar='COLS',
        type=parse_columns,
        help='Comma-separated list of columns to keep/select from the first catalogue.'
    )
    xmatch_group.add_argument(
        '--columns-2',
         metavar='COLS',
        type=parse_columns,
        help='Comma-separated list of columns to keep/select from the second catalogue.'
    )
    xmatch_group.add_argument(
        '--radius',
        '-r',
        metavar='ARCSEC',
        type=float,
        required=False, # Optional, checked later if needed
        help='Cross-match radius in arcseconds. Required for spatial matching unless --join-on-ids is used.'
    )
    # Added: ID Join
    xmatch_group.add_argument(
        '--join-on-ids',
        metavar='COL1:COL2',
        type=parse_id_join,
        help='Perform an ID-based join instead of spatial. Specify columns using format: `col_name_cat1:col_name_cat2`.'
    )
    # Added: Coordinate Column Overrides
    xmatch_group.add_argument('--ra1-col', metavar='COL_NAME', help='Override RA column name for first input (e.g., for local files).')
    xmatch_group.add_argument('--dec1-col', metavar='COL_NAME', help='Override Dec column name for first input.')
    xmatch_group.add_argument('--ra2-col', metavar='COL_NAME', help='Override RA column name for second input.')
    xmatch_group.add_argument('--dec2-col', metavar='COL_NAME', help='Override Dec column name for second input.')
    # Added: Method Hint
    xmatch_group.add_argument(
        '--method',
        help='Suggest a specific cross-match method (e.g., stilts_sky, cds, tap_join). See documentation for available methods.'
    )
    xmatch_group.add_argument(
        '--output',
        '-o',
        metavar='OUTPUT_FILE',
        help='Path to save the cross-matched results (e.g., matched.csv, matched.parquet).'
    )

    # --- Input Catalogues (Positional) ---
    parser.add_argument(
        'catalog_inputs',
        metavar='CATALOGUE',
        nargs='*',
        help='Input catalogues: Two required for cross-matching. Can be file paths '
             '(e.g., /path/to/cat.fits) or catalogue entry names defined in config.'
    )

    # --- Other Options ---
    parser.add_argument(
        '--verbose',
        '-v',
        action='count',
        default=0,
        help='Increase logging verbosity (-v for INFO, -vv for DEBUG).'
    )
    parser.add_argument(
        '--log-file',
        metavar='LOG_PATH',
        help='Path to write detailed logs to a file.'
    )


    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = logging.WARNING # Default level if not verbose
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout
    )

    # Configure file logging if requested
    if args.log_file:
        try:
            log_path = Path(args.log_file).resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            file_handler = logging.FileHandler(log_path, mode='a') # Append mode
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            # Log DEBUG level and above to file
            file_handler.setLevel(logging.DEBUG)
            logging.getLogger().addHandler(file_handler) # Add handler to root logger
            logging.getLogger().setLevel(min(log_level, logging.DEBUG)) # Ensure root level is low enough for file handler
            log.info(f"Logging DEBUG+ level output to {log_path}")
        except Exception as e:
            log.error(f"Failed to set up log file handler for {args.log_file}: {e}")
            # Continue without file logging

    log.debug(f"Parsed arguments: {vars(args)}")

    # --- Instantiate CrossMatch Class ---
    try:
        log.debug(f"Initializing CrossMatch with config: {args.config}")
        # Pass config path if provided, otherwise CrossMatch uses its default
        cross_matcher = CrossMatch(config_file=args.config) if args.config else CrossMatch()
        log.info(f"CrossMatch initialized using config: {cross_matcher.config_file}")
    except CrossMatchError as e:
        log.error(f"Failed to initialize CrossMatch: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
         log.error(f"Configuration file error: {e}")
         sys.exit(1)
    except Exception as e:
         log.error(f"Unexpected error during CrossMatch initialization: {e}", exc_info=log_level <= logging.DEBUG)
         sys.exit(1)

    # --- Determine Operating Mode ---
    info_commands_provided = [
        args.list_catalogues, args.list_archives,
        args.describe_catalogue, args.describe_archive
    ]
    num_info_commands = sum(bool(cmd) for cmd in info_commands_provided) # Count True/non-None values

    # Check for create mode
    is_create_mode = args.create_catalogue_entry is not None

    is_info_mode = num_info_commands > 0 and not is_create_mode # Info mode excludes create mode
    # Assume cross-match mode if no info/create command is given AND inputs are provided
    is_potential_xmatch_mode = not is_info_mode and not is_create_mode and len(args.catalog_inputs) > 0

    log.debug(f"Info mode: {is_info_mode}, Create mode: {is_create_mode}, Potential xmatch mode: {is_potential_xmatch_mode}, Num info commands: {num_info_commands}")

    # --- Validate Arguments Based on Mode ---
    if num_info_commands > 1:
        parser.error("Please specify only one informational command (--list-*, --describe-*) at a time.")
    if is_info_mode and is_create_mode:
        parser.error("Cannot combine informational commands with --create-catalogue-entry.")
    if is_create_mode and len(args.catalog_inputs) > 0:
        parser.error("Catalogue inputs should not be provided with --create-catalogue-entry.")
    if is_create_mode and (not args.archive or not args.access_id):
        parser.error("--archive and --access-id are required when using --create-catalogue-entry.")

    # --- Execute Action ---
    try:
        if is_info_mode:
            if args.catalog_inputs:
                parser.error("Catalogue inputs should not be provided with informational commands.")
            # Cross-match specific args are ignored, maybe warn?
            ignored_args = [arg for arg, val in [
                ('radius', args.radius), ('output', args.output),
                ('columns-1', args.columns_1), ('columns-2', args.columns_2),
                ('join-on-ids', args.join_on_ids),
                ('ra1-col', args.ra1_col), ('dec1-col', args.dec1_col),
                ('ra2-col', args.ra2_col), ('dec2-col', args.dec2_col),
                ('method', args.method)
                ] if val is not None]
            if ignored_args:
                log.warning(f"Cross-match specific arguments ({', '.join(ignored_args)}) are ignored in informational mode.")

            # --- Run Informational Command using CrossMatch instance ---
            if args.list_catalogues:
                catalogues = cross_matcher.catalogues_config
                if not catalogues:
                    print("No catalogues defined in the configuration.")
                else:
                    print("Available Catalogues:")
                    for name in sorted(catalogues.keys()):
                        desc = catalogues[name].get('description', 'No description')
                        print(f"  - {name}: {desc}")

            elif args.list_archives:
                archives = cross_matcher.archives_config
                if not archives:
                     print("No archives defined in the configuration.")
                else:
                     print("Available Archives:")
                     # Archives structure might be nested {archive_name: {service_id: details}}
                     for archive_name in sorted(archives.keys()):
                          print(f"  Archive: {archive_name}")
                          archive_details = archives[archive_name]
                          if isinstance(archive_details, dict):
                               desc = archive_details.get('description', 'No description') # Top-level desc?
                               print(f"    Description: {desc}")
                               # List services within the archive
                               for service_id in sorted(archive_details.keys()):
                                    if service_id != 'description': # Skip description key
                                        service_desc = archive_details[service_id].get('description', 'No service description')
                                        print(f"    - Service: {service_id} ({service_desc})") # Assuming services are dicts
                          else:
                              print(f"    (Invalid structure for archive {archive_name})")

            elif args.describe_catalogue:
                cat_name = args.describe_catalogue
                try:
                    # Use the existing method to get fully resolved config
                    cat_config = cross_matcher.get_catalogue_config(cat_name)
                    print(f"\nDetails for Catalogue '{cat_name}':")
                    print_dict_details(cat_config)
                except CrossMatchError as e:
                    log.error(e)
                    # Suggest alternatives if possible
                    available = sorted(cross_matcher.catalogues_config.keys())
                    if available:
                        print("\nAvailable catalogue entries:")
                        for avail_name in available:
                            print(f"  - {avail_name}")
                    sys.exit(1)

            elif args.describe_archive:
                archive_name = args.describe_archive
                if archive_name not in cross_matcher.archives_config:
                     log.error(f"Archive entry '{archive_name}' not found in configuration.")
                     available = sorted(cross_matcher.archives_config.keys())
                     if available:
                         print("\nAvailable archive entries:")
                         for avail_name in available:
                             print(f"  - {avail_name}")
                     sys.exit(1)
                else:
                     print(f"\nDetails for Archive Structure '{archive_name}':")
                     print_dict_details(cross_matcher.archives_config[archive_name])

            sys.exit(0) # Exit cleanly after info command

        elif is_create_mode:
            log.info(f"--- Creating Catalogue Entry: {args.create_catalogue_entry} ---")
            # Call the new method on the CrossMatch instance
            # This method will handle querying TAP_SCHEMA and updating the YAML file
            success = cross_matcher.create_catalogue_entry(
                new_catalogue_name=args.create_catalogue_entry,
                archive_name=args.archive,
                access_identifier=args.access_id,
                service_id=args.service_id,
                description_override=args.description
            )
            if success:
                log.info(f"Successfully created entry '{args.create_catalogue_entry}' in {cross_matcher.config_file}")
                sys.exit(0)
            else:
                log.error(f"Failed to create catalogue entry '{args.create_catalogue_entry}'.")
                sys.exit(1)

        elif is_potential_xmatch_mode:
            # --- Cross-Match Mode ---
            if len(args.catalog_inputs) != 2:
                 # Should not happen if logic is correct, but catch just in case
                 parser.error("Internal Error or incorrect usage: Expected exactly two catalogue inputs for cross-matching.")

            # Conditional radius check: Required only if NOT doing an ID join
            if args.join_on_ids is None and args.radius is None:
                 parser.error("Either --radius (for spatial join) or --join-on-ids (for ID join) must be provided.")
            if args.join_on_ids and args.radius is not None:
                 log.warning("Both --radius and --join-on-ids provided. --radius will be ignored for ID join.")
            if args.radius is not None and args.radius <= 0:
                 parser.error("The --radius must be a positive value.")

            # Output is recommended but not strictly required by the class method
            if not args.output:
                log.warning("No --output file specified. Results will be returned as a DataFrame (and potentially lost if not handled).")

            # --- Prepare parameters for cross_matcher.crossmatch ---
            xmatch_params = {
                "catalogue_1_input": args.catalog_inputs[0],
                "catalogue_2_input": args.catalog_inputs[1],
                "output_file": args.output,
                "radius_arcsec": args.radius,
                "columns1": args.columns_1, # Pass the list from parse_columns (or None)
                "columns2": args.columns_2, # Pass the list from parse_columns (or None)
                "join_on_ids": args.join_on_ids,
                "method": args.method,
                "ra1_col": args.ra1_col,
                "dec1_col": args.dec1_col,
                "ra2_col": args.ra2_col,
                "dec2_col": args.dec2_col,
            }
            # Filter out None values, as crossmatch method uses defaults
            xmatch_params = {k: v for k, v in xmatch_params.items() if v is not None}

            log.info("--- Starting Cross-Match ---")
            log.info(f"Input 1: {args.catalog_inputs[0]}")
            log.info(f"Input 2: {args.catalog_inputs[1]}")
            log.info(f"Parameters: {xmatch_params}") # Log the actual params passed

            # --- Call the core crossmatch method ---
            results_df = cross_matcher.crossmatch(**xmatch_params)

            log.info("--- Cross-Match Completed Successfully ---")

            if results_df is not None:
                log.info(f"Cross-match returned a DataFrame with {len(results_df)} rows.")
                if not args.output:
                    # If no output file, maybe print head? Be careful with large tables.
                    print("\nCross-match Results Preview:")
                    try:
                        # Requires pandas to be installed
                        import pandas as pd
                        with pd.option_context('display.max_rows', 10, 'display.max_columns', 10):
                             print(results_df)
                    except ImportError:
                         print("(Install pandas to see DataFrame preview)")
                    except Exception as e:
                         print(f"(Error generating preview: {e})")
            elif args.output:
                 log.info(f"Results saved to {Path(args.output).resolve()}")


            sys.exit(0) # Success

        else:
            # No informational command and no input catalogues provided
            log.info("No command or catalogue inputs provided.")
            parser.print_help()
            sys.exit(0)

    except CrossMatchError as e:
         log.error(f"Cross-matching Error: {e}", exc_info=log_level <= logging.DEBUG)
         sys.exit(1)
    except FileNotFoundError as e: # Catch file errors during matching too
         log.error(f"File Error: {e}")
         sys.exit(1)
    except ImportError as e: # Catch missing dependencies like stilts
         log.error(f"Import Error: {e}. Please ensure all required libraries (astropy, pandas, pyvo, etc.) are installed.")
         sys.exit(1)
    except Exception as e:
         log.error(f"An unexpected error occurred: {e}", exc_info=log_level <= logging.DEBUG)
         sys.exit(1)


if __name__ == "__main__":
    main() 