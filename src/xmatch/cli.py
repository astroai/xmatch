import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .crossmatch import CrossMatch

logger = logging.getLogger(__name__)


def resolve_catalogue_name(name: str, cm: CrossMatch) -> str:
    """Resolves a catalogue alias to its full name.

    Checks if the name is an alias and returns the resolved name,
    otherwise returns the original name.

    Args:
        name: The catalogue name or alias
        cm: CrossMatch instance with loaded config

    Returns:
        The resolved catalogue name
    """
    if not hasattr(cm, "config") or not cm.config:
        return name

    # Check catalogue aliases if available
    aliases = cm.config.get("catalogue_aliases", {})
    if name.lower() in aliases:
        return aliases[name.lower()]

    return name


def handle_archive_override(name: str, archive_prefix: str, cm: CrossMatch) -> str:
    """Apply archive override to catalogue name if needed.

    Args:
        name: Original catalogue name
        archive_prefix: Archive prefix to apply (e.g., 'esa', 'cds')
        cm: CrossMatch instance with config

    Returns:
        Updated catalogue name with archive prefix if applicable
    """
    resolved = resolve_catalogue_name(name, cm)

    # Handle the case where a simple name is given but archive override is specified
    if resolved == name:  # No alias found
        # Try constructing a catalogue name with archive prefix
        with_archive = f"{name}_{archive_prefix}"
        if with_archive in cm.config.get("catalogues", {}):
            logger.info(f"Using {with_archive} based on archive override")
            return with_archive

    return name


def parse_args(args: List[str] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="xmatch",
        description="Cross-match astronomical catalogues locally or via remote services",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Catalogue inputs (positional arguments)
    parser.add_argument(
        "catalogue_1",
        nargs="?",
        help="First catalogue: name of configured catalogue (e.g., 'gaia', 'galex') or path to local file",
    )
    parser.add_argument(
        "catalogue_2",
        nargs="?",
        help="Second catalogue: name of configured catalogue (e.g., 'gaia', 'galex') or path to local file",
    )

    # Archive selection for catalogue lookup
    parser.add_argument(
        "--archive-1",
        help="Specify archive for first catalogue (e.g., 'esa', 'cds', 'noao')",
        dest="archive_1",
    )
    parser.add_argument(
        "--archive-2",
        help="Specify archive for second catalogue (e.g., 'esa', 'cds', 'noao')",
        dest="archive_2",
    )

    # Column selection for both catalogues
    parser.add_argument(
        "--columns-1",
        help="Comma-separated list of columns to select from first catalogue (e.g., 'ra,dec,mag_g')",
        dest="columns_1",
    )
    parser.add_argument(
        "--columns-2",
        help="Comma-separated list of columns to select from second catalogue (e.g., 'ra,dec,mag_r')",
        dest="columns_2",
    )

    # Option to show default columns
    parser.add_argument(
        "--show-default-columns",
        action="store_true",
        help="Show default columns for catalogues and proceed with crossmatch",
        dest="show_default_columns",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Output file path (.parquet, .fits, or .csv)",
    )

    # Matching parameters
    parser.add_argument(
        "-r",
        "--radius",
        dest="radius_arcsec",
        type=float,
        default=1.0,
        help="Match radius in arcseconds",
    )

    # Column overrides for local files
    parser.add_argument("--ra1", dest="ra_column_1", help="RA column name for first catalogue")
    parser.add_argument("--dec1", dest="dec_column_1", help="Dec column name for first catalogue")
    parser.add_argument("--ra2", dest="ra_column_2", help="RA column name for second catalogue")
    parser.add_argument("--dec2", dest="dec_column_2", help="Dec column name for second catalogue")

    # For ID-based joins
    parser.add_argument("--id1", dest="id_column_1", help="ID column name for first catalogue")
    parser.add_argument("--id2", dest="id_column_2", help="ID column name for second catalogue")
    parser.add_argument(
        "--join-on-ids",
        dest="join_on_ids",
        action="store_true",
        help="Perform ID-based join instead of spatial join",
    )

    # Join type
    parser.add_argument(
        "--join",
        dest="join_type",
        choices=["1and2", "1or2", "all", "1not2", "2not1", "all1", "all2"],
        default="1and2",
        help="Join type: 1and2=inner, 1or2=outer, 1not2=left anti, 2not1=right anti, all=full",
    )

    # For specifying spatial region
    parser.add_argument("--ra", dest="ra", type=float, help="RA of region center in degrees")
    parser.add_argument("--dec", dest="dec", type=float, help="Dec of region center in degrees")

    # Advanced error handling options
    parser.add_argument(
        "--matcher",
        dest="matcher",
        choices=["sky", "skyerr", "skyellipse"],
        help="Matcher algorithm: 'sky' for fixed radius, 'skyerr' for symmetric errors, 'skyellipse' for error ellipses",
    )
    parser.add_argument(
        "--max-error",
        dest="max_error",
        type=float,
        help="Maximum separation in units of sigma (for skyerr/skyellipse matchers)",
    )

    # Strategy selection
    parser.add_argument("--strategy", dest="strategy", help="Force specific matching strategy")

    # Performance tuning
    parser.add_argument(
        "--n-workers",
        dest="n_workers",
        type=int,
        help="Number of worker processes for parallel matching",
    )
    parser.add_argument(
        "--chunk-size", dest="chunk_size", type=int, help="Chunk size for processing large catalogs"
    )

    # Info commands
    parser.add_argument(
        "--list-catalogues",
        dest="list_catalogues",
        action="store_true",
        help="List available catalogues",
    )
    parser.add_argument(
        "--describe", dest="describe_catalogue", help="Describe a specific catalogue"
    )

    # Config handling
    parser.add_argument(
        "--config", dest="config_file", help="Path to config file (default: use builtin config)"
    )

    # Authentication
    parser.add_argument(
        "--auth-config", dest="auth_config", help="Path to authentication config file"
    )

    # Verbosity/logging
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)",
    )
    parser.add_argument("--log-file", dest="log_file", help="Log file path")

    # Dry run
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # Version
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Parse args
    parsed_args = parser.parse_args(args)

    # Add default output file if not provided but catalogues are
    if (
        parsed_args.catalogue_1
        and parsed_args.catalogue_2
        and not parsed_args.output_file
        and not parsed_args.list_catalogues
        and not parsed_args.describe_catalogue
    ):
        # Generate default output name based on input catalogues
        cat1_name = (
            Path(parsed_args.catalogue_1).stem
            if Path(parsed_args.catalogue_1).suffix
            else parsed_args.catalogue_1
        )
        cat2_name = (
            Path(parsed_args.catalogue_2).stem
            if Path(parsed_args.catalogue_2).suffix
            else parsed_args.catalogue_2
        )
        parsed_args.output_file = f"{cat1_name}_{cat2_name}.parquet"
        logger.info(f"No output file specified, using default: {parsed_args.output_file}")

    return parsed_args


def setup_logging(args: argparse.Namespace) -> None:
    """Set up logging based on command-line arguments."""
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])

    # Add file handler if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

    # Set level for some noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pyvo").setLevel(max(logging.INFO, log_level))


def list_catalogues(cm: CrossMatch) -> None:
    """Print a list of available catalogues."""
    catalogues = {}

    # Get catalogue definitions
    if hasattr(cm, "config") and "catalogues" in cm.config:
        catalogues = cm.config["catalogues"]
    else:
        print("No catalogues configured.", file=sys.stderr)
        return

    # Get aliases for more helpful output
    aliases = {}
    if "catalogue_aliases" in cm.config:
        # Invert the aliases mapping for display
        for alias, cat_name in cm.config["catalogue_aliases"].items():
            if cat_name not in aliases:
                aliases[cat_name] = []
            aliases[cat_name].append(alias)

    # Print header
    print("\nAvailable catalogues:")
    print(f"{'NAME':<16} {'ALIASES':<25} {'ARCHIVE':<12} {'DESCRIPTION':<40}\n{'-' * 93}")

    # Sort catalogues by name
    for cat_name in sorted(catalogues.keys()):
        cat_config = catalogues[cat_name]
        cat_aliases = ", ".join(aliases.get(cat_name, []))
        archive = cat_config.get("archive", "unknown")
        description = cat_config.get("description", "No description")

        print(f"{cat_name:<16} {cat_aliases:<25} {archive:<12} {description:<40}")

    print("\nUse 'xmatch --describe CATALOGUE' for more details on a specific catalogue.")


def describe_catalogue(cm: CrossMatch, catalogue_name: str) -> bool:
    """Print detailed information about a specific catalogue."""
    # Resolve alias if needed
    original_name = catalogue_name
    catalogue_name = resolve_catalogue_name(catalogue_name, cm)

    # Get catalogue definition
    if not hasattr(cm, "config") or "catalogues" not in cm.config:
        print("No catalogues configured.", file=sys.stderr)
        return

    catalogues = cm.config["catalogues"]
    if catalogue_name not in catalogues:
        print(f"Catalogue '{catalogue_name}' not found.", file=sys.stderr)
        if original_name != catalogue_name:
            print(f"Note: '{original_name}' was resolved to '{catalogue_name}'.", file=sys.stderr)

        import difflib

        available_names = list(catalogues.keys())
        if "catalogue_aliases" in cm.config:
            available_names.extend(cm.config["catalogue_aliases"].keys())

        # Perform case-insensitive matching
        lower_to_original = {name.lower(): name for name in available_names}
        suggestions = difflib.get_close_matches(
            catalogue_name.lower(), list(lower_to_original.keys()), n=3, cutoff=0.5
        )
        if suggestions:
            original_suggestions = [lower_to_original[s] for s in suggestions]
            print(f"Did you mean: {', '.join(original_suggestions)}?", file=sys.stderr)
        return False

    cat_config = catalogues[catalogue_name]

    # Print catalogue details
    print(f"\nCatalogue: {catalogue_name}")
    if original_name != catalogue_name:
        print(f"Alias: {original_name}")

    # Get all aliases for this catalogue
    if "catalogue_aliases" in cm.config:
        aliases = [
            alias
            for alias, name in cm.config["catalogue_aliases"].items()
            if name == catalogue_name and alias != original_name
        ]
        if aliases:
            print(f"Other aliases: {', '.join(aliases)}")

    # Print general info
    print(f"Description: {cat_config.get('description', 'No description')}")
    print(f"Archive: {cat_config.get('archive', 'unknown')}")
    print(f"Service: {cat_config.get('service_id', 'unknown')}")
    print(f"Release: {cat_config.get('release', 'unknown')}")

    # Print table info
    print(f"Table identifier: {cat_config.get('access_identifier', 'unknown')}")

    # Print spatial column info
    print("\nSpatial columns:")
    print(f"  RA: {cat_config.get('ra_column', 'unknown')}")
    print(f"  Dec: {cat_config.get('dec_column', 'unknown')}")
    if "id_column" in cat_config:
        print(f"  ID: {cat_config.get('id_column')}")
    if "pm_ra_column" in cat_config and "pm_dec_column" in cat_config:
        print(
            f"  Proper motion: {cat_config.get('pm_ra_column')} / {cat_config.get('pm_dec_column')}"
        )
    if "epoch_column" in cat_config or "epoch" in cat_config:
        print(f"  Epoch: {cat_config.get('epoch_column', cat_config.get('epoch', 'unknown'))}")

    # Print error info
    if "ra_err_column" in cat_config or "dec_err_column" in cat_config:
        print("\nPosition errors:")
        print(f"  RA error: {cat_config.get('ra_err_column', 'unknown')}")
        print(f"  Dec error: {cat_config.get('dec_err_column', 'unknown')}")
        if "corr_column" in cat_config:
            print(f"  Correlation: {cat_config.get('corr_column', 'unknown')}")
        if "pos_err_units" in cat_config:
            print(f"  Units: {cat_config.get('pos_err_units', 'unknown')}")
        if "default_pos_error_arcsec" in cat_config:
            print(f"  Default error: {cat_config.get('default_pos_error_arcsec')} arcsec")

    # Print default columns
    if "default_columns" in cat_config:
        print("\nDefault columns:")
        for col in cat_config["default_columns"]:
            print(f"  - {col}")


    return True


def prepare_crossmatch_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Prepare parameters for the crossmatch function based on command-line arguments."""
    params = {}

    # Basic matching params
    if args.radius_arcsec is not None:
        params["radius_arcsec"] = args.radius_arcsec

    # Column selection
    if args.columns_1:
        params["columns_1"] = [col.strip() for col in args.columns_1.split(",") if col.strip()]
    if args.columns_2:
        params["columns_2"] = [col.strip() for col in args.columns_2.split(",") if col.strip()]

    # Column overrides for local files
    if args.ra_column_1:
        params["ra_column_1"] = args.ra_column_1
    if args.dec_column_1:
        params["dec_column_1"] = args.dec_column_1
    if args.ra_column_2:
        params["ra_column_2"] = args.ra_column_2
    if args.dec_column_2:
        params["dec_column_2"] = args.dec_column_2
    if args.id_column_1:
        params["id_column_1"] = args.id_column_1
    if args.id_column_2:
        params["id_column_2"] = args.id_column_2

    # Join type
    if args.join_type:
        params["join_type"] = args.join_type

    # ID join
    if args.join_on_ids:
        if args.id_column_1 and args.id_column_2:
            params["join_on_ids"] = {"cat1": args.id_column_1, "cat2": args.id_column_2}
        else:
            logger.warning(
                "ID-based join requested but one or more ID columns not specified. "
                "Will attempt to use default ID columns from catalog configuration "
                "if available. Specify with --id1 and --id2 for explicit control."
            )
            params["join_on_ids"] = {}

    # Spatial region
    if args.ra is not None and args.dec is not None:
        params["ra"] = args.ra
        params["dec"] = args.dec

    # Error handling
    if args.matcher:
        params["matcher"] = args.matcher
    if args.max_error is not None:
        params["max_error"] = args.max_error

    # Strategy
    if args.strategy:
        params["strategy"] = args.strategy

    # Performance tuning
    if args.n_workers is not None:
        params["n_workers"] = args.n_workers
    if args.chunk_size is not None:
        params["chunk_size"] = args.chunk_size

    return params


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the command-line interface."""
    parsed_args = parse_args(args)

    # Set up logging
    setup_logging(parsed_args)

    try:
        # Import exceptions here, just before they might be caught
        from .exceptions import ConfigError, CrossMatchError, StiltsError, TapError

        # Initialize CrossMatch with config
        cm = CrossMatch(config_file=parsed_args.config_file)

        # Handle info commands
        if parsed_args.list_catalogues:
            list_catalogues(cm)
            return 0

        if parsed_args.describe_catalogue:
            success = describe_catalogue(cm, parsed_args.describe_catalogue)
            return 0 if success else 1

        # Check if we have catalogues to match
        if not parsed_args.catalogue_1 or not parsed_args.catalogue_2:
            print("Error: Two catalogues are required for matching.", file=sys.stderr)
            print("Use 'xmatch --help' for usage information.", file=sys.stderr)
            print("Use 'xmatch --list-catalogues' to see available catalogues.", file=sys.stderr)
            return 1

        # Resolve catalogue names/aliases
        cat1_name = parsed_args.catalogue_1
        cat2_name = parsed_args.catalogue_2

        # Handle archive overrides
        if parsed_args.archive_1:
            cat1_name = handle_archive_override(
                cat1_name, parsed_args.archive_1.lower().rstrip("_"), cm
            )

        if parsed_args.archive_2:
            cat2_name = handle_archive_override(
                cat2_name, parsed_args.archive_2.lower().rstrip("_"), cm
            )

        # Show default columns if requested
        if parsed_args.show_default_columns:
            # For catalogue 1
            cat1_resolved = resolve_catalogue_name(cat1_name, cm)
            cat1_is_catalogue = Path(cat1_name).suffix not in [".csv", ".fits", ".parquet"]
            if cat1_is_catalogue and cat1_resolved in cm.config.get("catalogues", {}):
                cat1_config = cm.config["catalogues"][cat1_resolved]
                print(f"\nDefault columns for {cat1_name}:")
                if "default_columns" in cat1_config and cat1_config["default_columns"]:
                    for col in cat1_config["default_columns"]:
                        print(f"  - {col}")
                else:
                    print("  No default columns specified.")
            else:
                print(f"\n{cat1_name} is a local file, no default columns available.")

            # For catalogue 2
            cat2_resolved = resolve_catalogue_name(cat2_name, cm)
            cat2_is_catalogue = Path(cat2_name).suffix not in [".csv", ".fits", ".parquet"]
            if cat2_is_catalogue and cat2_resolved in cm.config.get("catalogues", {}):
                cat2_config = cm.config["catalogues"][cat2_resolved]
                print(f"\nDefault columns for {cat2_name}:")
                if "default_columns" in cat2_config and cat2_config["default_columns"]:
                    for col in cat2_config["default_columns"]:
                        print(f"  - {col}")
                else:
                    print("  No default columns specified.")
            else:
                print(f"\n{cat2_name} is a local file, no default columns available.")

        # Prepare crossmatch parameters
        params = prepare_crossmatch_params(parsed_args)

        # Print what will be done
        logger.info(f"Matching '{cat1_name}' with '{cat2_name}'")
        logger.info(f"Parameters: {params}")
        if parsed_args.output_file:
            logger.info(f"Output will be written to: {parsed_args.output_file}")

        # Skip actual execution in dry run mode
        if parsed_args.dry_run:
            logger.info("Dry run requested, skipping execution.")
            return 0

        # Execute the crossmatch
        result_df = cm.crossmatch(
            catalogue_1_input=cat1_name,
            catalogue_2_input=cat2_name,
            output_file=parsed_args.output_file,
            **params,
        )

        # If output_file is None, the result is returned and should be printed
        if result_df is not None and parsed_args.output_file is None:
            logger.info(f"Crossmatch returned {len(result_df)} rows.")
            if not result_df.empty:
                print("\nResults preview:")
                print(result_df.head().to_string(index=False))
            else:
                print("No matches found.")

        return 0

    except CrossMatchError as e:
        logger.error(f"CrossMatch error: {str(e)}")
        return 1
    except ConfigError as e:
        logger.error(f"Configuration Error: {e}")
        return 1
    except TapError as e:
        logger.error(f"TAP Service Error: {e}")
        return 1
    except StiltsError as e:
        logger.error(f"STILTS Execution Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
