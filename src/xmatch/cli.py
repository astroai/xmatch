"""Command-line interface: ``xmatch CAT1 CAT2``."""

import argparse
import difflib
import logging
import sys
from typing import List, Optional

from . import __version__
from .crossmatch import CrossMatch
from .exceptions import CrossMatchError

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmatch",
        description="Cross-match two astronomical catalogues (local files or remote archives).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "catalogues",
        nargs="*",
        help="Two or more catalogues: files, HATS dirs, or configured names.",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Output file (.parquet/.csv/.fits/.hats). If omitted, CSV is written to stdout.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        dest="radius_arcsec",
        type=float,
        default=1.0,
        help="Match radius in arcseconds.",
    )

    parser.add_argument(
        "--matcher", choices=["sky", "skyerr", "skyellipse", "lr", "ml", "xgb", "auf", "macauff"], help="Match algorithm (default: sky). lr=Likelihood Ratio, ml=Random Forest, xgb=XGBoost, auf=AUF empirical error model, macauff=AUF+flux."
    )
    parser.add_argument(
        "--max-error",
        dest="max_error",
        type=float,
        default=3.0,
        help="N-sigma cap for skyerr/skyellipse.",
    )
    parser.add_argument(
        "--join",
        dest="join_type",
        default="1and2",
        choices=["1and2", "1or2", "all", "1not2", "2not1", "all1", "all2"],
        help="Join type (1and2=inner, 1or2=outer, 1not2/2not1=anti, all1/all2=outer side).",
    )
    parser.add_argument(
        "--union",
        dest="union_match",
        action="store_true",
        help="Build a master union catalogue (full outer join across all catalogues, adds _src_cats column).",
    )
    parser.add_argument(
        "--fof",
        dest="fof_match",
        action="store_true",
        help="Friends-of-Friends transitive closure: merge all catalogues into object bundles via pairwise matching and connected-component clustering.",
    )
    parser.add_argument(
        "--find",
        choices=["best", "all"],
        default="best",
        help="Keep the best match or all matches within the radius.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "stilts", "astropy", "fast", "zone", "ray"],
        default="auto",
        help="Sky-match engine (fast=scipy.cKDTree, zone=HEALPix pixellated, ray=distributed).",
    )

    # ID join.
    parser.add_argument(
        "--id-join",
        dest="id_join",
        action="store_true",
        help="Join on id columns instead of sky position.",
    )
    parser.add_argument("--id1", dest="id_column_1", help="ID column for catalogue 1.")
    parser.add_argument("--id2", dest="id_column_2", help="ID column for catalogue 2.")

    # Column overrides for local files.
    parser.add_argument("--ra1", dest="ra_column_1", help="RA column name for catalogue 1.")
    parser.add_argument("--dec1", dest="dec_column_1", help="Dec column name for catalogue 1.")
    parser.add_argument("--ra2", dest="ra_column_2", help="RA column name for catalogue 2.")
    parser.add_argument("--dec2", dest="dec_column_2", help="Dec column name for catalogue 2.")
    parser.add_argument(
        "--columns-1", dest="columns_1", help="Comma-separated columns from catalogue 1."
    )
    parser.add_argument(
        "--columns-2", dest="columns_2", help="Comma-separated columns from catalogue 2."
    )
    parser.add_argument(
        "--probabilistic",
        dest="probabilistic",
        action="store_true",
        help="Compute Budavari-style hierarchical Bayes factor (+ p_match column).",
    )
    parser.add_argument(
        "--priors",
        dest="priors",
        default="",
        help="Comma-separated photometric columns used as Bayesian priors (e.g. g,r).",
    )

    # --- advanced match features (v0.5+) ---
    parser.add_argument(
        "--target-epoch",
        dest="target_epoch",
        type=float,
        help="Julian-year epoch to propagate coordinates to via proper motion.",
    )
    parser.add_argument(
        "--pm-prior",
        dest="pm_prior",
        action="store_true",
        help="Inflate positional errors for sources without measured proper motions using a Galactic-latitude drift model (Wilson 2023). Requires --target-epoch.",
    )
    parser.add_argument(
        "--pm-prior-mag-col",
        dest="pm_prior_mag_col",
        help="Magnitude column for refining the PM drift dispersion estimate (brighter = closer = larger PM). Only effective with --pm-prior.",
    )
    parser.add_argument(
        "--filter-expr",
        dest="filter_expr",
        help="Polars SQL WHERE clause to post-filter matched pairs (e.g. 'abs(mag - mag_2) < 0.5').",
    )
    parser.add_argument(
        "--extra-distance-cols",
        dest="extra_distance_cols",
        help="Column:weight pairs for N-dimensional cKDTree ranking (e.g. 'g:0.5,bp_rp:0.3').",
    )
    parser.add_argument(
        "--lr-magnitude-column",
        dest="lr_magnitude_column",
        help="Magnitude column for Likelihood Ratio matcher (required when --matcher lr).",
    )
    parser.add_argument(
        "--lr-q",
        dest="lr_q",
        type=float,
        default=0.8,
        help="Prior Q factor for LR matcher: fraction of primary sources with detectable counterparts (0.5-1.0).",
    )
    parser.add_argument(
        "--ml-color-cols",
        dest="ml_color_cols",
        help="Comma-separated photometric columns for ML matcher features (e.g. 'g,r,i'). Required when --matcher ml.",
    )
    parser.add_argument(
        "--ml-model-path",
        dest="ml_model_path",
        help="Path to save/load a pre-trained Random Forest model (joblib). If the file exists it is loaded; otherwise a new model is trained and saved.",
    )
    parser.add_argument(
        "--xgb-model-path",
        dest="xgb_model_path",
        help="Path to save/load a pre-trained XGBoost model (joblib). If the file exists it is loaded; otherwise a new model is trained and saved.",
    )
    parser.add_argument(
        "--macauff-flux-cols",
        dest="macauff_flux_cols",
        help="Comma-separated magnitude columns for macauff flux likelihoods (e.g. 'g,r,i'). Used when --matcher macauff.",
    )
    parser.add_argument(
        "--hats-threshold",
        dest="hats_threshold",
        type=int,
        default=100_000,
        help="Max rows per HEALPix pixel for .hats output (HATS partitioning granularity).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="HEALPix pixel groups per batch for out-of-core processing (zone engine).",
    )

    # Region for remote downloads.
    parser.add_argument("--ra", type=float, help="Region center RA (deg) for remote downloads.")
    parser.add_argument("--dec", type=float, help="Region center Dec (deg) for remote downloads.")
    parser.add_argument(
        "--radius-deg",
        dest="radius_deg",
        type=float,
        help="Region radius (deg) for remote downloads.",
    )

    # Info / config.
    parser.add_argument(
        "--list",
        dest="list_catalogues",
        action="store_true",
        help="List configured catalogues and exit.",
    )
    parser.add_argument("--describe", dest="describe", help="Describe a catalogue and exit.")
    parser.add_argument(
        "--search",
        dest="search",
        nargs="?",
        const="*",
        metavar="PATTERN",
        help="Search remote TAP services for tables matching PATTERN (default: all tables).",
    )
    parser.add_argument(
        "--discover",
        dest="discover",
        metavar="ENDPOINT",
        help="Discover tables and schema on a remote TAP endpoint (e.g. vizier, gaia, noirlab).",
    )
    parser.add_argument(
        "--schema",
        dest="schema_table",
        metavar="TABLE",
        help="Show column schema for a specific remote table (use with --discover or --search).",
    )
    parser.add_argument("--config", dest="config_file", help="Path to a config YAML.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def list_catalogues(cm: CrossMatch) -> None:
    """Print all configured catalogues with archive, description, and size info."""
    print(f"{'Name':<24} {'Archive':<14} {'Access':<14} {'Rows':>8}  Description")
    print("-" * 100)
    for name in sorted(cm.catalogues_config):
        cat = cm.catalogues_config[name]
        archive = cat.get("archive", "?")
        access = cat.get("access_identifier", "?")
        size = cat.get("estimated_size", "?")
        desc = cat.get("description", "")
        print(f"  {name:<22} {archive:<14} {str(access)[:14]:<14} {str(size):>8}  {desc}")
    print(f"\n{len(cm.catalogues_config)} catalogues, {len(cm.aliases_config)} aliases.")
    print("Use 'xmatch --describe <name>' for column details.")


def describe(cm: CrossMatch, name: str) -> bool:
    """Print detailed configuration for a named catalogue."""
    resolved = cm.resolve_name(name)
    if resolved not in cm.catalogues_config:
        print(f"Catalogue '{name}' not found. {_suggest(cm, name)}", file=sys.stderr)
        return False
    cat = cm.catalogues_config[resolved]
    print(f"Catalogue: {resolved}")
    print(f"  Description:    {cat.get('description', '')}")
    print(f"  Archive:        {cat.get('archive', '?')}")
    print(f"  Service:        {cat.get('service_id', '?')}")
    print(f"  Table:          {cat.get('access_identifier', '?')}")
    print(f"  RA column:      {cat.get('ra_column', '—')}")
    print(f"  Dec column:     {cat.get('dec_column', '—')}")
    print(f"  ID column:      {cat.get('id_column', '—')}")
    if cat.get("ra_err_column"):
        print(f"  RA err column:  {cat['ra_err_column']}")
    if cat.get("dec_err_column"):
        print(f"  Dec err column: {cat['dec_err_column']}")
    print(f"  Epoch:          {cat.get('epoch', '—')}")
    if cat.get("default_columns"):
        cols = ", ".join(str(c) for c in cat["default_columns"])
        print(f"  Default cols:   {cols}")
    if cat.get("estimated_size"):
        print(f"  Est. rows:      {cat['estimated_size']}")
    return True


def _suggest(cm: CrossMatch, name: str, *, n: int = 3) -> str:
    """Format the closest known catalogue names to ``name`` for hinting.

    Returns a leading-space " Did you mean: X, Y, Z?" string ready to be
    appended to an existing error message, or an empty string when nothing is
    close enough. Callers can append it unconditionally.
    """
    matches = cm.suggest(name, n=n)
    return f"Did you mean: {', '.join(matches)}?" if matches else ""



def _suggest_endpoint(endpoint: str, cm: CrossMatch, *, n: int = 3, cutoff: float = 0.4) -> str:
    """Format the closest known endpoint names to ``endpoint`` for hinting."""
    from .discovery import get_public_endpoints

    endpoints = get_public_endpoints()
    pool = list(endpoints.keys())
    for archive_key, archive in cm.archives_config.items():
        pool.append(archive_key)
        for svc_key, svc in archive.items():
            if isinstance(svc, dict) and "access_url" in svc:
                pool.append(svc_key)

    if not pool:
        return ""

    lower_to_orig = {name.lower(): name for name in pool}
    matches = difflib.get_close_matches(
        endpoint.lower(), list(lower_to_orig), n=n, cutoff=cutoff,
    )
    return f" Did you mean: {', '.join(lower_to_orig[m] for m in matches)}?" if matches else ""


def _resolve_discovery_endpoint(name: str, cm: CrossMatch) -> str:
    """Resolve a short endpoint name (e.g. 'vizier') to a full TAP URL."""
    from .discovery import get_public_endpoints

    endpoints = get_public_endpoints()
    if name.lower() in endpoints:
        return endpoints[name.lower()]["url"]
    # If it looks like a URL, return as-is.
    if name.startswith("http://") or name.startswith("https://"):
        return name
    # Check configured archives.
    for archive_key, archive in cm.archives_config.items():
        for svc_key, svc in archive.items():
            if isinstance(svc, dict) and "access_url" in svc:
                if svc_key == name.lower() or archive_key == name.lower():
                    return svc["access_url"]
    return ""


def handle_search(cm: CrossMatch, pattern: str) -> int:
    """Search remote TAP services for tables matching a pattern."""
    from .discovery import discover_tables, get_public_endpoints

    endpoints = get_public_endpoints()
    found_any = False
    effective_pattern = pattern if pattern != "*" else "%"
    if not effective_pattern.startswith("%"):
        effective_pattern = f"%{effective_pattern}%"

    for name, info in sorted(endpoints.items()):
        print(f"\n--- {name} ({info['description']}) ---", flush=True)
        try:
            tables = discover_tables(
                info["url"],
                name_filter=effective_pattern if effective_pattern != "%" else None,
            )
        except Exception as exc:
            print(f"  [unreachable: {exc}]", file=sys.stderr)
            continue
        if tables.height == 0:
            print("  (no matching tables)")
            continue
        found_any = True
        for row in tables.iter_rows(named=True):
            schema = str(row.get("schema_name", ""))
            table = str(row.get("table_name", ""))
            desc = str(row.get("description", ""))
            full = f"{schema}.{table}" if schema else table
            desc_short = desc[:60] + "..." if len(desc) > 60 else desc
            print(f"  {full:<40} {desc_short}")
    # Also search configured archives when they have TAP endpoints.
    for archive_key, archive in sorted(cm.archives_config.items()):
        for svc_key, svc in archive.items():
            if not isinstance(svc, dict) or "access_url" not in svc:
                continue
            url = svc["access_url"]
            try:
                tables = discover_tables(
                    url,
                    name_filter=effective_pattern if effective_pattern != "%" else None,
                )
            except Exception:
                continue
            if tables.height:
                found_any = True
                print(f"\n--- {archive_key}/{svc_key} ({url}) ---", flush=True)
                for row in tables.iter_rows(named=True):
                    schema = str(row.get("schema_name", ""))
                    table = str(row.get("table_name", ""))
                    desc = str(row.get("description", ""))
                    full = f"{schema}.{table}" if schema else table
                    desc_short = desc[:60] + "..." if len(desc) > 60 else desc
                    print(f"  {full:<40} {desc_short}")
    if not found_any:
        print("\nNo matching tables found.")
        print("Tip: use 'xmatch --discover <endpoint>' for known endpoint names.")
    return 0


def handle_discover(cm: CrossMatch, endpoint: str, schema_table: Optional[str] = None) -> int:
    """Discover tables and optionally column schema on a remote TAP endpoint."""
    from .discovery import discover_tables, get_public_endpoints, get_table_schema

    url = _resolve_discovery_endpoint(endpoint, cm)
    if not url:
        suggestion = _suggest_endpoint(endpoint, cm)
        print(f"Unknown endpoint '{endpoint}'.{suggestion}", file=sys.stderr)
        print("Known endpoints:", file=sys.stderr)
        for name, info in sorted(get_public_endpoints().items()):
            print(f"  {name:<12} {info['description']}", file=sys.stderr)
        # Also show configured archives.
        for archive_key in sorted(cm.archives_config):
            print(f"  {archive_key:<12} (configured archive)", file=sys.stderr)
        return 1

    print(f"Endpoint: {url}")

    if schema_table:
        # Show column schema for a specific table.
        try:
            schema = get_table_schema(url, schema_table)
        except Exception as exc:
            print(f"Failed to query schema for '{schema_table}': {exc}", file=sys.stderr)
            return 1
        cols = schema["columns"]
        print(f"\nTable: {schema_table}  ({cols.height} columns)")
        if schema["ra_column"]:
            print(f"  Guessed RA column:  {schema['ra_column']}")
        if schema["dec_column"]:
            print(f"  Guessed Dec column: {schema['dec_column']}")
        print()
        print(f"{'Column':<30} {'Type':<20} {'UCD':<30} {'Unit':<12}")
        print("-" * 95)
        for row in cols.iter_rows(named=True):
            cname = str(row.get("column_name", ""))
            dtype = str(row.get("datatype", ""))
            ucd = str(row.get("ucd", ""))
            unit = str(row.get("unit", ""))
            print(f"  {cname:<28} {dtype:<20} {ucd:<30} {unit:<12}")

        # --- suggested YAML config snippet ---------------------------------
        ra_guess = schema.get("ra_column")
        dec_guess = schema.get("dec_column")
        col_list = schema.get("columns_list", [])
        id_candidates = [c for c in col_list if c.lower() in ("source_id", "id", "objid", "object_id", "allwise", "usno-b1.0", "desig")]
        id_guess = id_candidates[0] if id_candidates else None
        err_candidates = [c for c in col_list if "err" in c.lower() or "error" in c.lower()]
        ra_err_guess = next((c for c in err_candidates if "ra" in c.lower()), None)
        dec_err_guess = next((c for c in err_candidates if "dec" in c.lower() or "de" in c.lower()), None)

        short_name = schema_table.rsplit(".", 1)[-1] if "." in schema_table else schema_table
        print("\n# --- Suggested xmatch.yaml entry (copy into your config) ---")
        print(f"  {short_name}:")
        print("    archive: <archive_name>")
        print("    service_id: <service_id>")
        print(f"    description: \"{schema_table}\"")
        print(f"    access_identifier: \"{schema_table}\"")
        if ra_guess:
            print(f"    ra_column: \"{ra_guess}\"")
        if dec_guess:
            print(f"    dec_column: \"{dec_guess}\"")
        if id_guess:
            print(f"    id_column: \"{id_guess}\"")
        if ra_err_guess:
            print(f"    ra_err_column: \"{ra_err_guess}\"")
        if dec_err_guess:
            print(f"    dec_err_column: \"{dec_err_guess}\"")
        print(f"    # {cols.height} columns discovered")
    else:
        # List tables.
        try:
            tables = discover_tables(url)
        except Exception as exc:
            print(f"Failed to discover tables: {exc}", file=sys.stderr)
            return 1
        print(f"\n{tables.height} table(s) on this endpoint:\n")
        for row in tables.iter_rows(named=True):
            schema_name = str(row.get("schema_name", ""))
            table_name = str(row.get("table_name", ""))
            desc = str(row.get("description", ""))
            full = f"{schema_name}.{table_name}" if schema_name else table_name
            desc_short = desc[:80] + "..." if len(desc) > 80 else desc
            print(f"  {full:<45} {desc_short}")
        print(f"\nTip: use 'xmatch --discover {endpoint} --schema <table>' for column details.")
    return 0


def _parse_extra_distance_cols(raw):
    """Parse 'col:weight,col2:weight2' into dict[str, float]."""
    if not raw:
        return {}
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" in pair:
            col, _, w = pair.partition(":")
            try:
                result[col.strip()] = float(w.strip())
            except ValueError:
                logger.warning(
                    "Invalid weight in --extra-distance-cols '%s'; skipping.", pair
                )
        else:
            result[pair] = 1.0
    return result


def _build_params(args: argparse.Namespace) -> dict:
    """Extract the shared match parameters from CLI args as a kwargs dict."""
    prior_columns = [c.strip() for c in (args.priors or "").split(",") if c.strip()]
    extra_distance = _parse_extra_distance_cols(args.extra_distance_cols)
    ml_color_columns = [c.strip() for c in (args.ml_color_cols or "").split(",") if c.strip()]
    macauff_flux_cols = [c.strip() for c in (args.macauff_flux_cols or "").split(",") if c.strip()]
    return dict(
        radius_arcsec=args.radius_arcsec,
        matcher=args.matcher or "sky",
        max_error=args.max_error,
        join_type=args.join_type,
        find=args.find,
        engine=args.engine,
        id_join=args.id_join,
        id_column_1=args.id_column_1,
        id_column_2=args.id_column_2,
        ra_column_1=args.ra_column_1,
        dec_column_1=args.dec_column_1,
        ra_column_2=args.ra_column_2,
        dec_column_2=args.dec_column_2,
        columns_1=args.columns_1,
        columns_2=args.columns_2,
        prior_columns=prior_columns,
        target_epoch=args.target_epoch,
        filter_expr=args.filter_expr,
        extra_distance_cols=extra_distance,
        batch_size=args.batch_size,
        lr_magnitude_column=args.lr_magnitude_column,
        lr_q=args.lr_q,
        ml_color_columns=ml_color_columns,
        ml_model_path=args.ml_model_path,
        xgb_model_path=args.xgb_model_path,
        macauff_flux_columns=macauff_flux_cols,
        pm_prior=args.pm_prior,
        pm_prior_magnitude_column=args.pm_prior_mag_col,
        ra=args.ra,
        dec=args.dec,
        radius_deg=args.radius_deg,
        probabilistic=args.probabilistic,
        hats_threshold=args.hats_threshold,
    )




def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    try:
        cm = CrossMatch(config_file=args.config_file)

        if args.list_catalogues:
            list_catalogues(cm)
            return 0
        if args.describe:
            success = describe(cm, args.describe)
            return 0 if success else 1
        if args.discover:
            return handle_discover(cm, args.discover, args.schema_table)
        if args.search is not None:
            return handle_search(cm, args.search)
        if len(args.catalogues) < 2:
            print("Error: at least two catalogues are required.", file=sys.stderr)
            print("Try 'xmatch --help' for usage, 'xmatch --list' for catalogues,", file=sys.stderr)
            print("     'xmatch --discover <endpoint>' to explore remote tables.", file=sys.stderr)
            return 2

        params = _build_params(args)

        if args.fof_match:
            # Friends-of-Friends transitive closure across all catalogues.
            result = cm.fof_match(
                args.catalogues,
                output_file=args.output_file,
                radius_arcsec=params.pop("radius_arcsec", 1.0),
                hats_threshold=params.pop("hats_threshold", 100_000),
                **params,
            )
        elif len(args.catalogues) == 2 and not args.union_match:
            # Classic two-catalogue match (backward-compatible path).
            result = cm.crossmatch(
                args.catalogues[0],
                args.catalogues[1],
                output_file=args.output_file,
                **params,
            )
        elif args.union_match:
            # Union catalogue: sequential full outer joins.
            result = cm.union_match(
                args.catalogues,
                output_file=args.output_file,
                **params,
            )
        else:
            # N-catalogue multi-way match.
            result = cm.crossmatch_multi(
                args.catalogues,
                output_file=args.output_file,
                **params,
            )

        if args.output_file is None and result is not None:
            print(f"Matched {result.height} rows.", file=sys.stderr)
            result.write_csv(sys.stdout)
        elif args.output_file:
            print(f"Results written to {args.output_file}.", file=sys.stderr)
        return 0

    except CrossMatchError as exc:
        hint = ""
        source = getattr(exc, "source", None)
        if source:
            hint = _suggest(cm, source)
        print(f"Error: {exc} {hint}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
