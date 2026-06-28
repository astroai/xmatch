"""Command-line interface: ``xmatch CAT1 CAT2``."""

import argparse
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
    parser.add_argument("catalogue_1", nargs="?", help="File path, HATS dir, or catalogue name.")
    parser.add_argument("catalogue_2", nargs="?", help="File path, HATS dir, or catalogue name.")

    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Output file (.parquet/.csv/.fits). If omitted, CSV is written to stdout.",
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
        "--matcher", choices=["sky", "skyerr", "skyellipse"], help="Match algorithm (default: sky)."
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
        "--find",
        choices=["best", "all"],
        default="best",
        help="Keep the best match or all matches within the radius.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "stilts", "astropy", "fast", "zone"],
        default="auto",
        help="Sky-match engine (fast=scipy.cKDTree, zone=HEALPix pixellated).",
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
    print("Available catalogues:")
    for name in sorted(cm.catalogues_config):
        cat = cm.catalogues_config[name]
        print(f"  {name:<24} {cat.get('description', '')}")


def describe(cm: CrossMatch, name: str) -> bool:
    resolved = cm.resolve_name(name)
    if resolved not in cm.catalogues_config:
        print(f"Catalogue '{name}' not found. {_suggest(cm, name)}", file=sys.stderr)
        return False
    cat = cm.catalogues_config[resolved]
    print(f"Catalogue: {resolved}")
    for key in (
        "description",
        "archive",
        "service_id",
        "access_identifier",
        "ra_column",
        "dec_column",
        "id_column",
        "epoch",
        "default_columns",
    ):
        if key in cat:
            print(f"  {key}: {cat[key]}")
    return True


def _suggest(cm: CrossMatch, name: str, *, n: int = 3) -> str:
    """Format the closest known catalogue names to ``name`` for hinting.

    Returns a leading-space " Did you mean: X, Y, Z?" string ready to be
    appended to an existing error message, or an empty string when nothing is
    close enough. Callers can append it unconditionally.
    """
    matches = cm.suggest(name, n=n)
    return f"Did you mean: {', '.join(matches)}?" if matches else ""


def _params(args: argparse.Namespace) -> dict:
    params = {
        "radius_arcsec": args.radius_arcsec,
        "join_type": args.join_type,
        "find": args.find,
        "engine": args.engine,
        "max_error": args.max_error,
    }
    if args.probabilistic:
        params["probabilistic"] = True
    prior_columns = [c.strip() for c in (args.priors or "").split(",") if c.strip()]
    if prior_columns:
        params["prior_columns"] = prior_columns
    if args.matcher:
        params["matcher"] = args.matcher
    if args.id_join:
        params["id_join"] = True
    for key in (
        "ra_column_1",
        "dec_column_1",
        "ra_column_2",
        "dec_column_2",
        "id_column_1",
        "id_column_2",
        "ra",
        "dec",
        "radius_deg",
    ):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    for key in ("columns_1", "columns_2"):
        value = getattr(args, key)
        if value:
            params[key] = [c.strip() for c in value.split(",") if c.strip()]
    return params


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
        if not args.catalogue_1 or not args.catalogue_2:
            print("Error: two catalogues are required.", file=sys.stderr)
            print("Try 'xmatch --list' or 'xmatch --help'.", file=sys.stderr)
            return 2

        result = cm.crossmatch(
            args.catalogue_1, args.catalogue_2, output_file=args.output_file, **_params(args)
        )

        if args.output_file is None and result is not None:
            print(f"Matched {result.height} rows.", file=sys.stderr)
            result.write_csv(sys.stdout)
        elif args.output_file:
            print(f"Results written to {args.output_file}.", file=sys.stderr)
        return 0

    except CrossMatchError as exc:
        # When the exception carries an `InputError.source` (or any future
        # exception that exposes `.source`), append a "did you mean?" hint
        # generated from the catalogue name space.
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
