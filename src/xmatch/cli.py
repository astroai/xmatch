"""Command-line interface for the *xmatch* cross-match tool.

This module exposes a single entry point, :func:`main`, that powers the
``xmatch`` console script defined in ``pyproject.toml``.

The CLI is split into:

* :class:`Console` — a tiny ANSI-colour helper that auto-disables for
  non-TTY streams, ``NO_COLOR`` (https://no-color.org/), ``XMATCH_NO_COLOR``,
  or ``--no-color``.
* Nine dedicated subcommand parsers (``match``, ``list``, ``describe``,
  ``discover``, ``search``, ``adopt``, ``sync``, ``completion``, ``doctor``).
  Each lives in its own function that produces
  an argparse ``Namespace`` plus a colour-aware printer so help, output and
  errors are easy to scan.
* A backwards-compatible legacy flat parser that retains every original
  flag (still used by users who run ``xmatch cat1 cat2`` without an
  explicit ``match`` keyword).
* :func:`main` which dispatches between legacy and subcommand mode based
  on the first non-flag token.

The modernized interface reuses the existing :class:`xmatch.crossmatch.CrossMatch`
engine — no logic changes; only the surface that the user types.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:
    import polars as pl

import yaml

from . import __version__
from .crossmatch import CrossMatch
from .discovery import (
    catalogue_entry_from_schema,
    discover_tables,
    endpoint_archive,
    get_public_endpoints,
    get_table_schema,
)
from .exceptions import ConfigError, CrossMatchError
from .user_config import append_catalogue_to_user_config, format_catalogue_yaml, user_config_path

logger = logging.getLogger(__name__)

# Match ANSI CSI sequences (colour, bold, dim, reset, …).  Used by the
# ``_pad`` / ``_visible_len`` helpers so column widths in coloured tables
# stay aligned on a real terminal.
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _pad(s: str, width: int, align: str = "<") -> str:
    """Right/left/centre-pad ``s`` to *visible* ``width`` (ANSI-aware)."""
    pad_needed = max(0, width - _visible_len(s))
    if align == "<":
        return s + " " * pad_needed
    if align == ">":
        return " " * pad_needed + s
    left = pad_needed // 2
    right = pad_needed - left
    return " " * left + s + " " * right


# ────────────────────────────────────────────────────────────────────────────
# ANSI colour helper
# ────────────────────────────────────────────────────────────────────────────


class Console:
    """A minimal, dependency-free ANSI-colour helper.

    Colours are emitted only when *enabled* is True.  The default detection
    follows the `no-color <https://no-color.org/>`_ standard plus a couple
    of project-specific escape hatches:

    * ``NO_COLOR`` environment variable set to any non-empty value
      → colour disabled.
    * ``XMATCH_NO_COLOR`` environment variable (project equivalent) →
      colour disabled.
    * Both ``stdout`` and ``stderr`` being a TTY → colour enabled.
    * Anywhere else (pipe, redirect, pytest capsys, CI logs) →
      colour disabled to keep machine output clean.
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"

    def __init__(self, *, enabled: Optional[bool] = None) -> None:
        if enabled is None:
            # Honour https://no-color.org/ (presence/any value disables) plus
            # the project-specific XMATCH_NO_COLOR.  Falls back to a TTY
            # check on stdout AND stderr so pipes, CI logs, and pytest
            # capsys fixtures all see plain output.
            disabled = (
                "NO_COLOR" in os.environ
                or "XMATCH_NO_COLOR" in os.environ
                or not sys.stdout.isatty()
                or not sys.stderr.isatty()
            )
            enabled = not disabled
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return str(text)
        return f"{code}{text}{self.RESET}"

    def cyan(self, t: str) -> str:
        return self._wrap(self.CYAN, t)

    def green(self, t: str) -> str:
        return self._wrap(self.GREEN, t)

    def yellow(self, t: str) -> str:
        return self._wrap(self.YELLOW, t)

    def red(self, t: str) -> str:
        return self._wrap(self.RED, t)

    def dim(self, t: str) -> str:
        return self._wrap(self.DIM, t)

    def bold(self, t: str) -> str:
        return self._wrap(self.BOLD, t)

    def header(self, text: str, *, file=None) -> None:
        print(self.cyan(self.bold(text)), file=file or sys.stdout)

    def hint(self, text: str, *, file=None) -> None:
        print(self.yellow(text), file=file or sys.stderr)

    def error(self, text: str, *, file=None) -> None:
        print(self.red(text), file=file or sys.stderr)

    def dim_print(self, text: str, *, file=None) -> None:
        print(self.dim(text), file=file or sys.stdout)

    def info(self, text: str, *, file=None) -> None:
        print(text, file=file or sys.stdout)


# ────────────────────────────────────────────────────────────────────────────
# Progress spinner (dependency-free, TTY-aware)
# ────────────────────────────────────────────────────────────────────────────


class Progress:
    """A tiny, dependency-free spinner for long-running operations.

    Renders a Unicode-Braille spinner + label + elapsed time + status to
    stderr, in-place via CR + clear-to-EOL.  Silently no-ops when
    ``enabled`` is False (non-TTY, CI logs, ``XMATCH_NO_PROGRESS=1``).

    Status updates are emitted via :meth:`update` — backends pass the
    callback through to the engine to surface server-side or client-side
    state changes ("submitting", "phase: queued", "fetching 12 345 rows")
    without needing threads of their own.

    Threading model: only the *animation thread* runs in the background.
    It does nothing except read ``self._status`` (written atomically under
    the GIL by any thread) and rewrite a single stderr line.  Backend
    I/O stays on the main thread.  This avoids races with pyvo /
    astroquery network calls which the upstream libraries do not
    guarantee thread-safe.
    """

    # Unicode Braille pattern spinner — works in nearly every modern TTY.
    _SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _REFRESH_HZ = 10  # Animation frame rate (10 fps).

    def __init__(self, label: str = "", *, enabled: bool = True) -> None:
        self.label = label
        self.enabled = enabled
        self._status = ""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ts = 0.0

    @property
    def status(self) -> str:
        """The current status string (read by the animation loop)."""
        return self._status

    def update(self, status: str) -> None:
        """Push a status string from the backend; replaces the previous one."""
        if self.enabled:
            self._status = status

    def __enter__(self) -> "Progress":
        if not self.enabled:
            return self
        self._start_ts = time.monotonic()
        self._thread = threading.Thread(
            target=self._animate,
            name="xmatch-progress",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.enabled or self._thread is None:
            return
        # Render one final time with the current status so the user's last
        # ``update('phase: running 42 rows')`` is visible *before* we blank
        # the line.  Without this the animation loop exits on the next tick
        # and any update posted in the last ~100 ms is lost.
        try:
            elapsed = time.monotonic() - self._start_ts
            sys.stderr.write(f"\r\033[K✓ {self.label} [{elapsed:5.1f}s] {self._status}")
            sys.stderr.flush()
        except Exception:
            pass
        self._stop.set()
        self._thread.join()
        # CR + clear-to-EOL cleans up the spinner line.  No trailing newline
        # — the *next* stderr write starts at column 1.
        try:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        except Exception:
            pass

    def _animate(self) -> None:
        frames = self._SPINNER_FRAMES
        idx = 0
        while not self._stop.wait(1.0 / self._REFRESH_HZ):
            try:
                elapsed = time.monotonic() - self._start_ts
                spin = frames[idx % len(frames)]
                sys.stderr.write(f"\r\033[K{spin} {self.label} [{elapsed:5.1f}s] {self._status}")
                sys.stderr.flush()
                idx += 1
            except Exception:
                # Never let a transient stderr exception kill the daemon.
                break


def _make_console(no_color: bool = False) -> Console:
    """Construct a :class:`Console` with the user's preference applied.

    Honour ``--no-color``, the ``NO_COLOR`` and ``XMATCH_NO_COLOR``
    environment variables (presence-based per https://no-color.org/),
    and fall back to auto-detection (TTY on both stdout and stderr).
    """
    if no_color or "NO_COLOR" in os.environ or "XMATCH_NO_COLOR" in os.environ:
        return Console(enabled=False)
    return Console()


# ────────────────────────────────────────────────────────────────────────────
# Constants — used by every subcommand's help text
# ────────────────────────────────────────────────────────────────────────────


SUBCOMMANDS = (
    "match",
    "list",
    "describe",
    "discover",
    "search",
    "adopt",
    "completion",
    "sync",
    "doctor",
)

TAGLINE = (
    "Cross-match two or more astronomical catalogues — local files, HATS "
    "directories,\nor the name of a configured remote catalogue (TAP or "
    "CDS XMatch). The\nCLI auto-detects input types, coordinate columns, "
    "and matching strategy."
)

TOP_EXAMPLES = """\
Examples:
  # Match two local files; CSV result on stdout
  xmatch a.parquet b.csv

  # Save matches to a Parquet file
  xmatch a.parquet b.csv -o matches.parquet -r 1.5 --engine fast

  # Match a local file against a configured remote catalogue
  xmatch my_sources.csv gaia -o my_gaia.parquet -r 2.0

  # Build a master union catalogue across N catalogues
  xmatch match gaia allwise.csv twomass.csv --union -o master.parquet

  # Inspection / discovery
  xmatch list
  xmatch describe gaia
  xmatch discover noirlab --schema nsc_dr2.object
  xmatch search gaia
  xmatch adopt vizier II/349/ps1 --name ps1

  # Shell tab completion (bash | zsh | fish) — emit, then `eval` or save
  eval "$(xmatch completion bash)"
  # or: xmatch completion zsh > "$HOME/.zsh/completions/_xmatch"
  # or: xmatch completion fish > "$HOME/.config/fish/completions/xmatch.fish"

  # When piping into files or CI logs, disable colour:
  xmatch --no-color list > catalogues.txt
"""

MATCH_EXAMPLES = """\
Examples:
  # Two local files, default radius (1 arcsec)
  xmatch match a.parquet b.csv

  # Choose engine, output, radius
  xmatch match a.parquet b.csv --engine fast -r 1.5 -o matches.parquet

  # Bayesian p_match with photometric priors
  xmatch match a.parquet b.csv --engine fast --probabilistic --priors g,r -o m.parquet

  # ID join instead of sky position
  xmatch match a.csv b.csv --id-join --id1 source_id --id2 source_id

  # N-way pairwise match across 3 catalogues
  xmatch match gaia allwise.csv twomass.csv -r 1.5 -o nway.parquet

  # Master union catalogue (-o master.hats preferred for spatial queries)
  xmatch match gaia allwise.csv twomass.csv --union -o master.hats --hats-threshold 50000

  # Friends-of-Friends transitive closure
  xmatch match gaia allwise.csv twomass.csv --fof -o bundles.parquet
"""

LIST_EXAMPLES = """\
Examples:
  xmatch list
  xmatch list --no-color    # plain text, friendly to pipes
"""

DESCRIBE_EXAMPLES = """\
Examples:
  xmatch describe gaia
  xmatch describe nsc_noao
"""

DISCOVER_EXAMPLES = """\
Examples:
  xmatch discover noirlab
  xmatch discover noirlab --schema nsc_dr2.object
  xmatch discover https://gea.esac.esa.int/tap-server/tap
"""

SEARCH_EXAMPLES = """\
Examples:
  xmatch search           # all tables on every known endpoint
  xmatch search gaia      # only tables whose name matches "gaia"
"""

ADOPT_EXAMPLES = """\
Examples:
  # Probe VizieR and append to ~/.config/xmatch/xmatch.yaml
  xmatch adopt vizier II/349/ps1 --name ps1

  # Preview the YAML without writing
  xmatch adopt noirlab catwise2020.main --name catwise --dry-run

  # Match an ad-hoc table id without adopting (endpoint auto-guessed for VizieR)
  xmatch match sources.csv II/349/ps1 --ra 150.1 --dec 2.18 --radius-deg 0.05
"""

COMPLETION_EXAMPLES = """\
Examples:
  # bash — source directly into the current shell
  eval "$(xmatch completion bash)"

  # bash — save into the user-level completion directory (loads at login)
  xmatch completion bash > ~/.local/share/bash-completion/completions/xmatch

  # zsh — drop into a $fpath directory (e.g. $ZDOTDIR/completions)
  xmatch completion zsh > "${ZDOTDIR:-$HOME}/.zsh/completions/_xmatch"

  # fish — standard user-level completion path
  xmatch completion fish > ~/.config/fish/completions/xmatch.fish

  # Inspect the script before installing
  xmatch completion bash | less
"""

DOCTOR_EXAMPLES = """\
Examples:
  # Human-readable drift report against the bundled baseline
  xmatch doctor

  # Machine-readable JSON for CI
  xmatch doctor --json

  # Strict mode — exit 1 on any drift (including user-only or missing entries)
  xmatch doctor --strict

  # Quiet — just emit a single summary line
  xmatch doctor --quiet
"""

# Field classifications used by `_diff_configs` to bucket catalogue / archive
# differences.  A field listed in OUTDATED_FIELDS contributes to exit code 1
# (silent breakage risk on the user side).  Informational fields are still
# reported (so the user *sees* them) but do not flip the exit status by
# themselves.  Any field not in either set is treated as outdated — the
# surplus is then surfaced without ambiguity.
OUTDATED_CATALOGUE_FIELDS: frozenset[str] = frozenset(
    {
        "archive",
        "service_id",
        "access_identifier",
        "table_name",
        "release",
        "ra_column",
        "dec_column",
        "id_column",
        "ra_err_column",
        "dec_err_column",
        "pm_ra_column",
        "pm_dec_column",
        "parallax_column",
        "radial_velocity_column",
        "epoch_column",
        "pos_err_units",
        "default_pos_error_arcsec",
        "default_columns",
    }
)
INFORMATIONAL_CATALOGUE_FIELDS: frozenset[str] = frozenset(
    {"description", "estimated_size", "epoch"}
)


# Fallback catalogue / endpoint lists used when the active xmatch.yaml
# cannot be loaded at completion-emission time.  Kept small so emitted
# shell scripts stay lightweight; users with extensive local configs
# will see their full catalogue set when ``xmatch completion`` is run
# from that environment, so the fallback is only here to guarantee
# completion never breaks.
_FALLBACK_CATALOGUES: tuple[str, ...] = (
    "gaia",
    "gaia_esa",
    "allwise",
    "allwise_noao",
    "twomass",
    "nsc",
    "nsc_noao",
    "ps1",
    "sdss",
    "des",
    "des_noao",
    "vizier",
)
_FALLBACK_ENDPOINTS: tuple[str, ...] = (
    "gaia",
    "noirlab",
    "vizier",
    "cadc",
)


def _collect_completion_names(
    cm: Optional["CrossMatch"],
) -> tuple[List[str], List[str]]:
    """Return ``(catalogues, endpoints)`` lists for shell-tab completion.

    ``catalogues`` is the union of catalogue names, aliases, and public
    TAP endpoint short-names (so a user can complete any of them when
    typing a positional for ``xmatch match`` / ``xmatch describe`` /
    ``xmatch discover``).  ``endpoints`` is the smaller set of TAP
    endpoint short-names used by ``xmatch discover`` specifically.

    Falls back to :data:`_FALLBACK_CATALOGUES` /
    :data:`_FALLBACK_ENDPOINTS` whenever any lookup fails — so the
    completion subcommand never breaks (the worst-case is offering a
    short static list rather than the user's full catalogue set).
    """
    catalogues: set[str] = set(_FALLBACK_CATALOGUES)
    endpoints: set[str] = set(_FALLBACK_ENDPOINTS)
    if cm is not None:
        try:
            catalogues.update(cm.catalogues_config.keys())
            catalogues.update(cm.aliases_config.keys())
        except Exception as exc:  # noqa: BLE001
            logger.debug("completion: could not enumerate catalogues: %s", exc)
        try:
            for ep_name in get_public_endpoints():
                endpoints.add(ep_name)
                # Public TAP endpoints are also valid positional names
                # for ``xmatch match`` (e.g. ``xmatch a.csv gaia``).
                catalogues.add(ep_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("completion: could not enumerate endpoints: %s", exc)
    return sorted(catalogues), sorted(endpoints)


# ────────────────────────────────────────────────────────────────────────────
# Helpers shared by parsers
# ────────────────────────────────────────────────────────────────────────────


def _add_global_options(parser: argparse.ArgumentParser) -> None:
    """Add the flags every subcommand supports.

    ``-v/--verbose``, ``--config``, ``--no-color``, ``--version``, and
    ``--help`` are attached to every subparser so users can use them
    either before or after the subcommand keyword (e.g. both
    ``xmatch -v list`` and ``xmatch list --no-color`` work).
    """
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG).",
    )
    parser.add_argument(
        "--config",
        dest="config_file",
        help="Path to a custom xmatch.yaml (default: bundled or ~/.config/xmatch/xmatch.yaml).",
    )
    parser.add_argument(
        "--no-color",
        dest="no_color",
        action="store_true",
        default=False,
        help="Disable ANSI colour output. NO_COLOR=1 also disables.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"xmatch {__version__}",
    )


def _suggest_endpoint(name: str, cm: CrossMatch, n: int = 3, cutoff: float = 0.4) -> str:
    endpoints = get_public_endpoints()
    pool = list(endpoints.keys())
    for archive_key, archive in cm.archives_config.items():
        pool.append(archive_key)
        for svc_key, svc in archive.items():
            if isinstance(svc, dict) and "access_url" in svc:
                pool.append(svc_key)

    if not pool:
        return ""

    lower_to_orig = {cand.lower(): cand for cand in pool}
    matches = difflib.get_close_matches(
        name.lower(),
        list(lower_to_orig),
        n=n,
        cutoff=cutoff,
    )
    if matches:
        suggestions = [lower_to_orig[m] for m in matches]
        return f"Did you mean: {', '.join(suggestions)}?"
    return ""


def _resolve_discovery_endpoint(name: str, cm: CrossMatch) -> str:
    """Resolve a short endpoint name (e.g. 'vizier') to a full TAP URL."""
    endpoints = get_public_endpoints()
    if name.lower() in endpoints:
        return endpoints[name.lower()]["url"]
    if name.startswith("http://") or name.startswith("https://"):
        return name
    for archive_key, archive in cm.archives_config.items():
        for svc_key, svc in archive.items():
            if (
                isinstance(svc, dict)
                and "access_url" in svc
                and (svc_key == name.lower() or archive_key == name.lower())
            ):
                return svc["access_url"]
    return ""


def _suggest(cm: CrossMatch, name: str, *, n: int = 3) -> str:
    matches = cm.suggest(name, n=n)
    return f"Did you mean: {', '.join(matches)}?" if matches else ""


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


def _parse_extra_distance_cols(raw):
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
                logger.warning("Invalid weight in --extra-distance-cols '%s'; skipping.", pair)
        else:
            result[pair] = 1.0
    return result


def _build_params(args) -> dict:
    """Extract the shared match parameters from CLI args as a kwargs dict.

    Works with both legacy and subcommand-mode namespaces, since both use
    identical ``dest=`` names.
    """
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
        memory_budget_bytes=args.memory_budget_bytes,
        scratch_dir=args.scratch_dir,
        partition_order=args.partition_order,
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
        endpoint=getattr(args, "endpoint", None),
        no_sync=bool(getattr(args, "no_sync", False)),
        synclimit=getattr(args, "synclimit", None),
        task_rows=getattr(args, "task_rows", None),
        cache_root=getattr(args, "cache_root", None),
        max_tuples=getattr(args, "max_tuples", None),
        chunk_memory_gb=getattr(args, "chunk_memory_gb", None),
        retries=getattr(args, "retries", None),
        fresh_after=getattr(args, "fresh_after", None),
        min_free_gb=getattr(args, "min_free_gb", None),
    )


# ────────────────────────────────────────────────────────────────────────────
# Parsers
# ────────────────────────────────────────────────────────────────────────────


def build_legacy_parser() -> argparse.ArgumentParser:
    """The legacy flat-form parser.

    Every original flag is preserved byte-for-byte (with identical ``dest=``
    names) so that callers using the old form — ``xmatch --list``,
    ``xmatch --describe NAME``, ``xmatch cat1 cat2 -r 1.0`` — keep working
    unchanged.  Only the *display* of options changes, via grouped help.
    """
    parser = argparse.ArgumentParser(
        prog="xmatch",
        description=TAGLINE,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "catalogues",
        nargs="*",
        help="Two or more catalogues: files, HATS dirs, or configured names.",
    )

    g_out = parser.add_argument_group("Output")
    g_out.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Output file (.parquet/.csv/.fits/.hats). If omitted, CSV is written to stdout.",
    )
    g_out.add_argument(
        "--hats-threshold",
        dest="hats_threshold",
        type=int,
        default=100_000,
        help="Max rows per HEALPix pixel for .hats output.",
    )
    g_out.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="HEALPix pixel groups per batch for out-of-core processing.",
    )
    g_out.add_argument(
        "--memory-budget-bytes",
        type=int,
        help="Spill pairwise local CSV/Parquet matching above this memory budget.",
    )
    g_out.add_argument(
        "--scratch-dir",
        help="Parent directory for bounded-memory partition files.",
    )
    g_out.add_argument(
        "--partition-order",
        default="auto",
        help="Sky-zone partition order (non-negative integer or 'auto').",
    )

    g_geom = parser.add_argument_group("Geometry (sky match)")
    g_geom.add_argument(
        "-r",
        "--radius",
        dest="radius_arcsec",
        type=float,
        default=1.0,
        help="Match radius in arcseconds.",
    )
    g_geom.add_argument("--ra1", dest="ra_column_1", help="RA column name for catalogue 1.")
    g_geom.add_argument("--dec1", dest="dec_column_1", help="Dec column name for catalogue 1.")
    g_geom.add_argument("--ra2", dest="ra_column_2", help="RA column name for catalogue 2.")
    g_geom.add_argument("--dec2", dest="dec_column_2", help="Dec column name for catalogue 2.")
    g_geom.add_argument(
        "--columns-1",
        dest="columns_1",
        help="Comma-separated columns from catalogue 1.",
    )
    g_geom.add_argument(
        "--columns-2",
        dest="columns_2",
        help="Comma-separated columns from catalogue 2.",
    )

    g_alg = parser.add_argument_group("Match algorithm")
    g_alg.add_argument(
        "--matcher",
        choices=["sky", "skyerr", "skyellipse", "lr", "ml", "xgb", "auf", "macauff"],
        help="Match algorithm (default: sky). lr=Likelihood Ratio, ml=Random Forest, "
        "xgb=XGBoost, auf=AUF empirical error model, macauff=AUF+flux.",
    )
    g_alg.add_argument(
        "--max-error",
        dest="max_error",
        type=float,
        default=3.0,
        help="N-sigma cap for skyerr/skyellipse.",
    )
    g_alg.add_argument(
        "--join",
        dest="join_type",
        default="1and2",
        choices=["1and2", "1or2", "all", "1not2", "2not1", "all1", "all2"],
        help="Join type (1and2=inner, 1or2=outer, 1not2/2not1=anti, all1/all2=outer side).",
    )
    g_alg.add_argument(
        "--union",
        dest="union_match",
        action="store_true",
        help="Build a master union catalogue (full outer join across all catalogues).",
    )
    g_alg.add_argument(
        "--fof",
        dest="fof_match",
        action="store_true",
        help="Friends-of-Friends transitive closure across all catalogues.",
    )

    g_ray = parser.add_argument_group("Distributed (engine=ray-union)")
    g_ray.add_argument(
        "--no-sync",
        dest="no_sync",
        action="store_true",
        help="Skip auto-mirroring of remote inputs; require cached HATS copies.",
    )
    g_ray.add_argument(
        "--synclimit",
        dest="synclimit",
        type=float,
        help="Requests/sec rate limit for mirroring remote catalogues (default 1.0).",
    )
    g_ray.add_argument(
        "--cache-root",
        dest="cache_root",
        help="Durable cache root for mirrored HATS catalogues (default: $XMATCH_CACHE_ROOT "
        "or ~/.cache/xmatch).",
    )
    g_ray.add_argument(
        "--task-rows",
        dest="task_rows",
        type=int,
        help="Target rows per Ray chunk task (default 2,000,000).",
    )
    g_ray.add_argument(
        "--max-tuples",
        dest="max_tuples",
        type=int,
        help="Max output tuples per hub source row (default 10,000).",
    )
    g_ray.add_argument(
        "--chunk-memory-gb",
        dest="chunk_memory_gb",
        type=float,
        help="Per-chunk candidate-pool memory guard in GiB (default 8.0).",
    )
    g_ray.add_argument(
        "--retries",
        dest="retries",
        type=int,
        default=0,
        help="Driver-level retries for the distributed union: on failure, re-run "
        "the same request after a backoff (mirror gaps refill incrementally, "
        "finished chunks are skipped). Default 0 (no retries).",
    )
    g_ray.add_argument(
        "--fresh-after",
        dest="fresh_after",
        type=float,
        help="Skip the mirror change-probe for copies fully synced within this many "
        "days (default: always probe).",
    )
    g_ray.add_argument(
        "--min-free-gb",
        dest="min_free_gb",
        type=float,
        help="Fail fast when the cache root or output filesystem has less free space "
        "than this (default 10.0; env XMATCH_MIN_FREE_GB overrides).",
    )
    g_alg.add_argument(
        "--find",
        choices=["best", "all"],
        default="best",
        help="Keep the best match or all matches within the radius.",
    )
    g_alg.add_argument(
        "--engine",
        choices=["auto", "stilts", "astropy", "fast", "torchsky", "zone", "ray", "ray-union"],
        default="auto",
        help=(
            "Sky-match engine (auto=default dispatch, stilts=Java tmatch2, "
            "astropy=pure-Python KD-tree, fast=scipy.cKDTree, torchsky=tensor-native, "
            "zone=HEALPix, ray=distributed zone match, ray-union=distributed N-way "
            "outer join on Ray with mirrored HATS inputs)."
        ),
    )

    g_id = parser.add_argument_group("ID join")
    g_id.add_argument(
        "--id-join",
        dest="id_join",
        action="store_true",
        help="Join on id columns instead of sky position.",
    )
    g_id.add_argument("--id1", dest="id_column_1", help="ID column for catalogue 1.")
    g_id.add_argument("--id2", dest="id_column_2", help="ID column for catalogue 2.")

    g_prob = parser.add_argument_group("Probabilistic / Bayesian")
    g_prob.add_argument(
        "--probabilistic",
        dest="probabilistic",
        action="store_true",
        help="Compute Budavari-style hierarchical Bayes factor (+ p_match column).",
    )
    g_prob.add_argument(
        "--priors",
        dest="priors",
        default="",
        help="Comma-separated photometric columns used as Bayesian priors (e.g. g,r).",
    )

    g_pm = parser.add_argument_group("Proper motion")
    g_pm.add_argument(
        "--target-epoch",
        dest="target_epoch",
        type=float,
        help="Julian-year epoch to propagate coordinates to via proper motion.",
    )
    g_pm.add_argument(
        "--pm-prior",
        dest="pm_prior",
        action="store_true",
        help="Inflate positional errors for sources without measured proper motions using "
        "a Galactic-latitude drift model (Wilson 2023). Requires --target-epoch.",
    )
    g_pm.add_argument(
        "--pm-prior-mag-col",
        dest="pm_prior_mag_col",
        help="Magnitude column for refining the PM drift dispersion estimate "
        "(brighter = closer = larger PM). Only effective with --pm-prior.",
    )

    g_filt = parser.add_argument_group("Advanced filters")
    g_filt.add_argument(
        "--filter-expr",
        dest="filter_expr",
        help="Polars SQL WHERE clause to post-filter matched pairs "
        "(e.g. 'abs(mag - mag_2) < 0.5').",
    )
    g_filt.add_argument(
        "--extra-distance-cols",
        dest="extra_distance_cols",
        help="Column:weight pairs for N-dimensional cKDTree ranking (e.g. 'g:0.5,bp_rp:0.3').",
    )

    g_msm = parser.add_argument_group("Matcher-specific")
    g_msm.add_argument(
        "--lr-magnitude-column",
        dest="lr_magnitude_column",
        help="Magnitude column for Likelihood Ratio matcher (required when --matcher lr).",
    )
    g_msm.add_argument(
        "--lr-q",
        dest="lr_q",
        type=float,
        default=0.8,
        help="Prior Q factor for LR matcher: fraction of primary sources with "
        "detectable counterparts (0.5-1.0).",
    )
    g_msm.add_argument(
        "--ml-color-cols",
        dest="ml_color_cols",
        help="Comma-separated photometric columns for ML matcher features "
        "(e.g. 'g,r,i'). Required when --matcher ml.",
    )
    g_msm.add_argument(
        "--ml-model-path",
        dest="ml_model_path",
        help="Path to save/load a pre-trained Random Forest model (joblib). If the file "
        "exists it is loaded; otherwise a new model is trained and saved.",
    )
    g_msm.add_argument(
        "--xgb-model-path",
        dest="xgb_model_path",
        help="Path to save/load a pre-trained XGBoost model (joblib).",
    )
    g_msm.add_argument(
        "--macauff-flux-cols",
        dest="macauff_flux_cols",
        help="Comma-separated magnitude columns for macauff flux likelihoods.",
    )

    g_reg = parser.add_argument_group("Region (remote downloads)")
    g_reg.add_argument("--ra", type=float, help="Region center RA (deg) for remote downloads.")
    g_reg.add_argument("--dec", type=float, help="Region center Dec (deg) for remote downloads.")
    g_reg.add_argument(
        "--radius-deg",
        dest="radius_deg",
        type=float,
        help="Region radius (deg) for remote downloads.",
    )
    g_reg.add_argument(
        "--endpoint",
        help="TAP endpoint for ad-hoc table ids (vizier, noirlab, gaia).",
    )

    g_info = parser.add_argument_group("Inspection / discovery (legacy flags)")
    g_info.add_argument(
        "--list",
        dest="list_catalogues",
        action="store_true",
        help="List configured catalogues and exit.",
    )
    g_info.add_argument(
        "--describe",
        dest="describe",
        help="Describe a catalogue and exit.",
    )
    g_info.add_argument(
        "--search",
        dest="search",
        nargs="?",
        const="*",
        metavar="PATTERN",
        help="Search remote TAP services for tables matching PATTERN.",
    )
    g_info.add_argument(
        "--discover",
        dest="discover",
        metavar="ENDPOINT",
        help="Discover tables and schema on a remote TAP endpoint.",
    )
    g_info.add_argument(
        "--schema",
        dest="schema_table",
        metavar="TABLE",
        help="Show column schema for a specific remote table.",
    )

    _add_global_options(parser)
    return parser


def _build_match_subparser() -> argparse.ArgumentParser:
    """Standalone parser for the ``match`` subcommand.

    It is *not* parented to a subparsers tree — it stands alone so each
    subcommand can have its own argparse context.  :func:`main` calls
    it directly when the user runs ``xmatch match ...``.
    """
    parser = argparse.ArgumentParser(
        prog="xmatch match",
        description=(
            "Cross-match two or more catalogues.\n\n"
            "Inputs may be local Parquet/CSV/FITS files, polars/pandas frames,\n"
            "HATS directories, or configured remote catalogues (TAP / CDS XMatch).\n"
            "Coordinate columns and matching strategy are auto-detected; flags\n"
            "below refine geometry, algorithm, and output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=MATCH_EXAMPLES,
    )
    parser.add_argument(
        "catalogues",
        nargs="+",
        help="Two or more catalogues: files, HATS dirs, or configured names.",
    )

    g_out = parser.add_argument_group("Output")
    g_out.add_argument(
        "-o",
        "--output",
        dest="output_file",
        metavar="FILE",
        help="Output file (.parquet/.csv/.fits/.hats). Default: CSV to stdout.",
    )
    g_out.add_argument(
        "--hats-threshold",
        dest="hats_threshold",
        type=int,
        default=100_000,
        help="Max rows per HEALPix pixel for .hats output.",
    )
    g_out.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="HEALPix pixel groups per batch for out-of-core processing.",
    )
    g_out.add_argument(
        "--memory-budget-bytes",
        type=int,
        help="Spill pairwise local CSV/Parquet matching above this memory budget.",
    )
    g_out.add_argument(
        "--scratch-dir",
        help="Parent directory for bounded-memory partition files.",
    )
    g_out.add_argument(
        "--partition-order",
        default="auto",
        help="Sky-zone partition order (non-negative integer or 'auto').",
    )

    g_geom = parser.add_argument_group("Geometry (sky match)")
    g_geom.add_argument(
        "-r",
        "--radius",
        dest="radius_arcsec",
        type=float,
        default=1.0,
        help="Match radius (arcsec) for the sky matcher.",
    )
    g_geom.add_argument("--ra1", dest="ra_column_1", help="RA column name for catalogue 1.")
    g_geom.add_argument("--dec1", dest="dec_column_1", help="Dec column name for catalogue 1.")
    g_geom.add_argument("--ra2", dest="ra_column_2", help="RA column name for catalogue 2.")
    g_geom.add_argument("--dec2", dest="dec_column_2", help="Dec column name for catalogue 2.")
    g_geom.add_argument(
        "--columns-1",
        dest="columns_1",
        help="Comma-separated columns from catalogue 1.",
    )
    g_geom.add_argument(
        "--columns-2",
        dest="columns_2",
        help="Comma-separated columns from catalogue 2.",
    )

    g_alg = parser.add_argument_group("Match algorithm")
    g_alg.add_argument(
        "--matcher",
        choices=["sky", "skyerr", "skyellipse", "lr", "ml", "xgb", "auf", "macauff"],
        help="Match algorithm (default: sky).",
    )
    g_alg.add_argument(
        "--max-error",
        dest="max_error",
        type=float,
        default=3.0,
        help="N-sigma cap for skyerr/skyellipse.",
    )
    g_alg.add_argument(
        "--join",
        dest="join_type",
        default="1and2",
        choices=["1and2", "1or2", "all", "1not2", "2not1", "all1", "all2"],
        help="Join type (1and2=inner, 1or2=outer, etc.).",
    )
    g_alg.add_argument(
        "--find",
        choices=["best", "all"],
        default="best",
        help="Keep the best match or all matches within the radius.",
    )
    g_alg.add_argument(
        "--engine",
        choices=["auto", "stilts", "astropy", "fast", "torchsky", "zone", "ray", "ray-union"],
        default="auto",
        help=(
            "Sky-match engine (auto=default dispatch, stilts=Java tmatch2, "
            "astropy=pure-Python KD-tree, fast=scipy.cKDTree, torchsky=tensor-native, "
            "zone=HEALPix, ray=distributed zone match, ray-union=distributed N-way "
            "outer join on Ray with mirrored HATS inputs)."
        ),
    )
    g_alg.add_argument(
        "--union",
        dest="union_match",
        action="store_true",
        help="Build a master union catalogue (outer join across all).",
    )
    g_alg.add_argument(
        "--fof",
        dest="fof_match",
        action="store_true",
        help="Friends-of-Friends transitive closure across all catalogues.",
    )

    g_ray = parser.add_argument_group("Distributed (engine=ray-union)")
    g_ray.add_argument(
        "--no-sync",
        dest="no_sync",
        action="store_true",
        help="Skip auto-mirroring of remote inputs; require cached HATS copies.",
    )
    g_ray.add_argument(
        "--synclimit",
        dest="synclimit",
        type=float,
        help="Requests/sec rate limit for mirroring remote catalogues (default 1.0).",
    )
    g_ray.add_argument(
        "--cache-root",
        dest="cache_root",
        help="Durable cache root for mirrored HATS catalogues (default: $XMATCH_CACHE_ROOT "
        "or ~/.cache/xmatch).",
    )
    g_ray.add_argument(
        "--task-rows",
        dest="task_rows",
        type=int,
        help="Target rows per Ray chunk task (default 2,000,000).",
    )
    g_ray.add_argument(
        "--max-tuples",
        dest="max_tuples",
        type=int,
        help="Max output tuples per hub source row (default 10,000).",
    )
    g_ray.add_argument(
        "--chunk-memory-gb",
        dest="chunk_memory_gb",
        type=float,
        help="Per-chunk candidate-pool memory guard in GiB (default 8.0).",
    )
    g_ray.add_argument(
        "--retries",
        dest="retries",
        type=int,
        default=0,
        help="Driver-level retries for the distributed union: on failure, re-run "
        "the same request after a backoff (mirror gaps refill incrementally, "
        "finished chunks are skipped). Default 0 (no retries).",
    )
    g_ray.add_argument(
        "--fresh-after",
        dest="fresh_after",
        type=float,
        help="Skip the mirror change-probe for copies fully synced within this many "
        "days (default: always probe).",
    )
    g_ray.add_argument(
        "--min-free-gb",
        dest="min_free_gb",
        type=float,
        help="Fail fast when the cache root or output filesystem has less free space "
        "than this (default 10.0; env XMATCH_MIN_FREE_GB overrides).",
    )

    g_id = parser.add_argument_group("ID join")
    g_id.add_argument(
        "--id-join",
        dest="id_join",
        action="store_true",
        help="Join on id columns instead of sky position.",
    )
    g_id.add_argument("--id1", dest="id_column_1", help="ID column for catalogue 1.")
    g_id.add_argument("--id2", dest="id_column_2", help="ID column for catalogue 2.")

    g_prob = parser.add_argument_group("Probabilistic / Bayesian")
    g_prob.add_argument(
        "--probabilistic",
        dest="probabilistic",
        action="store_true",
        help="Compute Budavari-style hierarchical Bayes factor (+ p_match).",
    )
    g_prob.add_argument(
        "--priors",
        dest="priors",
        default="",
        help="Comma-separated photometric columns used as priors (e.g. g,r).",
    )

    g_pm = parser.add_argument_group("Proper motion")
    g_pm.add_argument(
        "--target-epoch",
        dest="target_epoch",
        type=float,
        help="Julian-year epoch to propagate coordinates via proper motion.",
    )
    g_pm.add_argument(
        "--pm-prior",
        dest="pm_prior",
        action="store_true",
        help="Inflate errors for sources lacking proper motions (Wilson 2023). "
        "Requires --target-epoch.",
    )
    g_pm.add_argument(
        "--pm-prior-mag-col",
        dest="pm_prior_mag_col",
        help="Magnitude column for refining the PM drift dispersion estimate.",
    )

    g_filt = parser.add_argument_group("Advanced filters")
    g_filt.add_argument(
        "--filter-expr",
        dest="filter_expr",
        help="Polars SQL WHERE clause to post-filter matched pairs.",
    )
    g_filt.add_argument(
        "--extra-distance-cols",
        dest="extra_distance_cols",
        help="Column:weight pairs for N-dim cKDTree ranking (e.g. 'g:0.5,bp_rp:0.3').",
    )

    g_msm = parser.add_argument_group("Matcher-specific")
    g_msm.add_argument(
        "--lr-magnitude-column",
        dest="lr_magnitude_column",
        help="Magnitude column for LR matcher (required when --matcher lr).",
    )
    g_msm.add_argument(
        "--lr-q",
        dest="lr_q",
        type=float,
        default=0.8,
        help="Prior Q factor for LR matcher (0.5-1.0).",
    )
    g_msm.add_argument(
        "--ml-color-cols",
        dest="ml_color_cols",
        help="Comma-separated photometric columns for ML matcher features "
        "(required when --matcher ml).",
    )
    g_msm.add_argument(
        "--ml-model-path",
        dest="ml_model_path",
        help="Path to save/load a pre-trained Random Forest model (joblib).",
    )
    g_msm.add_argument(
        "--xgb-model-path",
        dest="xgb_model_path",
        help="Path to save/load a pre-trained XGBoost model (joblib).",
    )
    g_msm.add_argument(
        "--macauff-flux-cols",
        dest="macauff_flux_cols",
        help="Comma-separated magnitude columns for macauff flux likelihoods.",
    )

    g_reg = parser.add_argument_group("Region (remote downloads)")
    g_reg.add_argument("--ra", type=float, help="Region center RA (deg).")
    g_reg.add_argument("--dec", type=float, help="Region center Dec (deg).")
    g_reg.add_argument(
        "--radius-deg",
        dest="radius_deg",
        type=float,
        help="Region radius (deg) for remote cone downloads.",
    )
    g_reg.add_argument(
        "--endpoint",
        help="TAP endpoint for ad-hoc table ids (vizier, noirlab, gaia).",
    )

    _add_global_options(parser)
    return parser


def _build_sync_subparser() -> argparse.ArgumentParser:
    """Standalone parser for the ``sync`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="xmatch sync",
        description=(
            "Mirror remote catalogues (TAP, or HATS over HTTP / vos:) into the durable "
            "cache root as HATS, with incremental re-sync on later runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  xmatch sync gaia allwise twomass\n"
            "  xmatch sync my_tap_table --rate-limit 2 --threads 4 --cache-root /data/xmatch-cache\n"
            "  xmatch sync --force allwise        # refetch everything\n"
        ),
    )
    parser.add_argument(
        "catalogues",
        nargs="+",
        help="Catalogues to mirror: configured names, HATS URLs (https://…, vos:…), "
        "or remote table ids.",
    )
    parser.add_argument(
        "--rate-limit",
        dest="rate_limit_rps",
        type=float,
        help="Requests/sec per remote host (default 1.0; 0 disables throttling).",
    )
    parser.add_argument(
        "--force",
        dest="force_sync",
        action="store_true",
        help="Refetch every file/page, ignoring the incremental manifest.",
    )
    parser.add_argument(
        "--cache-root",
        dest="cache_root",
        help="Durable cache root for mirrored HATS catalogues (default: $XMATCH_CACHE_ROOT "
        "or ~/.cache/xmatch).",
    )
    parser.add_argument(
        "--threads",
        dest="workers",
        type=int,
        default=8,
        help="Parallel download workers (default 8).",
    )
    parser.add_argument(
        "--hats-threshold",
        dest="hats_threshold",
        type=int,
        default=100_000,
        help="Max rows per HEALPix pixel for the converted HATS catalogue.",
    )
    parser.add_argument(
        "--fresh-after",
        dest="fresh_after",
        type=float,
        help="Skip the change-probe for copies fully synced within this many days "
        "(default: always probe).",
    )
    parser.add_argument(
        "--min-free-gb",
        dest="min_free_gb",
        type=float,
        help="Fail fast when the cache root has less free space than this "
        "(default 10.0; env XMATCH_MIN_FREE_GB overrides).",
    )
    _add_global_options(parser)
    return parser


def _build_list_subparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmatch list",
        description="List every catalogue defined in the active config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LIST_EXAMPLES,
    )
    _add_global_options(parser)
    return parser


def _build_describe_subparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmatch describe",
        description="Print full metadata (columns, access info, defaults) for a named catalogue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DESCRIBE_EXAMPLES,
    )
    parser.add_argument("name", help="Catalogue name or alias (e.g. gaia, nsc_noao).")
    _add_global_options(parser)
    return parser


def _build_discover_subparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmatch discover",
        description="Browse tables on a remote TAP endpoint; with --schema, dump column metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DISCOVER_EXAMPLES,
    )
    parser.add_argument(
        "endpoint",
        help="Endpoint short-name (gaia, noirlab, vizier) or full TAP URL.",
    )
    parser.add_argument(
        "--schema",
        dest="schema_table",
        metavar="TABLE",
        help="Dump column schema for a single table and suggest a YAML snippet.",
    )
    _add_global_options(parser)
    return parser


def _build_search_subparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmatch search",
        description="Search every known TAP endpoint for tables whose name matches a substring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SEARCH_EXAMPLES,
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default="*",
        help="Substring to search for (default: '*' = all tables).",
    )
    _add_global_options(parser)
    return parser


def _build_adopt_subparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmatch adopt",
        description=(
            "Probe a remote TAP table and append a catalogue entry to "
            "~/.config/xmatch/xmatch.yaml (merged over the bundled config)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=ADOPT_EXAMPLES,
    )
    parser.add_argument(
        "endpoint",
        help="Endpoint short-name (vizier, noirlab, gaia) or a known public TAP name.",
    )
    parser.add_argument(
        "table",
        help="Table id (e.g. II/349/ps1 or ls_dr10.tractor).",
    )
    parser.add_argument(
        "--name",
        dest="catalogue_name",
        help="Catalogue key to write (default: derived from the table id).",
    )
    parser.add_argument(
        "--alias",
        help="Optional short alias pointing at the new catalogue.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the YAML snippet without writing the user config.",
    )
    _add_global_options(parser)
    return parser


def _build_completion_subparser() -> argparse.ArgumentParser:
    """Standalone parser for the ``completion`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="xmatch completion",
        description=(
            "Emit a shell-completion script for the requested shell.\n\n"
            "Catalogue names and TAP endpoints are resolved from the active\n"
            "xmatch.yaml at emission time and embedded into the script, so\n"
            "`xmatch completion` always reflects the user's current config.\n"
            "If the config can't be loaded a small hardcoded list is used\n"
            "so completion is never broken."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=COMPLETION_EXAMPLES,
    )
    parser.add_argument(
        "shell",
        choices=("bash", "zsh", "fish"),
        help="Shell to emit a completion script for.",
    )
    _add_global_options(parser)
    return parser


def _build_doctor_subparser() -> argparse.ArgumentParser:
    """Standalone parser for the ``doctor`` subcommand.

    The ``doctor`` subcommand compares the user's active ``xmatch.yaml``
    against the bundled baseline and reports drift.  Output is human-readable
    by default; ``--json`` emits a structured report on ``stdout``;
    ``--strict`` elevates intentional-override drift (missing-in-user /
    user-only) to exit code 1, useful for CI templates that want to enforce
    an exact match to the baseline.
    """
    parser = argparse.ArgumentParser(
        prog="xmatch doctor",
        description=(
            "Validate the active xmatch.yaml against the bundled baseline\n"
            "and report drift.\n\n"
            "Three categories of drift are tracked:\n\n"
            "  - outdated fields    : a catalogue/alias exists in both\n"
            "                         configs but differs in a structural\n"
            "                         field (ra_column, default_columns, …).\n"
            "                         May indicate silent breakage.\n"
            "  - missing-in-user    : a bundled entry has been removed.\n"
            "                         Usually an intentional trim.\n"
            "  - user-only          : an entry is present in the user's\n"
            "                         config but not the baseline.\n"
            "                         Usually a deliberate local addition.\n\n"
            "Default exit code is 0 unless at least one outdated field is\n"
            "found, or ``--strict`` is set and the topology has drifted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DOCTOR_EXAMPLES,
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=False,
        help="Exit 1 on ANY drift (including missing-in-user / user-only).",
    )
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="Suppress the per-entry report; emit only a one-line summary.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        default=False,
        help="Emit the drift report as JSON on stdout (machine-readable).",
    )
    _add_global_options(parser)
    return parser


# ────────────────────────────────────────────────────────────────────────────
# Shell-completion script emitters
# ────────────────────────────────────────────────────────────────────────────
# These functions generate self-contained POSIX-style shell scripts that
# the user installs via `eval` (bash) or by saving into the appropriate
# completion directory (zsh, fish).  All three accept the same two
# sequences (catalogues, endpoints) so the embedded completion list is
# the user's active config, captured at emission time.
#
# Quoting strategy per shell:
#   bash  — wrap each name in single quotes, escape internal ' as '\''.
#            Names with whitespace are stored as separate array entries,
#            preserving round-trip fidelity through `compgen -W`.
#   zsh   — single-quoted array entries; zsh `_describe` handles quoting
#            internally so we don't need shell-side escaping tricks.
#   fish  — single-quoted with `\'` for embedded apostrophes (fish's
#            only escape inside single quotes).


def _bash_quote_array_entry(s: str) -> str:
    """Single-quote-quote a string for inclusion in a bash array literal.

    Replaces internal single-quotes with ``'\\''`` (the standard POSIX
    safe escape sequence: close-quote, escaped-quote, re-open).
    """
    if not s:
        return "''"
    return "'" + s.replace("'", "'\\''") + "'"


def _zsh_quote(s: str) -> str:
    """Quote a string for inclusion in a zsh array literal."""
    if not s:
        return "''"
    return "'" + s.replace("'", "'\\''") + "'"


def _fish_quote(s: str) -> str:
    """Quote a string for inclusion in a fish single-quoted literal.

    Fish uses ``\\'`` (backslash-escaped) for embedded apostrophes inside
    single-quoted strings.
    """
    if not s:
        return "''"
    return "'" + s.replace("'", "\\'") + "'"


def _emit_bash(catalogues: Sequence[str], endpoints: Sequence[str]) -> str:
    """Return the bash completion-script body."""
    cat_array = "\n".join(f"    {_bash_quote_array_entry(c)}" for c in catalogues)
    ep_array = "\n".join(f"    {_bash_quote_array_entry(c)}" for c in endpoints)
    return f"""\
# bash completion for xmatch — generated by `xmatch completion bash`.
# Install (current shell only):
#   eval "$(xmatch completion bash)"
# Install persistently:
#   xmatch completion bash > ~/.local/share/bash-completion/completions/xmatch

_xmatch_catalogues=(
{cat_array}
)
_xmatch_endpoints=(
{ep_array}
)
_xmatch_shells=( bash zsh fish )
_xmatch_subcommands=( match list describe discover search adopt completion sync doctor )

_xmatch() {{
    local cur="" words="" cword=0 i subcmd=""
    # Use bash-completion's _init_completion if available, otherwise
    # fall back to raw COMP_WORDS parsing so we never require an extra
    # dependency for tab completion.
    if type _init_completion &>/dev/null; then
        _init_completion 2>/dev/null || {{
            cur="${{COMP_WORDS[COMP_CWORD]}}"
            words=( "${{COMP_WORDS[@]}}" )
            cword=$COMP_CWORD
        }}
    else
        cur="${{COMP_WORDS[COMP_CWORD]}}"
        words=( "${{COMP_WORDS[@]}}" )
        cword=$COMP_CWORD
    fi

    # Skip past leading flag tokens to locate the subcommand keyword.
    for ((i = 1; i < cword; i++)); do
        case "${{words[i]}}" in
            match|list|describe|discover|search|adopt|completion|sync)
                subcmd="${{words[i]}}"; break ;;
        esac
    done

    # Subcommand keyword position (1st positional, before any subcmd).
    if [[ -z "$subcmd" && "$cur" != -* ]]; then
        COMPREPLY=( $(compgen -W "${{_xmatch_subcommands[*]}}" -- "$cur") )
        return 0
    fi

    # Completing an option: offer the standard global flag set.
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "-h --help -v --verbose --config --no-color --version" -- "$cur") )
        return 0
    fi

    case "$subcmd" in
        match|describe|sync)
            COMPREPLY=( $(compgen -W "${{_xmatch_catalogues[*]}}" -- "$cur") )
            ;;
        discover|adopt)
            COMPREPLY=( $(compgen -W "${{_xmatch_endpoints[*]}}" -- "$cur") )
            ;;
        completion)
            COMPREPLY=( $(compgen -W "${{_xmatch_shells[*]}}" -- "$cur") )
            ;;
    esac
    return 0
}}
complete -F _xmatch xmatch
"""


def _emit_zsh(catalogues: Sequence[str], endpoints: Sequence[str]) -> str:
    """Return the zsh completion-script body."""
    cat_array = " ".join(_zsh_quote(c) for c in catalogues)
    ep_array = " ".join(_zsh_quote(c) for c in endpoints)
    # The literal ``{{``/``}}`` in the script below are escaped because
    # this Python string is an f-string; raw zsh sees single braces.
    return f"""\
#compdef xmatch
# zsh completion for xmatch — generated by `xmatch completion zsh`.
# Install:
#   xmatch completion zsh > "${{ZDOTDIR:-$HOME}}/.zsh/completions/_xmatch"

_xmatch_subcommands=(
    'match:cross-match two or more catalogues'
    'sync:mirror remote catalogues into the local HATS cache'
    'list:list every configured catalogue'
    'describe:show columns and access info for a catalogue'
    'discover:browse tables on a remote TAP endpoint'
    'search:search every endpoint for tables matching a substring'
    'adopt:save a remote table into the user config overlay'
    'completion:emit a shell completion script'
    'doctor:report drift against the bundled baseline'
)
_xmatch_catalogues=( {cat_array} )
_xmatch_endpoints=( {ep_array} )
_xmatch_shells=( bash zsh fish )

_xmatch() {{
    local state
    _arguments -C \\
        '(-h --help)'{{-h,--help}}'[show help and exit]' \\
        '(-v --verbose)'{{-v,--verbose}}'[increase verbosity]' \\
        '--config[custom xmatch.yaml]:file:_files' \\
        '--no-color[disable ANSI colours]' \\
        '--version[show version and exit]' \\
        '*:: :->rest'
    case $state in
        rest)
            if (( CURRENT == 1 )); then
                _describe 'subcommand' _xmatch_subcommands
                return
            fi
            case $words[1] in
                match|describe|sync)
                    _describe 'catalogue' _xmatch_catalogues
                    ;;
                discover|adopt)
                    _describe 'endpoint' _xmatch_endpoints
                    ;;
                completion)
                    _describe 'shell' _xmatch_shells
                    ;;
            esac
            ;;
    esac
}}
_xmatch "$@"
"""


def _emit_fish(catalogues: Sequence[str], endpoints: Sequence[str]) -> str:
    """Return the fish completion-script body."""
    cat_lines = "\n".join(f"set -a _xmatch_catalogues {_fish_quote(c)}" for c in catalogues)
    ep_lines = "\n".join(f"set -a _xmatch_endpoints {_fish_quote(c)}" for c in endpoints)
    return f"""\
# fish completion for xmatch — generated by `xmatch completion fish`.
# Install:
#   xmatch completion fish > ~/.config/fish/completions/xmatch.fish

function _xmatch_subcmd
    for token in $argv
        switch $token
            case match list describe discover search adopt completion sync
                echo $token
                return
        end
    end
end

function _xmatch_needs_catalogue
    set -l cmd (_xmatch_subcmd $argv)
    contains -- $cmd match describe
end

function _xmatch_needs_endpoint
    set -l cmd (_xmatch_subcmd $argv)
    contains -- $cmd discover adopt
end

function _xmatch_needs_shell
    set -l cmd (_xmatch_subcmd $argv)
    test "$cmd" = completion
end

set -l _xmatch_subcommands match list describe discover search adopt completion sync doctor

{cat_lines}

{ep_lines}

# Subcommand keyword (1st positional, no subcommand typed yet).
complete -c xmatch -f -n 'test (count (commandline -opc)) -eq 1' -a '$_xmatch_subcommands'

# Catalogue names for `xmatch match <TAB>` and `xmatch describe <TAB>`.
complete -c xmatch -f -n '_xmatch_needs_catalogue; and test (count (commandline -opc)) -eq 2' -a '$_xmatch_catalogues'

# Endpoint names for `xmatch discover <TAB>`.
complete -c xmatch -f -n '_xmatch_needs_endpoint; and test (count (commandline -opc)) -eq 2' -a '$_xmatch_endpoints'

# Shell names for `xmatch completion <TAB>`.
complete -c xmatch -f -n '_xmatch_needs_shell; and test (count (commandline -opc)) -eq 2' -a 'bash zsh fish'
"""


def _build_top_parser() -> argparse.ArgumentParser:
    """The top-level parser shown by ``xmatch --help``.

    Carries *no positional* and only the global options
    (``-v``, ``--config``, ``--no-color``, ``--version``).  Subcommand
    dispatch is handled in :func:`main` via dedicated subparsers, so we
    don't use ``add_subparsers`` here — that decision keeps the help
    output clean (subcommands are listed in the description) and avoids
    argparse's subparsers "ambiguous option" restrictions.  This parser
    also serves as the top-level help blurb so users running
    ``xmatch --help`` see the friendly one-screen summary instead of the
    full flat-form help.
    """
    subcommand_blurb = (
        "\n\nAvailable commands:\n"
        "  match       cross-match catalogues (the default for `xmatch CAT1 CAT2`)\n"
        "  list        list every configured catalogue\n"
        "  describe    show columns & access info for a catalogue\n"
        "  discover    browse tables and columns on a remote TAP endpoint\n"
        "  search      search every endpoint for tables matching a substring\n"
        "  adopt       save a remote table into ~/.config/xmatch/xmatch.yaml\n"
        "  sync        mirror remote catalogues into the local HATS cache\n"
        "  completion  emit a shell tab-completion script (bash | zsh | fish)\n"
        "  doctor      report drift between your xmatch.yaml and the bundled baseline\n"
        "\n"
        "Run `xmatch COMMAND --help` for command-specific options.  The legacy\n"
        "flat form (`xmatch --list`, `xmatch --describe NAME`, `xmatch --search`,\n"
        "`xmatch --discover ENDPOINT`) keeps working for backward compatibility."
    )
    parser = argparse.ArgumentParser(
        prog="xmatch",
        description=TAGLINE + subcommand_blurb,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TOP_EXAMPLES,
    )
    _add_global_options(parser)
    return parser


# ────────────────────────────────────────────────────────────────────────────
# Pretty-printer helpers (use Console)
# ────────────────────────────────────────────────────────────────────────────


def list_catalogues(cm: CrossMatch, console: Console) -> None:
    """Print all configured catalogues — colour-aware, table-formatted.

    Shows memorable survey names first. Archive/table details live in
    ``xmatch describe`` so users are not nudged to paste ACCESS ids.
    """
    widths = (20, 8, 56)
    header = f"{'NAME':<{widths[0]}}  {'ROWS':>{widths[1]}}  DESCRIPTION"
    console.header(header)
    console.dim_print("\u2500" * _visible_len(header))
    # Prefer primary surveys; demote archive mirrors of the same survey.
    # With `gaia` defaulting to CDS VizieR, gaia_cds is primary and the
    # ESA / NOIRLab mirrors are demoted.
    primary: list[str] = []
    mirrors: list[str] = []
    for name in sorted(cm.catalogues_config):
        if name in {"gaia_esa", "gaia_noao"}:
            mirrors.append(name)
        else:
            primary.append(name)

    def _emit(name: str, *, mirror: bool = False) -> None:
        cat = cm.catalogues_config[name]
        size = str(cat.get("estimated_size", "?"))
        desc = cat.get("description", "")
        if mirror:
            desc = f"(mirror) {desc}"
        size_styled = console.green(size) if size in {"small", "medium", "large", "huge"} else size
        label = console.dim(name) if mirror else console.bold(name)
        sys.stdout.write(
            f"  {_pad(label, widths[0])}  {_pad(size_styled, widths[1], '>')}  {desc}\n"
        )

    for name in primary:
        _emit(name)
    if mirrors:
        console.dim_print("")
        console.dim_print("  Alternative Gaia archives (gaia defaults to CDS VizieR):")
        for name in mirrors:
            _emit(name, mirror=True)
    sys.stdout.write("\n")
    console.info(f"{len(cm.catalogues_config)} catalogues, {len(cm.aliases_config)} aliases.")
    console.hint("Use 'xmatch describe <name>' for archive/table/column details.")
    console.hint("Match with the NAME column (e.g. ps1, desils), not the remote table id.")


def describe(cm: CrossMatch, name: str, console: Console) -> bool:
    resolved = cm.resolve_name(name)
    if resolved not in cm.catalogues_config:
        console.error(f"  Catalogue '{name}' not found.")
        suggestion = _suggest(cm, name)
        if suggestion:
            console.hint(f"  {suggestion}")
        return False
    cat = cm.catalogues_config[resolved]
    # Single styled line so the contiguous substring ``"Catalogue: NAME"``
    # is visible on real terminals too (ANSI escapes wrap the whole line,
    # not interleaved between the label and the name).
    console.header(f"Catalogue: {resolved}")
    label_width = 18

    def kv(label: str, value) -> None:
        sys.stdout.write(
            f"  {_pad(console.dim(label + ':'), label_width + 2)}{console.cyan(str(value))}\n"
        )

    kv("Description", cat.get("description", "—"))
    kv("Archive", cat.get("archive", "?"))
    kv("Service", cat.get("service_id", "?"))
    kv("Table", cat.get("access_identifier", "?"))
    kv("RA column", cat.get("ra_column", "—"))
    kv("Dec column", cat.get("dec_column", "—"))
    kv("ID column", cat.get("id_column", "—"))
    if cat.get("ra_err_column"):
        kv("RA err column", cat["ra_err_column"])
    if cat.get("dec_err_column"):
        kv("Dec err column", cat["dec_err_column"])
    kv("Epoch", cat.get("epoch", "—"))
    if cat.get("default_columns"):
        cols = ", ".join(str(c) for c in cat["default_columns"])
        kv("Default cols", console.yellow(cols))
    if cat.get("estimated_size"):
        kv("Est. rows", cat["estimated_size"])

    if cat.get("pm_ra_column") or cat.get("pm_dec_column"):
        sys.stdout.write("\n")
        console.dim_print("  (proper-motion columns available)")
    if cat.get("parallax_column"):
        kv("Parallax column", cat["parallax_column"])
    if cat.get("radial_velocity_column"):
        kv("Radial velocity", cat["radial_velocity_column"])
    return True


def handle_search(cm: CrossMatch, pattern: str, console: Console) -> int:
    endpoints = get_public_endpoints()
    found_any = False
    effective_pattern = pattern if pattern != "*" else "%"
    if not effective_pattern.startswith("%"):
        effective_pattern = f"%{effective_pattern}%"

    for name, info in sorted(endpoints.items()):
        console.header(f"── {console.bold(name)}  {console.dim(info['description'])} ──")
        try:
            tables = discover_tables(
                info["url"],
                name_filter=effective_pattern if effective_pattern != "%" else None,
            )
        except Exception as exc:
            console.error(f"  [{console.dim('unreachable')}: {exc}]")
            continue
        if tables.height == 0:
            console.dim_print("  (no matching tables)")
            continue
        found_any = True
        for row in tables.iter_rows(named=True):
            schema = str(row.get("schema_name", ""))
            table = str(row.get("table_name", ""))
            desc = str(row.get("description", ""))
            full = f"{schema}.{table}" if schema else table
            desc_short = (desc[:60] + "…") if len(desc) > 60 else desc
            sys.stdout.write(f"  {_pad(console.cyan(full), 40)} {desc_short}\n")
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
                console.header(
                    f"\u2500\u2500 {console.bold(archive_key + '/' + svc_key)}  "
                    f"{console.dim(url)} \u2500\u2500"
                )
                for row in tables.iter_rows(named=True):
                    schema = str(row.get("schema_name", ""))
                    table = str(row.get("table_name", ""))
                    desc = str(row.get("description", ""))
                    full = f"{schema}.{table}" if schema else table
                    desc_short = (desc[:60] + "…") if len(desc) > 60 else desc
                    sys.stdout.write(f"  {_pad(console.cyan(full), 40)} {desc_short}\n")
    if not found_any:
        console.error("\nNo matching tables found.")
        console.hint("Tip: use 'xmatch discover <endpoint>' for known endpoint names.")
        console.hint("     or 'xmatch adopt <endpoint> <table> --name <short>' to save one.")
        return 1
    return 0


def handle_discover(
    cm: CrossMatch, endpoint: str, schema_table: Optional[str], console: Console
) -> int:
    url = _resolve_discovery_endpoint(endpoint, cm)
    if not url:
        console.error(f"Unknown endpoint '{endpoint}'.")
        suggestion = _suggest_endpoint(endpoint, cm)
        if suggestion:
            console.hint(f"  {suggestion}")
        console.hint("Known endpoints:")
        for name, info in sorted(get_public_endpoints().items()):
            sys.stderr.write(f"  {_pad(console.cyan(name), 14)}{info['description']}\n")
        for archive_key in sorted(cm.archives_config):
            sys.stderr.write(
                f"  {_pad(console.cyan(archive_key), 14)}{console.dim('(configured archive)')}\n"
            )
        return 1

    console.info(f"Endpoint: {console.cyan(url)}")

    if schema_table:
        try:
            schema = get_table_schema(url, schema_table)
        except Exception as exc:
            console.error(f"Failed to query schema for '{schema_table}': {exc}")
            return 1
        cols = schema["columns"]
        if cols.height == 0:
            console.error(f"\nTable '{schema_table}' not found or has no columns.")
            return 1
        console.header(f"\nTable: {console.bold(schema_table)}  ({cols.height} columns)")
        if schema["ra_column"]:
            console.info(f"  Guessed RA column:  {console.green(schema['ra_column'])}")
        if schema["dec_column"]:
            console.info(f"  Guessed Dec column: {console.green(schema['dec_column'])}")
        sys.stdout.write("\n")
        console.header(
            f"  {_pad('COLUMN', 30)}{_pad('TYPE', 20)}{_pad('UCD', 30)}{_pad('UNIT', 12)}"
        )
        for row in cols.iter_rows(named=True):
            cname = str(row.get("column_name", ""))
            dtype = str(row.get("datatype", ""))
            ucd = str(row.get("ucd", ""))
            unit = str(row.get("unit", ""))
            sys.stdout.write(
                f"  {_pad(console.cyan(cname), 30)}"
                f"{_pad(dtype, 20)}"
                f"{_pad(ucd, 30)}"
                f"{_pad(unit, 12)}\n"
            )

        # ── suggested YAML config snippet ───────────────────────────────
        try:
            archive_name, service_id, _ = endpoint_archive(endpoint)
        except Exception:
            archive_name, service_id = None, "tap_service"
        if archive_name is None:
            archive_name = endpoint.lower()
        try:
            short_name, entry = catalogue_entry_from_schema(
                schema_table,
                schema,
                archive=archive_name,
                service_id=service_id,
            )
            tip = f"# ─── Suggested entry (or: xmatch adopt {endpoint} {schema_table}) ───"
            sys.stdout.write(f"\n{console.dim(tip)}\n")
            sys.stdout.write(format_catalogue_yaml(short_name, entry))
        except Exception as exc:
            console.hint(f"Could not build a full snippet ({exc}); showing minimal stub.")
            short_name = schema_table.rsplit(".", 1)[-1] if "." in schema_table else schema_table
            sys.stdout.write(f"  {short_name}:\n")
            sys.stdout.write(f'    archive: "{archive_name}"\n')
            sys.stdout.write(f'    service_id: "{service_id}"\n')
            sys.stdout.write(f'    access_identifier: "{schema_table}"\n')
            if schema.get("ra_column"):
                sys.stdout.write(f'    ra_column: "{schema["ra_column"]}"\n')
            if schema.get("dec_column"):
                sys.stdout.write(f'    dec_column: "{schema["dec_column"]}"\n')
        console.hint(
            f"Tip: `xmatch adopt {endpoint} {schema_table}` writes this into {user_config_path()}."
        )
    else:
        try:
            tables = discover_tables(url)
        except Exception as exc:
            console.error(f"Failed to discover tables: {exc}")
            return 1
        if tables.height == 0:
            console.error("\nNo tables found on this endpoint.")
            return 1
        console.info(f"\n{console.bold(str(tables.height))} table(s) on this endpoint:\n")
        for row in tables.iter_rows(named=True):
            schema_name = str(row.get("schema_name", ""))
            table_name = str(row.get("table_name", ""))
            desc = str(row.get("description", ""))
            full = f"{schema_name}.{table_name}" if schema_name else table_name
            desc_short = (desc[:80] + "…") if len(desc) > 80 else desc
            sys.stdout.write(f"  {_pad(console.cyan(full), 45)}  {desc_short}\n")
        console.hint(f"Tip: use 'xmatch discover {endpoint} --schema <table>' for column details.")
        console.hint(f"     or 'xmatch adopt {endpoint} <table>' to save a user-config entry.")
    return 0


def handle_adopt(
    cm: CrossMatch,
    endpoint: str,
    table: str,
    console: Console,
    *,
    catalogue_name: Optional[str] = None,
    alias: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Probe *table* on *endpoint* and append it to the user config overlay."""
    try:
        archive_name, service_id, tap_url = endpoint_archive(endpoint)
    except Exception as exc:
        console.error(str(exc))
        suggestion = _suggest_endpoint(endpoint, cm)
        if suggestion:
            console.hint(f"  {suggestion}")
        return 1
    if archive_name is None:
        console.error(
            f"Endpoint '{endpoint}' has no bundled archive mapping; "
            "cannot adopt into xmatch.yaml yet."
        )
        return 1
    if archive_name not in cm.archives_config:
        console.error(f"Archive '{archive_name}' is not in the active config.")
        return 1

    console.info(f"Probing {console.cyan(table)} on {console.cyan(tap_url)} …")
    try:
        schema = get_table_schema(tap_url, table)
    except Exception as exc:
        console.error(f"Failed to query schema for '{table}': {exc}")
        return 1
    if schema["columns_count"] == 0:
        console.error(f"Table '{table}' not found or has no columns on '{endpoint}'.")
        return 1

    try:
        name, entry = catalogue_entry_from_schema(
            table,
            schema,
            archive=archive_name,
            service_id=service_id,
            name=catalogue_name,
        )
    except Exception as exc:
        console.error(str(exc))
        return 1

    snippet = format_catalogue_yaml(name, entry)
    if dry_run:
        console.info(f"Dry run — would write to {user_config_path()}:\n")
        sys.stdout.write(snippet)
        if alias:
            sys.stdout.write(f"  # alias: {alias.lower()} -> {name}\n")
        return 0

    if name in cm.catalogues_config:
        console.error(
            f"Catalogue '{name}' already exists in the active config. "
            "Pass --name to choose a different key."
        )
        return 1

    if alias:
        alias_key = alias.lower()
        if alias_key in cm.catalogues_config:
            console.error(
                f"Alias '{alias_key}' collides with an existing catalogue name. "
                "Choose a different --alias."
            )
            return 1
        existing = cm.aliases_config.get(alias_key)
        if existing is not None and existing != name:
            console.error(
                f"Alias '{alias_key}' already points at '{existing}'. Choose a different --alias."
            )
            return 1

    try:
        path = append_catalogue_to_user_config(
            name,
            entry,
            alias=alias,
            archives={archive_name: cm.archives_config[archive_name]},
        )
    except ConfigError as exc:
        console.error(str(exc))
        return 1

    console.info(f"Adopted {console.bold(name)} → {path}")
    if alias:
        console.info(f"Alias {console.cyan(alias.lower())} → {name}")
    console.hint(f"Try: xmatch describe {alias or name}")
    console.hint(f"     xmatch match <local.parquet> {alias or name} --ra … --dec … --radius-deg …")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# Match execution (shared between legacy and subcommand modes)
# ────────────────────────────────────────────────────────────────────────────


def _execute_match(args, cm: CrossMatch, console: Console) -> int:
    """Drive the crossmatch orchestration common to legacy and subcommand modes."""
    if len(args.catalogues) < 2:
        console.error("Error: at least two catalogues are required for a crossmatch.")
        console.hint("Try 'xmatch --help' for usage, 'xmatch list' for catalogues,")
        console.hint("     'xmatch discover <endpoint>' to explore remote tables.")
        return 2

    params = _build_params(args)
    n = len(args.catalogues)
    label = f"Cross-matching {n} catalogue{'s' if n != 1 else ''}"
    progress = Progress(label, enabled=console.enabled)

    with progress:
        progress.update("preparing inputs")
        if args.fof_match:
            result: Optional[Union[pl.DataFrame, pl.LazyFrame]] = cm.fof_match(
                args.catalogues,
                output_file=args.output_file,
                radius_arcsec=params.pop("radius_arcsec", 1.0),
                hats_threshold=params.pop("hats_threshold", 100_000),
                progress_cb=progress.update,
                **params,
            )
        elif len(args.catalogues) == 2 and not args.union_match:
            result = cm.crossmatch(
                args.catalogues[0],
                args.catalogues[1],
                output_file=args.output_file,
                progress_cb=progress.update,
                **params,
            )
        elif args.union_match:
            result = cm.union_match(
                args.catalogues,
                output_file=args.output_file,
                progress_cb=progress.update,
                **params,
            )
        else:
            result = cm.crossmatch_multi(
                args.catalogues,
                output_file=args.output_file,
                progress_cb=progress.update,
                **params,
            )

    if args.output_file is None and result is not None:
        print(f"Matched {result.height} rows.", file=sys.stderr)
        result.write_csv(sys.stdout)
    elif args.output_file:
        print(f"Results written to {args.output_file}.", file=sys.stderr)
    return 0


# ────────────────────────────────────────────────────────────────────────────
# Dispatchers for each mode
# ────────────────────────────────────────────────────────────────────────────


def _guarded(
    cm: CrossMatch,
    console: Console,
    body: Callable[[], int],
) -> int:
    """Run ``body()`` with the CLI's standard exception handlers.

    Used by every subcommand runner so that ``CrossMatchError``,
    ``KeyboardInterrupt``, argparse's ``SystemExit`` (from ``-h`` /
    ``--version``), and unexpected errors are reported consistently.
    """
    try:
        return body()
    except CrossMatchError as exc:
        hint = ""
        source = getattr(exc, "source", None)
        if source:
            hint = _suggest(cm, source)

        console.error(f"Error: {exc}")
        if hint:
            console.hint(f"  {hint}")
        return 1
    except KeyboardInterrupt:
        console.error("Interrupted.")
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error")
        console.error(f"Unexpected error: {exc}")
        return 1


def _run_legacy(argv: Sequence[str]) -> int:
    """Run with the original flat-form parser — every flag preserved.

    Includes ``--version`` and ``--help`` which argparse handles
    natively via ``action="version"``.  Any SystemExit from those
    propagates through :func:`_guarded` unchanged.
    """
    parser = build_legacy_parser()
    args = parser.parse_args(list(argv))
    setup_logging(args.verbose)
    console = _make_console(no_color=bool(getattr(args, "no_color", False)))
    cm = CrossMatch(config_file=args.config_file)

    def body() -> int:
        if args.list_catalogues:
            list_catalogues(cm, console)
            return 0
        if args.describe:
            return 0 if describe(cm, args.describe, console) else 1
        if args.discover:
            return handle_discover(cm, args.discover, args.schema_table, console)
        if args.search is not None:
            return handle_search(cm, args.search, console)

        # Empty positional or 1 catalogue: emit the well-known error
        # message that legacy callers (and our tests) pin.  Note: this is
        # the safest fallback for users who migrated from very old forms
        # like `xmatch cat1` without an explicit subcommand.
        if len(args.catalogues) < 2:
            console.error("Error: at least two catalogues are required.")
            console.hint("Try 'xmatch --help' for usage,")
            console.hint("     'xmatch list' to list configured catalogues,")
            console.hint("     'xmatch discover <endpoint>' for remote tables.")
            return 2

        return _execute_match(args, cm, console)

    return _guarded(cm, console, body)


def _run_match_subcommand(argv: Sequence[str]) -> int:
    parser = _build_match_subparser()
    args = parser.parse_args(list(argv))
    setup_logging(args.verbose)
    console = _make_console(no_color=bool(getattr(args, "no_color", False)))
    cm = CrossMatch(config_file=args.config_file)
    return _guarded(cm, console, lambda: _execute_match(args, cm, console))


def _run_sync_subcommand(argv: Sequence[str]) -> int:
    parser = _build_sync_subparser()
    args = parser.parse_args(list(argv))
    setup_logging(args.verbose)
    console = _make_console(no_color=bool(getattr(args, "no_color", False)))
    cm = CrossMatch(config_file=args.config_file)

    def body() -> int:
        from .mirror import SyncStats, sync_catalogue
        from .storage import default_cache_root

        cache_cfg = cm.config.get("cache", {}) if isinstance(cm.config, dict) else {}
        cache_root = (
            args.cache_root
            or os.environ.get("XMATCH_CACHE_ROOT")
            or cache_cfg.get("root")
            or default_cache_root()
        )
        rate_limit = args.rate_limit_rps
        if rate_limit is None:
            rate_limit = float(cache_cfg.get("rate_limit_rps", 1.0))
        total = SyncStats()
        for name in args.catalogues:
            resolved = cm.resolve_name(name)
            cat_cfg = cm.catalogues_config.get(resolved, {})
            src = cm.resolve_source(name, {})
            one_limit = rate_limit
            if args.rate_limit_rps is None:
                one_limit = float((cat_cfg.get("sync") or {}).get("rate_limit_rps", rate_limit))
            console.info(f"Syncing {name} (cache: {cache_root}) …")
            stats = sync_catalogue(
                src,
                cache_root=cache_root,
                rate_limit_rps=one_limit,
                workers=int(args.workers),
                force=bool(args.force_sync),
                hats_threshold=int(args.hats_threshold),
                fresh_after=args.fresh_after,
                min_free_gb=args.min_free_gb,
            )
            total.merge(stats)
            console.info(
                f"  {name}: {stats.bytes_downloaded} bytes, {stats.files_downloaded} file(s), "
                f"{stats.pages} page(s) ({stats.files_skipped} skipped, {stats.failed} failed)"
            )
        console.info(
            f"Sync complete: {total.files_downloaded} file(s), {total.bytes_downloaded} bytes "
            f"({total.files_skipped} skipped, {total.failed} failed)."
        )
        return 0 if total.failed == 0 else 1

    return _guarded(cm, console, body)


def _run_simple_describe(argv: Sequence[str]) -> int:
    parser = _build_describe_subparser()
    args = parser.parse_args(list(argv))
    setup_logging(args.verbose)
    console = _make_console(no_color=bool(getattr(args, "no_color", False)))
    cm = CrossMatch(config_file=args.config_file)
    return _guarded(cm, console, lambda: 0 if describe(cm, args.name, console) else 1)


def _run_completion(argv: Sequence[str]) -> int:
    """Dispatch ``xmatch completion <shell>``: emit a completion script to stdout.

    Catalogue and endpoint names are pulled from the active ``xmatch.yaml``
    so ``xmatch match <TAB>``, ``xmatch describe <TAB>``, and
    ``xmatch discover <TAB>`` complete against the user's *actual*
    catalogue set.  When the config can't be loaded (:func:`CrossMatch`
    raises on a missing/broken file), the emitters fall back to a
    short hardcoded list so completion is never broken.

    The script is written to stdout so users can either ``eval`` it (bash)
    or redirect to the completion-destined file (zsh, fish).
    """
    parser = _build_completion_subparser()
    args = parser.parse_args(list(argv))
    setup_logging(args.verbose)
    console = _make_console(no_color=bool(getattr(args, "no_color", False)))
    cm: Optional[CrossMatch] = None
    try:
        cm = CrossMatch(config_file=args.config_file)
    except Exception as exc:  # noqa: BLE001
        console.hint(
            f"# Couldn't load config ({exc}); using fallback catalogue list.",
            file=sys.stderr,
        )
    catalogues, endpoints = _collect_completion_names(cm)
    if args.shell == "bash":
        sys.stdout.write(_emit_bash(catalogues, endpoints))
    elif args.shell == "zsh":
        sys.stdout.write(_emit_zsh(catalogues, endpoints))
    elif args.shell == "fish":
        sys.stdout.write(_emit_fish(catalogues, endpoints))
    # argparse already constrains args.shell to {bash,zsh,fish}, so any
    # other branch is unreachable in practice.
    return 0


def _run_adopt_subcommand(argv: Sequence[str]) -> int:
    parser = _build_adopt_subparser()
    args = parser.parse_args(list(argv))
    setup_logging(args.verbose)
    console = _make_console(no_color=bool(getattr(args, "no_color", False)))
    cm = CrossMatch(config_file=args.config_file)
    return _guarded(
        cm,
        console,
        lambda: handle_adopt(
            cm,
            args.endpoint,
            args.table,
            console,
            catalogue_name=args.catalogue_name,
            alias=args.alias,
            dry_run=args.dry_run,
        ),
    )


def _run_no_pos_subcommand(argv: Sequence[str], name: str) -> int:
    """Dispatch a subcommand with no required positional (list, discover, search)."""
    parser_builders = {
        "list": _build_list_subparser,
        "discover": _build_discover_subparser,
        "search": _build_search_subparser,
    }
    if name not in parser_builders:
        raise ValueError(f"_run_no_pos_subcommand got unknown name {name!r}")
    parser = parser_builders[name]()
    args = parser.parse_args(list(argv))
    setup_logging(args.verbose)
    console = _make_console(no_color=bool(getattr(args, "no_color", False)))
    cm = CrossMatch(config_file=args.config_file)

    def body() -> int:
        if name == "list":
            list_catalogues(cm, console)
            return 0
        if name == "discover":
            return handle_discover(cm, args.endpoint, args.schema_table, console)
        if name == "search":
            return handle_search(cm, args.pattern, console)
        raise AssertionError(f"unreachable: {name}")  # pragma: no cover

    return _guarded(cm, console, body)


# ────────────────────────────────────────────────────────────────────────────
# Top-level dispatcher: main()
# ────────────────────────────────────────────────────────────────────────────


def _split_subcommand(argv: Sequence[str]) -> tuple[Optional[str], List[str]]:
    """Locate the first subcommand keyword in ``argv`` and split it off.

    Returns ``(keyword, remaining)`` where ``remaining`` is ``argv`` with
    the located keyword removed and any leading flag-only tokens kept (so
    ``xmatch -v list`` becomes ``("list", ["-v"])``).

    Flag tokens are skipped while looking for the keyword — so
    ``xmatch --no-color describe gaia`` is correctly recognised as
    ``describe`` with remaining ``["--no-color", "gaia"]``.

    Returns ``(None, list(argv))`` when no keyword is found, so callers
    can fall through to the legacy flat-form parser unchanged.
    """
    for i, tok in enumerate(argv):
        if tok.startswith("-"):
            continue
        if tok in SUBCOMMANDS:
            return tok, list(argv[:i]) + list(argv[i + 1 :])
        return None, list(argv)
    return None, list(argv)


def _load_bundled_config() -> tuple[Path, Dict[str, Any]]:
    """Load the bundled ``xmatch.yaml`` that ships with the package.

    This is the *baseline* used by :func:`_run_doctor` to detect drift in the
    user's active config.  We resolve it via ``importlib.resources`` so the
    lookup works whether the package was installed (wheel / sdist / pixi
    env), is loaded straight from a checkout, or is symlinked into a venv.
    """
    try:
        from importlib.resources import files
    except ImportError:  # pragma: no cover — Python < 3.9 fallback
        from importlib_resources import files  # type: ignore
    bundled_path = Path(str(files("xmatch") / "xmatch.yaml"))
    try:
        with open(bundled_path, "r") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read bundled xmatch.yaml: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Bundled xmatch.yaml is not a YAML mapping.")
    return bundled_path, data


def _normalise_for_compare(value: Any) -> Any:
    """Coerce YAML-loaded values into a canonical form for equality checks.

    Normalisations:

    * ``None`` and absent are both represented as ``None`` (the equality
      check in :func:`_compare_field` already handles this idempotently).
    * Lists are converted to frozensets of their element types so that
      ``default_columns: [ra, dec]`` and ``default_columns: [dec, ra]``
      compare equal (order is irrelevant for the engine).
    * Bare numeric scalars are coerced to ``float`` so ``epoch: 2016`` and
      ``epoch: 2016.0`` compare equal — YAML's int/float type leak is a
      common false-positive source.
    """
    if isinstance(value, list):
        # Tuples are not permitted in xmatch.yaml so a list is unambiguous.
        return frozenset(_normalise_for_compare(v) for v in value)
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _compare_field(bundled_value: Any, user_value: Any) -> bool:
    """Return True iff the two values should be treated as equivalent."""
    b = _normalise_for_compare(bundled_value)
    u = _normalise_for_compare(user_value)
    if b is None and u is None:
        return True
    if b is None or u is None:
        # Treat null and absent as equal — YAML allows either spelling and
        # users shouldn't see drift between them.
        return b is u
    return b == u


def _diff_field_lists(bundled_val: Any, user_val: Any) -> tuple[List[Any], List[Any]]:
    """Return ``(bundled_added, user_extra)`` as ordered lists.

    Used for lists such as ``default_columns`` — we report symmetric diff
    rather than a binary equal/unequal verdict because users want to see
    which columns were renamed / added upstream.
    """
    b_set = set(_normalise_for_compare(v) for v in (bundled_val or []))
    u_set = set(_normalise_for_compare(v) for v in (user_val or []))
    bundled_added = [v for v in (bundled_val or []) if _normalise_for_compare(v) not in u_set]
    user_extra = [v for v in (user_val or []) if _normalise_for_compare(v) not in b_set]
    return bundled_added, user_extra


def _diff_configs(bundled: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the bundled baseline against the user's active config.

    Returns a report dict with four top-level arrays:

    * ``outdated_fields`` — list of ``{section_type, name, field, …}``
      records where a field differs between baseline and user.  Bucketed
      into ``outdated`` (silent-breakage risk) and ``informational``
      (cosmetic change).
    * ``missing_in_user`` — baseline entries that the user has trimmed.
    * ``user_only`` — entries present in the user config but not the bundle.
    * ``alias_redirects`` — same as outdated_fields but flagged specially
      because they break positional-name resolution.

    The report is purely structural — no formatting or exit-code logic lives
    here so the same data can drive both human and ``--json`` output.
    """
    report: Dict[str, Any] = {
        "outdated_fields": [],
        "informational_fields": [],
        "missing_in_user": [],
        "user_only": [],
        "alias_redirects": [],
    }

    # ---- catalogues ---------------------------------------------------
    bundled_cats = dict(bundled.get("catalogues") or {})
    user_cats = dict(user.get("catalogues") or {})
    for name in sorted(bundled_cats):
        if name not in user_cats:
            report["missing_in_user"].append({"section_type": "catalogue", "name": name})
            continue
        bund = bundled_cats[name] or {}
        usr = user_cats[name] or {}
        # Union of all fields appearing in either side, plus the known
        # structural / informational fields so we never silently skip
        # them on the other side.
        keys = set(bund) | set(usr) | OUTDATED_CATALOGUE_FIELDS | INFORMATIONAL_CATALOGUE_FIELDS
        for field in sorted(keys):
            in_bund = field in bund
            in_user = field in usr
            bund_v = bund.get(field)
            user_v = usr.get(field)
            if (
                (in_bund and not in_user)
                or (not in_bund and in_user)
                or not _compare_field(bund_v, user_v)
            ):
                bucket = "informational" if field in INFORMATIONAL_CATALOGUE_FIELDS else "outdated"
                entry = {
                    "section_type": "catalogue",
                    "name": name,
                    "field": field,
                    "_bucket": bucket,
                }
                if field == "default_columns":
                    bundled_added, user_extra = _diff_field_lists(bund_v, user_v)
                    entry["bundled_added"] = bundled_added
                    entry["user_extra"] = user_extra
                else:
                    entry["bundled_value"] = bund_v
                    entry["user_value"] = user_v
                if bucket == "outdated":
                    report["outdated_fields"].append(entry)
                else:
                    report["informational_fields"].append(entry)

    for name in sorted(user_cats):
        if name not in bundled_cats:
            report["user_only"].append({"section_type": "catalogue", "name": name})

    # ---- aliases ------------------------------------------------------
    bundled_aliases = dict(bundled.get("catalogue_aliases") or {})
    user_aliases = dict(user.get("catalogue_aliases") or {})
    for alias in sorted(bundled_aliases):
        bund_target = bundled_aliases[alias]
        if alias not in user_aliases:
            report["missing_in_user"].append({"section_type": "alias", "name": alias})
            continue
        if bund_target != user_aliases[alias]:
            report["alias_redirects"].append(
                {
                    "section_type": "alias",
                    "name": alias,
                    "field": "target",
                    "_bucket": "outdated",
                    "bundled_value": bund_target,
                    "user_value": user_aliases[alias],
                }
            )
            # Alias redirects are surfaced separately because they change
            # positional-name resolution, which is more visible to the user
            # than a single-column rename.  Still count them as outdated so
            # the exit-code policy is symmetric.
            report["outdated_fields"].append(
                {
                    "section_type": "alias",
                    "name": alias,
                    "field": "target",
                    "_bucket": "outdated",
                    "bundled_value": bund_target,
                    "user_value": user_aliases[alias],
                }
            )

    for alias in sorted(user_aliases):
        if alias not in bundled_aliases:
            report["user_only"].append({"section_type": "alias", "name": alias})

    # ---- archives -----------------------------------------------------
    bundled_archives = dict(bundled.get("archives") or {})
    user_archives = dict(user.get("archives") or {})
    for archive in sorted(bundled_archives):
        if archive not in user_archives:
            report["missing_in_user"].append({"section_type": "archive", "name": archive})
        # We don't deep-compare nested service dicts (TAP URLs change too
        # often per release to be useful as drift signals).  Topology and
        # archive keys are what matter here.
    for archive in sorted(user_archives):
        if archive not in bundled_archives:
            report["user_only"].append({"section_type": "archive", "name": archive})

    return report


def _doctor_exit_code(report: Dict[str, Any], *, strict: bool = False) -> int:
    """Map a drift report to an exit code per the policy in the docstring."""
    if report["outdated_fields"]:
        return 1
    if strict and (report["missing_in_user"] or report["user_only"]):
        return 1
    return 0


def _emit_doctor_human(
    report: Dict[str, Any],
    *,
    console: Console,
    active_path: Optional[Path],
    bundled_path: Path,
    strict: bool,
    quiet: bool,
) -> int:
    """Render the drift report to stdout, then return the chosen exit code."""
    rc = _doctor_exit_code(report, strict=strict)

    if quiet:
        n_outdated = len(report["outdated_fields"])
        n_missing = len(report["missing_in_user"])
        n_user_only = len(report["user_only"])
        n_alias = len(report["alias_redirects"])
        marker = "[strict] " if strict and rc == 0 else ""
        verb = "drift" if rc else "matches baseline"
        sys.stdout.write(
            f"xmatch doctor: {n_outdated} outdated, "
            f"{n_alias} alias redirect(s), {n_missing} missing, "
            f"{n_user_only} user-only — {marker}{verb}.\n"
        )
        return rc

    # ---- header ---------------------------------------------------
    console.header("xmatch doctor — configuration drift report")
    console.info(
        f"  active config:   {console.cyan(str(active_path) if active_path else '(bundled default)')}"
    )
    console.info(f"  bundled baseline: {console.dim(str(bundled_path))}")
    sys.stdout.write("\n")

    # ---- outdated fields ------------------------------------------
    if report["outdated_fields"]:
        console.header(console.yellow("[ OUTDATED FIELDS ] — exit 1"))
        for entry in report["outdated_fields"]:
            name = entry["name"]
            field = entry["field"]
            label = f"  {name}.{field}"
            sys.stdout.write(console.bold(label) + "\n")
            if "bundled_added" in entry:
                added = entry["bundled_added"]
                extra = entry["user_extra"]
                if added:
                    sys.stdout.write(f"    + bundled added:   {console.green(repr(added))}\n")
                if extra:
                    sys.stdout.write(f"    - user has extra:  {console.yellow(repr(extra))}\n")
            else:
                sys.stdout.write(
                    f"    bundled: {console.dim(repr(entry['bundled_value']))}\n"
                    f"    user:    {console.cyan(repr(entry['user_value']))}\n"
                )
        sys.stdout.write("\n")
    else:
        console.dim_print("  [ OUTDATED FIELDS ] — none.\n")

    # ---- informational --------------------------------------------
    if report["informational_fields"]:
        console.header(console.dim("[ INFORMATIONAL ] — cosmetic only"))
        for entry in report["informational_fields"]:
            label = f"  {entry['name']}.{entry['field']}"
            sys.stdout.write(f"{console.dim(label)}\n")
            sys.stdout.write(
                f"    bundled: {console.dim(repr(entry['bundled_value']))}\n"
                f"    user:    {console.dim(repr(entry['user_value']))}\n"
            )
        sys.stdout.write("\n")
    else:
        console.dim_print("  [ INFORMATIONAL ] — none.\n")

    # ---- missing in user ------------------------------------------
    if report["missing_in_user"]:
        console.header(console.dim("[ MISSING IN USER ] — informational"))
        for entry in report["missing_in_user"]:
            kind = entry["section_type"]
            sys.stdout.write(f"  - {kind}: {console.dim(entry['name'])}\n")
        sys.stdout.write("\n")
    else:
        console.dim_print("  [ MISSING IN USER ] — none.\n")

    # ---- user-only -------------------------------------------------
    if report["user_only"]:
        console.header(console.cyan("[ USER-ONLY ] — local additions"))
        for entry in report["user_only"]:
            kind = entry["section_type"]
            sys.stdout.write(f"  + {kind}: {console.cyan(entry['name'])}\n")
        sys.stdout.write("\n")
    else:
        console.dim_print("  [ USER-ONLY ] — none.\n")

    # ---- verdict ---------------------------------------------------
    n_outdated = len(report["outdated_fields"])
    n_missing = len(report["missing_in_user"])
    n_user_only = len(report["user_only"])
    n_alias = len(report["alias_redirects"])
    if rc == 0:
        console.info(
            f"xmatch doctor: {n_outdated} outdated, {n_alias} alias redirect(s), "
            f"{n_missing} missing, {n_user_only} user-only — matches baseline."
        )
    else:
        console.error(
            f"xmatch doctor: {n_outdated} outdated, {n_alias} alias redirect(s), "
            f"{n_missing} missing, {n_user_only} user-only — drift detected."
        )
    return rc


def _emit_doctor_json(
    report: Dict[str, Any],
    *,
    active_path: Optional[Path],
    bundled_path: Path,
    strict: bool,
) -> int:
    """Render the drift report as JSON on stdout, then return the chosen exit code."""
    rc = _doctor_exit_code(report, strict=strict)
    payload = {
        "drift": rc != 0,
        "strict": strict,
        "config_active": str(active_path) if active_path else None,
        "config_bundled": str(bundled_path),
        "outdated_fields": [
            {k: v for k, v in e.items() if k != "_bucket"} for e in report["outdated_fields"]
        ],
        "informational_fields": [
            {k: v for k, v in e.items() if k != "_bucket"} for e in report["informational_fields"]
        ],
        "missing_in_user": report["missing_in_user"],
        "user_only": report["user_only"],
        "alias_redirects": report["alias_redirects"],
        "summary": {
            "n_outdated": len(report["outdated_fields"]),
            "n_informational": len(report["informational_fields"]),
            "n_missing": len(report["missing_in_user"]),
            "n_user_only": len(report["user_only"]),
            "n_alias_redirects": len(report["alias_redirects"]),
        },
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
    sys.stdout.write("\n")
    return rc


def _run_doctor(argv: Sequence[str]) -> int:
    """Dispatch ``xmatch doctor``: compare active config to bundled baseline.

    Loads the user's active ``xmatch.yaml`` (via :class:`CrossMatch` so the
    resolver logic matches every other subcommand) and the bundled yaml via
    :func:`_load_bundled_config`, builds a drift report, then renders it
    either human-readable or JSON based on the flags.

    Exit codes: 0 (matches baseline / informational only), 1 (drift detected),
    2 (cannot load / invalid — handled by argparse).
    """
    parser = _build_doctor_subparser()
    args = parser.parse_args(list(argv))
    setup_logging(args.verbose)
    console = _make_console(no_color=bool(getattr(args, "no_color", False)))

    # Load bundled baseline first — if this fails we surface a hard error
    # (rc=2) because there is no baseline to compare against.
    try:
        bundled_path, bundled_cfg = _load_bundled_config()
    except ConfigError as exc:
        console.error(f"Error: {exc}")
        return 2

    # Load the user's active config.  We use CrossMatch directly so the
    # default-resolution chain (--config > ~/.config/xmatch > bundled) is
    # shared with every other subcommand.  Falling back to the bundled
    # config here would be misleading — the user just asked "how does my
    # config differ?" and the answer is "there is no user config".
    user_path: Optional[Path] = None
    user_cfg: Dict[str, Any] = {}
    try:
        cm = CrossMatch(config_file=args.config_file)
        user_path = cm.config_file
        # Re-load raw YAML so we keep the user's field form (we want to
        # report what the user actually wrote, not the merged view).
        with open(user_path, "r") as fh:
            user_cfg = yaml.safe_load(fh) or {}
    except (ConfigError, OSError, yaml.YAMLError) as exc:
        console.error(f"Error: could not load active config: {exc}")
        return 2
    if not isinstance(user_cfg, dict):
        console.error("Error: active config is not a YAML mapping.")
        return 2

    report = _diff_configs(bundled_cfg, user_cfg)
    if args.as_json:
        return _emit_doctor_json(
            report,
            active_path=user_path,
            bundled_path=bundled_path,
            strict=getattr(args, "strict", False),
        )
    return _emit_doctor_human(
        report,
        console=console,
        active_path=user_path,
        bundled_path=bundled_path,
        strict=getattr(args, "strict", False),
        quiet=getattr(args, "quiet", False),
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``xmatch`` console script and ``python -m xmatch``.

    Dispatch rules (in order):

    * ``xmatch --help`` / ``xmatch -h``  → top-level help blurb showing the
      subcommands and a few examples (via :func:`_build_top_parser`).
    * ``xmatch --version``            → handled natively by every parser
      via ``argparse``'s ``action="version"`` (prints version, exits 0).
    * ``xmatch <subcommand> …``      → dedicated subcommand parser, with
      grouped options and colour-aware output.
    * Anything else                    → legacy flat-form parser.  This is
      how ``xmatch cat1 cat2``, ``xmatch --list``, ``xmatch --describe NAME``,
      and every historic invocation keep working unchanged.
    """
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    # Friendlier top-level help: when the user asks for ``-h``/``--help``
    # without a subcommand keyword (regardless of where in argv it sits),
    # route to the slim subcommand-blurb.  ``xmatch list --help`` and
    # ``xmatch -h list`` still dispatch to the per-subcommand argparse
    # help, which the subparser itself owns — fall through to the usual
    # dispatch so the subparser receives ``["--help"]`` (or ``["-h"]``)
    # and argparse handles everything natively.
    if any(t in ("-h", "--help") for t in argv_list):
        cmd_peek, _ = _split_subcommand(argv_list)
        if cmd_peek is None:
            parser = _build_top_parser()
            try:
                parser.parse_args(argv_list)
            except SystemExit:
                raise
            return 0

    cmd, remaining = _split_subcommand(argv_list)

    if cmd == "match":
        return _run_match_subcommand(remaining)
    if cmd == "sync":
        return _run_sync_subcommand(remaining)
    if cmd == "describe":
        return _run_simple_describe(remaining)
    if cmd == "adopt":
        return _run_adopt_subcommand(remaining)
    if cmd == "completion":
        return _run_completion(remaining)
    if cmd == "doctor":
        return _run_doctor(remaining)
    if cmd in ("list", "discover", "search"):
        return _run_no_pos_subcommand(remaining, cmd)

    # Fallback: legacy flat form (--list, --describe NAME, positional
    # catalogues, etc.).
    return _run_legacy(argv_list)


if __name__ == "__main__":
    sys.exit(main())
