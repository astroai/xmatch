"""Catalogue discovery via TAP_SCHEMA introspection (à la TOPCAT).

Queries remote TAP services for their available tables and column metadata.
The functions here are non-destructive (read-only ADQL queries) and are
safe to run against any TAP endpoint that supports ``TAP_SCHEMA``.

Modeled after TOPCAT's "TAP Table Browser" workflow:
1. Query ``TAP_SCHEMA.tables`` to discover available tables and their descriptions.
2. Query ``TAP_SCHEMA.columns`` to get per-table column names, types, and UCDs.

Results are returned as polars DataFrames for filtering/display.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import polars as pl

from .exceptions import InputError
from .tap import execute_tap_query, get_tap_service

logger = logging.getLogger(__name__)

# Known TAP endpoints that we can query without authentication.
_PUBLIC_TAP_ENDPOINTS: dict[str, dict[str, Any]] = {
    "vizier": {
        "url": "http://tapvizier.u-strasbg.fr/TAPVizieR/tap",
        "description": "CDS VizieR TAP service",
        "archive": "cds",
        "service_id": "tap_service",
    },
    "gaia": {
        "url": "https://gea.esac.esa.int/tap-server/tap",
        "description": "ESA Gaia Archive TAP service",
        "archive": "esa_gaia",
        "service_id": "tap_service",
    },
    "noirlab": {
        "url": "https://datalab.noirlab.edu/tap",
        "description": "NOIRLab Astro Data Lab TAP service",
        "archive": "noao_datalab",
        "service_id": "tap_service",
    },
    "cadc": {
        "url": "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/tap",
        "description": "Canadian Astronomy Data Centre TAP service",
        # Not in bundled archives yet — ad-hoc still works via tap_url override.
        "archive": None,
        "service_id": "tap_service",
    },
}

# VizieR-style catalogue ids: I/355/gaiadr3, II/349/ps1, J/A+A/588/A103/cat2rxs, VIII/65/nvss
# Require a catalogue-class prefix so paths like data/foo are not treated as TAP ids.
_VIZIER_TABLE_RE = re.compile(r"^(?:[IVX]+|J)/[A-Za-z0-9.+/_-]+$")
# Data Lab / ESA style: schema.table (single dot, no slash, no path separators)
_SCHEMA_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")

_ID_NAME_HINTS = (
    "source_id",
    "objid",
    "object_id",
    "obj_id",
    "ls_id",
    "targetid",
    "designation",
    "allwise",
    "sourceid",
    "coadd_object_id",
)


def get_public_endpoints() -> dict[str, dict[str, Any]]:
    """Return known public TAP endpoints."""
    return dict(_PUBLIC_TAP_ENDPOINTS)


def discover_tables(
    tap_url: str,
    *,
    schema_filter: str | None = None,
    name_filter: str | None = None,
    auth_session: Any | None = None,
    timeout_seconds: float = 30.0,
) -> pl.DataFrame:
    """Query ``TAP_SCHEMA.tables`` and return a polars DataFrame.

    Parameters
    ----------
    tap_url:
        TAP service URL.
    schema_filter:
        Optional SQL LIKE pattern for ``schema_name`` (e.g. ``'gaia%'``).
    name_filter:
        Optional SQL LIKE pattern for ``table_name`` (e.g. ``'%source%'``).
    auth_session:
        Authenticated requests session (for private TAP services).
    timeout_seconds:
        How long to wait for the TAP query.

    Returns
    -------
    polars.DataFrame
        Columns: ``schema_name``, ``table_name``, ``description``, ``table_type``.
    """
    query = "SELECT schema_name, table_name, description, table_type FROM TAP_SCHEMA.tables"
    conditions = []
    if schema_filter:
        safe = schema_filter.replace("'", "''")
        conditions.append(f"schema_name LIKE '{safe}'")
    if name_filter:
        safe = name_filter.replace("'", "''")
        conditions.append(f"table_name LIKE '{safe}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY schema_name, table_name"

    service = get_tap_service(tap_url, auth_session=auth_session)
    table = execute_tap_query(service, query)
    from .io_utils import astropy_table_to_polars

    return astropy_table_to_polars(table)


def split_table_id(table_id: str) -> tuple[str | None, str]:
    """Split a table id into ``(schema_name, table_name)`` for TAP_SCHEMA queries.

    * Data Lab / ESA: ``ls_dr10.tractor`` → ``('ls_dr10', 'tractor')``
    * VizieR: ``II/349/ps1`` → ``(None, 'II/349/ps1')`` (single identifier)
    """
    text = table_id.strip().strip('"')
    if "/" in text:
        return None, text
    if "." in text:
        schema, _, name = text.partition(".")
        if schema and name and "." not in name:
            return schema, name
    return None, text


def looks_like_table_id(value: str) -> bool:
    """True when *value* looks like a remote TAP/VizieR table id, not a local name."""
    text = value.strip().strip('"')
    if not text or " " in text or text.startswith(".") or text.startswith("/"):
        return False
    # Local data files must not be treated as TAP ids (e.g. sources.csv).
    lower = text.lower()
    if lower.endswith((".csv", ".parquet", ".fits", ".fit", ".tsv", ".txt", ".hats")):
        return False
    if _VIZIER_TABLE_RE.match(text):
        return True
    return bool(_SCHEMA_TABLE_RE.match(text))


def guess_endpoint(table_id: str) -> str | None:
    """Guess the public endpoint short-name for a table id."""
    text = table_id.strip().strip('"')
    if "/" in text:
        return "vizier"
    if _SCHEMA_TABLE_RE.match(text):
        # ESA Gaia uses gaiadr3.* ; everything else with schema.table → noirlab.
        if text.lower().startswith("gaiadr"):
            return "gaia"
        return "noirlab"
    return None


def endpoint_archive(endpoint: str) -> tuple[str | None, str, str]:
    """Return ``(archive_name, service_id, tap_url)`` for a public endpoint."""
    key = endpoint.lower().strip()
    info = _PUBLIC_TAP_ENDPOINTS.get(key)
    if info is None:
        raise InputError(f"Unknown TAP endpoint '{endpoint}'.")
    return info.get("archive"), info.get("service_id", "tap_service"), info["url"]


def discover_columns(
    tap_url: str,
    table_name: str,
    *,
    schema_name: str | None = None,
    auth_session: Any | None = None,
) -> pl.DataFrame:
    """Query ``TAP_SCHEMA.columns`` for a specific table.

    Parameters
    ----------
    tap_url:
        TAP service URL.
    table_name:
        Exact table name, or ``schema.table`` (schema is split automatically
        when *schema_name* is omitted).
    schema_name:
        Optional schema name qualifier.
    auth_session:
        Authenticated requests session.

    Returns
    -------
    polars.DataFrame
        Columns: ``column_name``, ``datatype``, ``ucd``, ``unit``, ``description``.
    """
    if schema_name is None:
        schema_name, table_name = split_table_id(table_name)
    safe_table = table_name.replace("'", "''")
    query = (
        "SELECT column_name, datatype, ucd, unit, description "
        "FROM TAP_SCHEMA.columns "
        f"WHERE table_name = '{safe_table}'"
    )
    if schema_name:
        safe = schema_name.replace("'", "''")
        query += f" AND schema_name = '{safe}'"
    query += " ORDER BY column_name"

    service = get_tap_service(tap_url, auth_session=auth_session)
    table = execute_tap_query(service, query)
    from .io_utils import astropy_table_to_polars

    df = astropy_table_to_polars(table)
    if df.height == 0 and not safe_table.startswith('"') and not safe_table.endswith('"'):
        # VizieR TAP_SCHEMA stores slash-delimited table names with literal double quotes: "II/349/ps1"
        q_quoted = (
            "SELECT column_name, datatype, ucd, unit, description "
            "FROM TAP_SCHEMA.columns "
            f"WHERE table_name = '\"{safe_table}\"'"
        )
        if schema_name:
            safe = schema_name.replace("'", "''")
            q_quoted += f" AND schema_name = '{safe}'"
        q_quoted += " ORDER BY column_name"
        try:
            t_quoted = execute_tap_query(service, q_quoted)
            df = astropy_table_to_polars(t_quoted)
        except Exception:
            pass
    if df.height > 0 and "column_name" in df.columns:
        df = df.with_columns(pl.col("column_name").str.strip_chars('"'))
    return df


def detect_radec_columns(cols_df: pl.DataFrame) -> tuple[str | None, str | None]:
    """Heuristic detection of RA/Dec columns from TAP_SCHEMA columns result.

    Looks at ``column_name`` and ``ucd`` fields to guess which columns hold
    right ascension and declination.

    Returns
    -------
    (ra_col, dec_col) or (None, None).
    """
    from .astro_utils import find_coord_columns

    if "column_name" not in cols_df.columns:
        return None, None
    col_names = cols_df["column_name"].to_list()
    ra_col, dec_col = find_coord_columns(col_names)
    if ra_col and dec_col:
        return ra_col, dec_col
    if "ucd" not in cols_df.columns:
        return ra_col, dec_col
    for name, ucd in cols_df.select("column_name", "ucd").iter_rows():
        tokens = {token.strip().lower() for token in str(ucd or "").split(";")}
        if ra_col is None and "pos.eq.ra" in tokens:
            ra_col = name
        if dec_col is None and "pos.eq.dec" in tokens:
            dec_col = name
    return ra_col, dec_col


def get_table_schema(
    tap_url: str,
    table_name: str,
    auth_session: Any | None = None,
) -> dict[str, Any]:
    """Get full column metadata for a table.

    Returns a dict with ``columns`` (DataFrame), ``columns_count`` (int),
    ``ra_column``, ``dec_column`` (guessed), ``columns_list``, and
    ``access_identifier`` (canonical FROM string).
    """
    schema_name, bare_table = split_table_id(table_name)
    cols = discover_columns(tap_url, bare_table, schema_name=schema_name, auth_session=auth_session)
    # VizieR TAP_SCHEMA often stores the bare id; if empty, retry without schema.
    if cols.height == 0 and schema_name is not None:
        cols = discover_columns(tap_url, table_name.strip().strip('"'), auth_session=auth_session)
    ra, dec = detect_radec_columns(cols)
    access = table_name.strip().strip('"')
    if schema_name and "." not in access and "/" not in access:
        access = f"{schema_name}.{bare_table}"
    return {
        "columns": cols,
        "columns_count": cols.height,
        "ra_column": ra,
        "dec_column": dec,
        "columns_list": cols["column_name"].to_list() if cols.height else [],
        "access_identifier": access,
        "schema_name": schema_name,
        "bare_table_name": bare_table,
    }


def _guess_id_column(columns: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for hint in _ID_NAME_HINTS:
        if hint in lower:
            return lower[hint]
    for c in columns:
        cl = c.lower()
        if cl.endswith("id") or cl.endswith("_id") or "objid" in cl:
            return c
    return None


def _guess_default_columns(
    columns: list[str],
    *,
    ra: str | None,
    dec: str | None,
    id_col: str | None,
    limit: int = 12,
) -> list[str]:
    """Pick a short default column list: coords + id + a few mag/flux columns."""
    chosen: list[str] = []
    for c in (id_col, ra, dec):
        if c and c not in chosen:
            chosen.append(c)
    for c in columns:
        if c in chosen:
            continue
        cl = c.lower()
        if any(k in cl for k in ("mag", "flux", "psf", "aper")) and "err" not in cl:
            chosen.append(c)
        if len(chosen) >= limit:
            break
    return chosen


def catalogue_entry_from_schema(
    table_id: str,
    schema: dict[str, Any],
    *,
    archive: str,
    service_id: str = "tap_service",
    name: str | None = None,
    description: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build ``(catalogue_name, entry_dict)`` from a :func:`get_table_schema` result."""
    access = schema.get("access_identifier") or table_id.strip().strip('"')
    ra = schema.get("ra_column")
    dec = schema.get("dec_column")
    if not ra or not dec:
        raise InputError(
            f"Could not detect RA/Dec columns for '{access}'. "
            "Pass --ra-column / --dec-column after adopting, or edit the YAML."
        )
    cols_list: list[str] = list(schema.get("columns_list") or [])
    id_col = _guess_id_column(cols_list)
    short = name or _default_catalogue_name(access)
    entry: dict[str, Any] = {
        "description": description
        or f"Ad-hoc catalogue {access} (adopted / resolved via TAP_SCHEMA)",
        "archive": archive,
        "service_id": service_id,
        "access_identifier": access,
        "table_name": access,
        "ra_column": ra,
        "dec_column": dec,
        "estimated_size": "huge",
    }
    if id_col:
        entry["id_column"] = id_col
    entry["default_columns"] = _guess_default_columns(cols_list, ra=ra, dec=dec, id_col=id_col)
    return short, entry


def _default_catalogue_name(access: str) -> str:
    """Derive a YAML-safe catalogue key from a table id."""
    text = access.strip().strip('"').lower()
    text = text.replace("/", "_").replace(".", "_").replace("+", "p").replace("-", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text or "adopted_table"
