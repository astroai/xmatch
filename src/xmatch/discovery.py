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
from typing import Any, Dict, Optional, Tuple

import polars as pl

from .exceptions import TapError
from .tap import execute_tap_query, get_tap_service

logger = logging.getLogger(__name__)

# Known TAP endpoints that we can query without authentication.
_PUBLIC_TAP_ENDPOINTS: Dict[str, Dict[str, Any]] = {
    "vizier": {
        "url": "http://tapvizier.u-strasbg.fr/TAPVizieR/tap",
        "description": "CDS VizieR TAP service",
    },
    "gaia": {
        "url": "https://gea.esac.esa.int/tap-server/tap",
        "description": "ESA Gaia Archive TAP service",
    },
    "noirlab": {
        "url": "https://datalab.noirlab.edu/tap",
        "description": "NOIRLab Astro Data Lab TAP service",
    },
    "cadc": {
        "url": "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/tap",
        "description": "Canadian Astronomy Data Centre TAP service",
    },
}


def get_public_endpoints() -> Dict[str, Dict[str, Any]]:
    """Return known public TAP endpoints."""
    return dict(_PUBLIC_TAP_ENDPOINTS)


def discover_tables(
    tap_url: str,
    *,
    schema_filter: Optional[str] = None,
    name_filter: Optional[str] = None,
    auth_session: Optional[Any] = None,
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


def discover_columns(
    tap_url: str,
    table_name: str,
    *,
    schema_name: Optional[str] = None,
    auth_session: Optional[Any] = None,
) -> pl.DataFrame:
    """Query ``TAP_SCHEMA.columns`` for a specific table.

    Parameters
    ----------
    tap_url:
        TAP service URL.
    table_name:
        Exact table name.
    schema_name:
        Optional schema name qualifier.
    auth_session:
        Authenticated requests session.

    Returns
    -------
    polars.DataFrame
        Columns: ``column_name``, ``datatype``, ``ucd``, ``unit``, ``description``.
    """
    query = (
        "SELECT column_name, datatype, ucd, unit, description "
        "FROM TAP_SCHEMA.columns "
        f"WHERE table_name = '{table_name.replace(chr(39), chr(39) + chr(39))}'"
    )
    if schema_name:
        safe = schema_name.replace("'", "''")
        query += f" AND schema_name = '{safe}'"
    query += " ORDER BY column_name"

    service = get_tap_service(tap_url, auth_session=auth_session)
    table = execute_tap_query(service, query)
    from .io_utils import astropy_table_to_polars

    return astropy_table_to_polars(table)


def detect_radec_columns(cols_df: pl.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Heuristic detection of RA/Dec columns from TAP_SCHEMA columns result.

    Looks at ``column_name`` and ``ucd`` fields to guess which columns hold
    right ascension and declination.

    Returns
    -------
    (ra_col, dec_col) or (None, None).
    """
    from .astro_utils import find_coord_columns

    col_names = cols_df["column_name"].to_list()
    return find_coord_columns(col_names)


def discover_archive(
    tap_url: str,
    auth_session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Get a summary of a TAP service: tables + top-level metadata.

    Returns a dict with ``tables`` (DataFrame) and ``tables_count`` (int).
    Raises ``TapError`` if the service is unreachable.
    """
    try:
        tables = discover_tables(tap_url, auth_session=auth_session)
    except Exception as exc:
        raise TapError(f"Failed to discover tables at {tap_url}: {exc}") from exc
    return {
        "tables": tables,
        "tables_count": tables.height,
    }


def get_table_schema(
    tap_url: str,
    table_name: str,
    auth_session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Get full column metadata for a table.

    Returns a dict with ``columns`` (DataFrame), ``columns_count`` (int),
    ``ra_column``, ``dec_column`` (guessed).
    """
    cols = discover_columns(tap_url, table_name, auth_session=auth_session)
    ra, dec = detect_radec_columns(cols)
    return {
        "columns": cols,
        "columns_count": cols.height,
        "ra_column": ra,
        "dec_column": dec,
        "columns_list": cols["column_name"].to_list(),
    }
