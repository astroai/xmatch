"""TAP backends: cone-search download and same-service ADQL self-join."""

import logging
from typing import Any, Callable, List, Optional

import polars as pl

from .exceptions import CrossMatchError
from .io_utils import astropy_table_to_polars
from .matchers import MatchSpec
from .sources import CatalogueSource
from .tap import execute_tap_query, get_tap_service

logger = logging.getLogger(__name__)


def _quote_id(name: str) -> str:
    """Quote an ADQL identifier (double-quoted per VO spec)."""
    return '"' + name.replace('"', '""') + '"'


def _select_columns(src: CatalogueSource, requested: Optional[List[str]]) -> str:
    cols = set(requested or src.default_columns or [])
    for essential in (src.ra_column, src.dec_column, src.id_column):
        if essential:
            cols.add(essential)
    return ", ".join(sorted(_quote_id(c) for c in cols)) if cols else "*"


def download_from_tap(
    src: CatalogueSource,
    *,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    radius_deg: Optional[float] = None,
    columns: Optional[List[str]] = None,
    auth_session: Optional[Any] = None,
    maxrec: Optional[int] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pl.DataFrame:
    """Download a (optionally cone-limited) catalogue from a TAP service.

    ``progress_cb`` (when supplied) is forwarded to
    :func:`xmatch.tap.execute_tap_query` so the CLI can surface
    ``"submitting"`` / ``"phase: queued"`` / ``"fetching N rows"`` updates
    as the async TAP job progresses.
    """
    if not src.tap_url or not src.access_identifier:
        raise CrossMatchError(f"Missing tap_url/table for '{src.name}'.")

    # All catalog-derived identifiers are double-quoted so columns whose names
    # collide with ADQL/SQL reserved words (or contain spaces) don't break the
    # query. Numeric values are floats; safe to interpolate.
    ra_q = _quote_id(src.ra_column)
    dec_q = _quote_id(src.dec_column)
    table_q = _quote_id(src.access_identifier)
    select = _select_columns(src, columns)
    query = f"SELECT {select} FROM {table_q} AS t"
    if ra is not None and dec is not None and radius_deg is not None:
        query += (
            f" WHERE 1=CONTAINS(POINT('ICRS', t.{ra_q}, t.{dec_q}),"
            f" CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))"
        )
    service = get_tap_service(src.tap_url, auth_session=auth_session)
    if progress_cb is not None:
        progress_cb(f"connecting to {src.tap_url}")
    table = execute_tap_query(service, query, maxrec=maxrec, progress_cb=progress_cb)
    logger.info("Downloaded %d rows from %s.", len(table), src.name)
    return astropy_table_to_polars(table)


def tap_self_join(
    src1: CatalogueSource,
    src2: CatalogueSource,
    spec: MatchSpec,
    *,
    auth_session: Optional[Any] = None,
    maxrec: Optional[int] = None,
) -> pl.DataFrame:
    """Spatial crossmatch of two tables on the *same* TAP service via ADQL."""
    if src1.tap_url != src2.tap_url:
        raise CrossMatchError("tap_self_join requires both catalogues on the same TAP service.")

    radius_deg = spec.radius_arcsec / 3600.0
    list1 = src1.default_columns or [c for c in (src1.ra_column, src1.dec_column) if c]
    list2 = src2.default_columns or [c for c in (src2.ra_column, src2.dec_column) if c]
    a_ra_q = _quote_id(src1.ra_column)
    a_dec_q = _quote_id(src1.dec_column)
    b_ra_q = _quote_id(src2.ra_column)
    b_dec_q = _quote_id(src2.dec_column)
    a_table_q = _quote_id(src1.access_identifier)
    b_table_q = _quote_id(src2.access_identifier)
    cols1 = ", ".join(f"a.{_quote_id(c)} AS a_{c}" for c in list1)
    cols2 = ", ".join(f"b.{_quote_id(c)} AS b_{c}" for c in list2)
    query = (
        f"SELECT {cols1}, {cols2}, "
        f"DISTANCE(POINT('ICRS', a.{a_ra_q}, a.{a_dec_q}),"
        f" POINT('ICRS', b.{b_ra_q}, b.{b_dec_q}))*3600 AS {spec.find}_sep_arcsec "
        f"FROM {a_table_q} AS a "
        f"JOIN {b_table_q} AS b "
        f"ON 1=CONTAINS(POINT('ICRS', a.{a_ra_q}, a.{a_dec_q}),"
        f" CIRCLE('ICRS', b.{b_ra_q}, b.{b_dec_q}, {radius_deg}))"
    )
    service = get_tap_service(src1.tap_url, auth_session=auth_session)
    table = execute_tap_query(service, query, maxrec=maxrec)
    logger.info("TAP self-join returned %d rows.", len(table))
    return astropy_table_to_polars(table)
