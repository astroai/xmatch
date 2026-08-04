"""TAP backends: cone-search download and same-service ADQL self-join."""

import logging
import math
from typing import Any, Callable, List, Optional

import polars as pl

from .exceptions import CrossMatchError, TapError
from .io_utils import astropy_table_to_polars
from .matchers import MatchSpec
from .sources import CatalogueSource
from .tap import execute_tap_query, get_tap_service

logger = logging.getLogger(__name__)


def _quote_id(name: str) -> str:
    """Quote an ADQL identifier (double-quoted per VO spec)."""
    # If the name contains a dot, it might be a schema-qualified table name (e.g. gaiadr3.gaia_source).
    # We should quote the parts separately so the database parser doesn't treat the dot as part of the name.
    return ".".join('"' + part.replace('"', '""') + '"' for part in name.split("."))


def _select_columns(src: CatalogueSource, requested: Optional[List[str]]) -> str:
    cols = set(requested or src.default_columns or [])
    for essential in (src.ra_column, src.dec_column, src.id_column):
        if essential:
            cols.add(essential)
    return ", ".join(sorted(_quote_id(c) for c in cols)) if cols else "*"


def _uses_box_cone(tap_url: str) -> bool:
    """Data Lab rejects ADQL CONTAINS/CIRCLE on many tables; use a RA/Dec box."""
    return "datalab.noirlab.edu" in (tap_url or "").lower()


def _table_ref(access_identifier: str, *, tap_url: str) -> str:
    """Format a FROM target for *tap_url*.

    NOIRLab Data Lab rejects quoted ``"schema"."table"`` (TAP job ERROR);
    VizieR needs quoting for slash ids like ``II/349/ps1``.
    """
    bare = access_identifier.strip().strip('"')
    if _uses_box_cone(tap_url):
        return bare
    return _quote_id(bare)


def _cone_predicate(
    ra_q: str, dec_q: str, ra: float, dec: float, radius_deg: float, *, box: bool
) -> str:
    if not box:
        return (
            f" WHERE 1=CONTAINS(POINT('ICRS', t.{ra_q}, t.{dec_q}),"
            f" CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))"
        )
    # Over-fetch slightly with a tangent-plane box; client match radius stays exact.
    cos_dec = max(abs(math.cos(math.radians(dec))), 0.1)
    dra = radius_deg / cos_dec
    ra_lo, ra_hi = ra - dra, ra + dra
    dec_lo, dec_hi = dec - radius_deg, dec + radius_deg
    # RA wraps at 0/360: split into two intervals instead of a single BETWEEN.
    if ra_lo < 0.0 or ra_hi > 360.0:
        ra_lo_m = ra_lo % 360.0
        ra_hi_m = ra_hi % 360.0
        return (
            f" WHERE ((t.{ra_q} >= {ra_lo_m}) OR (t.{ra_q} <= {ra_hi_m}))"
            f" AND t.{dec_q} BETWEEN {dec_lo} AND {dec_hi}"
        )
    return (
        f" WHERE t.{ra_q} BETWEEN {ra_lo} AND {ra_hi} AND t.{dec_q} BETWEEN {dec_lo} AND {dec_hi}"
    )


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
    table_q = _table_ref(src.access_identifier, tap_url=src.tap_url)
    select = _select_columns(src, columns)
    query = f"SELECT {select} FROM {table_q} AS t"
    if ra is not None and dec is not None and radius_deg is not None:
        query += _cone_predicate(
            ra_q,
            dec_q,
            ra,
            dec,
            radius_deg,
            box=_uses_box_cone(src.tap_url),
        )
    service = get_tap_service(src.tap_url, auth_session=auth_session)
    if progress_cb is not None:
        progress_cb(f"connecting to {src.tap_url}")
    try:
        table = execute_tap_query(service, query, maxrec=maxrec, progress_cb=progress_cb)
    except TapError as exc:
        hint = ""
        if ra is not None and not _uses_box_cone(src.tap_url):
            hint = (
                " Tip: pass an explicit --ra/--dec/--radius-deg region, or try a "
                "local file cone download first."
            )
        raise TapError(f"{exc}{hint}") from exc
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
    a_table_q = _table_ref(src1.access_identifier, tap_url=src1.tap_url)
    b_table_q = _table_ref(src2.access_identifier, tap_url=src2.tap_url)
    cols1 = ", ".join(f"a.{_quote_id(c)} AS a_{c}" for c in list1)
    cols2 = ", ".join(f"b.{_quote_id(c)} AS b_{c}" for c in list2)
    query = (
        f"SELECT {cols1}, {cols2}, "
        f"DISTANCE(POINT('ICRS', a.{a_ra_q}, a.{a_dec_q}),"
        f" POINT('ICRS', b.{b_ra_q}, b.{b_dec_q}))*3600 AS {spec.find}_sep_arcsec "
        f"FROM {a_table_q} AS a JOIN {b_table_q} AS b "
        f"ON 1=CONTAINS(POINT('ICRS', a.{a_ra_q}, a.{a_dec_q}),"
        f" CIRCLE('ICRS', b.{b_ra_q}, b.{b_dec_q}, {radius_deg}))"
    )
    service = get_tap_service(src1.tap_url, auth_session=auth_session)
    table = execute_tap_query(service, query, maxrec=maxrec)
    logger.info("TAP self-join returned %d rows.", len(table))
    return astropy_table_to_polars(table)
