"""TAP backends: cone-search download and same-service ADQL self-join."""

import logging
import math
from collections.abc import Callable
from typing import Any

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


def _select_columns(src: CatalogueSource, requested: list[str] | None) -> str:
    if requested is None and src.default_columns is None:
        return "*"
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
    ra = ra % 360.0
    if not box:
        return (
            f" WHERE 1=CONTAINS(POINT('ICRS', t.{ra_q}, t.{dec_q}),"
            f" CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))"
        )

    dec_lo = max(-90.0, dec - radius_deg)
    dec_hi = min(90.0, dec + radius_deg)
    dec_predicate = f"t.{dec_q} BETWEEN {dec_lo} AND {dec_hi}"
    # A cap reaching either pole spans every RA. Otherwise the spherical
    # tangent-meridian bound is asin(sin(radius) / cos(dec_center)); the
    # tangent-plane radius/cos(dec) approximation can under-fetch.
    if abs(dec) + radius_deg >= 90.0:
        return f" WHERE {dec_predicate}"
    ratio = math.sin(math.radians(radius_deg)) / math.cos(math.radians(dec))
    dra = math.degrees(math.asin(min(1.0, max(-1.0, ratio))))
    ra_lo, ra_hi = ra - dra, ra + dra
    if ra_lo < 0.0:
        ra_predicate = f"(t.{ra_q} >= {ra_lo % 360.0} OR t.{ra_q} <= {ra_hi})"
    elif ra_hi > 360.0:
        ra_predicate = f"(t.{ra_q} >= {ra_lo} OR t.{ra_q} <= {ra_hi % 360.0})"
    else:
        ra_predicate = f"t.{ra_q} BETWEEN {ra_lo} AND {ra_hi}"
    return f" WHERE {ra_predicate} AND {dec_predicate}"


def download_from_tap(
    src: CatalogueSource,
    *,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    columns: list[str] | None = None,
    auth_session: Any | None = None,
    maxrec: int | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> pl.DataFrame:
    """Download a (optionally cone-limited) catalogue from a TAP service.

    ``progress_cb`` (when supplied) is forwarded to
    :func:`xmatch.tap.execute_tap_query` so the CLI can surface
    ``"submitting"`` / ``"phase: queued"`` / ``"fetching N rows"`` updates
    as the async TAP job progresses.
    """
    if not src.tap_url or not src.access_identifier:
        raise CrossMatchError(f"Missing tap_url/table for '{src.name}'.")

    region = (ra, dec, radius_deg)
    if any(value is not None for value in region) and not all(
        value is not None for value in region
    ):
        raise CrossMatchError("TAP cone search requires ra, dec, and radius_deg together.")
    if all(value is not None for value in region):
        try:
            ra, dec, radius_deg = (float(value) for value in region)
        except (TypeError, ValueError) as exc:
            raise CrossMatchError(
                "TAP cone search coordinates and radius must be numeric."
            ) from exc
        if (
            not math.isfinite(ra)
            or not math.isfinite(dec)
            or not math.isfinite(radius_deg)
            or not -90.0 <= dec <= 90.0
            or not 0.0 <= radius_deg <= 180.0
        ):
            raise CrossMatchError(
                "TAP cone search requires finite RA, Dec in [-90, 90], and radius_deg in [0, 180]."
            )

    # All catalog-derived identifiers are double-quoted so columns whose names
    # collide with ADQL/SQL reserved words (or contain spaces) don't break the
    # query. Numeric values are floats; safe to interpolate.
    ra_q = _quote_id(src.ra_column)
    dec_q = _quote_id(src.dec_column)
    table_q = _table_ref(src.access_identifier, tap_url=src.tap_url)
    select = _select_columns(src, columns)
    query = f"SELECT {select} FROM {table_q} AS t"
    if ra is not None:
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
    auth_session: Any | None = None,
    maxrec: int | None = None,
) -> pl.DataFrame:
    """Spatial crossmatch of two tables on the *same* TAP service via ADQL."""
    if not src1.tap_url or not src2.tap_url:
        raise CrossMatchError("tap_self_join requires both catalogues to declare a TAP URL.")
    if src1.tap_url != src2.tap_url:
        raise CrossMatchError("tap_self_join requires both catalogues on the same TAP service.")
    if not src1.access_identifier or not src2.access_identifier:
        raise CrossMatchError(
            "tap_self_join requires both catalogues to declare a table identifier."
        )
    if not all((src1.ra_column, src1.dec_column, src2.ra_column, src2.dec_column)):
        raise CrossMatchError("tap_self_join requires RA and Dec columns on both catalogues.")
    unsupported = []
    if spec.matcher != "sky":
        unsupported.append(f"matcher={spec.matcher!r}")
    if spec.find != "all":
        unsupported.append("find='best'")
    if spec.join_type != "1and2":
        unsupported.append(f"join_type={spec.join_type!r}")
    if spec.probabilistic:
        unsupported.append("probabilistic")
    if spec.prior_columns:
        unsupported.append("prior_columns")
    if spec.extra_distance_cols:
        unsupported.append("extra_distance_cols")
    if spec.filter_expr:
        unsupported.append("filter_expr")
    if spec.target_epoch is not None or spec.pm_prior:
        unsupported.append("proper-motion options")
    if unsupported:
        raise CrossMatchError("tap_self_join cannot honor " + ", ".join(unsupported))

    radius_deg = spec.radius_arcsec / 3600.0
    list1 = src1.default_columns or [c for c in (src1.ra_column, src1.dec_column) if c]
    list2 = src2.default_columns or [c for c in (src2.ra_column, src2.dec_column) if c]
    a_ra_q = _quote_id(src1.ra_column)
    a_dec_q = _quote_id(src1.dec_column)
    b_ra_q = _quote_id(src2.ra_column)
    b_dec_q = _quote_id(src2.dec_column)
    a_table_q = _table_ref(src1.access_identifier, tap_url=src1.tap_url)
    b_table_q = _table_ref(src2.access_identifier, tap_url=src2.tap_url)
    cols1 = ", ".join(f"a.{_quote_id(c)} AS {_quote_id(f'a_{c}')}" for c in list1)
    cols2 = ", ".join(f"b.{_quote_id(c)} AS {_quote_id(f'b_{c}')}" for c in list2)
    sep_col_q = _quote_id(f"{spec.find}_sep_arcsec")
    query = (
        f"SELECT {cols1}, {cols2}, "
        f"DISTANCE(POINT('ICRS', a.{a_ra_q}, a.{a_dec_q}),"
        f" POINT('ICRS', b.{b_ra_q}, b.{b_dec_q}))*3600 AS {sep_col_q} "
        f"FROM {a_table_q} AS a JOIN {b_table_q} AS b "
        f"ON 1=CONTAINS(POINT('ICRS', a.{a_ra_q}, a.{a_dec_q}),"
        f" CIRCLE('ICRS', b.{b_ra_q}, b.{b_dec_q}, {radius_deg}))"
    )
    service = get_tap_service(src1.tap_url, auth_session=auth_session)
    table = execute_tap_query(service, query, maxrec=maxrec)
    logger.info("TAP self-join returned %d rows.", len(table))
    return astropy_table_to_polars(table)
