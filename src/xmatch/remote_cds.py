"""CDS backends: VizieR download and the CDS XMatch service (local vs remote)."""

import logging
from collections.abc import Callable
from typing import Any

import polars as pl

from .exceptions import CrossMatchError
from .io_utils import astropy_table_to_polars
from .matchers import MatchSpec
from .sources import CatalogueSource

logger = logging.getLogger(__name__)

_XMATCH_KEY = "__xmatch_row_id__"


def download_from_cds(
    src: CatalogueSource,
    *,
    ra: float | None = None,
    dec: float | None = None,
    radius_arcsec: float | None = None,
    columns: list[str] | None = None,
    progress_cb: Callable[[str], None] | None = None,
    **_: Any,
) -> pl.DataFrame:
    """Download a VizieR catalogue (cone-limited if a region is given).

    ``progress_cb`` (when supplied) is invoked with short status strings
    before (``"querying VizieR"``) and after (``"received N tables"`` /
    ``"converting to polars"``) the synchronous astroquery call so the
    CLI spinner can confirm the connection is alive.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    try:
        from astroquery.vizier import Vizier
    except ImportError as exc:
        raise CrossMatchError(
            "CDS/VizieR access requires the optional 'astroquery' package. "
            "Install it with `pip install 'xmatch[cds]'` or `pip install astroquery`."
        ) from exc

    if not src.access_identifier:
        raise CrossMatchError(f"Missing access_identifier for CDS catalogue '{src.name}'.")

    vizier = Vizier(columns=list(columns or src.default_columns or ["*"]), row_limit=-1)
    if ra is None or dec is None or not radius_arcsec:
        raise CrossMatchError(
            f"A spatial region (ra/dec/radius) is required to download '{src.name}' from CDS."
        )
    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    if progress_cb is not None:
        progress_cb("querying VizieR")
    tables = vizier.query_region(
        center, radius=radius_arcsec * u.arcsec, catalog=src.access_identifier
    )
    if not tables:
        if progress_cb is not None:
            progress_cb("no tables returned")
        return pl.DataFrame()
    if progress_cb is not None:
        progress_cb(f"received {len(tables)} table(s); converting to polars")
    return astropy_table_to_polars(tables[0])


def cds_xmatch_local_remote(
    local_src: CatalogueSource,
    remote_src: CatalogueSource,
    local_lf: pl.LazyFrame,
    spec: MatchSpec,
) -> pl.DataFrame:
    """Crossmatch a local catalogue against a VizieR table via the CDS XMatch service.

    A surrogate row id is attached before the upload so the service results can be
    reliably joined back to the original local rows (no fragile float-equality
    merge on RA/Dec).
    """
    import astropy.units as u

    try:
        from astroquery.xmatch import XMatch
    except ImportError as exc:
        raise CrossMatchError(
            "CDS XMatch access requires the optional 'astroquery' package. "
            "Install it with `pip install 'xmatch[cds]'` or `pip install astroquery`."
        ) from exc

    from .io_utils import polars_to_astropy

    if not remote_src.access_identifier:
        raise CrossMatchError(f"Missing access_identifier for CDS catalogue '{remote_src.name}'.")

    local = local_lf.collect().with_row_index(_XMATCH_KEY)
    ra_col, dec_col = local_src.ra_column, local_src.dec_column
    if ra_col not in local.columns or dec_col not in local.columns:
        raise CrossMatchError(f"RA/Dec columns '{ra_col}'/'{dec_col}' not in local catalogue.")

    upload = polars_to_astropy(local.select([_XMATCH_KEY, ra_col, dec_col]))

    result_table = XMatch().query(
        cat1=upload,
        cat2=f"vizier:{remote_src.access_identifier}",
        max_distance=spec.radius_arcsec * u.arcsec,
        colRA1=ra_col,
        colDec1=dec_col,
    )
    if result_table is None or len(result_table) == 0:
        return local.drop(_XMATCH_KEY).clear()

    matched = astropy_table_to_polars(result_table)
    if _XMATCH_KEY not in matched.columns:
        logger.warning("CDS XMatch result lacks surrogate id; returning raw service output.")
        return matched

    # Join remote match columns back onto the full local rows via the surrogate id.
    matched = matched.with_columns(pl.col(_XMATCH_KEY).cast(local[_XMATCH_KEY].dtype))
    remote_only = matched.drop([c for c in (ra_col, dec_col) if c in matched.columns])
    joined = local.join(remote_only, on=_XMATCH_KEY, how="inner", suffix="_remote")
    return joined.drop(_XMATCH_KEY)
