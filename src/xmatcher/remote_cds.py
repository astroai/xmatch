"""CDS backends: VizieR download and the CDS XMatch service (local vs remote)."""

import logging
import math
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

    if not src.access_identifier:
        raise CrossMatchError(f"Missing access_identifier for CDS catalogue '{src.name}'.")
    if ra is None or dec is None or radius_arcsec is None:
        raise CrossMatchError(
            f"A spatial region (ra/dec/radius) is required to download '{src.name}' from CDS."
        )
    try:
        ra, dec, radius_arcsec = float(ra), float(dec), float(radius_arcsec)
    except (TypeError, ValueError) as exc:
        raise CrossMatchError("CDS cone search coordinates and radius must be numeric.") from exc
    if (
        not math.isfinite(ra)
        or not math.isfinite(dec)
        or not math.isfinite(radius_arcsec)
        or not -90.0 <= dec <= 90.0
        or not 0.0 <= radius_arcsec <= 648_000.0
    ):
        raise CrossMatchError(
            "CDS cone search requires finite RA, Dec in [-90, 90], "
            "and radius_arcsec in [0, 648000]."
        )

    try:
        from astroquery.vizier import Vizier
    except ImportError as exc:
        raise CrossMatchError(
            "CDS/VizieR access requires the optional 'astroquery' package. "
            "Install it with `pip install 'xmatcher[cds]'` or `pip install astroquery`."
        ) from exc

    vizier = Vizier(columns=list(columns or src.default_columns or ["*"]), row_limit=-1)
    center = SkyCoord(ra=ra % 360.0 * u.deg, dec=dec * u.deg, frame="icrs")
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
    if not remote_src.access_identifier:
        raise CrossMatchError(f"Missing access_identifier for CDS catalogue '{remote_src.name}'.")
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
        raise CrossMatchError("CDS XMatch cannot honor " + ", ".join(unsupported))

    import astropy.units as u

    try:
        from astroquery.xmatch import XMatch
    except ImportError as exc:
        raise CrossMatchError(
            "CDS XMatch access requires the optional 'astroquery' package. "
            "Install it with `pip install 'xmatcher[cds]'` or `pip install astroquery`."
        ) from exc

    from .io_utils import polars_to_astropy

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
        raise CrossMatchError(
            "CDS XMatch result lacks its local-row identifier; cannot rejoin rows."
        )

    # Join remote match columns back onto the full local rows via the surrogate id.
    matched = matched.with_columns(pl.col(_XMATCH_KEY).cast(local[_XMATCH_KEY].dtype))
    remote_only = matched.drop([c for c in (ra_col, dec_col) if c in matched.columns])
    joined = local.join(remote_only, on=_XMATCH_KEY, how="inner", suffix="_remote")
    return joined.drop(_XMATCH_KEY)
