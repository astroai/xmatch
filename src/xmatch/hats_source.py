"""Optional HATS (Hierarchical Adaptive Tiling Scheme) support via LSDB.

LSDB is an optional dependency. Install it with ``pip install lsdb`` (in an
environment where the HATS stack is available). All entry points raise a clear
error if it is missing rather than failing at import time.
"""

import logging
from typing import Optional

import polars as pl

from .exceptions import CrossMatchError
from .matchers import MatchSpec
from .sources import CatalogueSource

logger = logging.getLogger(__name__)


def lsdb_available() -> bool:
    try:
        import lsdb  # noqa: F401

        return True
    except Exception:
        return False


def _require_lsdb():
    try:
        import lsdb

        return lsdb
    except Exception as exc:  # pragma: no cover - exercised only without lsdb
        raise CrossMatchError(
            "HATS support requires the optional 'lsdb' package. Install it with "
            "`pip install lsdb` (see https://lsdb.readthedocs.io)."
        ) from exc


def read_hats(src: CatalogueSource):
    """Open a HATS catalogue with LSDB."""
    lsdb = _require_lsdb()
    path = src.access_identifier or (str(src.path) if src.path else None)
    if not path:
        raise CrossMatchError(f"No HATS path for '{src.name}'.")
    logger.info("Opening HATS catalogue: %s", path)
    return lsdb.read_hats(path)


def hats_crossmatch(
    src1: CatalogueSource,
    src2: CatalogueSource,
    spec: MatchSpec,
    *,
    local_lf1: Optional[pl.LazyFrame] = None,
    local_lf2: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    """Crossmatch two catalogues where at least one is a HATS catalogue.

    A non-HATS side is imported into an in-memory LSDB catalogue. The result is
    returned as an eager polars DataFrame.
    """
    lsdb = _require_lsdb()

    def to_catalog(src, lf):
        if src.access_method == "hats":
            return read_hats(src)
        if lf is None:
            raise CrossMatchError(f"Cannot turn '{src.name}' into a HATS catalogue without data.")
        pdf = lf.collect().to_pandas()
        return lsdb.from_dataframe(pdf, ra_column=src.ra_column, dec_column=src.dec_column)

    cat1 = to_catalog(src1, local_lf1)
    cat2 = to_catalog(src2, local_lf2)

    matched = cat1.crossmatch(cat2, radius_arcsec=spec.radius_arcsec).compute()
    return pl.from_pandas(matched.reset_index(drop=True))
