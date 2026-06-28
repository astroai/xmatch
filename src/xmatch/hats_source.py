"""Optional HATS (Hierarchical Adaptive Tiling Scheme) support via LSDB.

LSDB is an optional dependency. Install it with ``pip install lsdb`` (in an
environment where the HATS stack is available). All entry points raise a clear
error if it is missing rather than failing at import time.
"""

import logging
from typing import Optional

import polars as pl

from .exceptions import CrossMatchError
from .matchers import _RIGHT_SUFFIX, SEP_COLUMN, MatchSpec
from .sources import CatalogueSource

logger = logging.getLogger(__name__)

# Mapping from xmatch join_type to LSDB's supported join modes.
# LSDB crossmatch is inherently inner-join (matched pairs only);
# we reject join types that LSDB cannot express natively.
_LSDB_SUPPORTED_JOINS = frozenset({"1and2"})


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
    right_suffix: str = _RIGHT_SUFFIX,
) -> pl.DataFrame:
    """Crossmatch two catalogues where at least one is a HATS catalogue.

    A non-HATS side is imported into an in-memory LSDB catalogue. The result is
    returned as an eager polars DataFrame with a ``sep_arcsec`` column.

    Parameters
    ----------
    right_suffix:
        Suffix appended to right-side columns that collide with left-side
        columns (default ``"_2"``).  For multi-way chains, pass ``"_3"``,
        ``"_4"``, etc.
    """
    lsdb = _require_lsdb()

    # Validate join type early — LSDB only supports inner joins natively.
    if spec.join_type not in _LSDB_SUPPORTED_JOINS:
        raise CrossMatchError(
            f"HATS/LSDB crossmatch only supports join_type='1and2' (inner join). "
            f"Got join_type='{spec.join_type}'."
        )

    # Map find mode to LSDB's n_neighbors parameter.
    n_neighbors = 1 if spec.find == "best" else None

    def to_catalog(src, lf):
        if src.access_method == "hats":
            return read_hats(src)
        if lf is None:
            raise CrossMatchError(f"Cannot turn '{src.name}' into a HATS catalogue without data.")
        pdf = lf.collect().to_pandas()
        return lsdb.from_dataframe(pdf, ra_column=src.ra_column, dec_column=src.dec_column)

    cat1 = to_catalog(src1, local_lf1)
    cat2 = to_catalog(src2, local_lf2)

    matched = cat1.crossmatch(
        cat2,
        radius_arcsec=spec.radius_arcsec,
        n_neighbors=n_neighbors,
        suffixes=("", right_suffix),
    ).compute()
    result = pl.from_pandas(matched.reset_index(drop=True))

    # Normalise to xmatch schema: rename LSDB's _dist_arcsec → sep_arcsec.
    if "_dist_arcsec" in result.columns:
        # Drop any pre-existing sep_arcsec (shouldn't exist, but be safe).
        if SEP_COLUMN in result.columns:
            result = result.drop(SEP_COLUMN)
        result = result.rename({"_dist_arcsec": SEP_COLUMN})
    else:
        logger.warning(
            "LSDB result missing expected '_dist_arcsec' column. "
            "The output may lack a 'sep_arcsec' separation column. "
            "This may indicate an LSDB API change."
        )

    # Warn if Bayesian priors were requested — not supported via LSDB.
    if spec.prior_columns:
        logger.warning(
            "Bayesian probabilistic qualification (prior_columns=%s) is not "
            "supported with HATS/LSDB. Use local files with engine='fast' or "
            "'astropy' for Tier-3 p_match output.",
            spec.prior_columns,
        )

    return result
