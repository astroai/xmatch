"""Spatial and id match engines operating on polars frames.

Two sky-match engines are available:

* ``stilts`` – shells out to STILTS ``tmatch2`` (sky/skyerr/skyellipse). Used by
  default when a STILTS command is available.
* ``astropy`` – a correct KD-tree matcher (``match_to_catalog_sky`` /
  ``search_around_sky``) used as a no-Java fallback.

Match criteria:

* ``sky`` – pairs within ``radius_arcsec``.
* ``skyerr`` / ``skyellipse`` – pairs with ``sep <= max_error * (e_left + e_right)``
  where ``e = hypot(ra_err, dec_err)`` (floored by the catalogue default). The
  errors define the search; ``radius_arcsec`` is not used. Correlation is not yet
  modelled, so ``skyellipse`` currently behaves like ``skyerr``.

Both engines (STILTS and astropy) produce identical schemas and a true
great-circle ``sep_arcsec`` column. Id joins are pure polars relational joins.
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import polars as pl

from .exceptions import CrossMatchError
from .sources import CatalogueSource

logger = logging.getLogger(__name__)

SEP_COLUMN = "sep_arcsec"
_RIGHT_SUFFIX = "_2"

_UNIT_TO_ARCSEC = {"arcsec": 1.0, "mas": 1e-3, "deg": 3600.0, "arcmin": 60.0}


@dataclass
class MatchSpec:
    radius_arcsec: float = 1.0
    matcher: str = "sky"  # "sky" | "skyerr" | "skyellipse"
    max_error: float = 3.0  # N-sigma for skyerr/skyellipse
    join_type: str = "1and2"
    find: str = "best"  # "best" | "all"


# --------------------------------------------------------------------------- #
# polars helpers
# --------------------------------------------------------------------------- #
def _gather(df: pl.DataFrame, idx: np.ndarray) -> pl.DataFrame:
    if len(idx) == 0:
        return df.clear()
    return df[pl.Series(values=np.asarray(idx, dtype=np.int64))]


def _rename_right(right: pl.DataFrame, left_cols, suffix: str = _RIGHT_SUFFIX) -> pl.DataFrame:
    overlap = set(left_cols) & set(right.columns)
    if overlap:
        right = right.rename({c: f"{c}{suffix}" for c in overlap})
    return right


def _build_result(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    spec: MatchSpec,
) -> pl.DataFrame:
    right_renamed = _rename_right(right, left.columns)

    matched = _gather(left, left_idx).hstack(_gather(right_renamed, right_idx))
    matched = matched.hstack(pl.DataFrame({SEP_COLUMN: np.asarray(seps, dtype=float)}))

    jt = spec.join_type
    parts = []
    if jt in ("1and2", "all1", "all2", "1or2", "all"):
        parts.append(matched)

    if jt in ("all1", "1or2", "all", "1not2"):
        mask = np.ones(left.height, dtype=bool)
        mask[np.asarray(left_idx, dtype=np.int64)] = False
        parts.append(left.filter(pl.Series(mask)))

    if jt in ("all2", "1or2", "all", "2not1"):
        mask = np.ones(right.height, dtype=bool)
        mask[np.asarray(right_idx, dtype=np.int64)] = False
        parts.append(right_renamed.filter(pl.Series(mask)))

    if not parts:
        return matched.clear()
    if len(parts) == 1:
        return parts[0]
    return pl.concat(parts, how="diagonal_relaxed")


# --------------------------------------------------------------------------- #
# positional errors (for skyerr / skyellipse sigma criterion)
# --------------------------------------------------------------------------- #
def _has_error_info(src: CatalogueSource) -> bool:
    return (
        bool(src.ra_err_column and src.dec_err_column) or src.default_pos_error_arcsec is not None
    )


def _pos_sigma_arcsec(df: pl.DataFrame, src: CatalogueSource) -> Optional[np.ndarray]:
    """Per-row positional error radius in arcsec: ``hypot(ra_err, dec_err)``.

    Floored by ``default_pos_error_arcsec * sqrt(2)`` when configured. Returns
    ``None`` if the catalogue carries no error information.
    """
    factor = _UNIT_TO_ARCSEC.get((src.pos_err_units or "arcsec").lower(), 1.0)
    floor = (
        float(src.default_pos_error_arcsec) * np.sqrt(2)
        if src.default_pos_error_arcsec is not None
        else None
    )
    if src.ra_err_column in df.columns and src.dec_err_column in df.columns:
        ra_e = df[src.ra_err_column].to_numpy().astype(float) * factor
        de_e = df[src.dec_err_column].to_numpy().astype(float) * factor
        sigma = np.sqrt(ra_e**2 + de_e**2)
        if floor is not None:
            sigma = np.maximum(np.nan_to_num(sigma, nan=floor), floor)
        return sigma
    if floor is not None:
        return np.full(df.height, floor)
    return None


# --------------------------------------------------------------------------- #
# astropy engine
# --------------------------------------------------------------------------- #
def _astropy_match(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import astropy.units as u
    from astropy.coordinates import SkyCoord, search_around_sky

    empty = (np.array([], int), np.array([], int), np.array([], float))
    if left.height == 0 or right.height == 0:
        return empty

    lcoord = SkyCoord(
        left[left_src.ra_column].to_numpy() * u.deg, left[left_src.dec_column].to_numpy() * u.deg
    )
    rcoord = SkyCoord(
        right[right_src.ra_column].to_numpy() * u.deg,
        right[right_src.dec_column].to_numpy() * u.deg,
    )

    # Plain radius match: fast nearest-neighbour path for find="best".
    if spec.matcher == "sky":
        if spec.find == "best":
            idx, sep2d, _ = lcoord.match_to_catalog_sky(rcoord)
            seps = sep2d.arcsec
            within = seps <= spec.radius_arcsec
            return np.nonzero(within)[0], idx[within], seps[within]
        left_idx, right_idx, sep2d, _ = search_around_sky(
            lcoord, rcoord, spec.radius_arcsec * u.arcsec
        )
        return left_idx, right_idx, sep2d.arcsec

    # Error-based match: a pair matches iff sep <= max_error * (e_left + e_right).
    lsig = _pos_sigma_arcsec(left, left_src)
    rsig = _pos_sigma_arcsec(right, right_src)
    if lsig is None or rsig is None:
        logger.warning(
            "Matcher '%s' needs positional errors present in the data; none found, no matches.",
            spec.matcher,
        )
        return empty
    search_radius = spec.max_error * (float(np.nanmax(lsig)) + float(np.nanmax(rsig)))
    if search_radius <= 0:
        return empty
    left_idx, right_idx, sep2d, _ = search_around_sky(lcoord, rcoord, search_radius * u.arcsec)
    seps = sep2d.arcsec
    combined = lsig[left_idx] + rsig[right_idx]
    keep = seps <= spec.max_error * combined
    left_idx, right_idx, seps, combined = (
        left_idx[keep],
        right_idx[keep],
        seps[keep],
        combined[keep],
    )

    if spec.find == "best" and len(left_idx) > 0:
        score = seps / np.where(combined > 0, combined, np.inf)
        order = np.argsort(score)
        # ⚡ Bolt Optimization: Use vectorized np.unique instead of a slow Python
        # loop with set() tracking. For large cross-matches, this C-level filtering
        # significantly speeds up removing duplicate left_idx matches while keeping
        # the best scores. Sorting unique_indices preserves the "best" score ordering.
        _, unique_indices = np.unique(left_idx[order], return_index=True)
        sel = order[np.sort(unique_indices)]
        left_idx, right_idx, seps = left_idx[sel], right_idx[sel], seps[sel]
    return left_idx, right_idx, seps


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def sky_match(
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left_lf: pl.LazyFrame,
    right_lf: pl.LazyFrame,
    spec: MatchSpec,
    *,
    engine: str = "auto",
    stilts_cmd_base: Optional[str] = None,
    java_opts: Optional[str] = None,
    tmpdir: Optional[str] = None,
) -> pl.LazyFrame:
    """Run a positional crossmatch and return a lazy result frame."""
    if not left_src.ra_column or not left_src.dec_column:
        raise CrossMatchError(f"RA/Dec columns unknown for '{left_src.name}'.")
    if not right_src.ra_column or not right_src.dec_column:
        raise CrossMatchError(f"RA/Dec columns unknown for '{right_src.name}'.")

    # Error-ellipse matchers need positional errors on *both* catalogues. If that
    # information is unavailable, fall back to a plain radius match so the result
    # does not depend on which engine runs.
    if spec.matcher != "sky" and not (_has_error_info(left_src) and _has_error_info(right_src)):
        logger.warning(
            "Matcher '%s' needs positional errors on both catalogues; falling back to 'sky'.",
            spec.matcher,
        )
        spec = dataclasses.replace(spec, matcher="sky")

    from . import stilts

    chosen = engine
    if chosen == "auto":
        chosen = "stilts" if stilts.stilts_available(stilts_cmd_base) else "astropy"

    if chosen == "stilts":
        try:
            return stilts.stilts_sky_match(
                left_src,
                right_src,
                left_lf.collect(),
                right_lf.collect(),
                spec,
                stilts_cmd_base=stilts_cmd_base,
                java_opts=java_opts,
                tmpdir=tmpdir,
            ).lazy()
        except Exception as exc:
            logger.warning("STILTS match failed (%s); falling back to astropy engine.", exc)

    left = left_lf.collect()
    right = right_lf.collect()
    left_idx, right_idx, seps = _astropy_match(left, right, left_src, right_src, spec)
    logger.info("astropy sky match: %d matched pairs.", len(left_idx))
    return _build_result(left, right, left_idx, right_idx, seps, spec).lazy()


_JOIN_HOW = {
    "1and2": "inner",
    "1or2": "full",
    "all": "full",
    "all1": "left",
    "all2": "right",
    "1not2": "anti",
    "2not1": "anti",  # handled by swapping operands
}


def id_join(
    left_lf: pl.LazyFrame,
    right_lf: pl.LazyFrame,
    id_left: str,
    id_right: str,
    join_type: str = "1and2",
) -> pl.LazyFrame:
    """Relational id join between two catalogues using polars."""
    how = _JOIN_HOW.get(join_type, "inner")
    if join_type == "2not1":
        return right_lf.join(
            left_lf, left_on=id_right, right_on=id_left, how="anti", suffix=_RIGHT_SUFFIX
        )
    return left_lf.join(right_lf, left_on=id_left, right_on=id_right, how=how, suffix=_RIGHT_SUFFIX)
