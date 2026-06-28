"""Spatial and id match engines operating on polars frames.

Sky-match engines (drop-in alternatives):

* ``stilts`` – shells out to STILTS ``tmatch2`` (sky/skyerr/skyellipse). Used by
  default when a STILTS command is available.
* ``astropy`` – ``match_to_catalog_sky`` / ``search_around_sky`` from astropy.
* ``fast``   – Tier 1: ``scipy.spatial.cKDTree`` on 3D Cartesian unit-sphere
  embeddings. Drop-in replacement for ``astropy`` in-memory; ~3-5x faster.
* ``zone``   – Tier 2: HEALPix-sharded cone match. Uses ``cdshealpix`` when
  importable (sub-pixel zonning + per-pixel ``cKDTree`` queries) and falls
  back to a single ``cKDTree`` query when ``cdshealpix`` is not available
  (with a logged warning).

All engines share an identical schema: source columns (left unchanged, right
collisions get a ``_2`` suffix) plus a true great-circle ``sep_arcsec``
column. ``id_join`` is a pure polars relational join and is engine-agnostic.

Match criteria:

* ``sky`` – pairs within ``radius_arcsec``.
* ``skyerr`` / ``skyellipse`` – pairs with ``sep <= max_error * (e_left + e_right)``
  where ``e = hypot(ra_err, dec_err)`` (floored by the catalogue default).
  Correlation is not yet modelled, so ``skyellipse`` behaves like ``skyerr``.

Tier 3 — Bayesian probabilistic qualification: when ``MatchSpec.prior_columns``
is non-empty (and the catalogue has those columns), every matched pair is
re-scored with a Budavári-style hierarchical Bayes factor that combines:

* a 2D Gaussian positional kernel with the joint (sigma_left + sigma_right)
  per-row uncertainty, and
* independent 1-D Gaussian-KDE prior densities on each requested magnitude /
  colour column, fit from a uniform random sample of both sides (capped at
  50 000 rows).

The output is a ``p_match`` column in [0, 1]. Numeric evaluation lives in
:mod:`xmatch.bayes`; this module only orchestrates data marshalling.
"""

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from .exceptions import CrossMatchError
from .sources import CatalogueSource

logger = logging.getLogger(__name__)

SEP_COLUMN = "sep_arcsec"
PMATCH_COLUMN = "p_match"
_RIGHT_SUFFIX = "_2"

_UNIT_TO_ARCSEC = {"arcsec": 1.0, "mas": 1e-3, "deg": 3600.0, "arcmin": 60.0}


@dataclass
class MatchSpec:
    radius_arcsec: float = 1.0
    matcher: str = "sky"  # "sky" | "skyerr" | "skyellipse"
    max_error: float = 3.0  # N-sigma for skyerr/skyellipse
    join_type: str = "1and2"
    find: str = "best"  # "best" | "all"
    # Bayesian-prior columns. When non-empty AND the catalogues carry those
    # columns, a ``p_match`` column is appended to the matched result.
    prior_columns: List[str] = field(default_factory=list)
    # Epoch to propagate coordinates to before spatial matching (Julian year).
    # Requires pm_ra_column / pm_dec_column + epoch metadata on the catalogue.
    # NaN proper motions are treated as zero (no propagation).
    target_epoch: Optional[float] = None
    # Optional polars expression string applied as a boolean post-filter on
    # the matched pairs BEFORE reducing find="all" → find="best".  Column
    # names from the left side are used as-is; right-side columns gain a
    # ``_2`` suffix (e.g. ``abs(mag_g - mag_g_2) < 0.5``).
    filter_expr: Optional[str] = None
    # Extra columns for N-dimensional cKDTree matching, mapping column name
    # to a dimensionless weight.  Columns are z-score normalized across the
    # union of both catalogues and appended to the 3-D Cartesian unit-sphere
    # embedding.  The spatial ``radius_arcsec`` is still enforced as a hard
    # bound; among candidates within that bound the nearest in N-d feature
    # space is chosen.
    extra_distance_cols: Dict[str, float] = field(default_factory=dict)
    # Maximum number of left HEALPix pixel groups to process in one batch.
    # When set, the zone engine processes pixel groups in chunks, freeing
    # intermediate results between batches (out-of-core friendly).  Defaults
    # to ``None`` (process all pixels in one pass).  Only effective with
    # ``engine="zone"`` and ``cdshealpix`` installed.
    batch_size: Optional[int] = None


# --------------------------------------------------------------------------- #
# polars helpers
# --------------------------------------------------------------------------- #
def _gather(df: pl.DataFrame, idx: np.ndarray) -> pl.DataFrame:
    """Return ``df`` rows indexed by *idx* via polars' Arrow-backed path."""
    if len(idx) == 0:
        return df.clear()
    return df.gather(np.asarray(idx, dtype=np.int64))


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
    p_match: Optional[np.ndarray] = None,
    right_suffix: str = _RIGHT_SUFFIX,
) -> pl.DataFrame:
    right_renamed = _rename_right(right, left.columns, suffix=right_suffix)

    matched = _gather(left, left_idx).hstack(_gather(right_renamed, right_idx))
    # Drop any previous sep_arcsec carried from the left side (accumulator in
    # multi-way crossmatching) so the latest match separation is unambiguous.
    if SEP_COLUMN in matched.columns:
        matched = matched.drop(SEP_COLUMN)
    matched = matched.hstack(pl.DataFrame({SEP_COLUMN: np.asarray(seps, dtype=float)}))
    if p_match is not None and p_match.size == matched.height:
        if PMATCH_COLUMN in matched.columns:
            matched = matched.drop(PMATCH_COLUMN)
        matched = matched.hstack(pl.DataFrame({PMATCH_COLUMN: np.asarray(p_match, dtype=float)}))

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
    """Per-row positional error radius in arcsec: ``hypot(ra_err, dec_err)``."""
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
# 3D Cartesian unit-sphere helpers (Tier 1 and Tier 2 share these)
# --------------------------------------------------------------------------- #
def _radec_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert (RA, Dec) in degrees to 3-D Cartesian unit vectors."""
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    cos_dec = np.cos(dec)
    xyz = np.empty((ra.shape[0], 3), dtype=float)
    xyz[:, 0] = cos_dec * np.cos(ra)
    xyz[:, 1] = cos_dec * np.sin(ra)
    xyz[:, 2] = np.sin(dec)
    return xyz


def _chord_to_arcsec(chord: np.ndarray) -> np.ndarray:
    """Chord length on unit-sphere → great-circle arc in arcsec."""
    chord = np.clip(np.asarray(chord, dtype=float), 0.0, 2.0)
    return np.degrees(2.0 * np.arcsin(chord * 0.5)) * 3600.0


def _arcsec_to_chord(arcsec: float) -> float:
    return float(2.0 * np.sin(np.radians(float(arcsec) / 3600.0) * 0.5))


# --------------------------------------------------------------------------- #
# proper motion correction
# --------------------------------------------------------------------------- #
def _apply_proper_motion(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    target_epoch: float,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Propagate coordinates to ``target_epoch`` using per-row PM + epoch.

    Mutates *left* and/or *right* in-place when both PM columns and epoch
    information are available on a side.  Returns the (possibly modified)
    pair.
    """
    from .astro_utils import propagate_proper_motion

    has_left_pm = bool(
        left_src.pm_ra_column and left_src.pm_dec_column
        and (left_src.epoch_column or left_src.epoch is not None)
    )
    has_right_pm = bool(
        right_src.pm_ra_column and right_src.pm_dec_column
        and (right_src.epoch_column or right_src.epoch is not None)
    )
    if not has_left_pm and not has_right_pm:
        logger.debug("No PM+epoch info on either side; skipping PM propagation.")
        return left, right

    def _propagate_side(
        df: pl.DataFrame, src: CatalogueSource, label: str
    ) -> pl.DataFrame:
        if not (src.pm_ra_column and src.pm_dec_column):
            logger.debug("No PM columns on %s side; skipping.", label)
            return df
        if src.pm_ra_column not in df.columns or src.pm_dec_column not in df.columns:
            logger.debug("PM columns missing from %s data; skipping.", label)
            return df

        if src.epoch_column and src.epoch_column in df.columns:
            epoch_arr = df[src.epoch_column].to_numpy().astype(float)
        elif src.epoch is not None:
            epoch_arr = np.full(df.height, float(src.epoch), dtype=float)
        else:
            logger.debug("No epoch info on %s side; skipping.", label)
            return df

        ra_arr = df[src.ra_column].to_numpy().astype(float)
        dec_arr = df[src.dec_column].to_numpy().astype(float)
        pmra = df[src.pm_ra_column].to_numpy().astype(float)
        pmde = df[src.pm_dec_column].to_numpy().astype(float)

        new_ra, new_dec = propagate_proper_motion(
            ra_arr, dec_arr, pmra, pmde, epoch_arr, target_epoch,
        )
        logger.info(
            "PM propagation %s: max ΔRA=%.4f arcsec, max ΔDec=%.4f arcsec",
            label,
            float(np.nanmax(np.abs(new_ra - ra_arr))) * 3600.0,
            float(np.nanmax(np.abs(new_dec - dec_arr))) * 3600.0,
        )
        return df.with_columns(
            pl.Series(src.ra_column, new_ra),
            pl.Series(src.dec_column, new_dec),
        )

    left = _propagate_side(left, left_src, "left")
    right = _propagate_side(right, right_src, "right")
    return left, right


# --------------------------------------------------------------------------- #
# N-dimensional feature helpers
# --------------------------------------------------------------------------- #
def _build_nd_features(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    df: pl.DataFrame,
    extra_cols: Dict[str, float],
    union_mean: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, Optional[Dict[str, Tuple[float, float]]]]:
    """Build N-d feature array: [x, y, z] + z-score normalised columns.

    Returns ``(features, stats)`` where *stats* maps column name to
    ``(mean, std)`` for reuse across left/right sides.  When *union_mean* is
    provided (from a previous call on the other side), the same normalisation
    constants are reused instead of being recomputed.
    """
    xyz = _radec_to_xyz(ra_deg, dec_deg)
    if not extra_cols:
        return xyz, None

    extra_parts: list = []
    stats: Dict[str, Tuple[float, float]] = {}
    for col, weight in extra_cols.items():
        if col not in df.columns:
            logger.warning("Extra distance column '%s' missing; skipping.", col)
            continue
        vals = df[col].to_numpy().astype(float)
        if union_mean is not None and col in union_mean:
            mean, std = union_mean[col]
        else:
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))
            if std == 0 or not np.isfinite(std):
                std = 1.0
        stats[col] = (mean, std)
        norm = np.nan_to_num((vals - mean) / std, nan=0.0) * weight
        extra_parts.append(norm.reshape(-1, 1))

    if not extra_parts:
        return xyz, None
    features = np.hstack([xyz] + extra_parts)
    return features, stats


def _apply_match_filter(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    filter_expr: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-filter matched pairs with a polars SQL WHERE clause.

    The *filter_expr* string is evaluated as ``SELECT * FROM tmp WHERE
    <filter_expr>`` using polars' ``SQLContext``.  Column names from the left
    side are used as-is; right-side collision columns gain a ``_2`` suffix.
    Returns the filtered index and separation arrays.
    """
    if left_idx.size == 0:
        return left_idx, right_idx, seps

    matched_left = _gather(left, left_idx)
    right_renamed = _rename_right(right, left.columns, suffix=_RIGHT_SUFFIX)
    matched_right = _gather(right_renamed, right_idx)

    # Build a temporary frame with a row-index column so we can recover which
    # original pairs survive the WHERE clause.
    tmp = matched_left.hstack(matched_right)
    tmp = tmp.with_row_index(name="_row_id")

    try:
        ctx = pl.SQLContext(tmp=tmp)
        filtered = ctx.execute(
            f"SELECT _row_id FROM tmp WHERE {filter_expr}"
        ).collect()
        keep_rows = set(int(r) for r in filtered["_row_id"].to_list())
        keep = np.array([i in keep_rows for i in range(tmp.height)], dtype=bool)
    except Exception as exc:
        logger.warning(
            "Filter expression '%s' failed (%s); keeping all pairs.",
            filter_expr, exc,
        )
        return left_idx, right_idx, seps

    n_kept = int(np.sum(keep))
    if n_kept == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
        )
    if n_kept < left_idx.size:
        logger.info("Filter expression kept %d/%d matched pairs.", n_kept, left_idx.size)
    return left_idx[keep], right_idx[keep], seps[keep]


# --------------------------------------------------------------------------- #
# Tier 1 — fast engine (scipy.spatial.cKDTree, no Java, no FITS I/O)
# --------------------------------------------------------------------------- #
def _scipy_match(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    empty = (np.array([], int), np.array([], int), np.array([], float))
    if left.height == 0 or right.height == 0:
        return empty

    l_ra = left[left_src.ra_column].to_numpy()
    l_dec = left[left_src.dec_column].to_numpy()
    r_ra = right[right_src.ra_column].to_numpy()
    r_dec = right[right_src.dec_column].to_numpy()

    l_xyz = _radec_to_xyz(l_ra, l_dec)
    r_xyz = _radec_to_xyz(r_ra, r_dec)

    if spec.matcher == "sky":
        chord_max = _arcsec_to_chord(spec.radius_arcsec)
    else:
        lsig = _pos_sigma_arcsec(left, left_src)
        rsig = _pos_sigma_arcsec(right, right_src)
        if lsig is None or rsig is None:
            logger.warning(
                "Matcher '%s' needs positional errors present in the data; none found, no matches.",
                spec.matcher,
            )
            return empty
        search_radius = spec.max_error * (float(np.nanmax(lsig)) + float(np.nanmax(rsig)))
        chord_max = _arcsec_to_chord(max(search_radius, 0.0))
    if chord_max <= 0:
        return empty

    # --- N-dimensional ranking (only affects find="best") ------------------
    if spec.extra_distance_cols and spec.find == "best":
        return _scipy_match_nd(
            l_xyz, r_xyz, l_ra, l_dec, r_ra, r_dec,
            left, right, chord_max, spec,
        )

    tree = cKDTree(r_xyz)
    if spec.find == "best":
        dist, idx = tree.query(l_xyz, k=1, distance_upper_bound=chord_max, workers=-1)
        # Out-of-bound indices are sentinel values (self.n == len(r_xyz)) and
        # distances are ``inf``; drop those rows.
        valid = np.isfinite(dist) & (idx < r_xyz.shape[0])
        left_idx = np.nonzero(valid)[0]
        right_idx = idx[valid].astype(np.int64)
        sep = _chord_to_arcsec(dist[valid])
        return left_idx, right_idx, sep

    # find == "all": per-left list of matched right indices.
    idx_lists = tree.query_ball_point(l_xyz, r=chord_max, workers=-1)
    l_parts, r_parts = [], []
    for i, neighbors in enumerate(idx_lists):
        if not neighbors:
            continue
        nb = np.asarray(neighbors, dtype=np.int64)
        l_parts.append(np.full(nb.shape, i, dtype=np.int64))
        r_parts.append(nb)
    if not r_parts:
        return empty
    left_idx = np.concatenate(l_parts)
    right_idx = np.concatenate(r_parts)
    sep = _chord_to_arcsec(np.linalg.norm(l_xyz[left_idx] - r_xyz[right_idx], axis=-1))
    return left_idx, right_idx, sep


# Module-level configuration — tunable for benchmarking / memory tuning.
_ND_CHUNK_SIZE = 50_000


def _scipy_match_nd(
    l_xyz: np.ndarray,
    r_xyz: np.ndarray,
    l_ra: np.ndarray,
    l_dec: np.ndarray,
    r_ra: np.ndarray,
    r_dec: np.ndarray,
    left: pl.DataFrame,
    right: pl.DataFrame,
    chord_max: float,
    spec: MatchSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N-dimensional cKDTree match for find="best" (vectorised + chunked).

    1. Query spatial cKDTree for up to *k_candidates* neighbours within
       ``chord_max``.
    2. Build N-d features (3-D spatial + z-score normalised extra columns).
    3. Compute N-d distances via broadcasting in slices of *CHUNK_SIZE*
       left rows, accumulating results incrementally — keeps the
       ``(n_left, k, ndim)`` intermediate array bounded to ≈ *CHUNK_SIZE*
       × k × ndim regardless of total catalogue size.
    """
    from scipy.spatial import cKDTree

    CHUNK_SIZE = _ND_CHUNK_SIZE

    empty = (np.array([], int), np.array([], int), np.array([], float))
    n_left = l_xyz.shape[0]
    n_right = r_xyz.shape[0]

    # Step 1: spatial candidate retrieval.
    k_candidates = min(max(10, int(spec.radius_arcsec * 2)), n_right)
    spatial_tree = cKDTree(r_xyz)
    dist_sp, idx_sp = spatial_tree.query(
        l_xyz, k=min(k_candidates, n_right),
        distance_upper_bound=chord_max, workers=-1,
    )
    if k_candidates == 1:
        dist_sp = dist_sp[:, None]
        idx_sp = idx_sp[:, None]
    k = idx_sp.shape[1]  # actual k used

    # Step 2: build N-d features once per side.
    l_feat, stats = _build_nd_features(l_ra, l_dec, left, spec.extra_distance_cols)
    r_feat, _ = _build_nd_features(r_ra, r_dec, right, spec.extra_distance_cols, union_mean=stats)

    # Step 3: chunked vectorised N-d distance computation.
    #          Process n_left in slices to bound memory at
    #          O(CHUNK_SIZE × k × ndim).
    left_idx_parts, right_idx_parts, sep_parts = [], [], []
    for sl_start in range(0, n_left, CHUNK_SIZE):
        sl_end = min(sl_start + CHUNK_SIZE, n_left)
        sl = slice(sl_start, sl_end)
        chunk_size = sl_end - sl_start

        valid_chunk = np.isfinite(dist_sp[sl]) & (idx_sp[sl] < n_right)
        if not np.any(valid_chunk):
            continue

        idx_safe = np.where(valid_chunk, idx_sp[sl], 0).astype(np.int64)
        r_candidates = r_feat[idx_safe]                 # (chunk, k, ndim)
        l_expanded = l_feat[sl, None, :]                 # (chunk, 1, ndim)
        nd_dists = np.linalg.norm(
            r_candidates - l_expanded, axis=-1
        )                                                # (chunk, k)
        nd_dists[~valid_chunk] = np.inf

        best_k = np.argmin(nd_dists, axis=-1)            # (chunk,)
        has_match = np.isfinite(nd_dists[np.arange(chunk_size), best_k])
        if not np.any(has_match):
            continue

        chunk_left = np.nonzero(has_match)[0].astype(np.int64) + sl_start
        chunk_right = idx_sp[chunk_left, best_k[has_match]].astype(np.int64)
        chunk_seps = _chord_to_arcsec(
            np.linalg.norm(l_xyz[chunk_left] - r_xyz[chunk_right], axis=-1)
        )
        left_idx_parts.append(chunk_left)
        right_idx_parts.append(chunk_right)
        sep_parts.append(np.asarray(chunk_seps, dtype=float))

    if not left_idx_parts:
        return empty
    return (
        np.concatenate(left_idx_parts),
        np.concatenate(right_idx_parts),
        np.concatenate(sep_parts),
    )


# --------------------------------------------------------------------------- #
# Tier 2 — HEALPix zone engine
# --------------------------------------------------------------------------- #
def _zone_match(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HEALPix zone cone-match (LSDB-orthogonal, lightweight).

    When ``cdshealpix`` is importable we shard both sides into HEALPix pixels
    (nside=32 by default), build a per-pixel cKDTree on the right, and only
    query neighbour pixels within the cone radius — classic HATS-style
    partitioning. When ``cdshealpix`` is not importable we transparently
    downgrade to :func:`_scipy_match` and log a warning.

    N-dimensional extra_distance_cols are delegated to :func:`_scipy_match`
    since the N-d ranking loop is simpler without pixel sharding.
    """
    if spec.extra_distance_cols:
        logger.info(
            "extra_distance_cols set; using Tier 1 cKDTree for N-d matching."
        )
        return _scipy_match(left, right, left_src, right_src, spec)
    try:
        import cdshealpix as hp  # noqa: F401
    except ImportError:
        logger.warning(
            "engine='zone' requested but 'cdshealpix' is unavailable; "
            "falling back to the Tier 1 cKDTree (no pixel-shard optimisation). "
            "Install with `pip install cdshealpix` to enable HEALPix-shard partitioning."
        )
        return _scipy_match(left, right, left_src, right_src, spec)
    return _zone_match_healpix(left, right, left_src, right_src, spec)


def _zone_match_healpix(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cdshealpix as hp
    from scipy.spatial import cKDTree

    empty = (np.array([], int), np.array([], int), np.array([], float))
    if left.height == 0 or right.height == 0:
        return empty

    l_ra = left[left_src.ra_column].to_numpy().astype(float)
    l_dec = left[left_src.dec_column].to_numpy().astype(float)
    r_ra = right[right_src.ra_column].to_numpy().astype(float)
    r_dec = right[right_src.dec_column].to_numpy().astype(float)

    if spec.matcher == "sky":
        radius_deg = spec.radius_arcsec / 3600.0
        chord_max = _arcsec_to_chord(spec.radius_arcsec)
    else:
        lsig = _pos_sigma_arcsec(left, left_src)
        rsig = _pos_sigma_arcsec(right, right_src)
        if lsig is None or rsig is None:
            logger.warning(
                "Matcher '%s' needs positional errors; none found.",
                spec.matcher,
            )
            return empty
        search_radius = spec.max_error * (float(np.nanmax(lsig)) + float(np.nanmax(rsig)))
        radius_deg = max(search_radius, 0.0) / 3600.0
        chord_max = _arcsec_to_chord(max(search_radius, 0.0))
    if radius_deg <= 0:
        return empty

    NSIDE = 32
    DEPTH = 5  # 2 ** 5 == 32
    l_pix = np.asarray(hp.lonlat_to_healpix(np.radians(l_ra), np.radians(l_dec), NSIDE), dtype=int)
    r_pix = np.asarray(hp.lonlat_to_healpix(np.radians(r_ra), np.radians(r_dec), NSIDE), dtype=int)

    # Cache per-pixel right xyz buffers and per-pixel cKDTree instances.
    unique_pix, pix_inv = np.unique(r_pix, return_inverse=True)
    r_groups = {int(pix): np.where(pix_inv == k)[0] for k, pix in enumerate(unique_pix)}
    tree_cache: dict[int, "cKDTree"] = {}
    xyz_cache: dict[int, np.ndarray] = {}
    for pix in unique_pix:
        idx = r_groups[int(pix)]
        xyz_cache[int(pix)] = _radec_to_xyz(r_ra[idx], r_dec[idx])
        tree_cache[int(pix)] = cKDTree(xyz_cache[int(pix)])

    l_xyz = _radec_to_xyz(l_ra, l_dec)

    # Group left points by HEALPix pixel so we can batch-query each right
    # pixel's tree once (instead of spawning worker threads per point).
    l_by_pix: dict[int, list] = {}
    for i, pix in enumerate(l_pix):
        l_by_pix.setdefault(int(pix), []).append(i)

    # Pre-compute per-right-pixel global-index arrays for fast margin merging.
    r_global_by_pix: dict[int, np.ndarray] = {
        int(pix): r_groups[int(pix)] for pix in unique_pix
    }

    # Sort left pixel groups largest-first so batches are balanced.
    pixel_items = sorted(
        l_by_pix.items(), key=lambda kv: len(kv[1]), reverse=True,
    )
    batch_size = spec.batch_size or len(pixel_items)

    l_parts, r_parts, sep_parts = [], [], []
    for batch_start in range(0, len(pixel_items), max(1, batch_size)):
        batch_pixels = pixel_items[batch_start : batch_start + max(1, batch_size)]
        batch_l = []
        batch_r = []
        batch_s = []

        for l_pix_int, left_indices in batch_pixels:
            indices_arr = np.asarray(left_indices, dtype=np.int64)
            # Cone search once per pixel (same for all points in pixel).
            mid = len(left_indices) // 2
            rep_i = left_indices[mid]
            npix = hp.cone_search_lonlat(
                lon=float(np.radians(l_ra[rep_i])),
                lat=float(np.radians(l_dec[rep_i])),
                radius=float(np.radians(radius_deg)),
                depth=DEPTH,
            )
            batch_xyz = l_xyz[indices_arr]  # (n_pix, 3)

            # --- margin caching: merge all neighbouring right pixels into one
            #     tree and query it once instead of querying each right pixel
            #     individually.
            margin_xyz_parts: list = []
            margin_global_parts: list = []
            for rpix in npix:
                rpix_int = int(rpix)
                if rpix_int not in r_global_by_pix:
                    continue
                margin_xyz_parts.append(xyz_cache[rpix_int])
                margin_global_parts.append(r_global_by_pix[rpix_int])

            if not margin_xyz_parts:
                continue

            margin_xyz = np.vstack(margin_xyz_parts)
            margin_global = np.concatenate(margin_global_parts)
            margin_tree = cKDTree(margin_xyz)

            if spec.find == "best":
                dist, local_idx = margin_tree.query(
                    batch_xyz, k=1, distance_upper_bound=chord_max, workers=-1,
                )
                valid = np.isfinite(dist) & (local_idx < margin_tree.n)
                for k in np.nonzero(valid)[0]:
                    k_idx = left_indices[k]
                    global_r = int(margin_global[local_idx[k]])
                    sep_arcsec = float(_chord_to_arcsec(float(dist[k])))
                    batch_l.append(np.array([k_idx], dtype=np.int64))
                    batch_r.append(np.array([global_r], dtype=np.int64))
                    batch_s.append(np.array([sep_arcsec], dtype=float))
            else:
                idx_lists = margin_tree.query_ball_point(
                    batch_xyz, r=chord_max, workers=-1,
                )
                for k, neighbors in enumerate(idx_lists):
                    if not neighbors:
                        continue
                    k_idx = left_indices[k]
                    nb = np.asarray(neighbors, dtype=np.int64)
                    global_r = margin_global[nb].astype(np.int64)
                    chords = np.linalg.norm(
                        margin_xyz[nb] - batch_xyz[k], axis=-1,
                    )
                    seps = _chord_to_arcsec(chords)
                    batch_l.append(np.full(len(neighbors), k_idx, dtype=np.int64))
                    batch_r.append(global_r)
                    batch_s.append(np.asarray(seps, dtype=float))

        # Flush batch to accumulator (out-of-core friendly — per-batch
        # margin trees and numpy arrays can be GC'd after this point).
        l_parts.extend(batch_l)
        r_parts.extend(batch_r)
        sep_parts.extend(batch_s)
        if spec.batch_size and batch_start + max(1, batch_size) < len(pixel_items):
            logger.debug(
                "Batch %d/%d complete (%d matched pairs so far).",
                batch_start // max(1, batch_size) + 1,
                (len(pixel_items) + max(1, batch_size) - 1) // max(1, batch_size),
                sum(p.size for p in l_parts),
            )

    if not l_parts:
        return empty
    return (
        np.concatenate(l_parts),
        np.concatenate(r_parts),
        np.concatenate(sep_parts),
    )


# --------------------------------------------------------------------------- #
# astropy engine (existing implementation, unchanged)
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

        _, first_occurrence_indices = np.unique(left_idx[order], return_index=True)
        sel = order[np.sort(first_occurrence_indices)]

        left_idx, right_idx, seps = left_idx[sel], right_idx[sel], seps[sel]
    return left_idx, right_idx, seps


# --------------------------------------------------------------------------- #
# Tier 3 — Bayesian probabilistic qualification
# --------------------------------------------------------------------------- #
def _bayesian_qualify(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
) -> Optional[np.ndarray]:
    """Compute a ``p_match`` column on the matched pairs.

    Returns ``None`` (no p_match) if either:

    * the spec has no prior columns, or
    * neither catalogue has the requested prior columns.
    """
    if not spec.prior_columns:
        return None
    if left_idx.size == 0:
        return np.zeros(0, dtype=float)

    from . import bayes

    sigma_left = _pos_sigma_arcsec(left, left_src)
    sigma_right = _pos_sigma_arcsec(right, right_src)
    # When a side has no per-row errors, fall back to a single 0.5 arcsec floor
    # so the posterior still has a meaningful width.
    if sigma_left is None:
        sigma_left = np.full(left.height, 0.5, dtype=float)
    if sigma_right is None:
        sigma_right = np.full(right.height, 0.5, dtype=float)
    sig_l = sigma_left[left_idx]
    sig_r = sigma_right[right_idx]

    prior_log_match: list = []
    prior_log_bg: list = []
    for col in spec.prior_columns:
        if col not in left.columns or col not in right.columns:
            logger.warning("Prior column '%s' missing on a side; skipping.", col)
            continue
        # Budavári et al. fit the prior on the unconditional union of
        # the full catalogues (the background density), not on the
        # matched-only subset.
        kde = bayes.fit_empirical_kde(
            left[col].to_numpy(),
            right[col].to_numpy(),
            sample_cap=50_000,
        )
        # --- match hypothesis: both sides are the same object --------------
        # Photometric scatter around the pair midpoint is drawn from the
        # population KDE.
        centre = 0.5 * (left[col].to_numpy()[left_idx] + right[col].to_numpy()[right_idx])
        prior_log_match.append(bayes.kde_log_at(kde, centre))
        # --- background hypothesis: two independent population draws --------
        # KDE(mag_left) × KDE(mag_right) in log space.
        log_left = bayes.kde_log_at(kde, left[col].to_numpy()[left_idx])
        log_right = bayes.kde_log_at(kde, right[col].to_numpy()[right_idx])
        prior_log_bg.append(log_left + log_right)

    p_match = bayes.compute_p_match(
        seps,
        sig_l,
        sig_r,
        spec.radius_arcsec,
        prior_log_match=sum(prior_log_match) if prior_log_match else None,
        prior_log_bg=sum(prior_log_bg) if prior_log_bg else None,
    )
    return p_match


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
    right_suffix: str = _RIGHT_SUFFIX,
) -> pl.LazyFrame:
    """Run a positional crossmatch and return a lazy result frame."""
    if not left_src.ra_column or not left_src.dec_column:
        raise CrossMatchError(f"RA/Dec columns unknown for '{left_src.name}'.")
    if not right_src.ra_column or not right_src.dec_column:
        raise CrossMatchError(f"RA/Dec columns unknown for '{right_src.name}'.")

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

    # --- proper motion propagation (common to all engines) -----------------
    if spec.target_epoch is not None:
        left_eager = left_lf.collect()
        right_eager = right_lf.collect()
        left_eager, right_eager = _apply_proper_motion(
            left_eager, right_eager, left_src, right_src, float(spec.target_epoch),
        )
        left_lf = left_eager.lazy()
        right_lf = right_eager.lazy()

    if chosen == "stilts":
        if right_suffix != _RIGHT_SUFFIX:
            logger.warning(
                "STILTS engine selected — right_suffix='%s' is ignored; "
                "STILTS always uses '_2'. Multi-way crossmatching with STILTS "
                "will produce duplicate column names for catalogues 3+.",
                right_suffix,
            )
        if spec.prior_columns:
            logger.warning(
                "STILTS engine selected — Bayesian probabilistic qualification "
                "(prior_columns=%s) is skipped. Use engine='fast' or 'astropy' "
                "for Tier-3 p_match output.",
                spec.prior_columns,
            )
        if spec.filter_expr:
            logger.warning(
                "STILTS engine selected — filter_expr='%s' is skipped. "
                "Use engine='fast', 'astropy', or 'zone' for post-match filtering.",
                spec.filter_expr,
            )
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

    if chosen == "astropy":
        left = left_lf.collect()
        right = right_lf.collect()
        l_idx, r_idx, seps = _astropy_match(left, right, left_src, right_src, spec)
        logger.info("astropy sky match: %d matched pairs.", len(l_idx))
    elif chosen == "fast":
        left = left_lf.collect()
        right = right_lf.collect()
        l_idx, r_idx, seps = _scipy_match(left, right, left_src, right_src, spec)
        logger.info("fast (cKDTree) sky match: %d matched pairs.", len(l_idx))
    elif chosen == "zone":
        left = left_lf.collect()
        right = right_lf.collect()
        l_idx, r_idx, seps = _zone_match(left, right, left_src, right_src, spec)
        logger.info("zone (HEALPix pixellated) sky match: %d matched pairs.", len(l_idx))
    elif chosen == "ray":
        from .ray_engine import ray_zone_match

        left = left_lf.collect()
        right = right_lf.collect()
        l_idx, r_idx, seps = ray_zone_match(left, right, left_src, right_src, spec)
        logger.info("ray (distributed) sky match: %d matched pairs.", len(l_idx))
    else:
        raise CrossMatchError(f"Unknown engine '{chosen}'.")

    # --- post-match boolean filter ----------------------------------------
    if spec.filter_expr and l_idx.size > 0:
        l_idx, r_idx, seps = _apply_match_filter(
            left, right, l_idx, r_idx, seps, spec.filter_expr,
        )

    p_match = _bayesian_qualify(left, right, l_idx, r_idx, seps, left_src, right_src, spec)
    return _build_result(left, right, l_idx, r_idx, seps, spec, p_match=p_match, right_suffix=right_suffix).lazy()


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
    suffix: str = _RIGHT_SUFFIX,
) -> pl.LazyFrame:
    """Relational id join between two catalogues using polars."""
    how = _JOIN_HOW.get(join_type, "inner")
    if join_type == "2not1":
        return right_lf.join(
            left_lf, left_on=id_right, right_on=id_left, how="anti", suffix=suffix
        )
    return left_lf.join(right_lf, left_on=id_left, right_on=id_right, how=how, suffix=suffix)
