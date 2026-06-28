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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
) -> pl.DataFrame:
    right_renamed = _rename_right(right, left.columns)

    matched = _gather(left, left_idx).hstack(_gather(right_renamed, right_idx))
    matched = matched.hstack(pl.DataFrame({SEP_COLUMN: np.asarray(seps, dtype=float)}))
    if p_match is not None and p_match.size == matched.height:
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

    l_xyz = _radec_to_xyz(
        left[left_src.ra_column].to_numpy(),
        left[left_src.dec_column].to_numpy(),
    )
    r_xyz = _radec_to_xyz(
        right[right_src.ra_column].to_numpy(),
        right[right_src.dec_column].to_numpy(),
    )

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
    """
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

    chord_max = _arcsec_to_chord(spec.radius_arcsec)
    l_xyz = _radec_to_xyz(l_ra, l_dec)

    l_parts, r_parts, sep_parts = [], [], []
    for i, _pix in enumerate(l_pix):
        # Overlapping right pixels = self + cone-neighbours.
        npix = hp.cone_search_lonlat(
            lon=float(np.radians(l_ra[i])),
            lat=float(np.radians(l_dec[i])),
            radius=float(np.radians(radius_deg)),
            depth=DEPTH,
        )
        # Collect candidates from every overlapping right pixel, dedup by
        # global right index (so a left point straddling pixel boundaries
        # never emits duplicate pairs), then argmin (find="best") or
        # keep-all (find="all").
        candidates = []  # (chord, rpix_int, local_idx)
        for rpix in npix:
            rpix_int = int(rpix)
            tree = tree_cache.get(rpix_int)
            if tree is None:
                continue
            if spec.find == "best":
                dist, idx = tree.query(
                    l_xyz[i : i + 1], k=1, distance_upper_bound=chord_max, workers=-1
                )
                if np.isfinite(dist[0]) and idx[0] < tree.n:
                    candidates.append((float(dist[0]), rpix_int, int(idx[0])))
            else:
                neighbors = tree.query_ball_point(l_xyz[i : i + 1], r=chord_max, workers=-1)[0]
                for n in neighbors:
                    dx = xyz_cache[rpix_int][int(n)] - l_xyz[i]
                    candidates.append((float(np.linalg.norm(dx)), rpix_int, int(n)))

        if not candidates:
            continue
        seen_r: set = set()
        dedup_chord = []
        dedup_global_r = []
        for chord, rpix_int, loc in candidates:
            global_r = int(r_groups[rpix_int][loc])
            if global_r in seen_r:
                continue
            seen_r.add(global_r)
            dedup_chord.append(chord)
            dedup_global_r.append(global_r)

        if spec.find == "best":
            j = int(np.argmin(dedup_chord))
            l_parts.append(np.asarray([i], dtype=np.int64))
            r_parts.append(np.asarray([dedup_global_r[j]], dtype=np.int64))
            sep_parts.append(np.asarray([float(_chord_to_arcsec(dedup_chord[j]))], dtype=float))
        else:
            chords_arr = np.asarray(dedup_chord, dtype=float)
            seps_arr = _chord_to_arcsec(chords_arr)
            l_parts.append(np.full(len(dedup_global_r), i, dtype=np.int64))
            r_parts.append(np.asarray(dedup_global_r, dtype=np.int64))
            sep_parts.append(seps_arr)

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

    prior_logs: list = []
    for col in spec.prior_columns:
        if col not in left.columns or col not in right.columns:
            logger.warning("Prior column '%s' missing on a side; skipping.", col)
            continue
        # Budavári et al. fit the prior on the unconditional union of
        # the full catalogues (the background density), not on the
        # matched-only subset. Evaluate at the per-pair average
        # ``0.5 * (left[idx] + right[idx])`` for the data term.
        kde = bayes.fit_empirical_kde(
            left[col].to_numpy(),
            right[col].to_numpy(),
            sample_cap=50_000,
        )
        centre = 0.5 * (left[col].to_numpy()[left_idx] + right[col].to_numpy()[right_idx])
        prior_logs.append(bayes.kde_log_at(kde, centre))

    p_match = bayes.compute_p_match(
        seps,
        sig_l,
        sig_r,
        spec.radius_arcsec,
        prior_logs,
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
    else:
        raise CrossMatchError(f"Unknown engine '{chosen}'.")

    p_match = _bayesian_qualify(left, right, l_idx, r_idx, seps, left_src, right_src, spec)
    return _build_result(left, right, l_idx, r_idx, seps, spec, p_match=p_match).lazy()


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
