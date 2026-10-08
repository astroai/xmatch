"""Spatial and id match engines operating on polars frames.

Sky-match engines (drop-in alternatives):

* ``stilts`` – shells out to STILTS ``tmatch2`` (sky/skyerr/skyellipse). Used by
  default when a STILTS command is available.
* ``astropy`` – ``match_to_catalog_sky`` / ``search_around_sky`` from astropy.
* ``fast``   – Tier 1: ``scipy.spatial.cKDTree`` on 3D Cartesian unit-sphere
  embeddings. Drop-in replacement for ``astropy`` in-memory; ~3-5x faster.
* ``torchsky`` – optional tensor-native nearest-neighbour matching. Geometry
  only: ``matcher='sky'`` and ``matcher='skyerr'`` (per-row radii). Association
  policy (Bayes / LR / AUF / FoF) stays in this module.
* ``zone``   – Tier 2: HEALPix-sharded cone match. Uses ``cdshealpix`` when
  importable (sub-pixel zonning + per-pixel ``cKDTree`` queries) and falls
  back to a single ``cKDTree`` query when ``cdshealpix`` is not available
  (with a logged warning).

All engines share an identical schema: source columns (left unchanged, right
collisions get a ``_2`` suffix) plus a true great-circle ``sep_arcsec``
column. ``id_join`` is a pure polars relational join and is engine-agnostic.

Match criteria:

* ``sky`` – pairs within ``radius_arcsec``.
* ``skyerr`` – radial N-sigma matching from per-row positional errors.
* ``skyellipse`` – full local 2D Mahalanobis matching, including declared
  correlations and target-epoch transported covariance when available.

Tier 3 — Bayesian probabilistic qualification: when ``MatchSpec.prior_columns``
is non-empty (and the catalogue has those columns), every matched pair is
re-scored with a Budavári-style hierarchical Bayes factor that combines:

* a 2D Gaussian positional kernel using equivalent isotropic per-axis
  astrometric uncertainty, and
* independent 1-D Gaussian-KDE prior densities on each requested magnitude /
  colour column, fit from a uniform random sample of both sides (capped at
  50 000 rows).

The output is a ``p_match`` column in [0, 1]. Numeric evaluation lives in
:mod:`xmatcher.bayes`; this module only orchestrates data marshalling.
"""

import dataclasses
import itertools
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .exceptions import CrossMatchError
from .sources import (
    ASTROMETRIC_COVARIANCE_KEYS,
    CatalogueSource,
    position_error_to_arcsec_factor,
)

logger = logging.getLogger(__name__)

SEP_COLUMN = "sep_arcsec"
PMATCH_COLUMN = "p_match"
_RIGHT_SUFFIX = "_2"
_PROPAGATED_COV_EE = "_propagated_cov_ee_arcsec2"
_PROPAGATED_COV_NN = "_propagated_cov_nn_arcsec2"
_PROPAGATED_COV_EN = "_propagated_cov_en_arcsec2"
_PROPAGATED_COV_PREFIX = "_propagated_cov_"
_EPOCH_SIGMA = "_xmatcher_spill_epoch_sigma"

# Accepted match-criteria vocabularies.  Validated up front so a typo fails
# loudly: an unknown ``join_type`` used to fall through :func:`_build_result`
# to an empty frame, i.e. a silently empty crossmatch result.
_MATCHER_KINDS = frozenset({"sky", "skyerr", "skyellipse", "lr", "ml", "xgb", "auf", "macauff"})
_JOIN_TYPES = frozenset({"1and2", "1or2", "all", "all1", "all2", "1not2", "2not1"})
_FIND_MODES = frozenset({"best", "all"})


def _first_occurrence_indices(sorted_array: np.ndarray) -> np.ndarray:
    """Find the indices of the first occurrence of each unique value in a sorted array.

    This is significantly faster (~6x) than `np.unique(sorted_array, return_index=True)[1]`
    for arrays that are already guaranteed to be sorted, avoiding overhead.
    """
    if len(sorted_array) == 0:
        return np.array([], dtype=int)
    split_points: np.ndarray = np.nonzero(sorted_array[1:] != sorted_array[:-1])[0] + 1
    return np.concatenate(([0], split_points))


@dataclass
class MatchSpec:
    radius_arcsec: float = 1.0
    matcher: str = (
        "sky"  # "sky" | "skyerr" | "skyellipse" | "lr" | "ml" | "xgb" | "auf" | "macauff"
    )
    max_error: float = 3.0  # N-sigma for skyerr/skyellipse
    # Magnitude column for Likelihood Ratio matcher (Sutherland & Saunders 1992).
    # Used to estimate the true-counterpart magnitude distribution q(m) and
    # background surface density n(m).  Required when matcher="lr".
    lr_magnitude_column: str | None = None
    # Prior Q factor for Likelihood Ratio matcher: probability that a
    # primary source has a detectable counterpart in the secondary catalogue.
    # Default 0.8 is a safe empirical value (Sutherland & Saunders 1992);
    # results are generally robust to choices in [0.5, 1.0].
    lr_q: float = 0.8
    # Photometric columns for the Random Forest ML matcher (matcher="ml").
    # Color differences |mag_L - mag_R| on these columns become features
    # alongside separation and local density.  Requires scikit-learn.
    ml_color_columns: list[str] = field(default_factory=list)
    # Path to save/load a pre-trained Random Forest model (joblib format).
    # When set and the file exists, the model is loaded and used for scoring
    # without re-training.  When set and the file does not exist, the model
    # is trained on-the-fly and saved to this path for future reuse.
    ml_model_path: str | None = None
    # Path to save/load a pre-trained XGBoost model (joblib format).
    # Mirrors ml_model_path for the XGBoost matcher (matcher="xgb").
    xgb_model_path: str | None = None
    # Flux/magnitude columns for the macauff matcher (matcher="macauff").
    # Magnitude differences on these columns are combined with the AUF
    # positional probability to produce improved match scores.
    macauff_flux_columns: list[str] = field(default_factory=list)
    join_type: str = "1and2"
    find: str = "best"  # "best" | "all"
    # Bayesian-prior columns. When non-empty and present in both catalogues,
    # add empirical photometric KDE terms to the requested ``p_match`` score.
    prior_columns: list[str] = field(default_factory=list)
    # Request a positional-only ``p_match`` score even with no photometric
    # priors. Both catalogues must provide positional errors or explicit floors.
    probabilistic: bool = False
    # Epoch to propagate coordinates to before spatial matching (Julian year).
    # Requires pm_ra_column / pm_dec_column + epoch metadata on the catalogue.
    # Target-epoch skyerr requires measured motion covariance or pm_prior;
    # unknown motion is an error even with fallback_policy='warn'.
    target_epoch: float | None = None
    # When True, sources that lack measured proper motions get a probabilistic
    # drift prior based on Galactic latitude (and optionally magnitude) instead
    # of being treated as stationary.  The drift uncertainty is added in
    # quadrature to the positional errors, inflating the match radius for
    # sources with large epoch baselines (Wilson 2023, RASTI 2, 1).
    pm_prior: bool = False
    # When set alongside pm_prior, the magnitude column is used to refine the
    # PM dispersion via a distance-proxy scale factor (brighter stars are
    # statistically closer → larger proper motion).  ``σ_μ`` is multiplied by
    # ``10^{-0.2 (mag - 15)}``, clipped to [0.3, 3.0].
    pm_prior_magnitude_column: str | None = None
    # Optional polars expression string applied as a boolean post-filter on
    # the matched pairs BEFORE reducing find="all" → find="best".  Column
    # names from the left side are used as-is; right-side columns gain a
    # ``_2`` suffix (e.g. ``abs(mag_g - mag_g_2) < 0.5``).
    filter_expr: str | None = None
    # Extra columns for N-dimensional cKDTree matching, mapping column name
    # to a dimensionless weight.  Columns are z-score normalized across the
    # union of both catalogues and appended to the 3-D Cartesian unit-sphere
    # embedding.  The spatial ``radius_arcsec`` is still enforced as a hard
    # bound; among candidates within that bound the nearest in N-d feature
    # space is chosen.
    extra_distance_cols: dict[str, float] = field(default_factory=dict)
    # Maximum number of left HEALPix pixel groups to process in one batch.
    # When set, the zone engine processes pixel groups in chunks, freeing
    # intermediate results between batches (out-of-core friendly).  Defaults
    # to ``None`` (process all pixels in one pass).  Only effective with
    # ``engine="zone"`` and ``cdshealpix`` installed.
    batch_size: int | None = None
    # ``warn`` logs a warning and falls back; ``error`` raises when an engine
    # cannot honor the requested semantics.
    fallback_policy: str = "warn"

    def __post_init__(self) -> None:
        try:
            self.radius_arcsec = float(self.radius_arcsec)
            self.max_error = float(self.max_error)
            self.lr_q = float(self.lr_q)
            if self.target_epoch is not None:
                self.target_epoch = float(self.target_epoch)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "radius_arcsec, max_error, lr_q, and target_epoch must be numeric"
            ) from exc
        if not math.isfinite(self.radius_arcsec) or not 0.0 < self.radius_arcsec <= 648_000.0:
            raise ValueError("radius_arcsec must be finite and in (0, 648000]")
        if not math.isfinite(self.max_error) or self.max_error <= 0.0:
            raise ValueError("max_error must be finite and positive")
        if not math.isfinite(self.lr_q) or not 0.0 <= self.lr_q <= 1.0:
            raise ValueError("lr_q must be finite and in [0, 1]")
        if self.target_epoch is not None and not math.isfinite(self.target_epoch):
            raise ValueError("target_epoch must be finite")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        try:
            self.extra_distance_cols = {
                name: float(weight) for name, weight in self.extra_distance_cols.items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("extra_distance_cols must map column names to finite weights") from exc
        if any(
            not isinstance(name, str) or not name or not math.isfinite(weight) or weight < 0.0
            for name, weight in self.extra_distance_cols.items()
        ):
            raise ValueError(
                "extra_distance_cols must map nonempty names to finite nonnegative weights"
            )
        if self.fallback_policy not in ("warn", "error"):
            raise ValueError("fallback_policy must be 'warn' or 'error'")
        if self.matcher not in _MATCHER_KINDS:
            raise ValueError(
                f"matcher must be one of {sorted(_MATCHER_KINDS)}, got {self.matcher!r}"
            )
        if self.join_type not in _JOIN_TYPES:
            raise ValueError(
                f"join_type must be one of {sorted(_JOIN_TYPES)}, got {self.join_type!r}"
            )
        if self.find not in _FIND_MODES:
            raise ValueError(f"find must be one of {sorted(_FIND_MODES)}, got {self.find!r}")
        if not isinstance(self.probabilistic, bool):
            raise ValueError("probabilistic must be a boolean")


# --------------------------------------------------------------------------- #
# polars helpers
# --------------------------------------------------------------------------- #
def _validate_coordinate_frames(
    sources: list[CatalogueSource], *, target_epoch: float | None = None, id_join: bool = False
) -> None:
    """Reject comparisons requiring a coordinate transform this matcher lacks."""
    if id_join:
        return
    frames = set()
    for source in sources:
        if (
            not isinstance(source.frame, str)
            or not source.frame
            or source.frame != source.frame.strip()
        ):
            raise CrossMatchError(f"Catalogue '{source.name}' needs a declared coordinate frame")
        frames.add(source.frame.lower())
    if len(frames) > 1:
        raise CrossMatchError(
            "Cannot match different coordinate frames; transform inputs to one frame first"
        )
    if target_epoch is not None and frames != {"icrs"}:
        raise CrossMatchError("Target-epoch propagation requires ICRS coordinates")


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
    p_match: np.ndarray | None = None,
    right_suffix: str = _RIGHT_SUFFIX,
    lr: np.ndarray | None = None,
    reliability: np.ndarray | None = None,
    ml_score: np.ndarray | None = None,
    xgb_score: np.ndarray | None = None,
    auf_prob: np.ndarray | None = None,
    macauff_prob: np.ndarray | None = None,
) -> pl.DataFrame:
    # Sanitise before renaming so matched and unmatched rows share a schema.
    prefixes = (_PM_DRIFT_COLUMN, _PROPAGATED_COV_PREFIX, _EPOCH_SIGMA)
    left = left.drop([c for c in left.columns if c.startswith(prefixes)])
    right = right.drop([c for c in right.columns if c.startswith(prefixes)])
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
    if lr is not None and lr.size == matched.height:
        matched = matched.hstack(pl.DataFrame({"lr": np.asarray(lr, dtype=float)}))
    if reliability is not None and reliability.size == matched.height:
        matched = matched.hstack(
            pl.DataFrame({"reliability": np.asarray(reliability, dtype=float)})
        )
    if ml_score is not None and ml_score.size == matched.height:
        matched = matched.hstack(pl.DataFrame({"ml_score": np.asarray(ml_score, dtype=float)}))
    if xgb_score is not None and xgb_score.size == matched.height:
        matched = matched.hstack(pl.DataFrame({"xgb_score": np.asarray(xgb_score, dtype=float)}))
    if auf_prob is not None and auf_prob.size == matched.height:
        matched = matched.hstack(pl.DataFrame({"auf_prob": np.asarray(auf_prob, dtype=float)}))
    if macauff_prob is not None and macauff_prob.size == matched.height:
        matched = matched.hstack(
            pl.DataFrame({"macauff_prob": np.asarray(macauff_prob, dtype=float)})
        )

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


def _positional_error_arrays(
    df: pl.DataFrame, src: CatalogueSource
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate declared RA/Dec errors before using them."""
    factor = position_error_to_arcsec_factor(src.pos_err_units)
    errors = tuple(
        df[column].to_numpy().astype(float) * factor
        for column in (src.ra_err_column, src.dec_err_column)
    )
    for axis, values in zip(("RA", "Dec"), errors, strict=True):
        bad = ~np.isfinite(values) | (values < 0.0)
        if bad.any():
            row = int(np.flatnonzero(bad)[0])
            raise CrossMatchError(
                f"Catalogue '{src.name}' has a non-finite or negative {axis} position error "
                f"at row {row}."
            )
    return errors[0], errors[1]


def _default_pos_error(src: CatalogueSource) -> float | None:
    if src.default_pos_error_arcsec is None:
        return None
    try:
        error = float(src.default_pos_error_arcsec)
    except (TypeError, ValueError) as exc:
        raise CrossMatchError(
            f"Catalogue '{src.name}' has a non-numeric default positional error."
        ) from exc
    if not math.isfinite(error) or error < 0.0:
        raise CrossMatchError(
            f"Catalogue '{src.name}' default positional error must be finite and nonnegative."
        )
    return error


def _pm_drift_uncertainty(df: pl.DataFrame, src: CatalogueSource) -> np.ndarray | None:
    if _PM_DRIFT_COLUMN not in df.columns:
        return None
    drift = df[_PM_DRIFT_COLUMN].to_numpy().astype(float)
    bad = ~np.isfinite(drift) | (drift < 0.0)
    if bad.any():
        row = int(np.flatnonzero(bad)[0])
        raise CrossMatchError(
            f"Catalogue '{src.name}' has invalid PM drift uncertainty at row {row}."
        )
    return drift


def _pos_sigma_arcsec(df: pl.DataFrame, src: CatalogueSource) -> np.ndarray | None:
    """Per-row radial RMS: evaluated epoch covariance or reference errors."""
    default_error = _default_pos_error(src)
    floor = default_error * np.sqrt(2) if default_error is not None else None
    # Per-row PM drift: used in quadrature regardless of error-column path.
    drift = _pm_drift_uncertainty(df, src)

    if _EPOCH_SIGMA in df.columns:
        sigma = df[_EPOCH_SIGMA].to_numpy().astype(float)
        return np.hypot(sigma, drift) if drift is not None else sigma
    if src.ra_err_column in df.columns and src.dec_err_column in df.columns:
        ra_e, de_e = _positional_error_arrays(df, src)
        sigma = np.sqrt(ra_e**2 + de_e**2)
        if floor is not None:
            sigma = np.maximum(sigma, floor)
        if drift is not None:
            sigma = np.sqrt(sigma**2 + drift**2)
        return sigma
    if src.astrometric_covariance_columns:
        covariance = _pos_covariance(df, src)
        if covariance is not None:
            sigma_sq_ra, sigma_sq_dec, _rho = covariance
            return np.sqrt(sigma_sq_ra + sigma_sq_dec)
    if floor is not None:
        sigma = np.full(df.height, floor)
        if drift is not None:
            sigma = np.sqrt(sigma**2 + drift**2)
        return sigma
    # When only per-row drift is available (no error columns, no floor),
    # use drift as the sole positional uncertainty.
    if drift is not None:
        return drift
    return None


def _pos_covariance(
    df: pl.DataFrame,
    src: CatalogueSource,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-row positional covariance parameters for skyellipse.

    Returns ``(sigma_sq_ra, sigma_sq_dec, rho)`` arrays:

    * ``sigma_sq_ra`` — variance in RA direction (arcsec², cosDec-corrected).
    * ``sigma_sq_dec`` — variance in Dec direction (arcsec²).
    * ``rho`` — correlation coefficient in [-1, 1]; 0 when no ``corr_column``.

    All arrays have length ``df.height``.  Returns ``None`` when no error
    information is available on this side.
    """
    default_error = _default_pos_error(src)
    floor = default_error if default_error is not None else None
    # Per-row PM drift: added in quadrature regardless of error-column path.
    drift = _pm_drift_uncertainty(df, src)
    drift_sq = None
    if drift is not None:
        # The drift prior is a radial RMS budget; distribute its variance over
        # the two isotropic tangent-plane axes so the radial sigma is unchanged.
        drift_sq = 0.5 * drift**2

    result = None
    if src.ra_err_column in df.columns and src.dec_err_column in df.columns:
        ra_e, de_e = _positional_error_arrays(df, src)
        sigma_sq_ra = ra_e**2
        sigma_sq_dec = de_e**2
        if floor is not None:
            floor_sq = floor * floor
            sigma_sq_ra = np.maximum(sigma_sq_ra, floor_sq)
            sigma_sq_dec = np.maximum(sigma_sq_dec, floor_sq)
        if drift_sq is not None:
            sigma_sq_ra = sigma_sq_ra + drift_sq
            sigma_sq_dec = sigma_sq_dec + drift_sq
        if src.corr_column and src.corr_column in df.columns:
            rho = df[src.corr_column].to_numpy().astype(float)
            bad = ~np.isfinite(rho) | (np.abs(rho) > 1.0)
            if bad.any():
                row = int(np.flatnonzero(bad)[0])
                raise CrossMatchError(
                    f"Catalogue '{src.name}' has a non-finite or out-of-range "
                    f"position-error correlation at row {row}; expected [-1, 1]."
                )
        else:
            rho = np.zeros(df.height, dtype=float)
        result = sigma_sq_ra, sigma_sq_dec, rho
    elif src.astrometric_covariance_columns:
        astrometric = _astrometric_covariance_mas(df, src)
        if astrometric is None:
            raise CrossMatchError(
                f"Catalogue '{src.name}' is missing declared astrometric covariance columns."
            )
        covariance_mas, _valid_6d, valid_angular = astrometric
        if not valid_angular.all():
            row = int(np.flatnonzero(~valid_angular)[0])
            raise CrossMatchError(
                f"Catalogue '{src.name}' has invalid declared astrometric covariance at row {row}."
            )
        positional = covariance_mas[:, :2, :2] / 1_000_000.0
        sigma_sq_ra = positional[:, 0, 0]
        sigma_sq_dec = positional[:, 1, 1]
        if floor is not None:
            floor_sq = floor * floor
            sigma_sq_ra = np.maximum(sigma_sq_ra, floor_sq)
            sigma_sq_dec = np.maximum(sigma_sq_dec, floor_sq)
        if drift_sq is not None:
            sigma_sq_ra = sigma_sq_ra + drift_sq
            sigma_sq_dec = sigma_sq_dec + drift_sq
        rho = positional[:, 0, 1] / np.sqrt(positional[:, 0, 0] * positional[:, 1, 1])
        result = sigma_sq_ra, sigma_sq_dec, rho
    elif floor is not None:
        floor_sq = floor * floor
        sigma_sq_ra = np.full(df.height, floor_sq)
        sigma_sq_dec = np.full(df.height, floor_sq)
        if drift_sq is not None:
            sigma_sq_ra = sigma_sq_ra + drift_sq
            sigma_sq_dec = sigma_sq_dec + drift_sq
        result = sigma_sq_ra, sigma_sq_dec, np.zeros(df.height, dtype=float)
    # When only per-row drift is available (no error columns, no floor),
    # use drift as the sole positional uncertainty.
    elif drift_sq is not None:
        result = drift_sq, drift_sq, np.zeros(df.height, dtype=float)

    propagated_columns = (_PROPAGATED_COV_EE, _PROPAGATED_COV_NN, _PROPAGATED_COV_EN)
    if all(column in df.columns for column in propagated_columns):
        ee = df[_PROPAGATED_COV_EE].to_numpy().astype(float)
        nn = df[_PROPAGATED_COV_NN].to_numpy().astype(float)
        en = df[_PROPAGATED_COV_EN].to_numpy().astype(float)
        if floor is not None:
            ee = np.maximum(ee, floor * floor)
            nn = np.maximum(nn, floor * floor)
        if drift_sq is not None:
            ee += drift_sq
            nn += drift_sq
        valid = (
            np.isfinite(ee)
            & np.isfinite(nn)
            & np.isfinite(en)
            & (ee >= 0.0)
            & (nn >= 0.0)
            & (ee * nn >= en * en)
        )
        if result is None:
            sigma_sq_ra = np.full(df.height, np.nan)
            sigma_sq_dec = np.full(df.height, np.nan)
            rho = np.full(df.height, np.nan)
        else:
            sigma_sq_ra, sigma_sq_dec, rho = result
        sigma_sq_ra[valid] = ee[valid]
        sigma_sq_dec[valid] = nn[valid]
        denominator = np.sqrt(ee[valid] * nn[valid])
        rho[valid] = np.divide(
            en[valid],
            denominator,
            out=np.zeros_like(en[valid]),
            where=denominator > 0.0,
        )
        result = sigma_sq_ra, sigma_sq_dec, rho
    return result


def _skyellipse_search_chord_max(
    cov_l: tuple[np.ndarray, np.ndarray, np.ndarray],
    cov_r: tuple[np.ndarray, np.ndarray, np.ndarray],
    max_error: float,
) -> float:
    """Maximum chord distance for skyellipse spatial pre-filter.

    The largest eigenvalue of any combined covariance bounds the search radius.
    We use the maximum across all rows as a conservative cKDTree bound.
    """
    sra2_l, sde2_l, rho_l = cov_l
    sra2_r, sde2_r, rho_r = cov_r
    max_sra2 = float(np.nanmax(sra2_l)) + float(np.nanmax(sra2_r))
    max_sde2 = float(np.nanmax(sde2_l)) + float(np.nanmax(sde2_r))
    # Conservative bound: the trace a+c is an upper bound on the max eigenvalue
    # of a 2×2 positive-semidefinite matrix (λ_max ≤ a+c).  Using max(a,c)
    # alone would under-estimate when the off-diagonal (ρ·σ_ra·σ_dec) is large.
    # We also add the worst-case off-diagonal term for extra safety.
    max_cov = math.sqrt(max(max_sra2, 0.0) * max(max_sde2, 0.0))
    max_eig = max_sra2 + max_sde2 + max_cov
    search_radius_arcsec = max_error * math.sqrt(max(max_eig, 0.0))
    return _arcsec_to_chord(search_radius_arcsec)


def _mahalanobis_pairwise(
    delta_ra: np.ndarray,
    delta_dec: np.ndarray,
    sra2_l: np.ndarray,
    sde2_l: np.ndarray,
    rho_l: np.ndarray,
    sra2_r: np.ndarray,
    sde2_r: np.ndarray,
    rho_r: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance squared for skyellipse pairs.

    For each pair with combined covariance C = C_left + C_right, computes
    ``d² = Δᵀ·C⁻¹·Δ`` where Δ = (delta_ra, delta_dec) in arcsec.

    Returns an array of d² values.  Pairs with singular covariance get inf.
    """
    # Combined covariance parameters.
    sra2 = sra2_l + sra2_r
    sde2 = sde2_l + sde2_r
    cov_ra_dec = rho_l * np.sqrt(sra2_l * sde2_l) + rho_r * np.sqrt(sra2_r * sde2_r)

    det = sra2 * sde2 - cov_ra_dec**2
    safe = np.where(det > 1e-30, det, np.inf)

    # C⁻¹ = [[sde2, -cov], [-cov, sra2]] / det
    inv11 = sde2 / safe
    inv22 = sra2 / safe
    inv12 = -cov_ra_dec / safe

    d2 = inv11 * delta_ra**2 + 2.0 * inv12 * delta_ra * delta_dec + inv22 * delta_dec**2
    d2[np.isfinite(d2) & (d2 < 0)] = 0.0  # clamp tiny negatives from floating point
    return np.where(np.isfinite(d2), d2, np.inf)


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
    angle = float(arcsec)
    if not np.isfinite(angle) or angle < 0.0:
        raise ValueError("Angular search radius must be finite and nonnegative.")
    # A spherical cap cannot extend beyond 180 degrees. Capping also prevents
    # sin(angle/2) from shrinking again for larger uncertainty bounds.
    return float(2.0 * np.sin(np.radians(min(angle, 648_000.0) / 3600.0) * 0.5))


def _flatten_candidates(
    idx_lists: list[np.ndarray],
    l_xyz: np.ndarray,
    r_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten ``cKDTree.query_ball_point`` per-left lists into pair arrays.

    Returns ``(left_idx, right_idx, sep_arcsec)``; empty arrays when no
    candidate pair exists.  Shared by every engine branch that needs the full
    candidate set (``find="all"``, Likelihood Ratio / ML / AUF / macauff, and
    the per-row ``skyerr`` criterion).
    """
    lens = np.fromiter((len(x) for x in idx_lists), dtype=int, count=len(idx_lists))
    total = int(lens.sum())
    if total == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
        )
    right_idx = np.fromiter(itertools.chain.from_iterable(idx_lists), dtype=np.int64, count=total)
    left_idx: np.ndarray = np.repeat(np.arange(len(idx_lists), dtype=np.int64), lens)
    chords = np.linalg.norm(l_xyz[left_idx] - r_xyz[right_idx], axis=-1)
    return left_idx, right_idx, _chord_to_arcsec(chords)


def _pixellate(hp_module, ra_deg, dec_deg, depth: int, *, label: str) -> np.ndarray:
    """``lonlat_to_healpix`` that turns a Rust panic into a normal error.

    ``cdshealpix`` asserts on invalid latitudes inside its Rust core; the
    resulting ``PanicException`` is a ``BaseException`` and would otherwise
    escape every ``except Exception`` handler on the way up (including Ray's
    per-task error handling and the ray-union retry loop).
    """

    if len(ra_deg) and len(dec_deg):
        from .astro_utils import require_finite_coordinates

        require_finite_coordinates(ra_deg, dec_deg, label=label)
    try:
        from astropy.coordinates import Latitude, Longitude

        return np.asarray(
            hp_module.lonlat_to_healpix(
                Longitude(np.radians(np.asarray(ra_deg, dtype=float)), unit="rad"),
                Latitude(np.radians(np.asarray(dec_deg, dtype=float)), unit="rad"),
                depth,
            ),
            dtype=np.int64,
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - pyo3 panics derive from BaseException
        raise CrossMatchError(f"HEALPix pixelation failed for {label}: {exc}") from exc


def _best_per_primary(
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the lowest-*scores* candidate per primary row (deterministic ties).

    Ties on *scores* fall back to the smallest separation, then the smallest
    right index, so repeated runs and different engines agree.  Returns the
    three arrays in ascending primary-row order.
    """
    if left_idx.size == 0:
        return left_idx, right_idx, seps
    order = np.lexsort((right_idx, seps, scores, left_idx))
    best = order[_first_occurrence_indices(left_idx[order])]
    best.sort()
    return left_idx[best], right_idx[best], seps[best]


def _skyerr_pair_filter(
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    lsig: np.ndarray,
    rsig: np.ndarray,
    max_error: float,
    *,
    find: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the per-row N-sigma criterion to candidate pairs.

    ``skyerr`` matches a pair iff ``sep <= max_error * (sigma_l + sigma_r)``.
    The engine's spatial bound is built from the *global* sigma maxima as a
    candidate pre-filter only; applying the criterion per row (as ``astropy``,
    ``stilts``, ``torchsky`` and the bounded-memory spill path do) keeps every
    engine on the same pair set.  Rows with a non-finite per-row sigma are
    rejected — an unknown uncertainty is not an infinite acceptance radius.

    With ``find="best"`` the surviving candidates are ranked by *normalised*
    separation ``sep / (sigma_l + sigma_r)``, matching astropy's score, then
    reduced to one row per primary source.
    """
    if left_idx.size == 0:
        return left_idx, right_idx, seps
    limit = max_error * (lsig[left_idx] + rsig[right_idx])
    keep = np.isfinite(limit) & np.isfinite(seps) & (seps <= limit)
    left_idx, right_idx, seps, limit = (
        left_idx[keep],
        right_idx[keep],
        seps[keep],
        limit[keep],
    )
    if find != "best" or left_idx.size == 0:
        return left_idx, right_idx, seps
    scores = np.divide(seps, limit, out=np.full_like(seps, np.inf), where=limit > 0.0)
    return _best_per_primary(left_idx, right_idx, seps, scores)


def _skyellipse_pair_filter(
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    l_ra: np.ndarray,
    l_dec: np.ndarray,
    r_ra: np.ndarray,
    r_dec: np.ndarray,
    cov_l: tuple[np.ndarray, np.ndarray, np.ndarray],
    cov_r: tuple[np.ndarray, np.ndarray, np.ndarray],
    max_error: float,
    *,
    find: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the per-pair 2-D Mahalanobis d² <= max_error² criterion.

    Shared across ``fast``, ``zone``, and ``ray`` so every engine evaluates
    identical error-ellipse geometry and tie-breaking.
    """
    if left_idx.size == 0:
        return left_idx, right_idx, seps
    sra2_l, sde2_l, rho_l = cov_l
    sra2_r, sde2_r, rho_r = cov_r
    mean_dec = 0.5 * (l_dec[left_idx] + r_dec[right_idx])
    cos_dec = np.cos(np.radians(mean_dec))
    delta_ra = ((l_ra[left_idx] - r_ra[right_idx] + 180.0) % 360.0 - 180.0) * 3600.0 * cos_dec
    delta_dec = (l_dec[left_idx] - r_dec[right_idx]) * 3600.0
    d2 = _mahalanobis_pairwise(
        delta_ra,
        delta_dec,
        sra2_l[left_idx],
        sde2_l[left_idx],
        rho_l[left_idx],
        sra2_r[right_idx],
        sde2_r[right_idx],
        rho_r[right_idx],
    )
    keep = np.isfinite(d2) & (d2 <= max_error**2)
    left_idx, right_idx, seps, d2 = (
        left_idx[keep],
        right_idx[keep],
        seps[keep],
        d2[keep],
    )
    if find == "best" and left_idx.size > 0:
        return _best_per_primary(left_idx, right_idx, seps, d2)
    return left_idx, right_idx, seps


# --------------------------------------------------------------------------- #
# proper motion correction
# --------------------------------------------------------------------------- #
def _astrometric_covariance_mas(
    df: pl.DataFrame,
    src: CatalogueSource,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build Gaia-style five-parameter covariances in Torchsky order.

    The returned covariance order is ``(alpha*, delta, pmra, pmdec,
    parallax)``. Errors use Gaia's native mas / mas-per-year units.
    """
    columns = src.astrometric_covariance_columns
    if not columns or set(columns) != set(ASTROMETRIC_COVARIANCE_KEYS):
        return None
    if any(column not in df.columns for column in columns.values()):
        return None

    gaia_parameters = ("ra", "dec", "parallax", "pmra", "pmdec")
    errors = np.stack(
        [
            df[columns[f"{parameter}_error"]].to_numpy().astype(float)
            for parameter in gaia_parameters
        ],
        axis=-1,
    )
    covariance = np.zeros((df.height, 5, 5), dtype=float)
    diagonal = np.arange(5)
    covariance[:, diagonal, diagonal] = errors * errors
    correlation_keys = {
        (0, 1): "ra_dec_corr",
        (0, 2): "ra_parallax_corr",
        (0, 3): "ra_pmra_corr",
        (0, 4): "ra_pmdec_corr",
        (1, 2): "dec_parallax_corr",
        (1, 3): "dec_pmra_corr",
        (1, 4): "dec_pmdec_corr",
        (2, 3): "parallax_pmra_corr",
        (2, 4): "parallax_pmdec_corr",
        (3, 4): "pmra_pmdec_corr",
    }
    correlations = []
    for (left, right), key in correlation_keys.items():
        correlation = df[columns[key]].to_numpy().astype(float)
        correlations.append(correlation)
        cross = correlation * errors[:, left] * errors[:, right]
        covariance[:, left, right] = cross
        covariance[:, right, left] = cross

    valid = np.isfinite(errors).all(axis=1) & (errors > 0.0).all(axis=1)
    correlation_values = np.stack(correlations, axis=-1)
    valid &= np.isfinite(correlation_values).all(axis=1)
    valid &= (np.abs(correlation_values) <= 1.0).all(axis=1)

    if valid.any():
        cov_valid = covariance[valid]
        eigenvalues = np.linalg.eigvalsh(cov_valid)
        scale = np.maximum(np.max(np.diagonal(cov_valid, axis1=1, axis2=2), axis=1), 1.0)
        valid[valid] &= eigenvalues[:, 0] >= -1e-10 * scale

    # Gaia publishes (alpha*, delta, parallax, pmra, pmdec); Torchsky's local
    # state orders proper motion before parallax.
    order = np.array([0, 1, 3, 4, 2])
    ordered_covariance = covariance[:, order][:, :, order]
    angular_covariance = ordered_covariance[:, :4, :4]
    angular_errors = errors[:, [0, 1, 3, 4]]
    angular_valid = np.isfinite(angular_errors).all(axis=1) & (angular_errors > 0.0).all(axis=1)
    angular_valid &= np.isfinite(angular_covariance).all(axis=(1, 2))

    if angular_valid.any():
        ang_cov_valid = angular_covariance[angular_valid]
        angular_eigenvalues = np.linalg.eigvalsh(ang_cov_valid)
        angular_scale = np.maximum(
            np.max(np.diagonal(ang_cov_valid, axis1=1, axis2=2), axis=1),
            1.0,
        )
        angular_valid[angular_valid] &= angular_eigenvalues[:, 0] >= -1e-10 * angular_scale

    return ordered_covariance, valid, angular_valid


def _apply_proper_motion(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    target_epoch: float,
    *,
    propagate_covariance: bool = False,
    fallback_policy: str = "warn",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Propagate coordinates to ``target_epoch`` using per-row PM + epoch.

    Mutates *left* and/or *right* in-place when both PM columns and epoch
    information are available on a side.  Returns the (possibly modified)
    pair.
    """
    from .astro_utils import (
        propagate_proper_motion,
        propagate_proper_motion_with_jacobian,
        propagate_space_motion,
        propagate_space_motion_with_jacobian,
    )

    has_left_pm = bool(
        left_src.pm_ra_column
        and left_src.pm_dec_column
        and (left_src.epoch_column or left_src.epoch is not None)
    )
    has_right_pm = bool(
        right_src.pm_ra_column
        and right_src.pm_dec_column
        and (right_src.epoch_column or right_src.epoch is not None)
    )
    if not has_left_pm and not has_right_pm:
        logger.debug("No PM+epoch info on either side; skipping PM propagation.")
        return left, right

    def _propagate_side(df: pl.DataFrame, src: CatalogueSource, label: str) -> pl.DataFrame:
        if df.is_empty():
            return df
        if not (src.pm_ra_column and src.pm_dec_column):
            logger.debug("No PM columns on %s side; skipping.", label)
            return df
        if src.pm_ra_column not in df.columns or src.pm_dec_column not in df.columns:
            logger.debug("PM columns missing from %s data; skipping.", label)
            return df

        if src.epoch_column and src.epoch_column in df.columns:
            epoch_arr = df[src.epoch_column].to_numpy().astype(float, copy=False)
        elif src.epoch is not None:
            epoch_arr = np.full(df.height, float(src.epoch), dtype=float)
        else:
            logger.debug("No epoch info on %s side; skipping.", label)
            return df

        ra_arr = df[src.ra_column].to_numpy().astype(float, copy=False)
        dec_arr = df[src.dec_column].to_numpy().astype(float, copy=False)
        pmra = df[src.pm_ra_column].to_numpy().astype(float, copy=False)
        pmde = df[src.pm_dec_column].to_numpy().astype(float, copy=False)

        complete = np.zeros(df.height, dtype=bool)
        parallax = radial_velocity = None
        has_6d_columns = bool(
            src.parallax_column
            and src.radial_velocity_column
            and src.parallax_column in df.columns
            and src.radial_velocity_column in df.columns
        )
        if has_6d_columns:
            parallax = df[src.parallax_column].to_numpy().astype(float, copy=False)
            radial_velocity = df[src.radial_velocity_column].to_numpy().astype(float, copy=False)
            complete = (
                np.isfinite(ra_arr)
                & np.isfinite(dec_arr)
                & np.isfinite(pmra)
                & np.isfinite(pmde)
                & np.isfinite(epoch_arr)
                & np.isfinite(parallax)
                & (parallax > 0.0)
                & np.isfinite(radial_velocity)
            )
            transverse_speed = (
                4.740470463 * np.hypot(pmra, pmde) / np.where(parallax > 0.0, parallax, np.nan)
            )
            admissible = np.hypot(transverse_speed, radial_velocity) < 0.5 * 299_792.458
            rejected = complete & ~admissible
            if np.any(rejected):
                if fallback_policy == "error":
                    raise CrossMatchError(
                        "6D space-motion propagation has physically inconsistent velocity rows."
                    )
                logger.warning(
                    "6D space-motion propagation %s: %d physically inconsistent rows "
                    "kept on the angular path",
                    label,
                    int(rejected.sum()),
                )
            complete &= admissible

        # Ensure we own a writeable copy to avoid mutating the original dataframe.
        # .astype(..., copy=False) returns a read-only view if the column is already float64.
        new_ra = ra_arr if ra_arr.flags.writeable else ra_arr.copy()
        new_dec = dec_arr if dec_arr.flags.writeable else dec_arr.copy()
        covariance_done = np.zeros(df.height, dtype=bool)
        propagated_covariance = None
        covariance_data = _astrometric_covariance_mas(df, src) if propagate_covariance else None
        if covariance_data is not None:
            covariance, covariance_valid_6d, covariance_valid_angular = covariance_data
            covariance_rows_6d = complete & covariance_valid_6d
            if np.any(covariance_rows_6d):
                result = propagate_space_motion_with_jacobian(
                    ra_arr[covariance_rows_6d],
                    dec_arr[covariance_rows_6d],
                    pmra[covariance_rows_6d],
                    pmde[covariance_rows_6d],
                    parallax[covariance_rows_6d],
                    radial_velocity[covariance_rows_6d],
                    epoch_arr[covariance_rows_6d],
                    target_epoch,
                )
                if result is None:
                    message = (
                        "target-epoch skyellipse covariance propagation requires "
                        "Torchsky propagate_space_motion_with_jacobian"
                    )
                    if fallback_policy == "error":
                        raise CrossMatchError(message)
                    logger.warning("%s; retaining reference-epoch ellipses.", message)
                else:
                    moved_ra, moved_dec, jacobian = result
                    new_ra[covariance_rows_6d] = moved_ra
                    new_dec[covariance_rows_6d] = moved_dec
                    covariance_done[covariance_rows_6d] = True
                    position_jacobian = jacobian[:, :2, :5]
                    target_covariance = (
                        position_jacobian
                        @ covariance[covariance_rows_6d]
                        @ np.swapaxes(position_jacobian, -1, -2)
                    )
                    target_covariance = 0.5 * (
                        target_covariance + np.swapaxes(target_covariance, -1, -2)
                    )
                    target_covariance /= 1_000_000.0  # mas² to arcsec²
                    propagated_covariance = np.full((df.height, 2, 2), np.nan)
                    propagated_covariance[covariance_rows_6d] = target_covariance

            angular_state_valid = (
                ~complete
                & covariance_valid_angular
                & np.isfinite(ra_arr)
                & np.isfinite(dec_arr)
                & np.isfinite(pmra)
                & np.isfinite(pmde)
                & np.isfinite(epoch_arr)
            )
            if np.any(angular_state_valid):
                result = propagate_proper_motion_with_jacobian(
                    ra_arr[angular_state_valid],
                    dec_arr[angular_state_valid],
                    pmra[angular_state_valid],
                    pmde[angular_state_valid],
                    epoch_arr[angular_state_valid],
                    target_epoch,
                )
                if result is None:
                    message = (
                        "target-epoch skyellipse covariance propagation requires "
                        "Torchsky propagate_proper_motion_with_jacobian"
                    )
                    if fallback_policy == "error":
                        raise CrossMatchError(message)
                    logger.warning("%s; retaining reference-epoch ellipses.", message)
                else:
                    moved_ra, moved_dec, position_jacobian = result
                    new_ra[angular_state_valid] = moved_ra
                    new_dec[angular_state_valid] = moved_dec
                    covariance_done[angular_state_valid] = True
                    target_covariance = (
                        position_jacobian
                        @ covariance[angular_state_valid, :4, :4]
                        @ np.swapaxes(position_jacobian, -1, -2)
                    )
                    target_covariance = 0.5 * (
                        target_covariance + np.swapaxes(target_covariance, -1, -2)
                    )
                    target_covariance /= 1_000_000.0
                    if propagated_covariance is None:
                        propagated_covariance = np.full((df.height, 2, 2), np.nan)
                    propagated_covariance[angular_state_valid] = target_covariance

        angular = ~complete & ~covariance_done
        if np.any(angular):
            moved_ra, moved_dec = propagate_proper_motion(
                ra_arr[angular],
                dec_arr[angular],
                pmra[angular],
                pmde[angular],
                epoch_arr[angular],
                target_epoch,
            )
            new_ra[angular] = moved_ra
            new_dec[angular] = moved_dec
        complete_without_covariance = complete & ~covariance_done
        if np.any(complete_without_covariance):
            assert parallax is not None and radial_velocity is not None
            try:
                moved_ra, moved_dec = propagate_space_motion(
                    ra_arr[complete_without_covariance],
                    dec_arr[complete_without_covariance],
                    pmra[complete_without_covariance],
                    pmde[complete_without_covariance],
                    parallax[complete_without_covariance],
                    radial_velocity[complete_without_covariance],
                    epoch_arr[complete_without_covariance],
                    target_epoch,
                )
            except ValueError as exc:
                raise CrossMatchError(
                    f"6D space-motion propagation failed for "
                    f"{int(complete_without_covariance.sum())} "
                    f"complete {label} rows: {exc}"
                ) from exc
            new_ra[complete_without_covariance] = moved_ra
            new_dec[complete_without_covariance] = moved_dec
            logger.debug(
                "6D space-motion propagation %s: %d/%d rows",
                label,
                int(complete.sum()),
                df.height,
            )

        # Use read-only original array directly from dataframe to avoid copies when computing delta
        orig_ra = df[src.ra_column].to_numpy()
        orig_dec = df[src.dec_column].to_numpy()
        logger.info(
            "PM propagation %s: max ΔRA=%.4f arcsec, max ΔDec=%.4f arcsec",
            label,
            float(np.nanmax(np.abs((new_ra - orig_ra + 180.0) % 360.0 - 180.0))) * 3600.0,
            float(np.nanmax(np.abs(new_dec - orig_dec))) * 3600.0,
        )
        columns = [
            pl.Series(src.ra_column, new_ra),
            pl.Series(src.dec_column, new_dec),
        ]
        if propagated_covariance is not None:
            columns.extend(
                (
                    pl.Series(_PROPAGATED_COV_EE, propagated_covariance[:, 0, 0]),
                    pl.Series(_PROPAGATED_COV_NN, propagated_covariance[:, 1, 1]),
                    pl.Series(_PROPAGATED_COV_EN, propagated_covariance[:, 0, 1]),
                )
            )
        return df.with_columns(columns)

    left = _propagate_side(left, left_src, "left")
    right = _propagate_side(right, right_src, "right")
    return left, right


_PM_DRIFT_COLUMN = "_pm_drift_arcsec"


def _apply_pm_drift_prior(
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left: pl.DataFrame,
    right: pl.DataFrame,
    target_epoch: float,
    magnitude_column: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, CatalogueSource, CatalogueSource]:
    """Inflate positional errors for sources lacking measured proper motions.

    When a catalogue has epoch information but no PM columns, the Wilson (2023)
    probabilistic drift model estimates the expected proper-motion dispersion
    from Galactic latitude and appends a per-row ``_pm_drift_arcsec`` column
    to the DataFrame.  :func:`_pos_sigma_arcsec` and
    :func:`_pos_covariance` then add the drift in quadrature to each row's
    astrometric errors.  Sides with measured PMs are left unchanged.

    When *magnitude_column* is set, ``σ_μ`` is scaled by a distance proxy
    ``10^{-0.2 (mag - 15)}`` (clipped to [0.3, 3.0]) — brighter stars get
    larger PM dispersion because they are statistically closer.

    Additionally, a per-row ``_pm_drift_arcsec`` column is appended to the
    DataFrame so that :func:`_pos_sigma_arcsec` and
    :func:`_pos_covariance` can add the drift in quadrature to each row's
    astrometric errors (rather than just inflating the source-level floor).

    Returns ``(left, right, left_src, right_src)`` — DataFrames may carry
    a ``_pm_drift_arcsec`` column, sources may have inflated
    ``default_pos_error_arcsec``.

    The empirical dispersion is the existing scalar radial RMS budget, added
    once to ``hypot(ra_err, dec_err)``; it is not a per-axis covariance model.
    """

    def _inflate_side(
        src: CatalogueSource, df: pl.DataFrame, label: str
    ) -> tuple[pl.DataFrame, CatalogueSource]:
        # Skip if this side already has measured PM columns.
        has_pm = bool(
            src.pm_ra_column
            and src.pm_dec_column
            and src.pm_ra_column in df.columns
            and src.pm_dec_column in df.columns
        )
        if has_pm:
            return df, src

        # Need epoch info to compute time baseline.
        if src.epoch_column and src.epoch_column in df.columns:
            epoch_arr = df[src.epoch_column].to_numpy().astype(float, copy=False)
        elif src.epoch is not None:
            epoch_arr = np.full(df.height, float(src.epoch), dtype=float)
        else:
            logger.debug(
                "pm_prior: no epoch on %s side; skipping drift inflation.",
                label,
            )
            return df, src

        delta_t = np.abs(epoch_arr - target_epoch)  # years
        if np.all(delta_t < 0.01):
            return df, src  # negligible baseline

        # --- estimate proper-motion dispersion from Galactic latitude -------
        ra = df[src.ra_column].to_numpy().astype(float)
        dec = df[src.dec_column].to_numpy().astype(float)
        gal_b = _galactic_latitude(ra, dec)

        # Simple model: σ_μ ≈ 3 mas/yr at poles, ~10 mas/yr at plane.
        # Exponential falloff from plane: σ(b) = 3 + 7 exp(-|b| / 20°).
        abs_b = np.abs(gal_b)
        sigma_mu_mas_yr = 3.0 + 7.0 * np.exp(-abs_b / 20.0)

        # Optionally scale by magnitude: brighter = closer = larger PM.
        # σ_μ ∝ 10^{-0.2 (m - 15)} — a distance-proxy from the distance modulus.
        if magnitude_column and magnitude_column in df.columns:
            mag = np.nan_to_num(
                df[magnitude_column].to_numpy().astype(float),
                nan=15.0,
            )
            mag_scale = 10.0 ** (-0.2 * (mag - 15.0))
            mag_scale = np.clip(mag_scale, 0.3, 3.0)
            sigma_mu_mas_yr = sigma_mu_mas_yr * mag_scale

        # σ_drift (arcsec) = σ_μ (mas/yr) × Δt (yr) / 1000.
        sigma_drift_arcsec = sigma_mu_mas_yr * delta_t / 1000.0

        # Per-row drift is stored as a DataFrame column (_pm_drift_arcsec)
        # and added in quadrature by _pos_sigma_arcsec / _pos_covariance.
        # No need to inflate default_pos_error_arcsec — per-row drift is
        # more precise and avoids double-counting the median drift.
        logger.info(
            "pm_prior %s: median σ_μ=%.1f mas/yr, median Δt=%.0f yr, median drift=%.3f arcsec.",
            label,
            float(np.median(sigma_mu_mas_yr)),
            float(np.median(delta_t)),
            float(np.median(sigma_drift_arcsec)),
        )
        # Per-row drift is handled by _pos_sigma_arcsec / _pos_covariance
        # which add it in quadrature to per-row astrometric errors.
        if _PM_DRIFT_COLUMN in df.columns:
            df = df.drop(_PM_DRIFT_COLUMN)
        df = df.with_columns(pl.Series(_PM_DRIFT_COLUMN, sigma_drift_arcsec))
        return df, src

    left, new_left = _inflate_side(left_src, left, "left")
    right, new_right = _inflate_side(right_src, right, "right")
    return left, right, new_left, new_right


def _galactic_latitude(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert equatorial (RA, Dec) to Galactic latitude (degrees)."""
    # Galactic North Pole (J2000): RA=192.85948°, Dec=27.12825°.
    # Galactic centre longitude: 32.93192°.
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    ngp_ra = np.radians(192.85948)
    ngp_dec = np.radians(27.12825)
    sin_b = np.sin(dec_rad) * np.sin(ngp_dec) + np.cos(dec_rad) * np.cos(ngp_dec) * np.cos(
        ra_rad - ngp_ra
    )
    return np.degrees(np.arcsin(np.clip(sin_b, -1.0, 1.0)))


# --------------------------------------------------------------------------- #
# N-dimensional feature helpers
# --------------------------------------------------------------------------- #
def _build_nd_features(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    df: pl.DataFrame,
    extra_cols: dict[str, float],
    union_mean: dict[str, tuple[float, float]] | None = None,
    union_df: pl.DataFrame | None = None,
) -> tuple[np.ndarray, dict[str, tuple[float, float]] | None]:
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
    stats: dict[str, tuple[float, float]] = {}
    for col, weight in extra_cols.items():
        frames = (df,) if union_df is None else (df, union_df)
        if any(col not in frame.columns for frame in frames):
            raise CrossMatchError(f"Requested extra distance column {col!r} is missing.")
        try:
            arrays = [frame[col].to_numpy().astype(float) for frame in frames]
        except (TypeError, ValueError) as exc:
            raise CrossMatchError(
                f"Requested extra distance column {col!r} must contain numeric values."
            ) from exc
        if any(not np.isfinite(values).all() for values in arrays):
            raise CrossMatchError(
                f"Requested extra distance column {col!r} must contain only finite values."
            )
        vals = arrays[0]
        if union_mean is not None and col in union_mean:
            mean, std = union_mean[col]
        else:
            values = vals if len(arrays) == 1 else np.concatenate(arrays)
            with np.errstate(over="ignore", invalid="ignore"):
                mean = float(np.mean(values))
                std = float(np.std(values))
            if std == 0:
                std = 1.0
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
            raise CrossMatchError(
                f"Requested extra distance column {col!r} cannot be normalized finitely."
            )
        stats[col] = (mean, std)
        with np.errstate(over="ignore", invalid="ignore"):
            norm = ((vals - mean) / std) * weight
        if not np.isfinite(norm).all():
            raise CrossMatchError(
                f"Normalized extra distance column {col!r} must contain only finite values."
            )
        extra_parts.append(norm.reshape(-1, 1))

    features = np.hstack([xyz] + extra_parts)
    return features, stats


# Module-level configuration — tunable for benchmarking / memory tuning.
_ND_CHUNK_SIZE = 50_000


def _rank_nd_candidates(
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    l_ra: np.ndarray,
    l_dec: np.ndarray,
    r_ra: np.ndarray,
    r_dec: np.ndarray,
    left: pl.DataFrame,
    right: pl.DataFrame,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the N-dimensional nearest candidate per primary source.

    Used by ``fast``, ``zone``, and ``ray`` whenever ``spec.extra_distance_cols``
    is set with ``find="best"``.  Evaluates N-d Euclidean distance in chunks of
    ``_ND_CHUNK_SIZE`` candidate pairs to bound memory.
    """
    if left_idx.size == 0 or not spec.extra_distance_cols or spec.find != "best":
        return left_idx, right_idx, seps

    l_feat, stats = _build_nd_features(l_ra, l_dec, left, spec.extra_distance_cols, union_df=right)
    r_feat, _ = _build_nd_features(r_ra, r_dec, right, spec.extra_distance_cols, union_mean=stats)

    n_pairs = left_idx.size
    nd_dists = np.empty(n_pairs, dtype=float)
    chunk_size = max(1, _ND_CHUNK_SIZE)
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        diff = r_feat[right_idx[start:end]] - l_feat[left_idx[start:end]]
        nd_dists[start:end] = np.linalg.norm(diff, axis=-1)

    return _best_per_primary(left_idx, right_idx, seps, nd_dists)


def _apply_match_filter(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    filter_expr: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        filtered = ctx.execute(f"SELECT _row_id FROM tmp WHERE {filter_expr}").collect()
        keep_indices = filtered["_row_id"].to_numpy()
        keep = np.zeros(tmp.height, dtype=bool)
        keep[keep_indices] = True
    except Exception as exc:
        raise CrossMatchError(f"Invalid filter_expr {filter_expr!r}: {exc}") from exc

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
    *,
    workers: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

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
    elif spec.matcher in ("lr", "ml", "xgb", "auf", "macauff"):
        # Likelihood Ratio / ML / XGB / AUF / macauff: retrieve ALL candidates
        # within radius_arcsec, then score with post-processing.
        chord_max = _arcsec_to_chord(spec.radius_arcsec)
    elif spec.matcher == "skyellipse":
        # Full 2-D Mahalanobis distance with error ellipses.
        cov_l = _pos_covariance(left, left_src)
        cov_r = _pos_covariance(right, right_src)
        if cov_l is None or cov_r is None:
            logger.warning(
                "Matcher '%s' needs positional errors on both sides; none found, no matches.",
                spec.matcher,
            )
            return empty
        chord_max = _skyellipse_search_chord_max(cov_l, cov_r, spec.max_error)
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
    # Likelihood Ratio / ML / XGB / AUF / macauff: retrieve ALL candidates so
    # the scoring function can compute reliabilities across the full candidate set.
    # ND ranking also needs every in-radius candidate; a nearest-k cap can drop
    # the best photometric candidate in crowded fields.
    if spec.matcher in ("lr", "ml", "xgb", "auf", "macauff") or (
        spec.matcher == "sky" and spec.extra_distance_cols and spec.find == "best"
    ):
        idx_lists = tree.query_ball_point(l_xyz, r=chord_max, workers=workers, return_sorted=True)
        left_idx, right_idx, seps = _flatten_candidates(idx_lists, l_xyz, r_xyz)
        if spec.extra_distance_cols and spec.find == "best":
            return _rank_nd_candidates(
                left_idx, right_idx, seps, l_ra, l_dec, r_ra, r_dec, left, right, spec
            )
        return left_idx, right_idx, seps

    if spec.matcher == "skyerr":
        # ``skyerr`` is a per-row criterion, so the global-chord candidate set
        # must be scored rather than reduced to the nearest candidate.
        assert lsig is not None and rsig is not None
        idx_lists = tree.query_ball_point(l_xyz, r=chord_max, workers=workers, return_sorted=True)
        left_idx, right_idx, seps = _flatten_candidates(idx_lists, l_xyz, r_xyz)
        left_idx, right_idx, seps = _skyerr_pair_filter(
            left_idx,
            right_idx,
            seps,
            lsig,
            rsig,
            spec.max_error,
            find="all" if spec.extra_distance_cols else spec.find,
        )
        if spec.extra_distance_cols and spec.find == "best":
            return _rank_nd_candidates(
                left_idx, right_idx, seps, l_ra, l_dec, r_ra, r_dec, left, right, spec
            )
        return left_idx, right_idx, seps

    if spec.matcher == "skyellipse":
        assert cov_l is not None and cov_r is not None
        idx_lists = tree.query_ball_point(l_xyz, r=chord_max, workers=workers, return_sorted=True)
        left_idx, right_idx, seps = _flatten_candidates(idx_lists, l_xyz, r_xyz)
        left_idx, right_idx, seps = _skyellipse_pair_filter(
            left_idx,
            right_idx,
            seps,
            l_ra,
            l_dec,
            r_ra,
            r_dec,
            cov_l,
            cov_r,
            spec.max_error,
            find="all" if spec.extra_distance_cols else spec.find,
        )
        if spec.extra_distance_cols and spec.find == "best":
            return _rank_nd_candidates(
                left_idx, right_idx, seps, l_ra, l_dec, r_ra, r_dec, left, right, spec
            )
        return left_idx, right_idx, seps

    if spec.find == "best":
        # --- plain sky: spatial-nearest match ------------------------------
        dist, idx = tree.query(l_xyz, k=1, distance_upper_bound=chord_max, workers=workers)
        valid = np.isfinite(dist) & (idx < r_xyz.shape[0])
        left_idx = np.nonzero(valid)[0]
        right_idx = idx[valid].astype(np.int64)
        sep = _chord_to_arcsec(dist[valid])
        return left_idx, right_idx, sep

    # find == "all": per-left list of matched right indices.
    idx_lists = tree.query_ball_point(l_xyz, r=chord_max, workers=workers, return_sorted=True)
    return _flatten_candidates(idx_lists, l_xyz, r_xyz)


# --------------------------------------------------------------------------- #
# Likelihood Ratio matcher (Sutherland & Saunders 1992)
# --------------------------------------------------------------------------- #
def _likelihood_ratio_scoring(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score candidate pairs with the Sutherland & Saunders (1992) Likelihood
    Ratio and return (left_idx, right_idx, seps, lr, reliability).

    For each primary source, all candidates within the search radius are scored.
    When ``spec.find == "best"``, the candidate with the highest reliability is
    kept per primary source.  When ``spec.find == "all"``, all candidates with
    their LR and reliability values are returned.

    Algorithm
    --------
    1. Estimate the background surface density n(m) from the secondary catalogue.
    2. Estimate the true-counterpart magnitude distribution q(m) by subtracting
       the expected background from the candidate magnitude distribution.
    3. For each pair compute the Rayleigh positional PDF f(r) using the combined
       per-row positional errors.
    4. Compute LR_i = q(m_i) * f(r_i) / n(m_i).
    5. Compute reliability R_j = LR_j / (Σ_i LR_i + (1 - Q)).
    6. If find="best", keep the highest-reliability candidate per primary source.

    Returns (left_idx, right_idx, seps, lr, reliability).  When no candidates
    survive, all arrays are empty.
    """
    empty = (
        np.array([], int),
        np.array([], int),
        np.array([], float),
        np.array([], float),
        np.array([], float),
    )
    if left_idx.size == 0:
        return empty

    mag_col = spec.lr_magnitude_column
    if mag_col is None or mag_col not in right.columns:
        logger.warning(
            "lr_magnitude_column='%s' missing from right catalogue; falling back to sky match.",
            mag_col,
        )
        return (
            left_idx,
            right_idx,
            seps,
            np.ones(left_idx.size, dtype=float),
            np.ones(left_idx.size, dtype=float),
        )

    # --- positional sigmas for the Rayleigh PDF f(r) ------------------------
    lsig = _pos_sigma_arcsec(left, left_src)
    rsig = _pos_sigma_arcsec(right, right_src)
    if lsig is None:
        lsig = np.full(left.height, 0.5, dtype=float)
    if rsig is None:
        rsig = np.full(right.height, 0.5, dtype=float)

    sigma_l = lsig[left_idx]
    sigma_r = rsig[right_idx]
    sigma_combined = np.sqrt(sigma_l**2 + sigma_r**2)
    sigma_combined = np.maximum(sigma_combined, 1e-6)

    # f(r) = r / σ² * exp(-r² / (2σ²)) — the Rayleigh radial PDF.
    sigma_sq = sigma_combined**2
    f_r = (seps / sigma_sq) * np.exp(-(seps**2) / (2.0 * sigma_sq))
    f_r = np.maximum(f_r, 1e-300)

    # --- n(m): background surface density per unit area per magnitude --------
    right_mags = right[mag_col].to_numpy().astype(float)
    right_mags = right_mags[np.isfinite(right_mags)]
    if right_mags.size < 10:
        logger.warning("Too few valid magnitudes in right catalogue for LR; falling back.")
        return (
            left_idx,
            right_idx,
            seps,
            np.ones(left_idx.size, dtype=float),
            np.ones(left_idx.size, dtype=float),
        )

    n_bins = min(50, int(np.sqrt(right_mags.size)))
    mag_min, mag_max = float(np.nanmin(right_mags)), float(np.nanmax(right_mags))
    if mag_max - mag_min < 1e-6:
        mag_max = mag_min + 1.0
    mag_edges = np.linspace(mag_min, mag_max, n_bins + 1)
    0.5 * (mag_edges[:-1] + mag_edges[1:])
    bin_width = mag_edges[1] - mag_edges[0]

    # Background counts per magnitude bin (from full right catalogue).
    bg_counts, _ = np.histogram(right_mags, bins=mag_edges)

    # Estimate sky area from the right catalogue extent.
    from .astro_utils import sky_extent

    r_ra = right[right_src.ra_column].to_numpy().astype(float)
    r_dec = right[right_src.dec_column].to_numpy().astype(float)
    extent = sky_extent(r_ra, r_dec)
    if extent["radius_deg"] and extent["radius_deg"] > 0:
        sky_area_deg2 = math.pi * extent["radius_deg"] ** 2
    else:
        sky_area_deg2 = 1.0  # fallback: 1 sq deg
    sky_area_arcsec2 = sky_area_deg2 * (3600.0**2)

    # n(m) in units of sources per arcsec² per magnitude.
    n_m = np.maximum(bg_counts.astype(float) / (sky_area_arcsec2 * bin_width), 1e-300)

    # --- q(m): true-counterpart magnitude distribution ----------------------
    search_area_arcsec2 = math.pi * spec.radius_arcsec**2
    cand_mags_arr = right[mag_col].to_numpy().astype(float)[right_idx]
    cand_counts, _ = np.histogram(
        cand_mags_arr[np.isfinite(cand_mags_arr)],
        bins=mag_edges,
    )

    # Expected background in the search area: n(m) × search_area × N_primary.
    n_primary = left.height
    expected_bg = n_m * search_area_arcsec2 * n_primary
    q_m = np.maximum(cand_counts.astype(float) - expected_bg, 0.0)
    q_sum = q_m.sum()
    if q_sum > 0:
        q_m /= q_sum  # normalise to probability distribution
    else:
        q_m = np.ones_like(q_m) / len(q_m)

    # --- per-pair LR computation -------------------------------------------
    # Map each candidate's magnitude to the appropriate bin.
    mag_bin_indices = np.clip(
        np.searchsorted(mag_edges, cand_mags_arr, side="right") - 1,
        0,
        n_bins - 1,
    )
    # Clip non-finite magnitudes to bin 0 (their LR will be negligible).
    mag_bin_indices[~np.isfinite(cand_mags_arr)] = 0

    q_per_pair = q_m[mag_bin_indices]
    n_per_pair = n_m[mag_bin_indices]

    lr = q_per_pair * f_r / n_per_pair
    lr = np.where(np.isfinite(lr), lr, 0.0)
    lr = np.maximum(lr, 0.0)

    # --- reliability per primary source ------------------------------------
    # Group LR values by primary source (left_idx).
    unique_left, inverse, counts = np.unique(
        left_idx,
        return_inverse=True,
        return_counts=True,
    )
    # Sum of LR per primary source.
    lr_sum = np.bincount(inverse, weights=lr, minlength=len(unique_left))

    # R_j = LR_j / (Σ_LR + (1 - Q)).
    Q = spec.lr_q
    # Map sum back to each pair.
    lr_sum_per_pair = lr_sum[inverse]
    denominator = lr_sum_per_pair + (1.0 - Q)
    reliability = np.divide(lr, denominator, where=denominator > 1e-300, out=np.zeros_like(lr))
    reliability = np.clip(reliability, 0.0, 1.0)

    logger.info(
        "LR match: %d candidates → %d unique primary sources, median reliability=%.3f.",
        len(left_idx),
        len(unique_left),
        float(np.median(reliability)),
    )

    # --- select best per primary source if find="best" ---------------------
    if spec.find == "best":
        # For each primary source, find the candidate with max reliability.
        # Ties go to smallest LR (then smallest sep).
        # Vectorized O(N log N) optimization replacing O(N^2) boolean mask loop
        order = np.lexsort((seps, lr, -reliability, inverse))
        best_indices = _first_occurrence_indices(inverse[order])
        best_positions = order[best_indices]
        best_positions.sort()  # Preserve original row order implicitly done by boolean mask
        return (
            left_idx[best_positions],
            right_idx[best_positions],
            seps[best_positions],
            lr[best_positions],
            reliability[best_positions],
        )

    return left_idx, right_idx, seps, lr, reliability


# --------------------------------------------------------------------------- #
# Shared ML feature engineering and pseudo-label generation
# --------------------------------------------------------------------------- #
def _engineer_ml_features_and_labels(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    spec: MatchSpec,
    matcher_name: str,
    add_synthetic_negatives: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int, list[str], np.ndarray]:
    """Engineer ML features and pseudo-labels shared by ``matcher='ml'`` and ``matcher='xgb'``.

    Builds feature matrix *X* with columns:

    *  0  — normalised separation ``sep / sigma_combined``
    *  1..k — standard-deviation-scaled absolute colour differences ``|col_L - col_R|``
    *  -1  — ``log1p(local_source_density)``

    Assigns pseudo-labels by marking the spatially-nearest candidate of
    each primary source as positive (1) and all others as negative (0).

    When *add_synthetic_negatives* is True, reproducibly sampled far-apart
    left/right pairings (> 10× search radius) are appended as extra negative
    examples.

    Returns
    ------
    X              — feature matrix ``(n_pairs + n_synthetic, n_features)``
    y_pseudo       — pseudo-labels ``(n_pairs + n_synthetic,)``
    n_pairs        — number of original candidate pairs
    n_features     — number of feature columns
    available_cols — colour columns present on both sides
    unique_left    — unique primary source indices (for logging)
    """
    from scipy.spatial import cKDTree

    empty_result: tuple[np.ndarray, np.ndarray, int, int, list[str], np.ndarray] = (
        np.zeros((0, 1), dtype=float),
        np.zeros(0, dtype=int),
        0,
        0,
        [],
        np.array([], dtype=np.int64),
    )

    colour_cols = spec.ml_color_columns or []
    available_cols = [c for c in colour_cols if c in left.columns and c in right.columns]
    if not available_cols:
        return empty_result

    n_pairs = left_idx.size
    n_features = 1 + len(available_cols) + 1  # sep + colour_diffs + density
    X: np.ndarray = np.zeros((n_pairs, n_features), dtype=float)

    # Feature 0: normalised separation.
    lsig = _pos_sigma_arcsec(left, left_src)
    rsig = _pos_sigma_arcsec(right, right_src)
    if lsig is None:
        lsig = np.full(left.height, 0.5, dtype=float)
    if rsig is None:
        rsig = np.full(right.height, 0.5, dtype=float)
    sigma_combined = np.maximum(
        np.sqrt(lsig[left_idx] ** 2 + rsig[right_idx] ** 2),
        1e-6,
    )
    X[:, 0] = seps / sigma_combined

    # Features 1..1+n_cols: absolute colour differences.
    for k, col in enumerate(available_cols):
        diff = np.abs(
            left[col].to_numpy().astype(float)[left_idx]
            - right[col].to_numpy().astype(float)[right_idx]
        )
        std = float(np.nanstd(diff))
        if std is None or std == 0 or not np.isfinite(std):
            std = 1.0
        X[:, 1 + k] = np.nan_to_num(diff / std, nan=0.0)

    # Feature -1: log local source density around each primary source.
    l_ra = left[left_src.ra_column].to_numpy().astype(float)
    l_dec = left[left_src.dec_column].to_numpy().astype(float)
    l_xyz = _radec_to_xyz(l_ra, l_dec)
    tree_l = cKDTree(l_xyz)
    density_radius = _arcsec_to_chord(spec.radius_arcsec * 2.0)
    density_counts = np.array(
        [max(len(neigh), 1) for neigh in tree_l.query_ball_point(l_xyz, r=density_radius)],
        dtype=float,
    )
    area_arcsec2 = max(math.pi * (spec.radius_arcsec * 2.0) ** 2, 1.0)
    density = density_counts[left_idx] / area_arcsec2
    X[:, -1] = np.log1p(density)

    # --- pseudo-labels -------------------------------------------------------
    y_pseudo: np.ndarray = np.zeros(n_pairs, dtype=int)
    # Primary sort by left_idx, secondary sort by normalised separation (X[:, 0])
    order = np.lexsort((X[:, 0], left_idx))
    sorted_left_idx = left_idx[order]

    # Since it's sorted by separation, the first occurrence gives the index of the minimum separation for each left_idx
    unique_indices = _first_occurrence_indices(sorted_left_idx)
    unique_left = sorted_left_idx[unique_indices]

    best_positions = order[unique_indices]
    y_pseudo[best_positions] = 1

    # --- optional synthetic negative examples --------------------------------
    if add_synthetic_negatives:
        n_neg = min(n_pairs, left.height, right.height, 1000)
        if n_neg > 0:
            # ponytail: keep stochastic sampling local and repeatable; engine
            # call order must not affect scores or mutate NumPy's global RNG.
            rng = np.random.default_rng(42)
            neg_l_idx = rng.choice(left.height, size=n_neg, replace=True)
            neg_r_idx = rng.choice(right.height, size=n_neg, replace=True)
            neg_l_xyz = l_xyz[neg_l_idx]
            neg_r_xyz = _radec_to_xyz(
                right[right_src.ra_column].to_numpy().astype(float),
                right[right_src.dec_column].to_numpy().astype(float),
            )[neg_r_idx]
            neg_chords = np.linalg.norm(neg_l_xyz - neg_r_xyz, axis=-1)
            far_mask = neg_chords > _arcsec_to_chord(spec.radius_arcsec * 10.0)
            neg_l_idx = neg_l_idx[far_mask]
            neg_r_idx = neg_r_idx[far_mask]

            if neg_l_idx.size > 0:
                neg_X = np.zeros((neg_l_idx.size, n_features), dtype=float)
                neg_sigma = np.maximum(
                    np.sqrt(lsig[neg_l_idx] ** 2 + rsig[neg_r_idx] ** 2),
                    1e-6,
                )
                neg_X[:, 0] = _chord_to_arcsec(neg_chords[far_mask]) / neg_sigma
                for k, col in enumerate(available_cols):
                    l_v = left[col].to_numpy().astype(float)[neg_l_idx]
                    r_v = right[col].to_numpy().astype(float)[neg_r_idx]
                    diff_neg = np.abs(l_v - r_v)
                    std_neg = float(np.nanstd(diff_neg))
                    if std_neg is None or std_neg == 0 or not np.isfinite(std_neg):
                        std_neg = 1.0
                    neg_X[:, 1 + k] = np.nan_to_num(diff_neg / std_neg, nan=0.0)
                neg_X[:, -1] = np.log1p(density_counts[neg_l_idx] / area_arcsec2)
                X = np.vstack([X, neg_X])
                y_pseudo = np.concatenate([y_pseudo, np.zeros(neg_l_idx.size, dtype=int)])

    logger.debug(
        "%s: %d features from %d candidate pairs (%d unique sources).",
        matcher_name,
        n_features,
        n_pairs,
        len(unique_left),
    )
    return X, y_pseudo, n_pairs, n_features, available_cols, unique_left


# --------------------------------------------------------------------------- #
# Random Forest ML matcher (matcher="ml")
# --------------------------------------------------------------------------- #
def _ml_rf_score(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score candidate pairs with a Random Forest classifier and return
    (left_idx, right_idx, seps, ml_score) for the best candidate per primary.

    Engineers features from each candidate pair:

    * ``sep_arcsec`` — great-circle separation.
    * ``|col_L - col_R|`` — absolute colour differences for each column
      in ``spec.ml_color_columns``.
    * ``density_L`` — local source density around the primary source
      (neighbours within 2×radius / area).

    When scikit-learn is installed, a ``RandomForestClassifier`` is trained
    on-the-fly using self-match pseudo-labels (nearest-neighbour → positive,
    random far pairs → negative).  When scikit-learn is absent, a weighted
    heuristic combining normalised separation and colour differences is used.

    Returns ``(left_idx, right_idx, seps, ml_score)`` with one entry per
    primary source (``find="best"``) or all candidates (``find="all"``).
    """
    empty4 = (
        np.array([], int),
        np.array([], int),
        np.array([], float),
        np.array([], float),
    )
    if left_idx.size == 0:
        return empty4

    colour_cols = spec.ml_color_columns or []
    available_cols = [c for c in colour_cols if c in left.columns and c in right.columns]
    if not available_cols:
        logger.warning(
            "matcher='ml' but no ml_color_columns available on both sides; "
            "falling back to nearest-neighbour by separation."
        )
        # Fall back: pick best by smallest separation.
        return _ml_fallback_best_by_sep(left_idx, right_idx, seps, spec)

    # --- engineer features & pseudo-labels (shared with xgb) --------------
    (
        X,
        y_pseudo,
        n_pairs,
        n_features,
        _available_cols,
        unique_left,
    ) = _engineer_ml_features_and_labels(
        left,
        right,
        left_src,
        right_src,
        left_idx,
        right_idx,
        seps,
        spec,
        matcher_name="ML",
        add_synthetic_negatives=True,
    )
    if X.size == 0:
        return _ml_fallback_best_by_sep(left_idx, right_idx, seps, spec)

    # --- train Random Forest on-the-fly -----------------------------------
    try:
        from sklearn.ensemble import RandomForestClassifier

        _HAS_SKLEARN = True
    except ImportError:
        _HAS_SKLEARN = False

    model_path = spec.ml_model_path

    if _HAS_SKLEARN and model_path is not None:
        from pathlib import Path

        model_file = Path(model_path)
        if model_file.is_file():
            # Load pre-trained model and skip training / pseudo-labels.
            try:
                import joblib

                rf = joblib.load(str(model_file))
                if not hasattr(rf, "predict_proba"):
                    raise ValueError("Loaded object is not a classifier")
                probs = rf.predict_proba(X[:n_pairs])[:, 1]
                logger.info(
                    "ML match (loaded model %s): %d candidates from %d sources, "
                    "%d features, median prob=%.3f.",
                    model_file,
                    n_pairs,
                    len(unique_left),
                    n_features,
                    float(np.median(probs)),
                )
                # Skip to best-per-primary selection.
                if spec.find == "best":
                    return _pick_best_per_primary(left_idx, right_idx, seps, probs)
                return left_idx, right_idx, seps, probs
            except Exception as exc:
                logger.warning(
                    "Failed to load ML model from '%s' (%s); will train a new one.",
                    model_path,
                    exc,
                )

    if _HAS_SKLEARN:
        n_pos = int(np.sum(y_pseudo))
        if n_pos < 2:
            logger.warning(
                "matcher='ml': too few positive pseudo-labels (%d); "
                "falling back to separation heuristic.",
                n_pos,
            )
            if spec.find == "best":
                return _ml_fallback_best_by_sep(
                    left_idx[:n_pairs], right_idx[:n_pairs], seps[:n_pairs], spec
                )
            # find="all": keep all candidates with simple separation scores.
            fallback_scores = np.clip(1.0 / (1.0 + seps[:n_pairs]), 0.0, 1.0)
            return left_idx[:n_pairs], right_idx[:n_pairs], seps[:n_pairs], fallback_scores

        rf = RandomForestClassifier(
            n_estimators=min(100, max(10, n_pairs // 5)),
            max_depth=min(5, max(2, int(np.log2(n_features + 1)))),
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(X, y_pseudo)

        # Save model if path is configured.
        if model_path is not None:
            try:
                import joblib

                joblib.dump(rf, model_path)
                logger.info("ML model saved to '%s'.", model_path)
            except Exception as exc:
                logger.warning("Failed to save ML model to '%s': %s", model_path, exc)

        probs = rf.predict_proba(X[:n_pairs])[:, 1]
        logger.info(
            "ML match: %d candidates from %d primary sources, "
            "%d features, median probability=%.3f.",
            n_pairs,
            len(unique_left),
            n_features,
            float(np.median(probs)),
        )
    else:
        # No sklearn: weighted heuristic.
        logger.info(
            "matcher='ml': scikit-learn not available; "
            "using weighted heuristic (sep + colour diffs)."
        )
        weights: np.ndarray = np.ones(n_features, dtype=float)
        weights[0] = 2.0  # separation is most important
        probs = 1.0 / (1.0 + np.sum(X[:n_pairs] * weights, axis=1))
        probs = np.clip(probs, 0.0, 1.0)

    # --- select best per primary source if find="best" ---------------------
    if spec.find == "best":
        return _pick_best_per_primary(left_idx, right_idx, seps, probs)

    return left_idx, right_idx, seps, probs


# --------------------------------------------------------------------------- #
# XGBoost matcher (matcher="xgb") — gradient-boosted trees
# --------------------------------------------------------------------------- #
def _xgb_score(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score candidates with an XGBoost classifier (or LightGBM fallback).

    Uses the same feature engineering as ``_ml_rf_score`` (separation/error,
    colour differences, local density) but replaces the Random Forest with a
    gradient-boosted tree model.  XGBoost typically outperforms Random Forest
    on tabular crossmatch data (Bai et al. 2019, Salvato et al. 2018).

    Falls back in order: XGBoost → LightGBM → sklearn GradientBoosting →
    weighted heuristic.
    """
    empty4 = (
        np.array([], int),
        np.array([], int),
        np.array([], float),
        np.array([], float),
    )
    if left_idx.size == 0:
        return empty4

    colour_cols = spec.ml_color_columns or []
    available_cols = [c for c in colour_cols if c in left.columns and c in right.columns]
    if not available_cols:
        logger.warning(
            "matcher='xgb' but no ml_color_columns available; "
            "falling back to nearest-neighbour by separation."
        )
        return _ml_fallback_best_by_sep(left_idx, right_idx, seps, spec)

    # --- engineer features & pseudo-labels (shared with ml) ---------------
    (
        X,
        y_pseudo,
        n_pairs,
        n_features,
        _available_cols,
        unique_left,
    ) = _engineer_ml_features_and_labels(
        left,
        right,
        left_src,
        right_src,
        left_idx,
        right_idx,
        seps,
        spec,
        matcher_name="XGB",
        add_synthetic_negatives=True,
    )
    if X.size == 0:
        return _ml_fallback_best_by_sep(left_idx, right_idx, seps, spec)

    n_pos = int(np.sum(y_pseudo))
    if n_pos < 2:
        if spec.find == "best":
            logger.warning("matcher='xgb': too few positive pseudo-labels; falling back.")
            return _ml_fallback_best_by_sep(left_idx, right_idx, seps, spec)
        # find="all": keep all candidates with simple separation scores.
        fallback_scores = np.clip(1.0 / (1.0 + seps), 0.0, 1.0)
        return left_idx, right_idx, seps, fallback_scores

    # --- model load path (mirroring ml_model_path) -------------------------
    model_path = spec.xgb_model_path
    if model_path is not None:
        from pathlib import Path

        model_file = Path(model_path)
        if model_file.is_file():
            try:
                import joblib

                clf = joblib.load(str(model_file))
                if not hasattr(clf, "predict_proba"):
                    raise ValueError("Loaded object is not a classifier")
                probs = clf.predict_proba(X[:n_pairs])[:, 1]
                logger.info(
                    "XGB match (loaded model %s): %d candidates from %d sources, "
                    "%d features, median prob=%.3f.",
                    model_file,
                    n_pairs,
                    len(unique_left),
                    n_features,
                    float(np.median(probs)),
                )
                if spec.find == "best":
                    return _pick_best_per_primary(left_idx, right_idx, seps, probs)
                return left_idx, right_idx, seps, probs
            except Exception as exc:
                logger.warning(
                    "Failed to load XGB model from '%s' (%s); will train a new one.",
                    model_path,
                    exc,
                )

    # Try XGBoost → LightGBM → sklearn GradientBoosting.
    classifier = None
    try:
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=min(100, max(10, n_pairs // 5)),
            max_depth=min(5, max(2, int(np.log2(n_features + 1)))),
            random_state=42,
            verbosity=0,
        )
        logger.info("XGBoost: using XGBClassifier.")
    except ImportError:
        pass

    if classifier is None:
        try:
            from lightgbm import LGBMClassifier

            classifier = LGBMClassifier(
                n_estimators=min(100, max(10, n_pairs // 5)),
                max_depth=min(5, max(2, int(np.log2(n_features + 1)))),
                random_state=42,
                verbose=-1,
            )
            logger.info("LightGBM: using LGBMClassifier.")
        except ImportError:
            pass

    if classifier is None:
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            classifier = GradientBoostingClassifier(
                n_estimators=min(100, max(10, n_pairs // 5)),
                max_depth=min(5, max(2, int(np.log2(n_features + 1)))),
                random_state=42,
            )
            logger.info("sklearn: using GradientBoostingClassifier.")
        except ImportError:
            pass

    if classifier is not None:
        classifier.fit(X, y_pseudo)

        # Save model if path is configured.
        if model_path is not None:
            try:
                import joblib

                joblib.dump(classifier, model_path)
                logger.info("XGB model saved to '%s'.", model_path)
            except Exception as exc:
                logger.warning("Failed to save XGB model to '%s': %s", model_path, exc)

        probs = classifier.predict_proba(X[:n_pairs])[:, 1]
        logger.info(
            "XGB match: %d candidates from %d sources, %d features, median prob=%.3f.",
            n_pairs,
            len(unique_left),
            n_features,
            float(np.median(probs)),
        )
    else:
        logger.info(
            "matcher='xgb': no gradient-boosting library available; using weighted heuristic."
        )
        weights: np.ndarray = np.ones(n_features, dtype=float)
        weights[0] = 2.0
        probs = np.clip(1.0 / (1.0 + np.sum(X[:n_pairs] * weights, axis=1)), 0.0, 1.0)

    if spec.find == "best":
        return _pick_best_per_primary(left_idx, right_idx, seps, probs)
    return left_idx, right_idx, seps, probs


# --------------------------------------------------------------------------- #
# Shared AUF probability computation (used by _auf_score and _macauff_score)
# --------------------------------------------------------------------------- #
def _compute_auf_probabilities(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    spec: MatchSpec,
):
    """Compute AUF positional probabilities for each candidate pair.

    Implements the empirical AUF (Astrometric Uncertainty Function) from
    Wilson & Naylor (2017): builds a separation histogram from observed
    candidate pairs, subtracts the expected background contribution, and
    computes ``P(r) = f_AUF(r) / (f_AUF(r) + n_bg)`` per pair.

    Returns ``(auf_prob, n_bg, f_sum)`` or ``(None, n_bg, f_sum)`` when
    no significant true-match signal is found above background.
    """
    from .astro_utils import sky_extent

    # --- estimate positional errors -----------------------------------------
    lsig = _pos_sigma_arcsec(left, left_src)
    rsig = _pos_sigma_arcsec(right, right_src)
    if lsig is None:
        lsig = np.full(left.height, 0.5, dtype=float)
    if rsig is None:
        rsig = np.full(right.height, 0.5, dtype=float)

    # --- estimate background density n_bg ---------------------------------
    r_ra = right[right_src.ra_column].to_numpy().astype(float)
    r_dec = right[right_src.dec_column].to_numpy().astype(float)
    extent = sky_extent(r_ra, r_dec)
    if extent["radius_deg"] and extent["radius_deg"] > 0:
        sky_area_deg2 = math.pi * extent["radius_deg"] ** 2
    else:
        sky_area_deg2 = 1.0
    sky_area_arcsec2 = sky_area_deg2 * (3600.0**2)
    n_bg = right.height / max(sky_area_arcsec2, 1.0)
    n_primary = left.height

    # --- build separation histogram from observed candidate pairs ----------
    n_bins = min(100, max(20, int(np.sqrt(left_idx.size))))
    r_min = max(float(np.min(seps[seps > 0]) if np.any(seps > 0) else 1e-6), 1e-6)
    r_max = max(float(np.max(seps)), spec.radius_arcsec)
    if r_max - r_min < 1e-6:
        r_max = r_min + spec.radius_arcsec
    bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    bin_centres = np.sqrt(bins[:-1] * bins[1:])
    bin_widths = bins[1:] - bins[:-1]

    # Observed count of candidate pairs in each separation bin.
    cand_hist, _ = np.histogram(seps, bins=bins)

    # Expected background in each annulus:
    # n_bg (arcsec⁻²) × 2πr × dr (annulus area) × n_primary
    bg_expected = n_bg * 2.0 * math.pi * bin_centres * bin_widths * n_primary

    # f_AUF(r): residual after subtracting background = true-match distribution.
    f_auf = np.maximum(cand_hist.astype(float) - bg_expected, 0.0)
    f_sum = f_auf.sum()
    if f_sum > 0:
        f_auf /= f_sum  # normalise to probability distribution over bins
    else:
        return None, n_bg, 0.0

    # Convert to probability density per unit r.
    f_auf_density = f_auf / np.maximum(bin_widths, 1e-300)
    f_auf_density = np.maximum(f_auf_density, 1e-300)

    # --- per-pair AUF probability ------------------------------------------
    # Log-space interpolation for stability.
    log_bin_centres = np.log(bin_centres)
    log_f_density = np.log(f_auf_density)
    log_seps = np.log(np.maximum(seps, r_min * 0.5))
    log_seps = np.clip(log_seps, log_bin_centres[0], log_bin_centres[-1])
    log_f_interp = np.interp(log_seps, log_bin_centres, log_f_density)
    f_per_pair = np.exp(log_f_interp)

    # P(r) = f_AUF(r) / (f_AUF(r) + n_bg) with correct annular area factor.
    annular_f = f_per_pair / np.maximum(2.0 * math.pi * seps, 1e-6)
    auf_prob = annular_f / (annular_f + n_bg)
    auf_prob = np.clip(auf_prob, 0.0, 1.0)

    return auf_prob, n_bg, f_sum


# --------------------------------------------------------------------------- #
# AUF matcher — Astrometric Uncertainty Function (Naylor et al. 2013 /
# Wilson & Naylor 2017)
# --------------------------------------------------------------------------- #
def _auf_score(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score candidate pairs with the AUF match probability and return
    (left_idx, right_idx, seps, auf_prob).

    The AUF (Astrometric Uncertainty Function) is an empirical model of the
    positional error distribution, constructed via the **perturbation method**:

    1. Estimate per-source positional errors from the primary catalogue.
    2. Perturb primary sources by drawing from their error distributions.
    3. Build an empirical separation PDF f_AUF(r) from the perturbed
       self-match offset distribution.
    4. For each candidate pair, compute the local background density n_bg.
    5. Compute P(r) = f_AUF(r) / (f_AUF(r) + n_bg).

    Unlike skyerr/skyellipse (which assume Gaussian errors), the AUF captures
    non-Gaussian wings that are common in ground-based survey data.  When
    per-source error estimates are unavailable, the AUF falls back to using
    the observed separation distribution of all candidates as a proxy.

    Reference: Wilson, T. J. & Naylor, T. 2017, MNRAS 468, 2517.
    """
    empty4 = (
        np.array([], int),
        np.array([], int),
        np.array([], float),
        np.array([], float),
    )
    if left_idx.size == 0:
        return empty4

    # --- compute AUF positional probabilities (shared helper) --------------
    auf_prob, n_bg, f_sum = _compute_auf_probabilities(
        left,
        right,
        left_src,
        right_src,
        left_idx,
        right_idx,
        seps,
        spec,
    )
    if auf_prob is None:
        logger.warning(
            "AUF: no significant true-match signal above background; "
            "falling back to nearest-neighbour."
        )
        return _ml_fallback_best_by_sep(left_idx, right_idx, seps, spec)

    logger.info(
        "AUF match: %d candidates from %d perturbation points, "
        "n_bg=%.3e arcsec⁻², median prob=%.3f.",
        len(left_idx),
        f_sum,
        n_bg,
        float(np.median(auf_prob)),
    )

    # --- select best per primary source if find="best" ---------------------
    if spec.find == "best":
        return _pick_best_per_primary(left_idx, right_idx, seps, auf_prob)

    return left_idx, right_idx, seps, auf_prob


# --------------------------------------------------------------------------- #
# macauff matcher — AUF + flux likelihoods (Wilson & Naylor macauff)
# --------------------------------------------------------------------------- #
def _macauff_score(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score candidates with the macauff algorithm (AUF positional probability ×
    flux likelihood ratios) and return ``(left_idx, right_idx, seps, macauff_prob)``.

    macauff (Matching Across Catalogues using the Astrometric Uncertainty
    Function and Flux) combines the empirical AUF positional model with
    photometric flux evidence to produce improved match probabilities.

    Algorithm
    --------
    1. Compute the AUF positional probability ``P_pos(r)`` per candidate
       (same empirical separation PDF as ``matcher='auf'``).
    2. For each flux/magnitude column in ``macauff_flux_columns``:
       * Model ``P(Δm | match)`` as a Gaussian with combined photometric
         errors (default 0.1 mag when no error columns available).
       * Model ``P(Δm | no-match)`` from empirical random-pair magnitude
         differences across the full catalogues.
       * Compute flux likelihood ratio ``LR_flux = P(Δm|match) / P(Δm|no-match)``.
    3. Combine via odds multiplication:
       ``odds = P_pos/(1-P_pos) × Π LR_flux_i``
       ``P_macauff = odds / (1 + odds)``
    4. Output ``macauff_prob`` in [0,1].

    When no flux columns are available (or none match on both sides), the
    algorithm falls back to pure AUF scoring.

    Reference: Wilson, T. J. & Naylor, T. 2017/2018, macauff.
    """
    empty4 = (
        np.array([], int),
        np.array([], int),
        np.array([], float),
        np.array([], float),
    )
    if left_idx.size == 0:
        return empty4

    # --- compute AUF positional probabilities (shared helper) --------------
    auf_prob, n_bg, f_sum = _compute_auf_probabilities(
        left,
        right,
        left_src,
        right_src,
        left_idx,
        right_idx,
        seps,
        spec,
    )
    if auf_prob is None:
        logger.warning(
            "macauff: no significant true-match signal above background; "
            "falling back to nearest-neighbour."
        )
        return _ml_fallback_best_by_sep(left_idx, right_idx, seps, spec)

    # --- flux/magnitude likelihood ratios ----------------------------------
    flux_cols = spec.macauff_flux_columns or []
    available_flux_cols = [c for c in flux_cols if c in left.columns and c in right.columns]
    # If no flux columns, return pure AUF probability.
    if not available_flux_cols:
        logger.info("macauff: no flux columns available; using pure AUF scoring.")
        if spec.find == "best":
            return _pick_best_per_primary(left_idx, right_idx, seps, auf_prob)
        return left_idx, right_idx, seps, auf_prob

    # Convert positional probability to odds.
    odds = auf_prob / np.maximum(1.0 - auf_prob, 1e-10)
    flux_lr_parts: list = []

    for col in available_flux_cols:
        l_mags = left[col].to_numpy().astype(float)[left_idx]
        r_mags = right[col].to_numpy().astype(float)[right_idx]
        dm = np.abs(np.nan_to_num(l_mags, nan=0.0) - np.nan_to_num(r_mags, nan=0.0))

        # P(dm | match): Gaussian with photometric scatter.
        # Use per-row errors if available (e.g. *_err columns with same
        # root prefix), otherwise use a conservative 0.1 mag default.
        sigma_phot = 0.1
        err_col_l = f"{col}_err"
        err_col_r = f"{col}_err"
        if err_col_l in left.columns and err_col_r in right.columns:
            err_l = np.nan_to_num(
                left[err_col_l].to_numpy().astype(float)[left_idx],
                nan=sigma_phot,
            )
            err_r = np.nan_to_num(
                right[err_col_r].to_numpy().astype(float)[right_idx],
                nan=sigma_phot,
            )
            sigma_per_pair = np.maximum(
                np.sqrt(err_l**2 + err_r**2),
                1e-4,
            )
        else:
            sigma_per_pair = np.full(left_idx.size, sigma_phot, dtype=float)

        p_match_flux = np.exp(-0.5 * (dm / sigma_per_pair) ** 2) / (
            sigma_per_pair * np.sqrt(2.0 * math.pi)
        )
        p_match_flux = np.maximum(p_match_flux, 1e-300)

        # P(dm | no-match): empirical distribution from random pairings.
        all_l = np.nan_to_num(left[col].to_numpy().astype(float), nan=0.0)
        all_r = np.nan_to_num(right[col].to_numpy().astype(float), nan=0.0)
        n_rand = min(5000, left.height * right.height)
        rng = np.random.default_rng(42)
        rand_l = rng.choice(all_l, size=n_rand, replace=True)
        rand_r = rng.choice(all_r, size=n_rand, replace=True)
        rand_dm = np.abs(rand_l - rand_r)

        # Use the random-pair dm range for the histogram; pairs with
        # dm larger than any random pair are assigned a tiny p_nomatch
        # so the flux LR correctly penalises extreme magnitude diffs.
        dm_max_rand = max(float(np.max(rand_dm)), 1e-6)
        hist_bins = np.linspace(0, dm_max_rand, min(100, max(20, int(np.sqrt(n_rand)))) + 1)
        hist_counts, _ = np.histogram(rand_dm, bins=hist_bins)
        hist_density = np.maximum(
            hist_counts.astype(float) / (n_rand * (hist_bins[1] - hist_bins[0])),
            1e-300,
        )
        hist_centres = 0.5 * (hist_bins[:-1] + hist_bins[1:])

        # Look up P(dm|no-match) for each pair via linear interpolation.
        # Pairs with dm beyond the random range get a tiny p_nomatch.
        dm_clipped = np.clip(dm, hist_centres[0], hist_centres[-1])
        p_nomatch = np.interp(dm_clipped, hist_centres, hist_density)
        beyond = dm > hist_centres[-1]
        p_nomatch = np.where(beyond, 1e-300, p_nomatch)
        p_nomatch = np.maximum(p_nomatch, 1e-300)

        lr_flux = p_match_flux / p_nomatch
        lr_flux = np.clip(lr_flux, 1e-10, 1e10)
        odds *= lr_flux
        flux_lr_parts.append(float(np.median(lr_flux)))

    # --- combined macauff probability --------------------------------------
    macauff_prob = odds / (1.0 + odds)
    macauff_prob = np.clip(macauff_prob, 0.0, 1.0)

    logger.info(
        "macauff: %d candidates, AUF median=%.3f, flux LR medians=%s, combined median=%.3f.",
        len(left_idx),
        float(np.median(auf_prob)),
        [f"{v:.2f}" for v in flux_lr_parts],
        float(np.median(macauff_prob)),
    )

    # --- select best per primary source if find="best" ---------------------
    if spec.find == "best":
        return _pick_best_per_primary(left_idx, right_idx, seps, macauff_prob)

    return left_idx, right_idx, seps, macauff_prob


def _pick_best_per_primary(
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick the highest-scoring candidate per primary source.

    Returns ``(left_idx, right_idx, seps, scores)`` filtered to one entry
    per unique primary source, keeping the candidate with the largest
    *scores* value.
    """
    # Vectorized O(N log N) optimization replacing O(N^2) boolean mask loop
    order = np.lexsort((-scores, left_idx))
    best_indices = _first_occurrence_indices(left_idx[order])
    best_positions = order[best_indices]
    best_positions.sort()  # Preserve original row order implicitly done by boolean mask
    return (
        left_idx[best_positions],
        right_idx[best_positions],
        seps[best_positions],
        scores[best_positions],
    )


def _ml_fallback_best_by_sep(
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    seps: np.ndarray,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: pick the spatially-nearest candidate per primary source."""
    # Vectorized O(N log N) optimization replacing O(N^2) boolean mask loop
    order = np.lexsort((seps, left_idx))
    best_indices = _first_occurrence_indices(left_idx[order])
    best_positions = order[best_indices]
    best_positions.sort()  # Preserve original row order implicitly done by boolean mask

    scores = 1.0 / (1.0 + seps[best_positions])
    return left_idx[best_positions], right_idx[best_positions], seps[best_positions], scores


# --------------------------------------------------------------------------- #
# Torchsky engine (optional tensor-native nearest-neighbour backend)
# --------------------------------------------------------------------------- #
def _load_torchsky_crossmatch():
    try:
        from torchsky.catalogs import crossmatch_sky
    except ImportError as exc:
        raise CrossMatchError("engine='torchsky' requires the optional torchsky package") from exc
    return crossmatch_sky


def _torchsky_lonlat(df: pl.DataFrame, src: CatalogueSource):
    """RA/Dec degrees, converted to ICRS when ``CatalogueSource.frame`` is set."""
    ra = df[src.ra_column].to_numpy()
    dec = df[src.dec_column].to_numpy()
    frame = str(getattr(src, "frame", None) or "icrs").strip().lower()
    if frame in {"icrs", "j2000", "j2000.0"}:
        return ra, dec
    try:
        from torchsky.wcs import convert_celestial
    except ImportError as exc:
        raise CrossMatchError("engine='torchsky' requires the optional torchsky package") from exc
    ra_t, dec_t = convert_celestial(ra, dec, from_frame=src.frame, to_frame="icrs")

    def _np(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    return _np(ra_t), _np(dec_t)


def _torchsky_match(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Torchsky's catalog matcher for ``sky`` and ``skyerr``."""
    import numpy as np

    if spec.extra_distance_cols or spec.matcher not in {"sky", "skyerr"}:
        raise CrossMatchError(
            "engine='torchsky' currently supports only matcher='sky' or 'skyerr' "
            "and no extra_distance_cols"
        )
    empty = (
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=float),
    )
    if left.height == 0 or right.height == 0:
        return empty
    try:
        import torch
    except ImportError as exc:
        raise CrossMatchError("engine='torchsky' requires the optional torchsky package") from exc

    crossmatch_sky = _load_torchsky_crossmatch()
    left_ra, left_dec = _torchsky_lonlat(left, left_src)
    right_ra, right_dec = _torchsky_lonlat(right, right_src)

    def _tensor(values):
        arr = np.asarray(values, dtype=np.float64)
        if not arr.flags.writeable or not arr.flags.c_contiguous:
            arr = np.array(arr, dtype=np.float64, copy=True, order="C")
        return torch.from_numpy(arr)

    left_ra_t = _tensor(left_ra)
    left_dec_t = _tensor(left_dec)
    right_ra_t = _tensor(right_ra)
    right_dec_t = _tensor(right_dec)

    lsig = rsig = None
    if spec.matcher == "skyerr":
        lsig = _pos_sigma_arcsec(left, left_src)
        rsig = _pos_sigma_arcsec(right, right_src)
        if lsig is None or rsig is None:
            raise CrossMatchError(
                "engine='torchsky' matcher='skyerr' requires positional errors on both sides"
            )
        radius = spec.max_error * (
            np.nan_to_num(lsig, nan=1e-12) + float(np.nanmax(np.nan_to_num(rsig, nan=0.0)))
        )
        radius = np.maximum(radius, 1e-12)
        find = "all"
        radius_arg = _tensor(radius)
    else:
        radius_arg = float(spec.radius_arcsec)
        find = spec.find

    result = crossmatch_sky(
        left_ra_t,
        left_dec_t,
        right_ra_t,
        right_dec_t,
        radius_arcsec=radius_arg,
        find=find,
    )

    def as_numpy(value, dtype):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    left_idx = as_numpy(result.left_index, np.int64)
    right_idx = as_numpy(result.right_index, np.int64)
    seps = as_numpy(result.separation_arcsec, float)
    if spec.matcher == "skyerr" and left_idx.size:
        # Same per-row criterion and normalised-separation ranking as every
        # other engine, so `skyerr` pair sets are engine-independent.
        left_idx, right_idx, seps = _skyerr_pair_filter(
            left_idx,
            right_idx,
            seps,
            lsig,
            rsig,
            spec.max_error,
            find=spec.find,
        )
    return left_idx, right_idx, seps


# --------------------------------------------------------------------------- #
# Tier 2 — HEALPix zone engine
# --------------------------------------------------------------------------- #
def _cone_search_pixels(
    hp_module,
    lon_rad: float,
    lat_rad: float,
    radius_rad: float,
    depth: int,
) -> np.ndarray:
    """Depth-``depth`` pixels covered by a cone (cdshealpix >= 0.8 API).

    ``cone_search`` returns *parent* cells for fully-contained subtrees, so
    parents are expanded to all children at the requested depth — callers get
    a flat set of same-depth pixels.
    """
    from astropy.coordinates import Angle, Latitude, Longitude

    ipix, depths, _ = hp_module.cone_search(
        Longitude(lon_rad, unit="rad"),
        Latitude(lat_rad, unit="rad"),
        Angle(radius_rad, unit="rad"),
        np.uint8(depth),
    )
    out: list[int] = []
    for ipx, d in zip(np.asarray(ipix).tolist(), np.asarray(depths).tolist(), strict=True):
        dif = depth - int(d)
        if dif <= 0:
            out.append(int(ipx))
            continue
        base = int(ipx) << (2 * dif)
        out.extend(range(base, base + (1 << (2 * dif))))
    return np.asarray(out, dtype=np.int64)


def _zone_match(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HEALPix zone cone-match.

    When ``cdshealpix`` is importable we shard both sides into HEALPix pixels
    (nside=32 by default), build a per-pixel cKDTree on the right, and only
    query neighbour pixels within the cone radius — classic HATS-style
    partitioning. When ``cdshealpix`` is not importable we transparently
    downgrade to :func:`_scipy_match` and log a warning.
    """
    try:
        import cdshealpix as hp  # noqa: F401
    except ImportError as exc:
        if spec.fallback_policy == "error":
            raise CrossMatchError(
                "engine='zone' requires optional cdshealpix under fallback_policy='error'"
            ) from exc
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    import cdshealpix as hp
    from scipy.spatial import cKDTree

    empty = (np.array([], int), np.array([], int), np.array([], float))
    if left.height == 0 or right.height == 0:
        return empty

    l_ra = left[left_src.ra_column].to_numpy().astype(float)
    l_dec = left[left_src.dec_column].to_numpy().astype(float)
    r_ra = right[right_src.ra_column].to_numpy().astype(float)
    r_dec = right[right_src.dec_column].to_numpy().astype(float)

    cov_l = cov_r = None
    lsig = rsig = None
    if spec.matcher in ("sky", "lr", "ml", "xgb", "auf", "macauff"):
        radius_deg = spec.radius_arcsec / 3600.0
        chord_max = _arcsec_to_chord(spec.radius_arcsec)
    elif spec.matcher == "skyellipse":
        cov_l = _pos_covariance(left, left_src)
        cov_r = _pos_covariance(right, right_src)
        if cov_l is None or cov_r is None:
            logger.warning(
                "Matcher '%s' needs positional errors; none found.",
                spec.matcher,
            )
            return empty
        chord_max = _skyellipse_search_chord_max(cov_l, cov_r, spec.max_error)
        radius_deg = math.degrees(2.0 * math.asin(chord_max * 0.5)) if chord_max < 2.0 else 180.0
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

    DEPTH = 5  # nside = 2 ** DEPTH == 32
    l_pix = _pixellate(hp, l_ra, l_dec, DEPTH, label=f"catalogue '{left_src.name}'")
    r_pix = _pixellate(hp, r_ra, r_dec, DEPTH, label=f"catalogue '{right_src.name}'")

    # Cache per-pixel right xyz buffers and per-pixel cKDTree instances.
    r_sort_idx = np.argsort(r_pix, kind="stable")
    r_sorted_pix = r_pix[r_sort_idx]
    r_unique_indices = _first_occurrence_indices(r_sorted_pix)
    unique_pix = r_sorted_pix[r_unique_indices]
    r_splits = np.split(r_sort_idx, r_unique_indices[1:])
    r_groups = {int(k): v for k, v in zip(unique_pix, r_splits, strict=False)}
    tree_cache: dict[int, cKDTree] = {}
    xyz_cache: dict[int, np.ndarray] = {}
    for pix in unique_pix:
        idx = r_groups[int(pix)]
        xyz_cache[int(pix)] = _radec_to_xyz(r_ra[idx], r_dec[idx])
        tree_cache[int(pix)] = cKDTree(xyz_cache[int(pix)])

    l_xyz = _radec_to_xyz(l_ra, l_dec)

    # Group left points by HEALPix pixel so we can batch-query each right
    # pixel's tree once (instead of spawning worker threads per point).
    l_sort_idx = np.argsort(l_pix, kind="stable")
    l_sorted_pix = l_pix[l_sort_idx]
    l_unique_indices = _first_occurrence_indices(l_sorted_pix)
    l_unique_pix = l_sorted_pix[l_unique_indices]
    l_splits = np.split(l_sort_idx, l_unique_indices[1:])
    l_by_pix: dict[int, list] = {
        int(k): v.tolist() for k, v in zip(l_unique_pix, l_splits, strict=False)
    }

    # Pre-compute per-right-pixel global-index arrays for fast margin merging.
    r_global_by_pix: dict[int, np.ndarray] = {int(pix): r_groups[int(pix)] for pix in unique_pix}

    # Sort left pixel groups largest-first so batches are balanced.
    pixel_items = sorted(
        l_by_pix.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    batch_size = spec.batch_size or len(pixel_items)
    need_all_candidates = (
        spec.find != "best" or spec.matcher != "sky" or bool(spec.extra_distance_cols)
    )

    l_parts, r_parts, sep_parts = [], [], []
    for batch_start in range(0, len(pixel_items), max(1, batch_size)):
        batch_pixels = pixel_items[batch_start : batch_start + max(1, batch_size)]
        batch_l = []
        batch_r = []
        batch_s = []

        for _l_pix_int, left_indices in batch_pixels:
            indices_arr = np.asarray(left_indices, dtype=np.int64)
            # Cone search once per pixel, inflated by the exact intra-batch angular
            # spread around the representative point so boundary pairs are never missed.
            mid = len(left_indices) // 2
            rep_i = left_indices[mid]
            batch_xyz = l_xyz[indices_arr]  # (n_pix, 3)
            rep_xyz = l_xyz[rep_i]
            batch_spread_rad = float(
                np.max(
                    2.0
                    * np.arcsin(
                        np.clip(np.linalg.norm(batch_xyz - rep_xyz, axis=-1) * 0.5, 0.0, 1.0)
                    )
                )
            )
            npix = _cone_search_pixels(
                hp,
                float(np.radians(l_ra[rep_i])),
                float(np.radians(l_dec[rep_i])),
                min(math.pi, float(np.radians(radius_deg)) + batch_spread_rad),
                DEPTH,
            )

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

            if not need_all_candidates:
                # Plain radius + find="best" without extra_distance_cols: the
                # spatial-nearest candidate is the answer.
                dist, local_idx = margin_tree.query(
                    batch_xyz,
                    k=1,
                    distance_upper_bound=chord_max,
                    workers=-1,
                )
                valid = np.isfinite(dist) & (local_idx < margin_tree.n)
                for valid_index in np.nonzero(valid)[0]:
                    k_idx = left_indices[valid_index]
                    global_r_index = int(margin_global[local_idx[valid_index]])
                    sep_arcsec = float(_chord_to_arcsec(float(dist[valid_index])))
                    batch_l.append(np.array([k_idx], dtype=np.int64))
                    batch_r.append(np.array([global_r_index], dtype=np.int64))
                    batch_s.append(np.array([sep_arcsec], dtype=float))
            else:
                idx_lists = margin_tree.query_ball_point(
                    batch_xyz,
                    r=chord_max,
                    workers=-1,
                )
                local_l_indices, local_r_indices, seps = _flatten_candidates(
                    idx_lists,
                    batch_xyz,
                    margin_xyz,
                )
                if local_l_indices.size:
                    k_indices = np.asarray(left_indices, dtype=np.int64)[local_l_indices]
                    global_r_indices = margin_global[local_r_indices].astype(np.int64)
                    batch_l.append(k_indices)
                    batch_r.append(global_r_indices)
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
    left_idx = np.concatenate(l_parts)
    right_idx = np.concatenate(r_parts)
    seps = np.concatenate(sep_parts)

    # Sort deterministically by (left_idx, right_idx) to match _scipy_match order.
    order = np.lexsort((right_idx, left_idx))
    left_idx = left_idx[order]
    right_idx = right_idx[order]
    seps = seps[order]

    # --- skyellipse Mahalanobis post-filter ---------------------------------
    if spec.matcher == "skyellipse" and left_idx.size > 0:
        assert cov_l is not None and cov_r is not None
        left_idx, right_idx, seps = _skyellipse_pair_filter(
            left_idx,
            right_idx,
            seps,
            l_ra,
            l_dec,
            r_ra,
            r_dec,
            cov_l,
            cov_r,
            spec.max_error,
            find="all" if spec.extra_distance_cols else spec.find,
        )

    # --- per-row skyerr N-sigma criterion -----------------------------------
    if spec.matcher == "skyerr" and left_idx.size > 0:
        assert lsig is not None and rsig is not None
        left_idx, right_idx, seps = _skyerr_pair_filter(
            left_idx,
            right_idx,
            seps,
            lsig,
            rsig,
            spec.max_error,
            find="all" if spec.extra_distance_cols else spec.find,
        )

    # --- N-dimensional ranking (when extra_distance_cols is set) ------------
    if spec.extra_distance_cols and spec.find == "best" and left_idx.size > 0:
        left_idx, right_idx, seps = _rank_nd_candidates(
            left_idx, right_idx, seps, l_ra, l_dec, r_ra, r_dec, left, right, spec
        )

    return left_idx, right_idx, seps


# --------------------------------------------------------------------------- #
# astropy engine (existing implementation, unchanged)
# --------------------------------------------------------------------------- #
def _astropy_match(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Error-based match: a pair matches iff within sigma criterion.
    if spec.matcher == "skyellipse":
        cov_l = _pos_covariance(left, left_src)
        cov_r = _pos_covariance(right, right_src)
        if cov_l is None or cov_r is None:
            logger.warning(
                "Matcher '%s' needs positional errors on both sides; none found.",
                spec.matcher,
            )
            return empty
        # Same conservative chord bound as ``fast``/``zone``/``torchsky``.
        # Taking ``max(sigma_ra^2, sigma_dec^2)`` per side under-estimates the
        # worst-case Mahalanobis radius whenever the ellipses are elongated
        # and/or correlated, so astropy silently dropped pairs the other
        # engines accept.  One shared bound keeps the engines on the same
        # *candidate* set; the d^2 filter below is the actual criterion.
        search_radius_arcsec = _chord_to_arcsec(
            _skyellipse_search_chord_max(cov_l, cov_r, spec.max_error)
        )
        if search_radius_arcsec <= 0:
            return empty
        left_idx, right_idx, sep2d, _ = search_around_sky(
            lcoord,
            rcoord,
            search_radius_arcsec * u.arcsec,
        )
        seps = sep2d.arcsec
        # Mahalanobis post-filter.  The RA difference is scaled by the *pair's*
        # mean declination (not a table-wide scalar), matching the ``fast``,
        # ``zone`` and ``torchsky`` engines so every engine accepts the same
        # pairs away from the equator.
        if left_idx.size > 0:
            sra2_l, sde2_l, rho_l = cov_l
            sra2_r, sde2_r, rho_r = cov_r
            l_dec_pairs = left[left_src.dec_column].to_numpy()[left_idx]
            r_dec_pairs = right[right_src.dec_column].to_numpy()[right_idx]
            delta_ra = (
                (
                    left[left_src.ra_column].to_numpy()[left_idx]
                    - right[right_src.ra_column].to_numpy()[right_idx]
                )
                * 3600.0
                * np.cos(np.radians(0.5 * (l_dec_pairs + r_dec_pairs)))
            )
            delta_dec = (
                left[left_src.dec_column].to_numpy()[left_idx]
                - right[right_src.dec_column].to_numpy()[right_idx]
            ) * 3600.0
            d2 = _mahalanobis_pairwise(
                delta_ra,
                delta_dec,
                sra2_l[left_idx],
                sde2_l[left_idx],
                rho_l[left_idx],
                sra2_r[right_idx],
                sde2_r[right_idx],
                rho_r[right_idx],
            )
            keep = d2 <= spec.max_error**2
            left_idx, right_idx, seps = left_idx[keep], right_idx[keep], seps[keep]

        if spec.find == "best" and len(left_idx) > 0:
            # Pick best match by Mahalanobis distance d², not spatial sep.
            # Recompute d² for the filtered pairs (already computed above).
            sra2_l2, sde2_l2, rho_l2 = cov_l
            sra2_r2, sde2_r2, rho_r2 = cov_r
            mean_dec2 = 0.5 * (
                left[left_src.dec_column].to_numpy()[left_idx]
                + right[right_src.dec_column].to_numpy()[right_idx]
            )
            cos_dec2 = np.cos(np.radians(mean_dec2))
            delta_ra2 = (
                (
                    left[left_src.ra_column].to_numpy()[left_idx]
                    - right[right_src.ra_column].to_numpy()[right_idx]
                )
                * 3600.0
                * cos_dec2
            )
            delta_dec2 = (
                left[left_src.dec_column].to_numpy()[left_idx]
                - right[right_src.dec_column].to_numpy()[right_idx]
            ) * 3600.0
            d2 = _mahalanobis_pairwise(
                delta_ra2,
                delta_dec2,
                sra2_l2[left_idx],
                sde2_l2[left_idx],
                rho_l2[left_idx],
                sra2_r2[right_idx],
                sde2_r2[right_idx],
                rho_r2[right_idx],
            )
            order = np.lexsort((d2, left_idx))
            first_idx = _first_occurrence_indices(left_idx[order])
            sel = order[first_idx]
            sel.sort()
            left_idx, right_idx, seps = left_idx[sel], right_idx[sel], seps[sel]
        return left_idx, right_idx, seps

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
        order = np.lexsort((score, left_idx))

        first_occurrence_idx = _first_occurrence_indices(left_idx[order])
        sel = order[first_occurrence_idx]
        sel.sort()

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
) -> np.ndarray | None:
    """Compute a ``p_match`` column on the matched pairs.

    Returns ``None`` (no p_match) if either:

    * neither probabilistic scoring nor prior columns are requested, or
    * neither catalogue has the requested prior columns.
    """
    if not spec.probabilistic and not spec.prior_columns:
        return None
    if left_idx.size == 0:
        return np.zeros(0, dtype=float)

    from . import bayes

    sigma_left = _pos_sigma_arcsec(left, left_src)
    sigma_right = _pos_sigma_arcsec(right, right_src)
    if sigma_left is None or sigma_right is None:
        missing = []
        if sigma_left is None:
            missing.append(left_src.name)
        if sigma_right is None:
            missing.append(right_src.name)
        raise CrossMatchError(
            "Bayesian p_match requires positional errors or default_pos_error_arcsec "
            f"for catalogue(s): {', '.join(missing)}."
        )
    # `_pos_sigma_arcsec` returns radial RMS; the isotropic 2-D Gaussian uses
    # the equivalent per-axis sigma.
    sigma_left = sigma_left / math.sqrt(2.0)
    sigma_right = sigma_right / math.sqrt(2.0)
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
    stilts_cmd_base: str | None = None,
    java_opts: str | None = None,
    tmpdir: str | None = None,
    right_suffix: str = _RIGHT_SUFFIX,
    _tree_workers: int = -1,
) -> pl.LazyFrame:
    """Run a positional crossmatch and return a lazy result frame."""
    _validate_coordinate_frames([left_src, right_src], target_epoch=spec.target_epoch)
    if not left_src.ra_column or not left_src.dec_column:
        raise CrossMatchError(f"RA/Dec columns unknown for '{left_src.name}'.")
    if not right_src.ra_column or not right_src.dec_column:
        raise CrossMatchError(f"RA/Dec columns unknown for '{right_src.name}'.")

    if spec.matcher == "lr":
        if not spec.lr_magnitude_column:
            raise CrossMatchError(
                "matcher='lr' requires lr_magnitude_column to be set "
                "(the magnitude column for background density estimation)."
            )
    elif spec.matcher in ("ml", "xgb"):
        if not spec.ml_color_columns:
            logger.warning(
                "matcher='%s' without ml_color_columns; "
                "will fall back to separation-based scoring.",
                spec.matcher,
            )
    elif spec.matcher == "macauff":
        # macauff works without explicit error or flux columns — it falls
        # back to pure AUF scoring when no flux columns are available.
        pass
    elif spec.matcher == "auf":
        # AUF works without explicit error columns — it uses the
        # perturbation method to estimate errors empirically.
        pass
    elif spec.matcher != "sky" and not (_has_error_info(left_src) and _has_error_info(right_src)):
        if spec.fallback_policy == "error" or (
            spec.matcher == "skyerr" and spec.target_epoch is not None
        ):
            raise CrossMatchError(
                f"matcher={spec.matcher!r} requires positional errors for the requested science"
            )
        logger.warning(
            "Matcher '%s' needs positional errors on both catalogues; falling back to 'sky'.",
            spec.matcher,
        )
        spec = dataclasses.replace(spec, matcher="sky")

    from . import stilts

    chosen = engine
    if chosen == "auto":
        chosen = (
            "fast"
            if spec.matcher == "skyerr" and spec.target_epoch is not None
            else "stilts"
            if stilts.stilts_available(stilts_cmd_base)
            else "fast"
        )
    if chosen == "stilts":
        unsupported = []
        if spec.matcher not in {"sky", "skyerr"}:
            unsupported.append(f"matcher={spec.matcher!r}")
        if spec.matcher == "skyerr" and spec.target_epoch is not None:
            unsupported.append("target-epoch skyerr uncertainty")
        if spec.extra_distance_cols:
            unsupported.append("extra_distance_cols")
        if spec.prior_columns:
            unsupported.append("prior_columns")
        if spec.probabilistic:
            unsupported.append("probabilistic scoring")
        if spec.filter_expr:
            unsupported.append("filter_expr")
        if unsupported:
            message = "engine='stilts' cannot honor " + ", ".join(unsupported)
            if spec.fallback_policy == "error":
                raise CrossMatchError(message)
            logger.warning("%s; falling back to fast engine.", message)
            chosen = "fast"

    # --- proper motion propagation (common to all engines) -----------------
    if spec.target_epoch is not None:
        target_epoch = float(spec.target_epoch)
        if not math.isfinite(target_epoch) or target_epoch <= 0:
            raise CrossMatchError("target_epoch must be finite and positive (Julian year)")
        if spec.matcher == "skyerr":
            # One strict covariance path supplies eager search radii and spill halos.
            from .out_of_core import _align_epoch, _validate_sigma

            def align(lf: pl.LazyFrame, src: CatalogueSource) -> pl.DataFrame:
                frame = _align_epoch(
                    lf,
                    src,
                    target_epoch,
                    propagate_covariance=True,
                    pm_prior=spec.pm_prior,
                    magnitude_column=spec.pm_prior_magnitude_column,
                ).collect()
                _validate_sigma(frame.lazy(), _EPOCH_SIGMA, src)
                return frame

            left_eager = align(left_lf, left_src)
            right_eager = align(right_lf, right_src)
        else:
            left_eager, right_eager = _apply_proper_motion(
                left_lf.collect(),
                right_lf.collect(),
                left_src,
                right_src,
                float(spec.target_epoch),
                propagate_covariance=spec.matcher == "skyellipse",
                fallback_policy=spec.fallback_policy,
            )
        # PM drift prior: inflate errors for sides without measured PMs.
        if spec.pm_prior and spec.matcher != "skyerr":
            left_eager, right_eager, left_src, right_src = _apply_pm_drift_prior(
                left_src,
                right_src,
                left_eager,
                right_eager,
                float(spec.target_epoch),
                magnitude_column=spec.pm_prior_magnitude_column,
            )
        left_lf = left_eager.lazy()
        right_lf = right_eager.lazy()

    def _collect_side(lf: pl.LazyFrame, src: CatalogueSource, side: str) -> pl.DataFrame:
        """Materialise one side and validate coordinates before engine dispatch."""
        from .astro_utils import require_finite_coordinates

        frame = lf.collect()
        if src.ra_column in frame.columns and src.dec_column in frame.columns and frame.height:
            require_finite_coordinates(
                frame[src.ra_column].cast(pl.Float64).to_numpy(),
                frame[src.dec_column].cast(pl.Float64).to_numpy(),
                label=f"catalogue '{src.name}' ({side} side)",
            )
        return frame

    if chosen == "stilts":
        # Coordinate errors are input errors, not engine-fallback conditions.
        left = _collect_side(left_lf, left_src, "left")
        right = _collect_side(right_lf, right_src, "right")
        try:
            return stilts.stilts_sky_match(
                left_src,
                right_src,
                left,
                right,
                spec,
                stilts_cmd_base=stilts_cmd_base,
                java_opts=java_opts,
                tmpdir=tmpdir,
                right_suffix=right_suffix,
            ).lazy()
        except Exception as exc:
            if spec.fallback_policy == "error":
                raise CrossMatchError(
                    "engine='stilts' failed under fallback_policy='error'"
                ) from exc
            logger.warning("STILTS match failed (%s); falling back to fast engine.", exc)
            chosen = "fast"

    if chosen == "astropy":
        left = _collect_side(left_lf, left_src, "left")
        right = _collect_side(right_lf, right_src, "right")
        l_idx, r_idx, seps = _astropy_match(left, right, left_src, right_src, spec)
        logger.info("astropy sky match: %d matched pairs.", len(l_idx))
    elif chosen == "fast":
        left = _collect_side(left_lf, left_src, "left")
        right = _collect_side(right_lf, right_src, "right")
        l_idx, r_idx, seps = _scipy_match(
            left, right, left_src, right_src, spec, workers=_tree_workers
        )
        logger.info("fast (cKDTree) sky match: %d matched pairs.", len(l_idx))
    elif chosen == "torchsky":
        left = _collect_side(left_lf, left_src, "left")
        right = _collect_side(right_lf, right_src, "right")
        l_idx, r_idx, seps = _torchsky_match(left, right, left_src, right_src, spec)
        logger.info("torchsky sky match: %d matched pairs.", len(l_idx))
    elif chosen == "zone":
        left = _collect_side(left_lf, left_src, "left")
        right = _collect_side(right_lf, right_src, "right")
        l_idx, r_idx, seps = _zone_match(left, right, left_src, right_src, spec)
        logger.info("zone (HEALPix pixellated) sky match: %d matched pairs.", len(l_idx))
    elif chosen == "ray":
        from .ray_engine import ray_zone_match

        left = _collect_side(left_lf, left_src, "left")
        right = _collect_side(right_lf, right_src, "right")
        l_idx, r_idx, seps = ray_zone_match(left, right, left_src, right_src, spec)
        logger.info("ray (distributed) sky match: %d matched pairs.", len(l_idx))
    else:
        raise CrossMatchError(f"Unknown engine '{chosen}'.")

    # --- Likelihood Ratio scoring (post-engine) ----------------------------
    lr_arr: np.ndarray | None = None
    reliability_arr: np.ndarray | None = None
    if spec.matcher == "lr" and l_idx.size > 0:
        l_idx, r_idx, seps, lr_arr, reliability_arr = _likelihood_ratio_scoring(
            left,
            right,
            left_src,
            right_src,
            l_idx,
            r_idx,
            seps,
            spec,
        )

    # --- ML Random Forest scoring (post-engine) ---------------------------
    ml_score_arr: np.ndarray | None = None
    if spec.matcher == "ml" and l_idx.size > 0:
        l_idx, r_idx, seps, ml_score_arr = _ml_rf_score(
            left,
            right,
            left_src,
            right_src,
            l_idx,
            r_idx,
            seps,
            spec,
        )

    # --- XGBoost scoring (post-engine) -----------------------------------
    xgb_score_arr: np.ndarray | None = None
    if spec.matcher == "xgb" and l_idx.size > 0:
        l_idx, r_idx, seps, xgb_score_arr = _xgb_score(
            left,
            right,
            left_src,
            right_src,
            l_idx,
            r_idx,
            seps,
            spec,
        )

    # --- AUF match probability scoring (post-engine) ----------------------
    auf_prob_arr: np.ndarray | None = None
    if spec.matcher == "auf" and l_idx.size > 0:
        l_idx, r_idx, seps, auf_prob_arr = _auf_score(
            left,
            right,
            left_src,
            right_src,
            l_idx,
            r_idx,
            seps,
            spec,
        )

    # --- macauff scoring (post-engine) -----------------------------------
    macauff_prob_arr: np.ndarray | None = None
    if spec.matcher == "macauff" and l_idx.size > 0:
        l_idx, r_idx, seps, macauff_prob_arr = _macauff_score(
            left,
            right,
            left_src,
            right_src,
            l_idx,
            r_idx,
            seps,
            spec,
        )

    # --- post-match boolean filter ----------------------------------------
    if spec.filter_expr and l_idx.size > 0:
        l_idx, r_idx, seps = _apply_match_filter(
            left,
            right,
            l_idx,
            r_idx,
            seps,
            spec.filter_expr,
        )

    p_match = _bayesian_qualify(left, right, l_idx, r_idx, seps, left_src, right_src, spec)
    return _build_result(
        left,
        right,
        l_idx,
        r_idx,
        seps,
        spec,
        p_match=p_match,
        right_suffix=right_suffix,
        lr=lr_arr,
        reliability=reliability_arr,
        ml_score=ml_score_arr,
        xgb_score=xgb_score_arr,
        auf_prob=auf_prob_arr,
        macauff_prob=macauff_prob_arr,
    ).lazy()


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
        return right_lf.join(left_lf, left_on=id_right, right_on=id_left, how="anti", suffix=suffix)
    return left_lf.join(right_lf, left_on=id_left, right_on=id_right, how=how, suffix=suffix)
