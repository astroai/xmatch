"""Bayesian probabilistic cross-matching (Tier 3).

Implements a Budavári-style hierarchical Bayes factor for paired catalogue
sources. The model has two components:

* a 2-D Gaussian positional kernel with the joint (sigma_left + sigma_right)
  per-row astrometric uncertainty, and
* independent 1-D Gaussian-KDE photometric priors on user-supplied columns,
  fit on a uniform random sample of both full catalogues (capped at 50 000 rows).

The posterior probability of a true match is computed by normalising
``p(match | data)`` against ``p(background | data)`` via a softmax over the
two log-joints. The result is a ``p_match`` value in [0, 1] per matched pair.
These probabilities use equal prior odds and a uniform background density
inside the search disc; they are not calibrated for population prevalence or
local source density. Photometric KDE terms are empirical heuristics, not
calibrated physical Bayes factors. Positional covariance is reduced to an
isotropic per-axis sigma, so use ``skyellipse`` when covariance orientation matters.
"""

from __future__ import annotations

import logging
import math
from copy import copy

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_CAP_DEFAULT = 50_000


# --------------------------------------------------------------------------- #
# Positional log likelihoods
# --------------------------------------------------------------------------- #
def positional_log_likelihood(
    sep_arcsec: np.ndarray,
    sigma_left: np.ndarray,
    sigma_right: np.ndarray,
) -> np.ndarray:
    """Log ``p(sep | match)`` under a 2-D isotropic Gaussian.

    Both sigmas are equivalent isotropic per-axis errors in arcsec. The
    combined variance is
    ``sigma² = sigma_left² + sigma_right²`` (errors add in quadrature for
    independent 2-D Gaussians — see Budavári & Szalay 2008, eq. 16).

    The log-density of a 2-D isotropic Gaussian with variance σ² is
    ``-ψ²/(2σ²) - log(2π) - 2·log(σ)``.
    """
    sep = np.asarray(sep_arcsec, dtype=float)
    sig_l = np.asarray(sigma_left, dtype=float)
    sig_r = np.asarray(sigma_right, dtype=float)
    try:
        sep, sig_l, sig_r = np.broadcast_arrays(sep, sig_l, sig_r)
    except ValueError as exc:
        raise ValueError("separation and positional-error arrays must be broadcastable") from exc
    if not np.isfinite(sep).all() or (sep < 0.0).any():
        raise ValueError("separations must be finite and nonnegative")
    if (
        not np.isfinite(sig_l).all()
        or not np.isfinite(sig_r).all()
        or (sig_l < 0.0).any()
        or (sig_r < 0.0).any()
    ):
        raise ValueError("positional errors must be finite and nonnegative")
    # An exact zero measurement is allowed; retain a small explicit numerical
    # floor while rejecting corrupt negative or non-finite measurements.
    sigma_safe = np.maximum(np.hypot(sig_l, sig_r), 1e-6)
    return -0.5 * (sep / sigma_safe) ** 2 - np.log(2.0 * math.pi) - 2.0 * np.log(sigma_safe)


def background_log_likelihood(
    sep_arcsec: np.ndarray,
    radius_arcsec: float,
) -> np.ndarray:
    """Log ``p(sep | background)`` for a uniform-density disc of given radius."""
    radius = float(radius_arcsec)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("search radius must be finite and nonnegative")
    radius = max(radius, 1e-6)
    return np.full(
        np.asarray(sep_arcsec, dtype=float).shape,
        -math.log(math.pi * radius * radius),
        dtype=float,
    )


# --------------------------------------------------------------------------- #
# Bayes-factor aggregator
# --------------------------------------------------------------------------- #
def compute_p_match(
    sep_arcsec: np.ndarray,
    sigma_left: np.ndarray,
    sigma_right: np.ndarray,
    radius_arcsec: float,
    prior_log_match: np.ndarray | None = None,
    prior_log_bg: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``p_match`` in [0, 1] per matched pair.

    Budavári & Szalay (2008) hierarchical model:

    * ``prior_log_match`` — per-pair log-density of photometric column(s)
      under the *match* hypothesis.  Typically ``log KDE(midpoint)`` where
      ``midpoint = 0.5*(mag_left + mag_right)``.
    * ``prior_log_bg`` — per-pair log-density under the *background*
      hypothesis.  Typically ``log KDE(mag_left) + log KDE(mag_right)``
      (two independent draws from the population KDE).

    Both are summed across columns by the caller; an empty or ``None`` array
    means a uniform photometric prior for the corresponding hypothesis.
    """
    sep = np.asarray(sep_arcsec, dtype=float)
    if sep.size == 0:
        return np.zeros(0, dtype=float)

    post = positional_log_likelihood(sep, sigma_left, sigma_right)
    bg = background_log_likelihood(sep, radius_arcsec)

    log_match = post
    log_bg = bg
    if prior_log_match is not None and prior_log_match.size:
        log_match = log_match + prior_log_match
    if prior_log_bg is not None and prior_log_bg.size:
        log_bg = log_bg + prior_log_bg

    stacked = np.stack([log_match, log_bg], axis=-1)
    stacked -= stacked.max(axis=-1, keepdims=True)
    weights = np.exp(stacked)
    p_match = weights[..., 0] / weights.sum(axis=-1)
    return np.clip(p_match, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Budavári-correct KDE: fit on unconditional sample, evaluate at matched pairs.
# --------------------------------------------------------------------------- #
def fit_empirical_kde(
    values_left: np.ndarray,
    values_right: np.ndarray,
    *,
    sample_cap: int = _SAMPLE_CAP_DEFAULT,
    bandwidth: str | None = None,
):
    """Return a scipy gaussian_kde fit on the UNION of values_left and
    values_right (subsampled to ``sample_cap`` rows). The Tier-3 Bayesian
    qualifier models the *unconditional* background density over a
    photometric/colour column — Budavári et al. (2008) require sampling
    from the full population, not the matched-only subset.

    Returns ``None`` if KDE construction fails (degenerate input, all NaN,
    <2 unique values); callers should treat ``None`` as a uniform prior.
    """
    from scipy.stats import gaussian_kde

    combined = np.concatenate(
        [np.asarray(values_left, dtype=float), np.asarray(values_right, dtype=float)]
    )
    combined = combined[np.isfinite(combined)]
    if combined.size < 2:
        return None
    if combined.size > sample_cap:
        rng = np.random.default_rng(seed=0)
        combined = rng.choice(combined, size=sample_cap, replace=False)
    try:
        return gaussian_kde(combined, bw_method=bandwidth or "scott")
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("KDE construction failed (%s); using uniform prior.", exc)
        return None


def kde_log_at(kde, centre_values: np.ndarray) -> np.ndarray:
    """Return ``kde.logpdf(centre_values)`` with a uniform fallback when
    ``kde is None``. Output shape matches ``centre_values``."""
    centre = np.asarray(centre_values, dtype=float)
    if not np.isfinite(centre).all():
        raise ValueError("KDE evaluation values must be finite")
    if kde is None:
        return np.zeros(centre.shape, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        log_density = kde.logpdf(centre)
    if not np.isfinite(log_density).all():
        raise ValueError("KDE log-density must be finite for evaluated values")
    return log_density


def _writable_kde(kde):
    """Copy SciPy KDE arrays when a distributed object-store made them read-only."""
    array_attributes = (
        "dataset",
        "_weights",
        "_data_covariance",
        "_data_cho_cov",
        "covariance",
        "cho_cov",
    )
    if all(
        not isinstance(values := getattr(kde, name, None), np.ndarray) or values.flags.writeable
        for name in array_attributes
    ):
        return kde
    writable = copy(kde)
    for name in array_attributes:
        values = getattr(writable, name, None)
        if isinstance(values, np.ndarray) and not values.flags.writeable:
            setattr(writable, name, values.copy())
    return writable


# --------------------------------------------------------------------------- #
# N-way Bayesian multi-catalogue crossmatching (Budavári & Szalay 2008)
# --------------------------------------------------------------------------- #
def compute_nway_bayes_factor(
    ras: list,
    decs: list,
    sigmas: list,
    radius_arcsec: float,
) -> np.ndarray:
    """Compute the N-way spatial Bayes factor for a set of tuples.

    Each tuple consists of N sources (one per catalogue).  *ras*, *decs*, and
    *sigmas* are lists of length N, each element being an array of shape
    ``(n_tuples,)`` with the RA (deg), Dec (deg), and equivalent isotropic
    per-axis positional uncertainty (arcsec) for each contribution.

    The spatial Bayes factor combines the positional agreement of all N
    catalogues using Budavári & Szalay (2008), equation 18:

        B = 2^(N−1) · ∏w_i / Σw_i · exp(−Σ_{i<j} w_i w_j ψ_ij² / (2Σw_i))

    Here ``w_i = 1 / sigma_i²`` uses radians and ``ψ_ij`` is the pairwise
    great-circle separation in radians. ``radius_arcsec`` is only relevant to
    candidate selection; it does not enter the Bayes factor.

    Returns log10(B) per tuple.
    """
    N = len(ras)
    if N < 2:
        return np.zeros(0, dtype=float)
    try:
        radius = float(radius_arcsec)
    except (TypeError, ValueError) as exc:
        raise ValueError("radius_arcsec must be finite and positive") from exc
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_arcsec must be finite and positive")
    if len(decs) != N or len(sigmas) != N:
        raise ValueError("ras, decs, and sigmas must have the same catalogue count")

    ra_arrays = [np.asarray(values, dtype=float) for values in ras]
    dec_arrays = [np.asarray(values, dtype=float) for values in decs]
    sigma_arrays = [np.asarray(values, dtype=float) for values in sigmas]
    n_tuples = len(ra_arrays[0])
    if n_tuples == 0:
        return np.zeros(0, dtype=float)
    expected_shape = (n_tuples,)
    for ra, dec, sigma in zip(ra_arrays, dec_arrays, sigma_arrays, strict=True):
        if (
            ra.shape != expected_shape
            or dec.shape != expected_shape
            or sigma.shape != expected_shape
        ):
            raise ValueError("each RA, Dec, and sigma array must be one-dimensional per tuple")
        if not np.isfinite(ra).all() or not np.isfinite(dec).all():
            raise ValueError("N-way coordinates must be finite")
        if ((dec < -90.0) | (dec > 90.0)).any():
            raise ValueError("N-way declinations must lie in [-90, 90] degrees")
        if not np.isfinite(sigma).all() or (sigma <= 0.0).any():
            raise ValueError("N-way positional errors must be finite and positive")

    # Pairwise unit-vector geometry handles RA wrap and polar coordinates
    # without selecting a privileged mean RA.
    xyz = []
    for ra, dec in zip(ra_arrays, dec_arrays, strict=True):
        ra_rad, dec_rad = np.radians(ra), np.radians(dec)
        cos_dec = np.cos(dec_rad)
        xyz.append(
            np.stack(
                (cos_dec * np.cos(ra_rad), cos_dec * np.sin(ra_rad), np.sin(dec_rad)),
                axis=-1,
            )
        )

    arcsec_to_rad = math.radians(1.0 / 3600.0)
    log_weights = np.stack(
        [-2.0 * (np.log(sigma) + math.log(arcsec_to_rad)) for sigma in sigma_arrays], axis=0
    )
    max_log_weight = log_weights.max(axis=0)
    scaled_weights = np.exp(log_weights - max_log_weight)
    log_sum_weight = max_log_weight + np.log(scaled_weights.sum(axis=0))
    log_factor = (N - 1) * math.log(2.0) + log_weights.sum(axis=0) - log_sum_weight

    half_chi_squared = np.zeros(n_tuples, dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            chord = np.linalg.norm(xyz[i] - xyz[j], axis=-1)
            psi = 2.0 * np.arcsin(np.clip(chord * 0.5, 0.0, 1.0))
            nonzero = psi > 0.0
            log_term = np.full(n_tuples, -np.inf)
            log_term[nonzero] = (
                log_weights[i, nonzero]
                + log_weights[j, nonzero]
                - log_sum_weight[nonzero]
                + 2.0 * np.log(psi[nonzero])
                - math.log(2.0)
            )
            too_large = log_term >= math.log(np.finfo(float).max)
            term = np.zeros(n_tuples, dtype=float)
            term[too_large] = np.inf
            finite = np.isfinite(log_term) & ~too_large
            term[finite] = np.exp(log_term[finite])
            half_chi_squared += term

    return (log_factor - half_chi_squared) / math.log(10.0)


def compute_nway_p_match(
    ras: list,
    decs: list,
    sigmas: list,
    radius_arcsec: float,
    prior_columns: list | None = None,
    prior_kdes: list | None = None,
) -> np.ndarray:
    """Compute N-way posterior match probability p ∈ [0, 1] per tuple.

    Combines the spatial Bayes factor with an optional photometric prior
    (same KDE-per-column approach as the pairwise Tier-3 qualifier).

    *ras*, *decs*, *sigmas* — per-catalogue arrays (each length n_tuples).
    *prior_columns* — if provided, a list of per-catalogue arrays for each
    photometric column. Supply *prior_kdes* fitted on full catalogues to keep
    results independent of candidate batching. If omitted, this direct-array
    helper fits from the supplied values, which may be candidate-selected.
    The midpoint KDE ratio is an empirical heuristic, not a calibrated
    photometric Bayes factor.

    Returns p_match ∈ [0, 1] per tuple under equal prior odds. This score does
    not account for sky density or population prevalence; it is not a
    population-calibrated probability.
    """
    log10_B = compute_nway_bayes_factor(ras, decs, sigmas, radius_arcsec)
    n_tuples = log10_B.size
    if n_tuples == 0:
        return np.zeros(0, dtype=float)

    # Everything below stays in log space: p = sigmoid(ln B).  Forming
    # ``10**log10_B`` first overflowed to ``inf`` for well-matched tuples
    # (and was only rescued by a clip to 1e300), and multiplying by
    # ``exp(log_match - log_bg)`` could overflow again.
    log_odds = log10_B * math.log(10.0)

    if prior_columns:
        if prior_kdes is not None and len(prior_kdes) != len(prior_columns):
            raise ValueError("prior_kdes must have one entry per prior column")
        # Empirical midpoint KDE ratio multiplies B (adds in log space).
        for col_idx in range(len(prior_columns)):
            col_values = [np.asarray(v, dtype=float) for v in prior_columns[col_idx]]
            if len(col_values) != len(ras) or any(
                values.shape != (n_tuples,) for values in col_values
            ):
                raise ValueError("each prior column needs one array per catalogue and tuple")
            if any(not np.isfinite(values).all() for values in col_values):
                raise ValueError("photometric prior values must be finite")
            if prior_kdes is None:
                # Direct-array convenience only; orchestration should pass a KDE
                # fitted once from the unconditional full-catalogue population.
                kde = fit_empirical_kde(np.concatenate(col_values), np.zeros(0))
            else:
                kde = _writable_kde(prior_kdes[col_idx])
            if kde is None:
                continue

            midpoint = np.array(np.mean(col_values, axis=0), dtype=float, copy=True)
            with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
                log_match = kde.logpdf(midpoint)
                log_bg = np.zeros(n_tuples, dtype=float)
                for values in col_values:
                    log_bg += kde.logpdf(np.array(values, dtype=float, copy=True))
            delta = log_match - log_bg
            if (
                not np.isfinite(log_match).all()
                or not np.isfinite(log_bg).all()
                or not np.isfinite(delta).all()
            ):
                raise ValueError("photometric prior log-density must be finite")
            log_odds = log_odds + delta

    # Numerically stable sigmoid: never evaluates exp of a positive number.
    p_match = np.empty_like(log_odds, dtype=float)
    non_neg = log_odds >= 0
    p_match[non_neg] = 1.0 / (1.0 + np.exp(-log_odds[non_neg]))
    exp_pos = np.exp(log_odds[~non_neg])
    p_match[~non_neg] = exp_pos / (1.0 + exp_pos)
    return np.clip(p_match, 0.0, 1.0)
