"""Bayesian probabilistic cross-matching (Tier 3).

Implements a Budavári-style hierarchical Bayes factor for paired catalogue
sources. The model has two components:

* a 2-D Gaussian positional kernel with the joint (sigma_left + sigma_right)
  per-row astrometric uncertainty, and
* independent 1-D Gaussian-KDE photometric priors on user-supplied columns,
  fit on a uniform random sample of both sides (capped at 50 000 rows).

The posterior probability of a true match is computed by normalising
``p(match | data)`` against ``p(background | data)`` via a softmax over the
two log-joints. The result is a ``p_match`` value in [0, 1] per matched pair.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

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
    """Log ``p(sep | match)`` under independent Gaussian errors.

    Both sigmas are in arcsec. The combined ``Rayleigh``-style sigma is
    ``sqrt(sigma_left^2 + sigma_right^2)``. ``sep`` is in arcsec.
    """
    sep = np.asarray(sep_arcsec, dtype=float)
    sig_l = np.asarray(sigma_left, dtype=float)
    sig_r = np.asarray(sigma_right, dtype=float)
    sigma = np.sqrt(sig_l**2 + sig_r**2)
    sigma_safe = np.where(sigma > 0, sigma, 1e-3)
    return -0.5 * (sep / sigma_safe) ** 2 - np.log(sigma_safe) - 0.5 * math.log(2 * math.pi)


def background_log_likelihood(
    sep_arcsec: np.ndarray,
    radius_arcsec: float,
) -> np.ndarray:
    """Log ``p(sep | background)`` for a uniform-density disc of given radius."""
    radius = max(float(radius_arcsec), 1e-6)
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
    prior_log_arrays: List[np.ndarray],
) -> np.ndarray:
    """Compute ``p_match`` in [0, 1] per matched pair.

    Each entry in ``prior_log_arrays`` is the per-pair log-prior evaluated
    by :func:`kde_log_at` on a :func:`fit_empirical_kde` model for one
    photometric / colour column. An empty list means uniform priors.
    """
    if sep_arcsec.size == 0:
        return np.zeros(0, dtype=float)

    post = positional_log_likelihood(sep_arcsec, sigma_left, sigma_right)
    bg = background_log_likelihood(sep_arcsec, radius_arcsec)
    log_match = post + sum(prior_log_arrays)
    log_bg = bg + sum(prior_log_arrays)  # same prior on both sides cancels in ratio

    stacked = np.stack([log_match, log_bg], axis=-1)
    stacked -= stacked.max(axis=-1, keepdims=True)
    weights = np.exp(stacked)
    p_match = weights[..., 0] / weights.sum(axis=-1)
    return np.clip(p_match, 0.0, 1.0)


def bayes_factor_log10(
    sep_arcsec: np.ndarray,
    sigma_left: np.ndarray,
    sigma_right: np.ndarray,
    radius_arcsec: float,
) -> np.ndarray:
    """Return ``log10 B = log10(p_match / p_background)`` per pair.

    A diagnostic only — :func:`compute_p_match` is the user-facing score.
    """
    post = positional_log_likelihood(sep_arcsec, sigma_left, sigma_right)
    bg = background_log_likelihood(sep_arcsec, radius_arcsec)
    return (post - bg) / math.log(10.0)


# --------------------------------------------------------------------------- #
# Budavári-correct KDE: fit on unconditional sample, evaluate at matched pairs.
# --------------------------------------------------------------------------- #
def fit_empirical_kde(
    values_left: np.ndarray,
    values_right: np.ndarray,
    *,
    sample_cap: int = _SAMPLE_CAP_DEFAULT,
    bandwidth: Optional[str] = None,
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
    except Exception as exc:
        logger.warning("KDE construction failed (%s); using uniform prior.", exc)
        return None


def kde_log_at(kde, centre_values: np.ndarray) -> np.ndarray:
    """Return ``kde.logpdf(centre_values)`` with a uniform fallback when
    ``kde is None``. Output shape matches ``centre_values``."""
    centre = np.asarray(centre_values, dtype=float)
    if kde is None:
        return np.zeros(centre.shape, dtype=float)
    return kde.logpdf(centre)
