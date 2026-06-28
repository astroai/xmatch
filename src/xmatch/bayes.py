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
    """Log ``p(sep | match)`` under a 2-D isotropic Gaussian.

    Both sigmas are in arcsec.  The combined variance is
    ``sigma² = sigma_left² + sigma_right²`` (errors add in quadrature for
    independent 2-D Gaussians — Budavári & Szalay 2008, eq. 25).

    The log-density of a 2-D isotropic Gaussian with variance σ² is
    ``-ψ²/(2σ²) - log(2π) - 2·log(σ)``.
    """
    sep = np.asarray(sep_arcsec, dtype=float)
    sig_l = np.asarray(sigma_left, dtype=float)
    sig_r = np.asarray(sigma_right, dtype=float)
    sigma_sq = sig_l**2 + sig_r**2
    sigma_safe = np.where(sigma_sq > 0, np.sqrt(sigma_sq), 1e-3)
    return -0.5 * (sep / sigma_safe) ** 2 - np.log(2.0 * math.pi) - 2.0 * np.log(sigma_safe)


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
    prior_log_match: Optional[np.ndarray] = None,
    prior_log_bg: Optional[np.ndarray] = None,
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
    if sep_arcsec.size == 0:
        return np.zeros(0, dtype=float)

    post = positional_log_likelihood(sep_arcsec, sigma_left, sigma_right)
    bg = background_log_likelihood(sep_arcsec, radius_arcsec)

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
