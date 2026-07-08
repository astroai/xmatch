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
from typing import Optional

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
    ``(n_tuples,)`` with the RA (deg), Dec (deg), and positional uncertainty
    (arcsec) for that catalogue's contribution to every tuple.

    The spatial Bayes factor combines the positional agreement of all N
    catalogues into a single scalar per tuple (Budavári & Szalay 2008, eq. 25):

        B = (2π)^(N−1) · ∏(σ_i^−2) / (Σ σ_i^−2)^(N−1) · exp(−χ²/2)

    where χ² = Σ (Δ_i − Δ̂)² / σ_i², and Δ̂ is the inverse-variance weighted
    mean position.

    Returns log10(B) per tuple.
    """
    N = len(ras)
    if N < 2:
        return np.zeros(0, dtype=float)

    n_tuples = len(ras[0])
    if n_tuples == 0:
        return np.zeros(0, dtype=float)

    # Convert sigmas to inverse-variance weights (1/σ²).
    inv_var: list = []
    for s in sigmas:
        s_arcsec = np.asarray(s, dtype=float)
        safe_s = np.where(s_arcsec > 1e-6, s_arcsec, 1e-6)
        inv_var.append(1.0 / safe_s**2)

    # Weighted mean position (inverse-variance weighted).
    sum_w = np.zeros(n_tuples, dtype=float)
    ra_weighted = np.zeros(n_tuples, dtype=float)
    dec_weighted = np.zeros(n_tuples, dtype=float)
    for i in range(N):
        w = inv_var[i]
        sum_w += w
        ra_weighted += w * np.asarray(ras[i], dtype=float)
        dec_weighted += w * np.asarray(decs[i], dtype=float)

    ra_hat = ra_weighted / sum_w
    dec_hat = dec_weighted / sum_w

    # χ² = Σ (x_i − x̂)² / σ_i².
    # Convert RA differences to arcsec (cosDec-corrected).
    cos_dec = np.cos(np.radians(dec_hat))
    chi2 = np.zeros(n_tuples, dtype=float)
    for i in range(N):
        dra = (np.asarray(ras[i], dtype=float) - ra_hat) * 3600.0 * cos_dec
        ddec = (np.asarray(decs[i], dtype=float) - dec_hat) * 3600.0
        chi2 += (dra**2 + ddec**2) * inv_var[i]

    # Log Bayes factor: log10 B.
    # log B = (N−1)·log(2π) + Σ(−2·log σ_i) − (N−1)·log(Σ 1/σ_i²) − χ²/2
    # Then convert to log10.
    log_factor = (N - 1) * np.log(2.0 * math.pi)
    log_factor -= (N - 1) * np.log(sum_w)
    for i in range(N):
        log_factor += np.log(inv_var[i])  # log(1/σ²) = −2·log σ
    log_factor -= 0.5 * chi2

    return log_factor / math.log(10.0)


def compute_nway_p_match(
    ras: list,
    decs: list,
    sigmas: list,
    radius_arcsec: float,
    prior_columns: list = None,
) -> np.ndarray:
    """Compute N-way posterior match probability p ∈ [0, 1] per tuple.

    Combines the spatial Bayes factor with an optional photometric prior
    (same KDE-per-column approach as the pairwise Tier-3 qualifier).

    *ras*, *decs*, *sigmas* — per-catalogue arrays (each length n_tuples).
    *prior_columns* — if provided, a list of per-catalogue arrays for each
    photometric column; the match hypothesis uses the KDE at the midpoint,
    the background uses independent KDE draws.

    Returns p_match ∈ [0, 1] per tuple.
    """
    n_tuples = len(ras[0])
    if n_tuples == 0:
        return np.zeros(0, dtype=float)

    log10_B = compute_nway_bayes_factor(ras, decs, sigmas, radius_arcsec)

    # Convert log10 Bayes factor to log posterior odds.
    # p_match = 1 / (1 + 1/B) = B / (1 + B)
    # For large B, this saturates at 1.
    B = 10.0**log10_B  # B can overflow for well-matched tuples; clip.
    B = np.clip(B, 0.0, 1e300)

    if prior_columns:
        # Photometric likelihood ratio multiplies B.
        from scipy.stats import gaussian_kde

        for col_idx in range(len(prior_columns)):
            col_values = prior_columns[col_idx]  # list of N arrays
            combined = np.concatenate([np.asarray(v, dtype=float) for v in col_values])
            combined = combined[np.isfinite(combined)]
            if combined.size < 2:
                continue
            try:
                kde = gaussian_kde(combined, bw_method="scott")
                # Match hypothesis: KDE at the midpoint.
                midpoint = np.mean([np.asarray(v, dtype=float) for v in col_values], axis=0)
                log_match = kde.logpdf(midpoint)
                # Background: product of independent KDE draws.
                log_bg = np.zeros(n_tuples, dtype=float)
                for v in col_values:
                    log_bg += kde.logpdf(np.asarray(v, dtype=float))
                B *= np.exp(log_match - log_bg)
            except Exception:
                pass  # uniform prior if KDE fails

    B_safe = np.clip(B, 0.0, 1e300)
    p_match = B_safe / (1.0 + B_safe)
    return np.clip(p_match, 0.0, 1.0)
