# Crossmatch Algorithms in xmatch

A survey of positional and probabilistic crossmatch algorithms implemented in
xmatch, with references to the astronomical and computer science literature.

---

## Implemented Algorithms

### 1. Simple Positional (matcher="sky")

The oldest and most widely-used method: a pair matches if their great-circle
separation ≤ `radius_arcsec`.

- **Engines**: scipy cKDTree (`fast`), HEALPix-sharded (`zone`), Ray-distributed
  (`ray`), astropy, STILTS.
- **Reference**: This is the baseline method used in virtually every
  cross-identification study; the HEALPix partitioning follows Górski et al.
  (2005, *ApJ* 622, 759).

### 2. Sigma-Based Positional (matcher="skyerr")

A pair matches when `sep ≤ N·(σ_left + σ_right)`, where σ is the
quadrature-sum of per-row RA/Dec positional errors. This adapts the search
radius per source, accepting larger separations for poorly-constrained sources.

- **Engines**: all engines except id_join.
- **Reference**: Common in Gaia-era catalogues with per-row astrometric errors
  (Lindegren et al. 2018, *A&A* 616, A2).

### 3. Sky Ellipse / Mahalanobis (matcher="skyellipse")

Models each source position as a 2-D Gaussian with covariance matrix derived
from `ra_error`, `dec_error`, and `ra_dec_corr`. Computes the Mahalanobis
distance `d² = Δᵀ·C⁻¹·Δ` where C = C_left + C_right. A pair matches when
`d² ≤ max_error²`. Queries k>1 candidates within a conservative spatial bound
and picks the best by Mahalanobis d².

- **Engines**: fast, astropy, zone.
- **Reference**: Pineau et al. (2017, *A&A* 597, A89) — the "χ-match"
  probabilistic framework with full covariance modelling.

### 4. Likelihood Ratio (matcher="lr")

Implements the Sutherland & Saunders (1992) method designed for multi-wavelength
counterpart identification (e.g. radio→optical, X-ray→IR). Computes:

- `f(r)` — Rayleigh radial PDF using combined per-row positional errors
- `q(m)` — true-counterpart magnitude distribution, estimated by subtracting
  the expected background from the candidate magnitude histogram
- `n(m)` — background surface density per magnitude bin, from the full
  secondary catalogue
- `LR = q(m)·f(r) / n(m)`
- `R_j = LR_j / (Σ_i LR_i + (1 - Q))` — reliability in [0,1]

**Reference**: Sutherland, W. & Saunders, W. 1992, *MNRAS* 259, 413.

### 5. Random Forest Classifier (matcher="ml")

Engineers per-candidate features (normalised separation, colour differences on
user-specified photometric columns, local source density) and scores candidates
with a `RandomForestClassifier` trained on-the-fly. When scikit-learn is
unavailable, a weighted-heuristic fallback is used.

- **Features**: `sep/σ_combined`, `|col_L - col_R|/σ` per colour, log(local density).
- **Training**: nearest-neighbour pseudo-labels (positive) + random far pairs
  (synthetic negatives, shared with XGB matcher).
- **Feature engineering**: shared with the XGB matcher via
  `_engineer_ml_features_and_labels()`; best-per-primary selection shared via
  `_pick_best_per_primary()`.
- **Model persistence**: `--ml-model-path` saves/loads trained RF models in
  joblib format for reuse across runs without re-training.
- **Output**: `ml_score` column in [0,1].

**Reference**: Bai, Y. et al. 2019, *MNRAS* 483, 4085 — Random Forest
cross-matching of crowded stellar fields.  See also Salvato, M. et al. 2018,
*MNRAS* 473, 4937 for NWAY with machine learning priors.

### 6. AUF Match Probability (matcher="auf")

Implements the Astrometric Uncertainty Function from Wilson & Naylor (2017).
Rather than assuming Gaussian positional errors, the AUF empirically models the
true-match separation distribution from observed candidate pairs:

- Builds a separation histogram of all candidate pairs within the search radius.
- Subtracts the expected background contribution (n_bg × 2πr·dr annulus area)
  to isolate the true-match distribution f_AUF(r).
- Computes `P(r) = f_AUF(r) / (f_AUF(r) + n_bg)` per candidate pair.
- When per-source error estimates are unavailable, uses the observed candidate
  separation distribution directly as a proxy for the perturbation PDF.

Captures non-Gaussian error wings common in ground-based survey data, making it
more robust than sigma-based Gaussian matching for surveys with extended PSFs.

- **Engines**: fast, zone (delegated to fast for post-processing).
- **Output**: `auf_prob` column in [0,1].
- **Reference**: Wilson, T. J. & Naylor, T. 2017, *MNRAS* 468, 2517.

### 7. XGBoost / LightGBM Classifier (matcher="xgb")

Gradient-boosted tree classifier that typically outperforms Random Forest on
tabular crossmatch data. Shares the same feature engineering pipeline as the
ML matcher (separation/error ratio, colour differences, local source density)
via a shared `_engineer_ml_features_and_labels()` helper.

- **Library fallback chain**: XGBoost → LightGBM → sklearn GradientBoosting →
  weighted heuristic (no external library required).
- **Training**: nearest-neighbour pseudo-labels (positive) + random far-apart
  synthetic negatives, shared with the ML matcher.
- **Best-per-primary selection**: shared `_pick_best_per_primary()` helper
  (argmax over scores per primary source), also used by ML and AUF matchers.
- **Model persistence**: `--xgb-model-path` saves/loads trained models in
  joblib format for reuse across runs (mirrors `--ml-model-path`).
- **Output**: `xgb_score` column in [0,1].

- **Reference**: Chen, T. & Guestrin, C. 2016, *KDD* — XGBoost: A Scalable
  Tree Boosting System.  See also Ke, G. et al. 2017, *NIPS* for LightGBM.

### 8. Bayesian Pairwise p_match (Tier 3 — `--probabilistic`)

Budavári-style hierarchical Bayes factor combining a 2-D Gaussian spatial
kernel with non-parametric KDE photometric priors. Produces a `p_match` column
in [0,1].

- **Reference**: Budavári, T. & Szalay, A. S. 2008, *ApJ* 679, 301.

### 9. N-Way Bayesian (nway_match)

Simultaneous N-catalogue crossmatching with the full Budavári & Szalay (2008)
N-way posterior. Builds Cartesian-product tuples from catalogue 1 × ... ×
catalogue N and scores each tuple with the combined spatial + photometric
Bayes factor.

- **Reference**: Budavári, T. & Szalay, A. S. 2008, *ApJ* 679, 301.

### 10. Friends-of-Friends Transitive Closure (fof_match)

Merges multi-survey detections into object bundles via pairwise spatial
matching and union-find transitive closure. Each connected component becomes
one output row with `bundle_id`, `n_cats`, and `_src_cats`.

- **Reference**: Huchra, J. P. & Geller, M. J. 1982, *ApJ* 257, 423 (friends-of-friends
  for galaxy groups).  Also Tempel, E. et al. 2014, *A&A* 572, A81.

### 11. ID Join (--id-join)

Pure polars relational join on user-specified ID columns. Engine-agnostic.

### 12. PM Drift Prior (`pm_prior=True`)

When catalogues are separated by a large epoch baseline and one side lacks
measured proper motions, treating those sources as stationary leads to false
negatives — the unknown stellar motion carries the counterpart outside the
match radius.  The **probabilistic PM drift model** from Wilson (2023)
estimates the expected proper-motion dispersion from Galactic latitude and
inflates the positional error budget accordingly.

Two application modes are selected automatically based on the catalogue's
positional-error metadata:

- **Per-row mode** (default when the catalogue exposes per-row `ra_error` /
  `dec_error` columns — typical for Gaia-class catalogues): drift is added
  in quadrature to **each row's** per-axis astrometric error. See the
  [Per-row PM drift inflation](#per-row-pm-drift-inflation) sub-section
  below for the formula, fallback paths, and the skyerr `chord_max`
  caveat.
- **Source-level mode** (when only ``default_pos_error_arcsec`` is
  configured — typical for legacy catalogues without per-row error
  columns): drift is added in quadrature to that source-wide floor to
  produce a single inflated radius for the whole catalogue.

- **σ_μ(b)** = 3 + 7·exp(−|b| / 20°) — PM dispersion in mas/yr as a function
  of Galactic latitude, ranging from ∼10 mas/yr at the plane to ∼3 mas/yr at
  the poles.
- **σ_drift** = σ_μ × |Δt| / 1000 — drift uncertainty in arcsec for an
  epoch baseline Δt in years.
- **σ_total²** = σ_ast² + σ_drift² — added in quadrature to whichever
  positional error source applies (per-row, or the source-wide fallback).
- **Optional magnitude refinement** (`pm_prior_magnitude_column`): scales σ_μ
  by ``10^{−0.2(m−15)}`` (clipped to [0.3, 3.0]) as a distance proxy — brighter
  stars are statistically closer and therefore have larger proper motions.

#### Per-row PM drift inflation

When the catalogue exposes per-row `ra_error` / `dec_error` columns (the
common Gaia-class case), `pm_prior=True` enables a more accurate **per-row**
drift inflation. The implementation in `matchers._apply_pm_drift_prior()`
proceeds in three steps:

1. **Compute drift per row.**  For each (RA, Dec) row, derive the
   Galactic latitude `b` using `astropy.coordinates.SkyCoord` and evaluate
   `σ_μ(b)`.  An optional magnitude scale
   `clip(10^{−0.2(m−15)}, 0.3, 3.0)` widens the budget for bright
   sources (proxy for short distance / larger PM).
   Result: `σ_drift_arcsec = σ_μ × |Δt| / 1000 × mag_scale`.

2. **Append a per-row drift column.**  `_inflate_side()` adds a hidden
   ``_pm_drift_arcsec`` column to each side's `DataFrame` carrying that
   row's drift value.  `_build_result()` strips this column from the
   matched output so it never leaks into the user-visible schema.  A
   side whose epoch equals `target_epoch` (Δt < 0.01 yr) silently
   skips drift inflation — the typical case of crossmatching an old
   survey (no PMs) against a modern reference catalogue at its
   reference epoch.  The **two-old-survey case** (both sides carry
   `epoch` gaps, e.g. USNO-B vs 2MASS at the Gaia DR3 reference
   epoch) inflates each side's per-row sigma independently.

3. **Quadrature-sum in the per-row σ.**  `_pos_sigma_arcsec` (skyerr)
   and `_pos_covariance` (skyellipse) read the per-row column and
   combine each row's drift with its per-axis error:

   ```
   σ_skyerr  = √(ra_err² + dec_err² + drift²)        (per-row combined)
   σ²_ra_skyellipse  = ra_err² + drift²
   σ²_dec_skyellipse = dec_err² + drift²
   ```

   This is **a true per-row quadrature addition** — a regression that
   switched `√(a² + d²)` for `a + d` (linear add) would inflate the
   chord budget and cause the matcher to silently accept candidates
   that should fall outside the error budget.  Three drift-addition
   paths are exercised per side in `matchers._pos_sigma_arcsec` and
   `matchers._pos_covariance`:

   * **Per-row** — drift added in quadrature to the row's combined
     `ra_err² + dec_err²` (skyerr) or each axis (skyellipse); used when
     the catalogue has per-row `ra_error` / `dec_error` columns.
   * **Floor-only** — drift added in quadrature to
     ``default_pos_error_arcsec``; used when only the source-wide floor
     is configured.
   * **Drift-only fallback** — drift used as the sole positional
     uncertainty (`_pos_sigma_arcsec` returns `drift`,
     `_pos_covariance` returns `drift²`); catalogues with only a
     `pm_prior` flag and an `epoch` column can now be matched against a
     reference catalogue without a separate error source.

**Skyerr `chord_max` caveat.**  Under the `skyerr` matcher,
`_scipy_match` queries spatial candidates using a single
`chord_max = max_error × (np.nanmax(lsig) + np.nanmax(rsig))`.  Within
a single query the row with the largest per-row sigma governs the
chord radius even if other rows in the same table have a tighter
budget.  `skyellipse` evaluates per-pair Mahalanobis distance instead
and is not subject to this `np.nanmax` aggregation.

- **Engines**: all sky-match engines (per-row mode applies in `fast`,
  `astropy`, `zone`, `ray`; source-level mode applies everywhere).
- **CLI flags**: ``--pm-prior`` (enable), ``--pm-prior-mag-col`` (optional magnitude column).
  Requires ``--target-epoch``.
- **Reference**: Wilson, T. J. 2023, *RASTI* 2, 1.
  *Overcoming Separation Between Counterparts Due to Unknown Proper Motions
  in Catalogue Cross-Matching.*

---

## Engines

| Engine | Algorithm | Strengths |
|---|---|---|
| `fast` | scipy.spatial.cKDTree on 3-D unit sphere | ~3-5× faster than astropy; best for in-memory data |
| `zone` | HEALPix-pixellated cone match | Out-of-core scalable via `--batch-size` |
| `ray` | Distributed HEALPix zone match | Horizontal scaling across a Ray cluster |
| `astropy` | `search_around_sky` / `match_to_catalog_sky` | Reference implementation; gold standard |
| `stilts` | STILTS `tmatch2` (Java) | Proven in production; handles FITS natively |

---

## Join Types

| Type | Operation | SQL Equivalent |
|---|---|---|
| `1and2` | Inner join | `INNER JOIN` |
| `1or2` | Full outer join (union) | `FULL OUTER JOIN` |
| `all` | Full outer join | `FULL OUTER JOIN` |
| `all1` | Left outer join | `LEFT OUTER JOIN` |
| `all2` | Right outer join | `RIGHT OUTER JOIN` |
| `1not2` | Left anti-join | `WHERE NOT EXISTS` |
| `2not1` | Right anti-join | `WHERE NOT EXISTS` |

---

## Design Notes

### Shared Feature Engineering (ML + XGB)

The Random Forest (`matcher="ml"`) and XGBoost (`matcher="xgb"`) matchers share
a common feature engineering and pseudo-label generation pipeline via
`_engineer_ml_features_and_labels()`.  This helper handles:

- Combined positional error computation (σ_combined per pair).
- Z-scored absolute colour differences for each `ml_color_columns` entry.
- Local source density around each primary source (neighbours within 2×radius).
- Pseudo-label assignment: the spatially-nearest candidate per primary source
  is marked positive (1), all others negative (0).
- Optional synthetic negative examples: random far-apart left/right pairings
  (> 10× search radius) added as extra negatives to improve classifier training.
  Enabled for both matchers post-v0.5.

### Shared Best-per-Primary Selection

The `_pick_best_per_primary()` helper selects the highest-scoring candidate
per primary source via `np.argmax` over score groups.  Used by the ML, XGB,
and AUF matchers.  (The Likelihood Ratio matcher uses its own lexsort-based
selection because it sorts by reliability/LR/separation jointly.)

---

## Roadmap: Future Algorithms

### High Priority

- **NWAY / χ-match (Salvato et al. 2018, Pineau et al. 2017)**
  Full-framework multi-wavelength crossmatching with photometric priors
  (already partially covered by nway_match + LR + ML).
  - Ref: Salvato, M. et al. 2018, *MNRAS* 473, 4937.

### Medium Priority

- **Probabilistic Record Linkage (Fellegi-Sunter)**
  The classical record-linkage framework from statistics, widely used in
  census and health data but less common in astronomy. Computes match
  probabilities from agreement/disagreement vectors on multiple fields.
  Would excel at merging heterogeneous catalogues with inconsistent columns.
  - Ref: Fellegi, I. P. & Sunter, A. B. 1969, *JASA* 64, 1183.

- **Set Similarity Joins (MinHash LSH)**
  Locality-sensitive hashing for approximate spatial joins. Potentially
  useful for extremely large catalogues (>10⁹ sources) where exact spatial
  joins become prohibitive.
  - Ref: Leskovec, J. et al. 2014, *Mining of Massive Datasets*, Ch. 3.

- **Incremental / Streaming Crossmatch**
  For time-domain surveys (Rubin/LSST), crossmatching alerts against a growing
  catalogue without recomputing from scratch each night.
  - Ref: Juric, M. et al. 2019, *AJ* 158, 222.

### Low Priority

- **Deep Learning (CNN / Transformer)**
  End-to-end crossmatch classifiers trained on image cutouts rather than
  catalogue columns. Promising for ambiguous fields but computationally
  expensive and requires imaging data.
  - Ref: Dey, A. et al. 2022, *MNRAS* 515, 4532.

- **Graph Neural Networks for Multi-Catalogue Merging**
  Treat catalogues as a heterogeneous graph and use GNN message-passing to
  learn edge weights for all catalogue pairs simultaneously.
  - Ref: Kipf, T. N. & Welling, M. 2017, *ICLR*.

---

## References

1. Sutherland, W. & Saunders, W. 1992, *MNRAS* 259, 413.
   *On the likelihood ratio for source identification.*
2. Budavári, T. & Szalay, A. S. 2008, *ApJ* 679, 301.
   *Probabilistic Cross-Identification of Astronomical Sources.*
3. Pineau, F.-X. et al. 2017, *A&A* 597, A89.
   *Probabilistic multi-catalogue positional cross-match.*
4. Salvato, M. et al. 2018, *MNRAS* 473, 4937.
   *Finding counterparts for all-sky X-ray surveys with NWAY.*
5. Naylor, T. et al. 2013, *MNRAS* 428, 1929.
   *Optimal photometry for colour-magnitude diagrams.*
6. Wilson, T. J. & Naylor, T. 2017, *MNRAS* 468, 2517.
   *A perturbation approach to optimal cross-matching.*
7. Bai, Y. et al. 2019, *MNRAS* 483, 4085.
   *Cross-matching with Random Forest: a crowded field approach.*
8. Górski, K. M. et al. 2005, *ApJ* 622, 759.
   *HEALPix: a framework for high-resolution discretization and fast analysis.*
9. Huchra, J. P. & Geller, M. J. 1982, *ApJ* 257, 423.
   *Groups of Galaxies. I. Nearby Groups.*
10. Fellegi, I. P. & Sunter, A. B. 1969, *JASA* 64, 1183.
    *A theory for record linkage.*
11. Chen, T. & Guestrin, C. 2016, *KDD*.
    *XGBoost: A Scalable Tree Boosting System.*
12. Wilson, T. J. 2023, *RASTI* 2, 1.
    *Overcoming Separation Between Counterparts Due to Unknown Proper Motions
    in Catalogue Cross-Matching.*
