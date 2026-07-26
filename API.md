# xmatch public API

Stable surface only. Anything not exported from `xmatch.__init__` is internal.

---

## Top-level entry points

### `xmatch.CrossMatch`

The orchestrator. Holds the YAML config and dispatches
`local × local`, `local × remote`, `remote × remote` (incl. HATS) work to the
right backend.

```python
from xmatch import CrossMatch

cm = CrossMatch()  # uses bundled xmatch.yaml
result = cm.crossmatch("a.parquet", "b.csv", radius_arcsec=1.0)
```

Methods:

| Method | Purpose |
|---|---|
| `crossmatch(cat1, cat2, output_file=None, lazy=False, **params)` | Legacy spread-args entry. For new code prefer `crossmatch_request()`. |
| `crossmatch_request(req: MatchRequest)` | **Preferred** entry point. Takes a typed `MatchRequest`. |
| `crossmatch_multi(catalogues, output_file=None, lazy=False, **params)` | **N-catalogue** sequential pairwise crossmatch. Cat1×cat2 first, then accumulator matched against cat3, cat4, … |
| `resolve_source(value, overrides)` | Build a `CatalogueSource` from a path / frame / configured name. |
| `resolve_name(name)` | Apply catalogue-alias indirection. |
| `get_catalogue_config(name)` | Resolve a `name` (incl. alias) to a fully-merged config dict. |

### `xmatch.CrossMatch.crossmatch_multi()` (new in v0.4)

N-catalogue sequential pairwise crossmatch. All pairwise steps share the same
`MatchSpec`; the first catalogue's RA/Dec columns serve as the spatial reference
throughout the chain. Column-name collisions on each successive right side get
`_2`, `_3`, `_4`, … suffixes.

```python
from xmatch import CrossMatch

cm = CrossMatch()

# 3-catalogue intersection: A × B × C
result = cm.crossmatch_multi(
    ["a.csv", "b.parquet", "c.csv"],
    radius_arcsec=1.0,
    join_type="1and2",
    engine="fast",
)

# Remote-first: download gaia_esa, then match against two local files
result = cm.crossmatch_multi(
    ["gaia_esa", "wise.csv", "twomass.parquet"],
    ra=279.23,
    dec=38.78,
    radius_deg=0.005,
    radius_arcsec=2.0,
)

# 4-way with per-stage column suffixes _2, _3, _4
result = cm.crossmatch_multi(
    [df_a, df_b, df_c, df_d],
    radius_arcsec=3.0,
    lazy=True,
)
```

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `catalogues` | `list[FrameInput]` | *(required)* | 2+ catalogues (paths, names, or frames). |
| `output_file` | `str`/`Path` | `None` | Stream final result to file. |
| `lazy` | `bool` | `False` | Return a `LazyFrame` instead of collecting. |
| `**params` | *(same as `crossmatch()`)* | | `radius_arcsec`, `matcher`, `engine`, `join_type`, `ra`/`dec`/`radius_deg`, … |

Column naming in the output:

* Cat-1 columns: **unchanged** (e.g. `ra`, `dec`, `source_id`).
* Cat-2 columns: overlapping ones get `_2` suffix (`ra_2`, `dec_2`).
* Cat-3 columns: overlapping ones get `_3` suffix (`ra_3`, `dec_3`).
* … and so on for cat-4 → `_4`, cat-5 → `_5`, etc.
* `sep_arcsec` always reflects the **last** pairwise match; prior separations
  are dropped.

### `xmatch.MatchRequest` (new in v0.4)

Typed dataclass that consolidates all match parameters. Replaces the legacy `**params` dict.

```python
from xmatch import CrossMatch, MatchRequest, MatchSpec, SideOverrides

cm = CrossMatch()
req = MatchRequest(
    cat1="gaia",
    cat2="wise.parquet",
    spec=MatchSpec(radius_arcsec=2.0, matcher="skyerr", max_error=3.0),
    engine="fast",
    side2=SideOverrides(ra_column="RAJ2000", dec_column="DEJ2000"),
)
result = cm.crossmatch_request(req)
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `cat1`, `cat2` | `FrameInput` | *(required)* | File path, configured name, or in-memory frame. |
| `spec` | `MatchSpec` | `MatchSpec()` | Match criteria (radius, matcher, join type, find, Bayes priors). |
| `output_file` | `str`/`Path` | `None` | Stream results to file instead of returning. |
| `lazy` | `bool` | `False` | Return a `LazyFrame` instead of collecting eagerly. |
| `engine` | `str` | `"auto"` | `"auto"` (STILTS if available), `"astropy"`, `"fast"` (cKDTree), `"zone"` (HEALPix), `"stilts"`. |
| `id_join` | `bool` | `False` | Switch from sky-match to a relational ID join. |
| `id_column_1/2` | `str` | `None` | ID column names (from config when `None`). |
| `side1`, `side2` | `SideOverrides` | `SideOverrides()` | Per-catalogue column overrides. |
| `ra`, `dec`, `radius_deg` | `float` | `None` | Region for remote downloads. |
| `probabilistic` | `bool` | `False` | Enable Tier-3 Bayesian qualification. |

### `xmatch.SideOverrides` (new in v0.4)

Per-catalogue column override bundle.

```python
SideOverrides(
    ra_column="ra_custom",  # override RA column name
    dec_column="dec_custom",  # override Dec column name
    id_column="objid",  # override ID column name
    columns=["ra", "dec"],  # restrict columns returned
)
```

### `xmatch.MatchSpec`

| Field | Type | Default | Notes |
|---|---|---|---|
| `radius_arcsec` | `float` | `1.0` | Search radius for `matcher="sky"`. |
| `matcher` | `str` | `"sky"` | `"sky"`, `"skyerr"`, `"skyellipse"`, `"lr"`, `"ml"`, `"xgb"`, `"auf"`, `"macauff"`. |
| `max_error` | `float` | `3.0` | N-sigma cap for `skyerr`/`skyellipse`. |
| `join_type` | `str` | `"1and2"` | `"1and2"`, `"1or2"`, `"all1"`, `"all2"`, `"1not2"`, `"2not1"`, `"all"`. |
| `find` | `str` | `"best"` | `"best"` (closest) or `"all"` (all within radius). |
| `prior_columns` | `list[str]` | `[]` | Photometric columns for Tier-3 Bayesian KDE prior. |
| `target_epoch` | `float` | `None` | Julian-year epoch for proper-motion propagation. |
| `filter_expr` | `str` | `None` | Polars SQL WHERE clause for post-match filtering. |
| `extra_distance_cols` | `dict[str,float]` | `{}` | Column→weight map for N-dimensional cKDTree ranking. |
| `batch_size` | `int` | `None` | HEALPix pixel groups per batch (out-of-core friendly). |
| `lr_magnitude_column` | `str` | `None` | Magnitude column for Likelihood Ratio matcher (`matcher="lr"`). |
| `lr_q` | `float` | `0.8` | Prior probability a primary source has a detectable counterpart (LR matcher). |
| `ml_color_columns` | `list[str]` | `[]` | Photometric columns for ML/XGB matcher feature engineering. |
| `ml_model_path` | `str` | `None` | Path to save/load a pre-trained Random Forest model (joblib). |
| `xgb_model_path` | `str` | `None` | Path to save/load a pre-trained XGBoost model (joblib). |
| `macauff_flux_columns` | `list[str]` | `[]` | Magnitude columns for macauff flux likelihood ratios (`matcher="macauff"`). |
| `pm_prior` | `bool` | `False` | Enable probabilistic PM drift prior (Wilson 2023). Drift is added in quadrature to each row's per-row astrometric error when `ra_error` / `dec_error` columns exist; otherwise to a source-wide `default_pos_error_arcsec` floor; alone when neither is configured (drift-only mode). Requires `target_epoch`. |
| `pm_prior_magnitude_column` | `str` | `None` | Magnitude column for the per-row magnitude scaling `10^{−0.2(m−15)}` clipped to [0.3, 3.0]. In per-row mode widens the row's sigma by its magnitude; in source-level mode adjusts the source-wide floor. Only effective with `pm_prior=True`. |

---

## Advanced match features

### Proper-motion correction

Propagates RA/Dec to a common Julian-year epoch using per-row proper motions
and epoch columns (or catalogue-level defaults). NaN proper motions are treated
as zero.

```python
from xmatch import MatchSpec

spec = MatchSpec(
    radius_arcsec=1.0,
    target_epoch=2016.0,  # Gaia DR3 reference epoch
)
result = cm.crossmatch(
    "wise.csv",
    "gaia_esa",
    ra=180,
    dec=-30,
    radius_deg=0.01,
    spec=spec,
)
```

Requires `pm_ra_column` / `pm_dec_column` in the catalogue config (e.g., Gaia
has `pmra`/`pmdec` registered). Uses astropy for the spatial kinematics.

### PM drift prior (Wilson 2023)

When catalogues are separated by a large epoch baseline and one side lacks
measured proper motions, unknown stellar motion can carry counterparts outside
the match radius. The probabilistic PM drift model from Wilson (2023) inflates
positional errors using a Galactic-latitude-based proper-motion dispersion
estimate.

- ``σ_μ(b) = 3 + 7·exp(−|b| / 20°)`` — PM dispersion in mas/yr.
- ``σ_drift = σ_μ × |Δt| / 1000`` — drift uncertainty in arcsec.

Three application modes (chosen automatically based on available error info):

* **Per-row mode** (default for modern catalogues such as Gaia with
  per-row `ra_error` / `dec_error` columns):
  `_apply_pm_drift_prior()` appends a per-row `_pm_drift_arcsec` column
  to each side's `DataFrame`. `_pos_sigma_arcsec` (skyerr) and
  `_pos_covariance` (skyellipse) add it in quadrature to that row's
  per-axis astrometric error — each row carries its own budget
  independently, and `_build_result` strips the internal column from
  the output.
* **Source-level mode** (fallback when per-row error columns are
  absent): drift is added in quadrature to `default_pos_error_arcsec`
  as a single source-wide floor on each side.
* **Drift-only mode** (when neither per-row errors nor a floor are
  configured): the per-row drift becomes the sole positional
  uncertainty — `_pos_sigma_arcsec` returns `drift`,
  `_pos_covariance` returns `drift²`. Catalogues with only `pm_prior`
  and an `epoch` column can now be matched against a reference
  catalogue without a separate positional error source.

**Asymmetry preserved**: a side whose epoch equals `target_epoch`
(Δt < 0.01 yr) silently skips drift inflation — typical when
crossmatching an old survey against a modern reference catalogue at
its reference epoch. Two-old-survey case (both sides have epoch gaps,
e.g., USNO-B vs 2MASS at the Gaia DR3 reference epoch) inflates each
side's per-row sigma independently and contributes to the joint
match budget. Note that under the `skyerr` matcher the `chord_max`
is derived from `np.nanmax(lsig) + np.nanmax(rsig)`, so within a
single query the row with the largest per-row sigma governs the
chord radius; `skyellipse` computes per-pair Mahalanobis distance
instead and is not subject to this `np.nanmax` aggregation.

Optionally refined by magnitude (`pm_prior_magnitude_column`) — brighter stars
get larger PM dispersion via a distance-proxy scale factor
``10^{−0.2(m−15)}`` clipped to [0.3, 3.0].  In per-row mode this widens
each row's sigma according to its magnitude; in source-level mode it
adjusts the single source-wide floor.

```python
spec = MatchSpec(
    radius_arcsec=2.0,
    matcher="skyerr",
    max_error=5.0,
    target_epoch=2016.0,
    pm_prior=True,
    pm_prior_magnitude_column="phot_g_mean_mag",  # optional magnitude refinement
)
result = cm.crossmatch(
    "old_survey.parquet",
    "gaia_esa",
    ra=180,
    dec=-30,
    radius_deg=0.01,
    spec=spec,
)
```

**Drift-only mode** — a catalogue with no positional error source at all
(no per-row errors, no `default_pos_error_arcsec`, just an `epoch` column)
can still be matched against a reference catalogue: each row's drift
becomes the sole positional uncertainty.

```python
# historical_catalogue.csv has columns ra, dec, epoch only — no
# ra_error / dec_error columns and no default_pos_error_arcsec on
# the CatalogueSource.  Drift-only mode applies: each row's drift
# is the sole positional uncertainty.
spec = MatchSpec(
    radius_arcsec=5.0,
    matcher="skyerr",
    max_error=5.0,
    target_epoch=2016.0,
    pm_prior=True,
)
result = cm.crossmatch(
    "historical_catalogue.csv",
    "gaia_esa",
    ra=180,
    dec=-30,
    radius_deg=0.01,
    spec=spec,
)
```

**Reference**: Wilson, T. J. 2023, *RASTI* 2, 1.  *Overcoming Separation
Between Counterparts Due to Unknown Proper Motions in Catalogue
Cross-Matching.*

### Multi-condition post-match filtering

Filter matched pairs with a Polars SQL WHERE clause before picking the best
match. Column names from the left side are used as-is; right-side collision
columns get a ``_2`` suffix.

```python
spec = MatchSpec(
    radius_arcsec=2.0,
    filter_expr="abs(pmra - pmra_2) < 2.0 AND abs(phot_g_mean_mag - phot_g_mean_mag_2) < 0.5",
    find="best",
)
result = cm.crossmatch("gaia_bright.parquet", "gaia_faint.parquet", spec=spec)
```

Works with `engine="fast"`, `"astropy"`, or `"zone"` (logged warning for STILTS).

### N-dimensional cKDTree ranking

When `extra_distance_cols` is set and `find="best"`, the engine retrieves
spatial candidates within `radius_arcsec` and picks the one nearest in N-d
feature space (3-D spatial + z-score normalised extra columns). Useful for
breaking degeneracies in dense fields by incorporating photometry or proper
motion into the distance metric.

```python
spec = MatchSpec(
    radius_arcsec=2.0,
    extra_distance_cols={"phot_g_mean_mag": 0.5, "bp_rp": 0.3},
    find="best",
)
result = cm.crossmatch("cat_a.parquet", "cat_b.parquet", spec=spec, engine="fast")
```

### Out-of-core batching

When `batch_size` is set with `engine="zone"`, the HEALPix zone matcher
processes pixel groups in configurable batches, flushing results incrementally
and releasing per-batch margin trees. Pixel groups sorted largest-first for
balanced memory use.

```python
spec = MatchSpec(radius_arcsec=1.0, batch_size=100)
result = cm.crossmatch("large_a.parquet", "large_b.parquet", spec=spec, engine="zone")
```

### Ray distributed engine

Fans out HEALPix pixel-batch matching across Ray workers for distributed
crossmatching. Right-side pixel data placed in Ray's object store for zero-copy
sharing. Falls back to single-machine zone match when Ray is unavailable.

```bash
pip install 'xmatch[ray]'
```

```python
result = cm.crossmatch(
    "huge_a.parquet",
    "huge_b.parquet",
    radius_arcsec=1.0,
    engine="ray",
)
```

### Likelihood Ratio matcher (`matcher="lr"`)

Sutherland & Saunders (1992) counterpart identification. Estimates the
true-counterpart magnitude distribution q(m) by subtracting the expected
background from the candidate magnitude histogram, computes the Rayleigh
positional PDF f(r), and produces `lr` and `reliability` columns in [0,1].
Requires `lr_magnitude_column`.

```python
spec = MatchSpec(
    radius_arcsec=2.0,
    matcher="lr",
    lr_magnitude_column="phot_g_mean_mag",
    lr_q=0.8,  # prior counterpart fraction
)
result = cm.crossmatch("radio.csv", "optical.parquet", spec=spec, engine="fast")
# Output columns: lr, reliability
```

### Random Forest / XGBoost matchers (`matcher="ml"`, `matcher="xgb"`)

Machine-learning classifiers trained on-the-fly using self-match pseudo-labels.
Feature engineering (normalised separation, colour differences, local density)
is shared between both matchers. Requires `ml_color_columns`.

- **ml**: `RandomForestClassifier` (scikit-learn). Falls back to weighted heuristic.
- **xgb**: XGBoost → LightGBM → sklearn GradientBoosting → weighted heuristic.
  Reuses the same `ml_color_columns` as the ML matcher.

Both support model save/load via `--ml-model-path` / `--xgb-model-path` for
reuse across runs without re-training.

```python
# Random Forest
spec = MatchSpec(
    radius_arcsec=2.0,
    matcher="ml",
    ml_color_columns=["g", "r", "i"],
    ml_model_path="my_rf_model.joblib",  # save/load
)
result = cm.crossmatch("cat_a.parquet", "cat_b.parquet", spec=spec, engine="fast")
# Output column: ml_score

# XGBoost with model persistence
spec = MatchSpec(
    radius_arcsec=2.0,
    matcher="xgb",
    ml_color_columns=["g", "r", "i"],
    xgb_model_path="my_xgb_model.joblib",  # save/load
)
result = cm.crossmatch("cat_a.parquet", "cat_b.parquet", spec=spec, engine="fast")
# Output column: xgb_score
```

### AUF matcher (`matcher="auf"`)

Astrometric Uncertainty Function (Wilson & Naylor 2017). Builds an empirical
separation PDF from observed candidate pairs, subtracts the expected background,
and computes `P(r) = f_AUF(r) / (f_AUF(r) + n_bg)`. Captures non-Gaussian
error wings common in ground-based surveys. No extra columns required.

```python
spec = MatchSpec(radius_arcsec=2.0, matcher="auf")
result = cm.crossmatch("cat_a.parquet", "cat_b.parquet", spec=spec, engine="fast")
# Output column: auf_prob
```

### macauff matcher (`matcher="macauff"`)

AUF + flux likelihood ratios. Combines the empirical AUF positional probability
with per-band magnitude-difference likelihood ratios for improved match scores.
Requires `macauff_flux_columns`. Falls back to pure AUF scoring when no flux
columns are available.

```python
spec = MatchSpec(
    radius_arcsec=2.0,
    matcher="macauff",
    macauff_flux_columns=["g", "r", "i"],
)
result = cm.crossmatch("cat_a.parquet", "cat_b.parquet", spec=spec, engine="fast")
# Output column: macauff_prob
```

---

## HATS catalogues

[HATS](https://lsdb.readthedocs.io) (Hierarchical Adaptive Tiling Scheme)
catalogues are supported via the optional **LSDB** package. Both pure-HATS and
mixed HATS+local chains work — xmatch routes them through LSDB's internal
crossmatch engine and normalises the output to the standard xmatch schema
(LSDB's ``_dist_arcsec`` column is renamed to ``sep_arcsec``; column collisions
are disambiguated with the standard ``_2``, ``_3``, … suffixes).

**Install LSDB:**  ``pip install lsdb`` (see `lsdb.readthedocs.io
<https://lsdb.readthedocs.io>`_).

Without LSDB installed, HATS directories are still detected and resolved, but
any crossmatch that actually touches a HATS catalogue will raise a clear
``CrossMatchError`` with a hint to install the package.

### Detection & resolution

A directory is recognised as a HATS catalogue when it contains any of the
marker files ``properties``, ``hats.properties``, ``catalog_info.json``, or
``_metadata``.  You can also call ``is_hats_dir(path)`` directly:

```python
from xmatch import is_hats_dir

assert is_hats_dir("/data/gaia_dr3_hats")
```

### Usage (Python)

```python
from xmatch import CrossMatch

cm = CrossMatch()

# Two HATS catalogues
result = cm.crossmatch(
    "/data/cat_a_hats",
    "/data/cat_b_hats",
    radius_arcsec=2.0,
)

# HATS × local file with column overrides (useful when HATS metadata is
# incomplete — the RA/Dec column names from the HATS catalogue can be
# given explicitly)
result = cm.crossmatch(
    "/data/gaia_dr3_hats",
    "wise_region.csv",
    radius_arcsec=1.0,
    ra_column1="raj2000",
    dec_column1="dej2000",  # override cat-1 columns
)
```

### Multi-way chains with HATS

HATS catalogues can appear at **any position** in an N-catalogue chain.
When cat-1 is HATS it acts as the spatial reference; when cat-3+ is HATS the
accumulated frame is converted into an in-memory LSDB catalogue and matched.

```python
# HATS as first catalogue → spatial reference for whole chain
result = cm.crossmatch_multi(
    ["/data/gaia_dr3_hats", "allwise_bright.csv", "twomass.csv"],
    radius_arcsec=2.0,
)

# HATS at position 3 — accumulator matched against HATS catalogue
result = cm.crossmatch_multi(
    ["local_a.csv", "local_b.csv", "/data/vizier_hats"],
    radius_arcsec=1.5,
)
```

Column suffixes work identically to non-HATS chains: cat-2 columns get ``_2``,
cat-3 get ``_3``, etc.  Lazy execution is **not** supported — HATS steps
always materialise eagerly.

### Limitations

| Feature | Status | Notes |
|---|---|---|
| `join_type` | ``1and2`` (inner) only | Other join types raise ``CrossMatchError``. Use post-hoc outer-join logic on local files if needed. |
| `find` | ``best`` and ``all`` supported | ``find="best"`` returns the nearest neighbour per left row; ``find="all"`` returns every pair within the radius (may produce more rows). Mapped to LSDB's ``n_neighbors`` parameter. |
| `matcher` / `engine` | Ignored | LSDB uses its own kd-tree engine internally. |
| Bayesian priors (`p_match`) | Not supported | Logs a warning. Use local files with ``engine="fast"`` for Tier‑3 output. |
| Proper-motion / epoch propagation | Not supported | LSDB does not propagate coordinates to a common epoch. |
| Lazy frames | Always eager | ``.compute()`` is called inside LSDB. |
| cat-1 = HATS + remote at position 3+ | Needs explicit ``ra``/``dec``/``radius_deg`` | When the first source is a HATS catalogue, region inference for remote TAP/CDS downloads at positions 3+ is not possible. Pass explicit ``ra``, ``dec``, and ``radius_deg`` in ``**params`` (or ``MatchRequest`` fields) as a workaround. |

---

## Remote catalogues

Configured remote catalogues are accessed via TAP/ADQL cone-search downloads.
Use catalogue names or aliases directly — xmatch fetches the region, downloads
it, and crossmatches locally.

### Available archives

| Archive | TAP URL | Catalogues (sample) |
|---|---|---|
| **ESA Gaia** | `gea.esac.esa.int` | `gaia_esa` (Gaia DR3), alias `gaia`, `gaia_dr3` |
| **CDS VizieR** | `tapvizier.u-strasbg.fr` | `gaia_cds`, `galex_cds`, `vhs_cds` |
| **NOIRLab Data Lab** | `datalab.noirlab.edu` | `nsc` (NSC DR2), `des` (DES DR2), `desils` / `decals` / `ls_dr10` (DECaLS DR10 tractor), `decals_objects` (DECaLS DR10 objects — no photometry), `smash` (SMASH DR2), `unwise` (unWISE DR1), `allwise_dl` (AllWISE), `delve`, `gaia_noao`, `ukidss` |

List all configured catalogues: ``xmatch --list`` or ``cm.catalogues_config``.

### Usage

```python
from xmatch import CrossMatch

cm = CrossMatch()

# Download a cone from Gaia DR3, match against a local file
result = cm.crossmatch(
    "gaia",
    "my_stars.csv",
    ra=279.23,
    dec=38.78,
    radius_deg=0.005,
    radius_arcsec=2.0,
)

# NSC DR2 × AllWISE (both via Data Lab TAP — same TAP self-join)
result = cm.crossmatch(
    "nsc",
    "allwise_dl",
    ra=180.0,
    dec=-30.0,
    radius_deg=0.01,
    radius_arcsec=1.5,
)

# Multi-way: Gaia × DES × DECaLS (remote first, two Data Lab catalogues)
result = cm.crossmatch_multi(
    ["gaia", "des", "ls_dr10"],
    ra=279.23,
    dec=38.78,
    radius_deg=0.005,
    radius_arcsec=1.0,
)
```

### Pre-computed crossmatch tables

Data Lab hosts pre-computed 1.5″ nearest-neighbour crossmatch tables that let
you **skip downloading** the full catalogues. Each table contains matched
pairs with ``ra1``/``dec1``/``id1`` (survey), ``ra2``/``dec2``/``id2`` (Gaia),
and ``distance`` (separation in arcsec).

```python
# Instead of downloading NSC and Gaia as two separate catalogues, use the
# pre-computed xmatch table — it already contains NSC×Gaia matched pairs.
# You download one table instead of two, and still sky-match against a local
# file.  The xmatch table's "distance" column is the NSC–Gaia separation;
# the output "sep_arcsec" is the separation from the local file.
result = cm.crossmatch(
    "nsc_x_gaia",
    "local_stars.csv",
    ra=279.23,
    dec=38.78,
    radius_deg=0.005,
    radius_arcsec=2.0,  # your desired local-match radius (the xmatch table's
    # 1.5″ NSC–Gaia pairing radius is baked in)
)
# Columns: ra1, dec1, id1 (NSC), ra2, dec2, id2 (Gaia), distance (NSC–Gaia),
#          plus local columns and sep_arcsec (local-match separation)
```

Available pre-computed xmatch tables: ``nsc_x_gaia``, ``des_x_gaia``,
``decals_x_gaia``, ``allwise_x_gaia`` (aliases for
``nsc_x_gaia_noao`` / ``des_x_gaia_noao`` / …).

CLI equivalent:

```bash
# Instant NSC×Gaia match — no full catalogue download
xmatch nsc_x_gaia my_stars.csv --ra 279.2 --dec 38.8 --radius-deg 0.005 -r 1.5
```

**Discovery:** Explore remote TAP tables interactively:

```bash
xmatch --discover noirlab                  # list all Data Lab tables
xmatch --discover noirlab --schema nsc_dr2.object  # NSC DR2 column schema
xmatch --search gaia                       # search all endpoints for "gaia" tables
```

---

## Data classes

```python
@dataclass
class CatalogueSource:
    name: str
    is_local: bool
    ra_column: Optional[str]
    dec_column: Optional[str]
    id_column: Optional[str]
    ra_err_column: Optional[str]  # for skyerr
    dec_err_column: Optional[str]  # for skyerr
    corr_column: Optional[str]  # for skyerr (not yet wired)
    pos_err_units: str = "arcsec"
    default_pos_error_arcsec: Optional[float]
    epoch: Optional[float]
    access_method: Optional[str]  # "tap" | "cds_xmatch" | "hats"
    ...
```

---

## I/O helpers (Arrow interop)

| Symbol | Signature | Notes |
|---|---|---|
| `FrameLike` | `Union[pl.DataFrame, pl.LazyFrame]` | Type alias exported for user code. |
| `is_hats_dir` | `(path) -> bool` | True if the directory looks like a HATS catalogue. |
| `scan_frame` | `(path) -> pl.LazyFrame` | Lazy scan of `.parquet` (predicate push-down), `.csv` (push-down), eager for `.fits`/`.fit`. |
| `write_frame` | `(frame, output_file)` | Streams `.parquet`/`.csv` via `sink_*` with streaming engine. FITS is eager. |
| `astropy_table_to_polars` | `(table) -> pl.DataFrame` | Arrow bridge (bytes → utf8 auto). |
| `polars_to_astropy` | `(df_or_lf) -> astropy.Table` | Arrow bridge; lazy frames are collected first. |

Performance note: the bridge reads/writes **polars' internal Arrow buffers**
directly, so a round-trip doesn't re-serialise through pandas — it just changes
the Python-level container.

---

## Exceptions (all inherit `CrossMatchError`)

| Class | When |
|---|---|
| `ConfigError` | YAML config missing, malformed, or invalid. |
| `InputError` | File/frame unreadable, unsupported extension, or auto-detection failure for local inputs. |
| `TapError` | TAP/ADQL failures (network, parse, async job failed). |
| `StiltsError` | STILTS not found or subprocess failed. |

Catch `CrossMatchError` for the full family.

---

## What is *not* public

* `matchers.sky_match`/`id_join` — implement ADQL/STILTS dialog; expose via
  `CrossMatch.crossmatch` instead.
* `remote_tap.download_from_tap`, `remote_cds.cds_xmatch_local_remote` — same.
* `astro_utils.sky_extent`, `find_coord_columns` — workhorse helpers; may
  become public on demand.
* `discovery.discover_tables`, `discovery.discover_columns` — internal; exposed
  via the CLI `--search` / `--discover` commands.
* The `cfg["stilts_config"]` block — use the `stilts_cmd_base`, `java_opts`,
  `tmpdir` kwargs on `CrossMatch(...)`.

---

## CLI

```bash
xmatch --help
xmatch --list                          # print configured catalogues in table format
xmatch --describe gaia                 # print merged config with column details
xmatch --search gaia                   # search remote TAP endpoints for matching tables
xmatch --discover vizier               # list all tables on CDS VizieR
xmatch --discover gaia --schema gaiaedr3.gaia_source  # show column schema for a table
xmatch a.csv b.csv -o out.parquet -r 1.5
xmatch --matcher skyerr --max-error 3.0 a.csv b.csv
xmatch --id-join --id1 objid --id2 objid a.csv b.csv
xmatch a.parquet b.parquet --engine fast --probabilistic --priors g,r -o out.parquet

# Advanced features
xmatch --target-epoch 2016.0 a.csv b.csv -r 1.5                        # PM correction
xmatch --target-epoch 2016.0 --pm-prior --matcher skyerr \
    --max-error 5.0 old_survey.csv gaia_esa -r 2.0                      # PM drift prior
xmatch --target-epoch 2016.0 --pm-prior --pm-prior-mag-col mag_g \
    old_survey.csv gaia_esa -r 2.0                                       # PM drift + magnitude refinement
xmatch --engine fast --filter-expr "abs(mag - mag_2) < 0.5" a.csv b.csv  # post-filter
xmatch --engine fast --extra-distance-cols g:0.5,bp_rp:0.3 a.csv b.csv  # N-d ranking
xmatch --engine zone --batch-size 100 large_a.csv large_b.csv            # out-of-core batching
xmatch --engine ray huge_a.parquet huge_b.parquet -r 1.0                 # Ray distributed

# Matcher-specific examples
xmatch a.csv b.csv --matcher lr --lr-magnitude-column mag_g -r 2.0      # Likelihood Ratio
xmatch a.csv b.csv --matcher ml --ml-color-cols g,r,i -r 2.0            # Random Forest
xmatch a.csv b.csv --matcher ml --ml-color-cols g,r,i -r 2.0 \
    --ml-model-path my_model.joblib                                     # RF with model save/load
xmatch a.csv b.csv --matcher xgb --ml-color-cols g,r,i -r 2.0           # XGBoost
xmatch a.csv b.csv --matcher xgb --ml-color-cols g,r,i -r 2.0 \
    --xgb-model-path my_xgb.joblib                                      # XGB with model save/load
xmatch a.csv b.csv --matcher auf -r 2.0                                 # AUF empirical error model
xmatch a.csv b.csv --matcher macauff --macauff-flux-cols g,r -r 2.0     # AUF + flux likelihoods

# N-catalogue (3+) crossmatching
xmatch a.csv b.csv c.csv -r 1.0                          # 3-way intersection
xmatch gaia_esa wise.csv twomass.parquet --ra 279.2 --dec 38.8 --radius-deg 0.005
xmatch a.csv b.csv c.csv d.csv -r 2.0 -o multi.parquet   # 4-way to file

# HATS catalogue crossmatching (requires: pip install lsdb)
xmatch /data/gaia_dr3_hats wise.csv -r 1.0               # HATS × local
xmatch /data/cat_a_hats /data/cat_b_hats -r 2.0          # HATS × HATS
xmatch /data/gaia_hats local.csv /data/allwise_hats -r 1.5   # HATS in 3-way chain

# Remote catalogue crossmatching (auto-downloads cone from TAP)
xmatch gaia my_stars.csv --ra 279.2 --dec 38.8 --radius-deg 0.005 -r 1.0
xmatch nsc allwise_dl --ra 180.0 --dec -30.0 --radius-deg 0.01 -r 1.5
xmatch gaia des ls_dr10 --ra 279.2 --dec 38.8 --radius-deg 0.005 -r 1.0  # 3-way

# Discovery and exploration
xmatch --discover noirlab                  # list Data Lab tables
xmatch --discover noirlab --schema nsc_dr2.object  # NSC DR2 column schema
xmatch --search des                        # search all TAP endpoints for DES tables
```

See `README.md` for full workflow examples.
