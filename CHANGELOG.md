# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — v0.5 Audit + HATS + Data Lab + New Engines

### Added — Versioned Association Contract

* **`xmatch.association.component.delta.release.v1`**: deterministic,
  content-addressed delta artifacts bind explicit parent/current association
  and component release namespaces and classify exact membership overlap as
  created, continued, merged, split, or retired. Component construction now
  requires explicit source/candidate input releases and included decisions;
  endpoint-aware verification recomputes classifications and rejects tampering
  or false declared lineage.
* **`xmatch.association.component.release.v1`**: deterministic sidecar releases
  bind exact, namespaced component membership to a verified association
  release. Component IDs cover the association release plus canonical members;
  manifests inherit executable provenance and bind optional parent component
  releases, while per-component parent IDs preserve merge/split lineage.
* **`xmatch.association.release.v1`**: deterministic release directories pair
  canonical `associations.jsonl` records with a content-addressed manifest,
  shared executable provenance, an optional parent release, and streaming
  checksum/count/schema verification. Publication is atomic and never
  overwrites an existing release path.
* **`xmatch.association.v1`**: strict candidate records with stable
  content-derived association IDs, release-scoped evidence/source/candidate
  IDs, explicit score semantics, true arcsecond separation, evaluation epoch,
  decisions, extensible flags, and content-addressed provenance.
* **Packaged conformance fixture**: four analytic/adversarial records cover a
  unique selection, crowded-field ambiguity, and unknown epoch/uncertainty;
  nine focused tests enforce deterministic round trips and reject invalid
  probabilities, calibration claims, identifiers, epochs, and separations.
* Documentation now describes `p_match` as an assumed-prior posterior rather
  than an empirically calibrated probability.

### Added — New Match Features

* **Proper motion correction**: `MatchSpec.target_epoch` propagates RA/Dec to a
  common Julian-year epoch using per-row `pm_ra_column` / `pm_dec_column` +
  `epoch_column` (or catalogue-level `epoch`). NaN proper motions treated as
  zero. Applies before any engine dispatch. Requires astropy.
* **Multi-condition filtering**: `MatchSpec.filter_expr` accepts a polars SQL
  WHERE clause (e.g. `"abs(mag_g - mag_g_2) < 0.5"`) applied as a post-match
  boolean filter. Works with `engine="fast"`, `"astropy"`, or `"zone"`.
  Left-side column names used as-is; right-side collisions get `_2` suffix.
* **N-dimensional cKDTree matching**: `MatchSpec.extra_distance_cols` maps
  column names to dimensionless weights. Columns are z-score normalized across
  the catalogue union and appended to the 3-D Cartesian unit-sphere embedding.
  Among spatial candidates within `radius_arcsec`, the nearest in N-d feature
  space is chosen. Only affects `find="best"`; `find="all"` unchanged.
* **Out-of-core batching**: `MatchSpec.batch_size` controls how many HEALPix
  pixel groups the zone engine processes per batch. Results flushed
  incrementally; per-batch margin trees GC'd between batches. Pixel groups
  sorted largest-first for balanced memory use.

### Added — Ray Distributed Engine

* **`ray_engine.py`**: new module with `ray_zone_match` entry point. Fans out
  HEALPix pixel-batch matching across Ray workers. Right-side pixel data placed
  in Ray's object store for zero-copy sharing. Gracefully falls back to
  single-machine zone match when Ray is unavailable. Lazy `@ray.remote`
  initialization avoids import-time failures without `ray` installed.
* **`pyproject.toml`**: `ray` added to `[project.optional-dependencies]` —
  install with `pip install 'xmatch[ray]'`.
* **`sky_match`**: accepts `engine="ray"`.

### Changed — Performance

* **HEALPix margin caching**: `_zone_match_healpix` now merges all neighbouring
  right-pixel data into a single cKDTree per left pixel group and queries it
  once (instead of querying each right pixel's tree individually). Significantly
  faster for dense fields with many overlapping neighbour pixels.

### Fixed — Correctness

* **PM propagation**: `_apply_proper_motion` now correctly processes both left
  and right sides (previously returned after first side due to early-return
  bug). Added debug log when neither side has PM+epoch info.

### Fixed — Math & Correctness

* **Bayes 2D Gaussian log-density** (`bayes.py`): `positional_log_likelihood` was
  using a 1D Gaussian (`-log(σ)`, `-0.5·log(2π)`) instead of the correct 2D
  isotropic Gaussian (`-2·log(σ)`, `-log(2π)` per Budavári & Szalay 2008 eq. 25).
  All Bayesian `p_match` values are now mathematically correct.
* **Photometric prior separation** (`bayes.py` + `matchers.py`):
  `compute_p_match` now receives separate match-hypothesis prior (KDE at
  magnitude midpoint) and background-hypothesis prior (KDE(left) + KDE(right)
  independently) instead of adding the same term to both and having them cancel.
* **HEALPix `chord_max` for skyerr** (`matchers.py`): `_zone_match_healpix` now
  uses the error-based `search_radius` for the chord distance bound instead of
  `spec.radius_arcsec`, matching `_scipy_match` behaviour.

### Fixed — Robustness

* **STILTS skyerr TypeError guard** (`stilts.py`): when no positional error
  columns exist, raises `StiltsError` instead of crashing on `np.nanmax(None)`.
* **Null-handling in sky extent** (`astro_utils.py`): antipodal fallback in
  `sky_extent_from_frame` now uses `drop_nulls().mean()` and guards against
  `None` before `float()` cast.
* **STILTS feature-suppression warnings** (`matchers.py`): now emits
  `WARNING`-level messages when `right_suffix` or `prior_columns` are silently
  ignored by the STILTS engine.

### Changed — Performance

* **Multi-way eager checkpoint** (`crossmatch.py`): `crossmatch_multi` now
  inserts `.collect().lazy()` inside the pairwise loop, preventing Polars from
  rebuilding the entire query graph on every iteration (eliminates O(N²)
  LazyFrame re-evaluation for N-catalogue chains).
* **HEALPix batched cKDTree queries** (`matchers.py`): `_zone_match_healpix`
  groups left points by HEALPix pixel and calls `tree.query(batch_xyz,
  workers=-1)` once per pixel-pair instead of spawning threads per single point
  in a Python loop.

### Added — HATS / LSDB Integration

* **`hats_crossmatch` enhancements** (`hats_source.py`): accepts `right_suffix`,
  maps `find`→`n_neighbors`, passes `suffixes` to LSDB, renames
  `_dist_arcsec`→`sep_arcsec`, validates `join_type` (only `1and2` supported),
  and warns when `prior_columns` are requested.
* **HATS in multi-way chains** (`crossmatch.py`): `_dispatch` passes
  `right_suffix`; `crossmatch_multi` handles HATS catalogues at any position
  (including 3+); download loop guards against HATS sources.
* **`_dist_arcsec` missing warning**: logs a warning when LSDB result lacks the
  expected column (e.g., future API change).* **16 HATS tests** (`tests/test_hats.py`): fully mocked (no LSDB required),
covering alias resolution, crossmatch parameter passthrough, multi-row
`find="all"` results, join type validation, local-frame conversion,
column renaming, and `crossmatch_multi` routing.  Names verified by
the bidirectional static-scan guards in `tests/test_ci_smoke.py`:
`test_lsdb_available_false`, `test_require_lsdb_raises_with_helpful_message`,
`test_read_hats_no_path`, `test_hats_crossmatch_unsupported_join_type_raises`,
`test_hats_crossmatch_passes_n_neighbors_best`,
`test_hats_crossmatch_passes_n_neighbors_all`,
`test_hats_crossmatch_find_all_multi_row_rename`,
`test_hats_crossmatch_custom_right_suffix`,
`test_hats_crossmatch_warns_on_missing_dist_arcsec`,
`test_hats_crossmatch_with_local_left_frame`,
`test_hats_crossmatch_warns_on_prior_columns`, `test_resolve_source_hats_dir`,
`test_resolve_source_hats_dir_with_overrides`,
`test_crossmatch_multi_hats_at_position_3_routes_via_hats_crossmatch`,
`test_crossmatch_two_hats_via_dispatch`, `test_is_hats_dir_multiple_markers`.

### Added — NOAO Data Lab Catalogues

* **6 new catalogues** in `xmatch.yaml`: `nsc_noao` (NSC DR2), `des_noao`
  (DES DR2), `decals_noao` (DECaLS DR10 objects), `smash_noao` (SMASH DR2),
  `unwise_noao` (unWISE DR1), `allwise_noao` (AllWISE).
* **4 pre-computed xmatch tables**: `nsc_x_gaia_noao`, `des_x_gaia_noao`,
  `decals_x_gaia_noao`, `allwise_x_gaia_noao` — 1.5″ nearest-neighbour
  pre-computed crossmatches (× Gaia DR3). Provides `ra1`/`dec1`/`ra2`/`dec2`/
  `id1`/`id2`/`distance` columns for instant matching without downloads.
* **14 new aliases**: `nsc`, `des`, `decals`→tractor, `decals_objects`→object
  table, `smash`, `unwise`, `allwise_dl`, `nsc_x_gaia`, `des_x_gaia`,
  `decals_x_gaia`, `allwise_x_gaia`, and qualifiers.* **2 Data Lab source resolution tests** (`tests/test_crossmatch.py`): verify aliases resolve to TAP-backed
`CatalogueSource` with correct table names, archive, and column
metadata.  Tractor vs object-table routing verified; the specific test
names are pinned by the bidirectional static-scan guards in
`tests/test_ci_smoke.py`: `test_resolve_source_datalab_alias`,
`test_resolve_source_datalab_aliases_point_to_tractor_not_object`.
* **28 aliases total**, **19 configured catalogues**.

### Changed — Documentation

* `API.md`: added **`crossmatch_multi()`** section (parameter table, 3-way /
  remote-first / 4-way examples, column naming guide); **HATS catalogues**
  section (detection, Python/CLI usage, multi-way chains, limitations table);
  **Remote catalogues** section (ESA Gaia, CDS VizieR, NOIRLab Data Lab
  archives with Python + CLI examples); updated CLI section with N-catalogue,
  HATS, Data Lab, and discovery examples.

### Notes

* 92 non-slow tests pass; 9 slow (real-catalogue) tests pass; 16 HATS tests;
  2 Data Lab tests. Zero regressions across Gaia DR3, AllWISE, USNO-B1.0.
* No breaking changes — all existing APIs unchanged.

### Added — Likelihood Ratio Matcher (`matcher="lr"`)

* **Sutherland & Saunders (1992) counterpart identification**: new
  `_likelihood_ratio_scoring()` estimates the true-counterpart magnitude
  distribution q(m) by background subtraction, computes Rayleigh positional
  PDF f(r) from combined per-row errors, and produces `lr` and `reliability`
  columns in [0,1]. Requires `lr_magnitude_column`; accepts `lr_q` prior.
* **CLI**: `--matcher lr`, `--lr-magnitude-column`, `--lr-q` flags.

### Added — ML & XGBoost Matchers (`matcher="ml"`, `matcher="xgb"`)

* **Random Forest classifier** (`matcher="ml"`): `_ml_rf_score()` engineers
  per-candidate features (separation/error ratio, colour differences, local
  density) and scores with a `RandomForestClassifier` trained on-the-fly.
  Falls back to weighted heuristic when scikit-learn unavailable.
  Outputs `ml_score` in [0,1]. Requires `ml_color_columns` (CLI:
  `--ml-color-cols`).
* **XGBoost classifier** (`matcher="xgb"`): `_xgb_score()` uses the same
  feature engineering with a gradient-boosted tree model (XGBoost → LightGBM
  → sklearn GradientBoosting → weighted heuristic fallback chain).
  Outputs `xgb_score` in [0,1]. Reuses `ml_color_columns` from ML matcher.
* **Shared feature engineering**: `_engineer_ml_features_and_labels()` helper
  extracts sigma computation, colour differences, local density, pseudo-label
  generation (nearest-neighbour = positive), and optional synthetic negatives
  for both matchers — eliminates ~150 lines of duplication.
* **Shared best-per-primary selection**: `_pick_best_per_primary()` helper
  (argmax over score groups) used by ML, XGB, AUF, and macauff matchers.
* **Model persistence**: `--ml-model-path` and `--xgb-model-path` CLI flags
  (and `ml_model_path` / `xgb_model_path` MatchSpec fields) save/load
  trained models in joblib format for reuse across runs.
* **Synthetic negatives**: both ML and XGB matchers now append random
  far-apart pairings (>10× search radius) as negative training examples
  for improved classifier discriminability.

### Added — AUF Matcher (`matcher="auf"`)

* **Astrometric Uncertainty Function** (Wilson & Naylor 2017): `_auf_score()`
  builds an empirical separation PDF from observed candidate pairs, subtracts
  the expected background (n_bg × 2πr·dr annulus area), and computes
  `P(r) = f_AUF/(f_AUF + n_bg)`. Captures non-Gaussian error wings common in
  ground-based survey data. Outputs `auf_prob` in [0,1].
* **CLI**: `--matcher auf`.

### Added — macauff Matcher (`matcher="macauff"`)

* **AUF + flux likelihood ratios**: `_macauff_score()` combines the empirical
  AUF positional probability with per-band magnitude-difference likelihood
  ratios (Gaussian match model with photometric errors, histogram-based
  background model). Outputs `macauff_prob` in [0,1].
* **CLI**: `--matcher macauff`, `--macauff-flux-cols` flags.

### Added — PM Drift Prior (`pm_prior=True`)

* **Wilson (2023) probabilistic PM drift model**: `_apply_pm_drift_prior()`
  inflates positional error floors for sources lacking measured proper motions
  based on Galactic latitude: `σ_μ(b) = 3 + 7·exp(−|b|/20°)` mas/yr,
  `σ_drift = σ_μ × |Δt| / 1000` arcsec, added in quadrature to
  `default_pos_error_arcsec`. Works with any sky-match engine + matcher.
* **Magnitude refinement** (`pm_prior_magnitude_column`): optionally scales
  `σ_μ` by `10^{−0.2(m−15)}` (clipped to [0.3, 3.0]) as a distance proxy —
  brighter stars are statistically closer and get larger PM dispersion.
* **CLI**: `--pm-prior` flag (requires `--target-epoch`), `--pm-prior-mag-col`
  flag for optional magnitude refinement.
* **`_galactic_latitude()`** helper: converts equatorial RA/Dec to Galactic
  latitude for the drift model.
* 5 PM drift prior baseline tests: skyerr inflation, no-epoch skip,
  large-baseline, magnitude scaling, missing magnitude column.

### Added — Per-Row PM Drift Mode (when `pm_prior=True`)

Builds on the PM Drift Prior entry above.  Per-row mode triggers when
the catalogue has per-row `ra_error` / `dec_error` columns — the
common case for modern catalogues such as Gaia.

* **Per-row drift column**: `_apply_pm_drift_prior()` appends a per-row
  `_pm_drift_arcsec` column to each side's `DataFrame` (drift computed
  per-row from Galactic latitude and `|Δt|`), and `_pos_sigma_arcsec`
  (skyerr) and `_pos_covariance` (skyellipse) add it in quadrature to
  that row's per-axis astrometric error.  Each row's sigma carries the
  inflation independently — a row with `mag=10` (with
  `pm_prior_magnitude_column`) gets a wider per-row sigma than a row
  with `mag=20` at the same epoch.  Note that skyerr's `chord_max` is
  derived from `np.nanmax(lsig) + np.nanmax(rsig)`, so within a
  single query the row with the largest per-row sigma governs the
  chord radius.
* **Source-level and drift-only fallbacks**: when per-row error
  columns are absent, `default_pos_error_arcsec` is still used as a
  floor (with drift added in quadrature), preserving the prior
  behaviour.  When neither per-row errors nor a floor are configured, the per-row
  drift becomes the sole positional uncertainty — `_pos_sigma_arcsec`
  returns `drift`, `_pos_covariance` returns `drift²` — so catalogues
  with only a `pm_prior` and an epoch column can now be matched
  without a separate positional error source.
* **Asymmetry preserved**: sides whose `epoch` equals `target_epoch`
  (Δt < 0.01 yr) silently skip drift inflation — typical when
  crossmatching an old survey against a modern reference catalogue at
  its reference epoch.  Two-old-survey case (e.g., USNO-B vs 2MASS
  at the Gaia DR3 reference epoch — both sides carry `epoch` gaps)
  lets each side contribute its own per-row drift to the joint
  skyerr `chord_max`; per-side budgets and the joint budget are
  independently testable at separations straddling the per-side chord
  boundaries.
* **Internals**: `_PM_DRIFT_COLUMN = "_pm_drift_arcsec"` module
  constant; `_build_result` strips the column from `matched`,
  `left`, and `right` output paths so it never leaks into the
  result DataFrame.  CLI unchanged (`--pm-prior`); per-row mode is
  automatic when per-row error columns are present.
* **3 new tests** (additive to the 5 baseline):
  `test_pm_prior_per_row_drift_added_to_astrometric_errors` —
  per-row drift added in quadrature to per-row astrometric errors
  with a 1.6" sep quadrature-vs-linear-add regression;
  `test_pm_prior_per_row_gaia_realistic_error_budgets` — Gaia-style
  per-row budgets at the Galactic plane with magnitude scaling,
  joint `chord_max`-vs-single-call isolation, and a 1.6" sep
  quadrature regression; `test_pm_prior_both_sides_drift_inflation`
  — two-old-survey case verifying both sides contribute drift
  independently and quadratically.

### Added — Comprehensive E2E Test

* `test_all_matchers_e2e_on_shared_catalogues`: runs all 8 matchers (sky,
  skyerr, skyellipse, lr, ml, xgb, auf, macauff) against shared catalogues,
  verifies output shapes, score columns, and `find="all"` ≥ `find="best"`
  row counts. 174 non-slow tests pass.

### Changed — Documentation

* `CROSSMATCH_ALGORITHMS.md`: added AUF, XGBoost, and macauff as implemented
  algorithms; updated ML section with model save/load and shared helpers;
  removed AUF and XGB from Roadmap; added Design Notes section documenting
  shared helpers; added PM Drift Prior (Wilson 2023) as algorithm #12 with
  formulas, magnitude refinement, and RASTI 2,1 reference.
* `API.md`: updated MatchSpec table with lr/ml/xgb/auf/macauff fields;
  added advanced match features subsections for LR, ML/XGB, AUF, and macauff
  matchers with Python examples; added CLI examples for all new matchers;
  added PM drift prior subsection (formulas, code example, reference) and
  `pm_prior`/`pm_prior_magnitude_column` MatchSpec fields.

---

### Added — Progress Spinner for Remote Downloads

* **`Progress` class** (`src/xmatch/cli.py`): minimal, dependency-free
  Unicode-Braille spinner rendered in-place on stderr via `‎\r\033[K‎`.
  Auto-disables on non-TTY streams and when `XMATCH_NO_PROGRESS` or
  `NO_COLOR` is set. Background animation thread reads an atomic
  `_status` field written by backend callbacks — backend I/O stays on
  the main thread so pyvo / astroquery races are not introduced.
* **`progress_cb` callback plumbing** (`crossmatch.py`,
  `remote_tap.py`, `remote_cds.py`, `tap.py`): every public crossmatch
  entry point (`crossmatch`, `crossmatch_multi`, `union_match`,
  `fof_match`, `nway_match`, `crossmatch_request`) accepts and forwards
  `progress_cb` to TAP/CDS downloads. TAP async jobs emit live
  `"phase: queued"` / `"phase: running"` updates via a main-thread
  polling loop replacing `job.wait()`; CDS VizieR queries emit
  `"querying VizieR"` and `"received N tables"` around the synchronous
  `astroquery` call. Backwards compatible — passing `None` is a true
  no-op.
* **CLI integration** (`src/xmatch/cli.py::_execute_match`):
  `_execute_match` wraps every crossmatch call in
  ``with Progress(label, enabled=console.enabled) as p`` so users get
  visible download progress on every `xmatch match` / `xmatch match
  --union` / `xmatch match --fof` invocation. Labelled with the
  number of catalogues being matched (e.g. ``Cross-matching 3
  catalogues``).
* **3 Progress unit tests** (`tests/test_cli.py`):
  `test_progress_disabled_is_silent` (zero frames, zero leftover
  threads), `test_progress_enabled_emits_animated_frames` (CR +
  clear-to-EOL frames + label + latest status), and
  `test_progress_update_replaces_status` (final status wins).

---

## [0.4.0] — v0.4 Lean API + Catalogue Discovery

### Added

* `xmatch.MatchRequest` and `xmatch.SideOverrides` — typed dataclasses that
  replace the ad-hoc `**params` dict previously passed to
  `CrossMatch.crossmatch()`. All match parameters now have mypy-visible types.
* `CrossMatch.crossmatch_request(req: MatchRequest)` — preferred typed entry
  point for new code. Old `crossmatch(**params)` preserved for backward compat.
* `CrossMatch.crossmatch_multi(catalogues, **params)` — N-catalogue sequential
  pairwise crossmatching. Cat1×cat2 via `_dispatch`, then accumulator matched
  against cat3, cat4, … with per-stage column suffixes (`_2`, `_3`, `_4`, …).
  First catalogue's RA/Dec used as spatial reference chain-wide. Remote
  catalogues in positions 3+ auto-downloaded via first-source region inference.
* `xmatch.discovery` module — TAP_SCHEMA introspection (à la TOPCAT):
  `discover_tables()`, `discover_columns()`, `get_table_schema()`, and
  `detect_radec_columns()` for querying remote TAP services.
* `--search PATTERN` CLI command — searches public TAP endpoints (VizieR, Gaia,
  NOIRLab, CADC) and configured archives for tables matching a pattern.
* `--discover ENDPOINT` CLI command — lists all tables on a remote TAP endpoint.
* `--schema TABLE` (pair with `--discover`) — shows per-table column schema.
* `xmatch cat1 cat2 cat3 ...` — CLI now accepts N catalogues (previously
  exactly 2). 2-catalogue path unchanged; 3+ auto-routes to `crossmatch_multi`.
* Enhanced `--list` (table-format output with archive, access, row estimates)
  and `--describe` (richer output with error columns, default columns).

### Changed

* `io_utils.write_frame()` now uses polars streaming engine for
  `sink_parquet`/`sink_csv` — large results stream disk-to-disk without RAM.
* `sky_match()` and `_build_result()` now accept `right_suffix` — enables
  per-stage column disambiguation (`_2`, `_3`, `_4`, …) in multi-way chains.
  `sep_arcsec` and `p_match` from prior stages are deduplicated.
* `id_join()` accepts an optional `suffix` parameter, forwarded from
  `_local_match` for consistent multi-way ID joins.
* `_download_remote()` column selection guards prefix `"3"`+ (defaults to all
  columns instead of `side2.columns`).
* `_local_match()` accepts `right_suffix` and imports `_RIGHT_SUFFIX` from
  matchers for consistency.
* CLIs `_build_request()` replaced by `_build_params()` — constructs a kwargs
  dict from argparse; `main()` routes 2-cat vs N-cat paths.
* CLI positional args changed from `catalogue_1`/`catalogue_2` to `catalogues`
  with `nargs='*'`, allowing info commands (`--list`/`--discover`) without
  required catalogues.
* Dead `_params()` helper removed.
* `_download_remote()` and all internal match dispatch methods now accept
  `MatchRequest` instead of `spec + params` dict.
* Removed `_side_overrides` helper (superseded by `SideOverrides.as_dict()`).
* Version bumped to `0.4.0`.

### Notes

* The legacy `**params` API is still fully supported via
  `MatchRequest.from_legacy()`. No breaking changes.
* `SideOverrides.as_dict()` no longer carries `columns` (consumed later).

---

## [Unreleased] — v0.3 audit batch

### Added

* `astro_utils.sky_extent_from_frame(frame, ra_col, dec_col) -> Dict | None`
  computes the bounding cone of a polars `DataFrame`/`LazyFrame` purely as
  polars aggregations. The eager numpy version (`sky_extent`) is unchanged.
* `MatchSpec` was *not* extended — placeholder kept open per `MatchRequest`
  in AUDIT.md (deferred to v0.4).
* `docs/API.md` — concise public surface reference.
* `docs/AUDIT.md` — the audit's findings + rollout. (`docs/` is a new top-level
  folder; not deleted from the repo root.)
* IO tests verify bytes-decoding and multi-D column dropping.
* `tests/test_astro_utils.py` covers the new lazy `sky_extent_from_frame`.

### Changed

* `io_utils.astropy_table_to_polars` and `polars_to_astropy` now use
  `Table.to_arrow` / `Table.from_arrow` (Apache Arrow) instead of pandas.
  Removes one full serialization per TAP/CDS round-trip.
* `matchers._gather` switched from `df[pl.Series(values=idx)]` to
  `df.take(idx)` — Arrow-backed positional selection.
* `crossmatch.CrossMatch._local_source` no longer hard-raises on missing
  RA/Dec; `matchers.sky_match` raises instead. **This unblocks `id_join` on
  non-spatial catalogues** (header `object_id` + `mag_g` no longer triggers an
  `InputError`).
* `crossmatch.CrossMatch._download_remote` uses `sky_extent_from_frame` on
  LazyFrame inputs so local RA/Dec columns never materialise into numpy.
* `remote_tap.py` wraps every catalogue-derived identifier in ADQL
  double-quotes (`"col_name"`), preventing parse failures on reserved-word or
  space-containing column names.
* `__init__.__all__` now exposes `FrameLike`, `is_hats_dir`, `scan_frame`,
  `write_frame`, `astropy_table_to_polars`, `polars_to_astropy`.

### Removed

* `exceptions.AuthError`, `exceptions.TapUploadUnsupportedError` — never
  raised anywhere in `src/`; pruned for a leaner public surface.
* `auth.cache_result` decorator — unused; was a credential-leak footgun.

### Performance (informational)

* Arrow interop: removes one pandas serialization per TAP/CDS data path.
* `_gather`: ~10–15% faster inside the hot match loop.
* `_download_remote` (LazyFrame branch): no longer materialises local
  RA/Dec columns for cone computation.

### Notes

* No deprecations yet — Tier 2 changes (the `MatchRequest` dataclass and the
  real-data benchmark suite) require breaking-change consent and remain on the
  deferred list (see `AUDIT.md`).
