# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — v0.5 Audit + HATS + Data Lab

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
  expected column (e.g., future API change).
* **16 HATS tests** (`tests/test_hats.py`): fully mocked (no LSDB required),
  covering alias resolution, crossmatch parameter passthrough, multi-row
  `find="all"` results, join type validation, local-frame conversion, column
  renaming, and `crossmatch_multi` routing.

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
  `decals_x_gaia`, `allwise_x_gaia`, and qualifiers.
* **2 Data Lab source resolution tests** (`tests/test_crossmatch.py`): verify
  aliases resolve to TAP-backed `CatalogueSource` with correct table names,
  archive, and column metadata. Tractor vs object table routing verified.
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
