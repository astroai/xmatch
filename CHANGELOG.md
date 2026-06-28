# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — v0.4 Lean API + Catalogue Discovery

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
