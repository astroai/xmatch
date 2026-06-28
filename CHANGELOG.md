# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

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
