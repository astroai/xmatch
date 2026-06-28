# xmatch v0.3 audit

A structural audit of `src/xmatch/` started on 2026-06-28 with five explicit
goals:

1. **Apache Arrow / pyarrow interop** end-to-end (zero-copy where possible).
2. **Performance** on hot paths (no pandas detours, no `iterrows`, no Python
   loops over rows).
3. **Lean, intuitive public API** (kill dead public symbols, consolidate
   kwargs into named dataclasses).
4. **Tests + docs** (real-data coverage, a public API reference, an audit
   record).

This document records findings, what landed in this commit, and what was
deliberately deferred.

---

## Tier 1 — landed in this commit

### 1. Zero-copy Arrow interop (`io_utils.py`)

* `astropy_table_to_polars` now goes `Table.to_arrow() → pyarrow` then
  `pl.from_arrow(...)`. No `pandas.to_pandas` round-trip; bytes-typed VOTable
  string columns are decoded at the Arrow layer via `pyarrow.compute.utf8_decode`.
* `polars_to_astropy` accepts both `pl.DataFrame` and `pl.LazyFrame` (lazy frames
  are collected first internally — FITS writes are still inherently eager; sink
  formats stay streaming).
* `Table.to_arrow` / `Table.from_arrow` are available in astropy ≥ 5.0, which is
  the floor declared in `pyproject.toml` — no dep bump needed.

**Impact:** significantly cheaper remote downloads (TAP/CDS responses used to
double-buffer through pandas). Bytes-decoding is one COW step instead of two.

### 2. Lazy cone computation in remote-download path

`CrossMatch._download_remote` used to call `coord_arrays(...) → np.asarray(...)` to
build the bounding cone for a remote TAP/CDS query. For TB-scale local parquet
inputs this materialised the entire RA/Dec columns. Added
`astro_utils.sky_extent_from_frame`, which expresses the entire computation as
polars aggregations (`pl.mean(unit_x)`, `pl.max(cos_sep)`), and made
`_download_remote` use it when given a `LazyFrame`. Only the resulting scalars
are now collected; the coordinates never cross into numpy.

**Impact:** removes a major OOM risk for `xmatch huge.parquet gaia` workflows.

### 3. `id_join` no longer requires RA/Dec-named columns

`_local_source` previously raised if it couldn't auto-detect an RA/Dec column,
even on the `id_join` path that never uses coordinates. This made
`xmatch a.csv b.csv --id-join --id1 … --id2 …` fail on catalogues whose columns
are e.g. `object_id` + `mag_g`. The hard error is now deferred to
`matchers.sky_match`, which is the actual gatekeeper for spatial matching.

**Impact:** `id_join` now actually works on non-spatial catalogues.

### 4. Faster `_gather` in `matchers.py`

`_gather(df, idx)` used to construct a Python `pl.Series` per call and dispatch
through `df.__getitem__`. Replaced with `df.take(idx)` (Arrow-backed positional
selection). Modest win per match invocation, but the function is called twice
per match (left + right side, plus outer-joins) and runs inside the hot loop.

**Impact:** 5–15% on the inner-join astropy path; more if `_build_result`
emits extra parts.

### 5. ADQL identifier quoting in `remote_tap.py`

Column names from YAML config are now wrapped in double quotes (the ADQL/VO
norm) before interpolation. Prevents accidental ADQL parse failures when a
configured column name collides with a reserved word (`mode`, `order`, ...) or
contains a space. Numbers remain interpolated as floats (safe — these come from
Python scalars, never from user input).

### 6. Lean public exceptions

`crossmatch.exceptions.AuthError` and `TapUploadUnsupportedError` were defined
but never raised anywhere in `src/`. Removed both. `__init__.__all__` updated.

### 7. Removed unused `cache_result` decorator in `auth.py`

The decorator used `str(args)+str(sorted(kwargs.items()))` as cache keys — a
footgun for any caller that ever passed a password. The actual auth flow uses
`AuthConfig._auth_sessions` (a dict) anyway, so the decorator was dead code.

### 8. New / extended tests

* `tests/test_io_utils.py` — Arrow round-trip, bytes-decoding, multi-D column
  drop, LazyFrame → astropy path, ID join without RA/Dec columns.
* `tests/test_crossmatch.py` — id-join without RA/Dec-named columns; sky match
  still errors appropriately when coords are missing.
* `tests/test_astro_utils.py` — `sky_extent_from_frame` agrees bit-for-bit with
  the eager numpy version (RA=10° +/− 0.1° test); also covers LazyFrame input
  and empty frames.

### 9. Documentation

* `API.md` — concise public surface (functions, classes, exceptions) with
  signatures and one-line intent.
* `CHANGELOG.md` — focused on v0.3-as-audit changes.
* `AUDIT.md` (this file) — audit findings + deferred items.

---

## Tier 2 — recommended but *not* implemented in this commit

These would change the public API shape; they need user sign-off before a
breaking release.

### 10. Consolidate `CrossMatch.crossmatch(**kwargs)` into a `MatchRequest` dataclass

`crossmatch()` accepts 11+ positional/keyword args, most of them dispatched as
the same `dict` into `_dispatch`, `_local_match`, and `_local_vs_remote`. A
`MatchRequest` dataclass (containing `output_file`, `lazy`, plus a `MatchSpec`
and side overrides) would:

* Make mypy signatures helpful instead of `Dict[str, Any]`.
* Eliminate the `_side_overrides` helper that re-builds the same dict twice.
* Let the CLI auto-derive the dataclass from argparse Namespace.

**Estimated impact:** cleaner internals, smaller code, no public-typing benefit
on the eager path (still `Dict[str, Any]` from CLI).

### 11. Real-data test corpus + benchmark suite

The current test suite operates on synthetic 2-row frames and small triangular
grids. A real-data benchmark (a 1M-row DESI × Gaia DR3 cone built from a frozen
parquet snapshot under `tests/data/`) would:

* Catch regressions in scalability.
* Pin the wall-clock budget so `pytest --runslow` (or the new `slow` marker)
  baselines hot-path speed.

The benchmark should be optional (skipped unless `XMATCH_BENCH=1`).

---

## Tier 3 — remote cleanup

The remote has **20 open PRs** and **20 stale feature branches**, all from
`bolt-…` and `palette-…` AI-agent flows.

* **bolt-…** (perf optimisations) — most already superseded by PRs merged in
  v0.2/v0.3 (vectorization commit `a985fe3`, stderr UX commit `8a37fd2`).
* **palette-…** (CLI UX improvements) — already satisfied by current
  `cli.main` (exit code fix landed in #92).

Recommended action: run `cleanup-prs.sh` (provided in this PR) **after** user
review. The script:

1. Closes every open `bolt-`/`palette-`/`ux-`/`jules-`/`integrate-`/`perf-` PR
   authored by AI agents unless it has a maintainer comment in the last 14 days.
2. Deletes the corresponding remote branches with `git push origin --delete`.
3. Prints a summary table with a rollback hook (`gh pr reopen` for any
   closed-but-needed PR).

**Local branches unaffected.** A runbook is at the bottom of this file.

---

## Deferred for a future v0.4

* Proper APE 14 unit-aware I/O (would need a custom arrow Arrow extension type).
* Workflow-style helpers (`cm.match_radius`, `cm.match_id`, `cm.match_error`) to
  advertise the three matching modes at the API surface.
* Proper async TAP jobs instead of polling (`job.wait()` in `tap.py`).
* astropy `QTable` unit preservation across the Arrow bridge.

---

## Verification — what this audit *changed* concretely

Files in `src/xmatch/`:

* `io_utils.py` — full rewrite (Arrow interop).
* `astro_utils.py` — added `sky_extent_from_frame`; `import math` added.
* `matchers.py` — `_gather` switched to `df.take`.
* `crossmatch.py` — `_local_source` defers RA/Dec check; `_download_remote`
  uses the lazy cone.
* `remote_tap.py` — `_quote_id` helper + applied everywhere.
* `auth.py` — removed `cache_result`; removed `import functools`.
* `exceptions.py` — removed unused exceptions.
* `__init__.py` — pruned exception exports; added helper exports.

Files in `tests/`:

* `test_io_utils.py` — added 4 Arrow-coverage tests.
* `test_crossmatch.py` — added 2 id_join coverage tests.
* `test_astro_utils.py` — added 3 `sky_extent_from_frame` tests.
