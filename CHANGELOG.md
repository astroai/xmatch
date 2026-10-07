# Changelog

All notable changes to `xmatch` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

No changes yet.

## [0.5.0] — 2026-10-07

Available from the public [GitHub release at tag `v0.5.0`](https://github.com/astroai/xmatch/releases/tag/v0.5.0).
No PyPI package is published because another project owns the `xmatch` name.

### Added

- **Source-Preserving Observation & Candidate Releases** (`observations.py`, `candidates.py`):
  - Publish source inventories with normalized photometry and property-evidence ledgers in immutable, checksummed Parquet releases (`write_observation_release`, `verify_observation_release`).
  - Publish scoped sparse candidate-pair releases (`write_candidate_release`, `verify_candidate_release`) and normalize caller-supplied per-target hypothesis weights with an explicit no-match alternative (`normalize_candidate_hypotheses`).
- **Versioned Association Contract** (`association.py`):
  - Deterministic `xmatch.association.v1` pair records, atomic release directories, component sidecar releases (`xmatch.association.component.release.v1`), component delta classifications (`created`, `continued`, `merged`, `split`, `retired`), and bijective member equivalence crosswalks.
- **HATS Mirroring & Distributed Full-Sky Union** (`mirror.py`, `ray_union.py`, `storage.py`):
  - `xmatch sync` CLI command and `sync_catalogue` / `ensure_mirrored` APIs to mirror remote TAP and HATS catalogues (over HTTP or CANFAR `vos:`) into a local HATS cache with keyset paging, token-bucket rate limiting, endpoint failover, and resumable checkpoints.
  - `engine="ray-union"`: distributed $N$-survey full-outer join over mirrored HATS inputs on Ray, producing partitioned HATS trees (`Norder=*/Dir=*/Npix=*.parquet`) with chunk-level resume state.
  - `ray_engine.py`: parallel HEALPix pixel-batch cone crossmatching (`engine="ray"`).
- **Astrometric Kinematics & Probabilistic Matchers** (`matchers.py`, `out_of_core.py`, `bayes.py`):
  - Six-dimensional epoch propagation (`target_epoch`) and 5-parameter astrometric covariance transport through motion Jacobians for both `skyellipse` and `skyerr` (eager and out-of-core spill paths).
  - Wilson (2023)-inspired proper-motion drift uncertainty model (`pm_prior=True`) with per-row error quadrature inflation and optional magnitude refinement.
  - Sutherland & Saunders (1992) Likelihood Ratio (`matcher="lr"`), Random Forest (`matcher="ml"`), XGBoost/LightGBM (`matcher="xgb"`), lightweight AUF-inspired empirical separation/background scoring (`matcher="auf"`), and a flux-augmented heuristic (`matcher="macauff"`); the latter two are not full reproductions of the published methods.
  - Multi-condition SQL post-filtering (`MatchSpec.filter_expr`), feature-weighted ranking of spatial candidates (`MatchSpec.extra_distance_cols`), and HEALPix pixel-group batching (`MatchSpec.batch_size`; this is not a general process-memory bound).
- **Formats & Catalogues** (`io_utils.py`, `xmatch.yaml`):
  - Native streaming `.tsv` and `.tab` reads and writes via Polars.
  - Added NOIRLab Data Lab catalogues (`nsc`, `des`, `decals`, `smash`, `unwise`, `allwise_dl`, and pre-computed `*_x_gaia` crossmatch tables) and COSMOS (`cosmos-web`, `zcosmos`).
  - Terminal progress indicator for remote TAP and VizieR downloads.

### Fixed

- Distributed HATS union now keeps output partitions spatially coherent across mixed input orders and RING/NESTED layouts, and reports non-null union routing coordinates. Tuple caps select the lowest-separation results from the complete candidate product; planner scans account for declared positional uncertainty and epoch motion.
- Friends-of-Friends matching searches cross-catalogue secondary links while retaining the primary-anchored component output policy.
- Probabilistic scores require declared positional errors or explicit error floors; the documentation states the equal-prior and isotropic-score assumptions. Remote `skyerr`, `skyellipse`, and `target_epoch` queries require an explicit search cone when service-side uncertainty or motion margins cannot be bounded.
- Invalid `filter_expr` SQL and missing, nonnumeric, or nonfinite declared ranking features now raise `CrossMatchError` instead of silently retaining rows or dropping/imputing features.
- ML synthetic negative sampling uses a local fixed seed, making repeated inputs reproducible without advancing the caller's NumPy random state.
- STILTS per-primary `find="best"` now uses `best1`, avoiding the symmetric one-to-one selection performed by STILTS `best`.
- An unreachable explicit Ray address no longer silently starts a local cluster under strict fallback policy, and the caller's address setting is preserved. Memory and storage budgets documented in GiB now use binary GiB.

### Changed

- Updated the Python requirement to $\ge 3.13$ and modernized type annotations to PEP 585 / PEP 604 built-ins. The lockfile targets Python 3.13. A targeted Python 3.14.8 / Polars 2 compatibility run passed 206 tests and skipped 24; it was not the complete release gate on that stack.
- Streamlined core runtime dependencies to `numpy`, `polars`, `pyarrow`, `astropy`, `scipy`, `pyyaml`, and `pyvo`, with direct PyArrow/NumPy `Table`/`MaskedColumn` conversions and optional `xmatch[cds]` extra for `astroquery`.
- Consolidated HATS reading and writing on the native HATS engine (`hats` + `cdshealpix`). Ray-union uses compressed interval planning and measured per-source halos for adaptive/mixed-order and RING inputs and uncertainty- or epoch-adjusted searches; it may scan additional partitions. The pairwise `hats_native` fast path remains fixed-radius, uniform-order NESTED and globally materializes other layouts or error/epoch-adjusted inputs.
- Removed the unusable `torchsky` package extra because that distribution is not published on PyPI. The tensor engine requires a separately installed compatible Torchsky checkout; the documented integration was tested with the sibling 0.4 development source installed editable and Torchfits 1.0.0 from PyPI.

---

## [0.4.0] — 2026-06-15

### Added

- Typed request API (`MatchRequest`, `SideOverrides`, `CrossMatch.crossmatch_request`).
- Multi-catalogue sequential crossmatching (`CrossMatch.crossmatch_multi`) and CLI support for $N \ge 3$ catalogues.
- Remote archive discovery module (`xmatch.discovery`) and CLI subcommands (`search`, `discover`, `adopt`, `list`, `describe`).
- Lazy bounding-cone estimation (`astro_utils.sky_extent_from_frame`) and streaming Parquet/CSV output sinks (`io_utils.write_frame`).

### Changed

- Switched Astropy–Polars table conversions (`astropy_table_to_polars`, `polars_to_astropy`) to Apache Arrow column buffers.
- Quoted ADQL identifiers in remote TAP queries to handle reserved words and special characters safely.
