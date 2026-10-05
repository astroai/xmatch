# Changelog

All notable changes to `xmatch` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0] — 2026-10-05

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
  - Probabilistic proper-motion drift prior (`pm_prior=True`, Wilson 2023) with per-row error quadrature inflation and optional magnitude refinement.
  - Sutherland & Saunders (1992) Likelihood Ratio (`matcher="lr"`), Random Forest (`matcher="ml"`), XGBoost/LightGBM (`matcher="xgb"`), Astrometric Uncertainty Function (`matcher="auf"`), and flux-enhanced AUF (`matcher="macauff"`).
  - Multi-condition SQL post-filtering (`MatchSpec.filter_expr`), N-dimensional feature-weighted `cKDTree` matching (`MatchSpec.extra_distance_cols`), and bounded-memory HEALPix batching (`MatchSpec.batch_size`).
- **Formats & Catalogues** (`io_utils.py`, `xmatch.yaml`):
  - Native streaming `.tsv` and `.tab` reads and writes via Polars.
  - Added NOIRLab Data Lab catalogues (`nsc`, `des`, `decals`, `smash`, `unwise`, `allwise_dl`, and pre-computed `*_x_gaia` crossmatch tables) and COSMOS (`cosmos-web`, `zcosmos`).
  - Terminal progress indicator for remote TAP and VizieR downloads.

### Changed

- Updated Python requirement to $\ge 3.13$ (with full Python 3.14 compatibility) and modernized all type annotations to PEP 585 / PEP 604 built-ins.
- Streamlined core runtime dependencies to `numpy`, `polars`, `pyarrow`, `astropy`, `scipy`, `pyyaml`, and `pyvo`, with direct PyArrow/NumPy `Table`/`MaskedColumn` conversions and optional `xmatch[cds]` extra for `astroquery`.
- Consolidated HATS reading, writing, and spatial crossmatching on the native HATS engine (`hats` + `cdshealpix`).

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
