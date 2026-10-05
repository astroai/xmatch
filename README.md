# xmatch

The high-performance, format-agnostic **Swiss-Army-knife for astronomical catalogue crossmatching**.

`xmatch` matches, joins, and aggregates astronomical catalogues across any scale—from interactive laptop tables to full-sky billions-of-rows surveys:

```bash
xmatch match catalog1.parquet catalog2.csv -o matches.parquet -r 1.0
```

`catalog1` and `catalog2` can each be a **local file** (Parquet, CSV, TSV, FITS), a **HATS spatial directory**, or a **remote TAP/ADQL catalogue** (Gaia, VizieR, NOIRLab Data Lab). `xmatch` auto-detects coordinate columns, resolves aliases, handles kinematics, and streams results directly through Polars.

---

## 📚 Complete Documentation Suite

All detailed architectural references, user guides, and roadmaps are organized in the [`docs/`](docs/) directory:

- **[User Guide & Cookbooks (`docs/usage.md`)](docs/usage.md)**: End-to-end recipes covering CLI commands, universal formats, remote archive federation, SOTA matching algorithms, proper motions, out-of-core scaling, and Ray clustering.
- **[Public API Reference (`docs/api.md`)](docs/api.md)**: Full Python API specifications for `CrossMatch`, `MatchRequest`, `MatchSpec`, observation/candidate release helpers, engines, storage abstractions, and exception hierarchies.
- **[Crossmatch Algorithms & Math (`docs/algorithms.md`)](docs/algorithms.md)**: Comprehensive survey of all 12 implemented algorithms, error covariance modeling, literature citations, and a deep dive into **Astropy vs. SciPy cKDTree**.
- **[Source & Measurement Releases (`docs/observations.md`)](docs/observations.md)**: Lazy, source-preserving measurement ledgers, photometric unit/limit normalization, property evidence, and checksummed Parquet releases.
- **[Candidate & Association Pipeline (`docs/catalogue-pipeline.md`)](docs/catalogue-pipeline.md)**: End-to-end workflow from pinned survey observation and sparse-candidate releases to per-target association hypotheses.
- **[Association Contract v1 (`docs/association-v1.md`)](docs/association-v1.md)**: Audit-grade versioned association record specification, release directories, component sidecars, and member equivalence tracking.

---

## Key Features

- **Universal Multi-Format I/O**:
  - Ingests and streams `.parquet`, `.csv`, `.tsv`, `.tab`, `.fits` (via high-speed Torchfits with Astropy fallback), and HATS.
  - Remote storage connectors: POSIX filesystem, CANFAR VOSpace (`vos:`), HTTP/HTTPS, and S3.
- **Polars Streaming Engine**:
  - Memory-bounded streaming execution via Polars `LazyFrame` sinks (`sink_parquet(streaming=True)`, `sink_csv(streaming=True)`).
  - Out-of-core memory budgeting (`--batch-size`) to process TB-scale catalogues on modest hardware.
- **SOTA Astrometric & Probabilistic Matchers**:
  - Simple radius (`sky`) and adaptive per-row astrometric uncertainties (`skyerr`).
  - 2D Gaussian error ellipses via Mahalanobis distance (`skyellipse`).
  - Proper-motion epoch propagation (`target_epoch`) and probabilistic PM drift prior for legacy catalogues (Wilson 2023 `pm_prior`).
  - Sutherland & Saunders (1992) Likelihood Ratio (`lr`) with magnitude background subtraction.
  - Machine learning classifiers (`ml` Random Forest, `xgb` XGBoost/LightGBM) with self-match pseudo-labeling.
  - Wilson & Naylor (2017) Astrometric Uncertainty Function (`auf`) and flux-enhanced AUF (`macauff`).
  - Budavári & Szalay (2008) Bayesian qualification (`p_match`) and simultaneous $N$-way posterior (`nway_match`).
  - Friends-of-Friends transitive closure object bundles (`fof_match`).
- **Complete Relational Joins**:
  - Inner (`1and2`), Left (`all1`), Right (`all2`), Full Outer (`1or2`, `all`), and Anti-joins (`1not2`, `2not1`).
  - Relational ID joins on survey identifiers (`--id-join`).
  - Multi-catalogue sequential chains (`crossmatch_multi`).
- **Local Mirroring & Offline Caching**:
  - `xmatch sync`: Mirror remote TAP and HATS surveys into a durable local HATS cache with keyset paging, rate limiting, and progress checkpoints.
- **Persistent Full-Sky Master Unions**:
  - `engine="ray-union"`: Distributed $N$-survey full-outer join producing partitioned HATS directories (`Norder/Dir/Npix.parquet`).

---

## Installation

Requires Python $\ge 3.10$.

### Recommended: Pixi

```bash
# Clone the repository
git clone https://github.com/astroai/xmatch.git
cd xmatch

# Install environment and dependencies
pixi install

# Run fast preflight check
pixi run preflight-push
```

### Standard: Pip

```bash
# Core package
pip install xmatch

# Full installation with all optional accelerators
pip install "xmatch[hats,ray,torchsky,torchfits,ml,nway]"
```

---

## Command Line Quick Reference

| Task | Command |
|---|---|
| Match two local files | `xmatch match cat1.parquet cat2.csv -o matches.parquet -r 1.5` |
| Match against remote Gaia DR3 | `xmatch match targets.csv gaia -o targets_gaia.parquet -r 1.0` |
| Astrometric error ellipse match | `xmatch match xray.fits optical.parquet --matcher skyellipse --max-error 3.0` |
| Proper-motion drift prior match | `xmatch match old_survey.csv gaia --target-epoch 2016.0 --pm-prior -r 2.0` |
| Relational ID join | `xmatch match cat1.parquet cat2.parquet --id-join --id1 objid --id2 objid` |
| Multi-survey 3-way chain | `xmatch match gaia wise.csv 2mass.parquet -r 1.5 -o 3way.parquet` |
| Distributed HATS master union | `xmatch match gaia allwise 2mass des --union --engine ray-union -o /data/master.hats` |
| Mirror remote surveys to local cache | `xmatch sync gaia allwise 2mass` |
| Search remote TAP archives | `xmatch search des` |
| Discover archive schemas | `xmatch discover noirlab --schema nsc_dr2.object` |

---

## Python API Quickstart

```python
from xmatch import CrossMatch, MatchRequest, MatchSpec

cm = CrossMatch()

# 1. Preferred typed request interface
req = MatchRequest(
    cat1="optical.parquet",
    cat2="infrared.csv",
    spec=MatchSpec(radius_arcsec=1.5, matcher="sky", find="best"),
    engine="fast",  # SciPy cKDTree multi-threaded on 3D unit sphere
)
result_df = cm.crossmatch_request(req)

# 2. Multi-catalogue sequential intersection
result_3way = cm.crossmatch_multi(
    ["optical.parquet", "infrared.csv", "radio.fits"],
    radius_arcsec=2.0,
    join_type="1and2",
)

# 3. Friends-of-Friends object bundles
bundles = cm.fof_match(
    ["survey_a.parquet", "survey_b.parquet", "survey_c.parquet"],
    radius_arcsec=1.5,
)
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

