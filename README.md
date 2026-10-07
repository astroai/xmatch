# xmatch

`xmatch` matches and joins astronomical catalogues from local files, HATS trees, and configured remote archives. Matching features depend on the selected engine and the metadata supplied for each catalogue; the guides document those limits and the memory trade-offs.

```bash
xmatch match catalog1.parquet catalog2.csv -o matches.parquet -r 1.0
```

Inputs can be local Parquet, CSV, TSV, or FITS files, HATS directories, Polars frames, or configured remote catalogues. Coordinate names can be configured or detected. Proper-motion propagation uses declared epoch and motion metadata; mixed coordinate frames are rejected and generic catalogue discovery does not invent positional uncertainties.

---

## 📚 Complete Documentation Suite

All detailed architectural references and user guides are organized in the [`docs/`](docs/) directory:

- **[User Guide & Cookbooks (`docs/usage.md`)](docs/usage.md)**: Installation, local and remote matching, explicit memory controls, and distributed workflows with their limits.
- **[Public API Reference (`docs/api.md`)](docs/api.md)**: Full Python API specifications for `CrossMatch`, `MatchRequest`, `MatchSpec`, observation/candidate release helpers, engines, storage abstractions, and exception hierarchies.
- **[Crossmatch Algorithms & Math (`docs/algorithms.md`)](docs/algorithms.md)**: Matcher definitions, uncertainty assumptions, score semantics, and engine behavior.
- **[Source & Measurement Releases (`docs/observations.md`)](docs/observations.md)**: Lazy, source-preserving measurement ledgers, photometric unit/limit normalization, property evidence, and checksummed Parquet releases.
- **[Candidate & Association Pipeline (`docs/catalogue-pipeline.md`)](docs/catalogue-pipeline.md)**: End-to-end workflow from pinned survey observation and sparse-candidate releases to per-target association hypotheses.
- **[Association Contract v1 (`docs/association-v1.md`)](docs/association-v1.md)**: Audit-grade versioned association record specification, release directories, component sidecars, and member equivalence tracking.

---

## Capabilities

- **Universal Multi-Format I/O**:
  - Ingests and streams `.parquet`, `.csv`, `.tsv`, `.tab`, `.fits` (via high-speed Torchfits with Astropy fallback), and HATS.
  - Remote storage connectors: POSIX filesystem, CANFAR VOSpace (`vos:`), HTTP/HTTPS, and S3.
- **File I/O and explicit memory controls**:
  - Parquet, CSV, TSV, FITS, and HATS readers with Parquet/CSV/TSV/FITS outputs.
  - Output sinks can stream serialization. Most matching engines still materialize inputs and matches; local pairwise CSV/Parquet requests can opt into the spill path with `--memory-budget-bytes` and file output. See [memory limits](docs/usage.md#pairwise-spill-matching-on-one-machine).
- **Astrometric and probabilistic matchers**:
  - Simple radius (`sky`) and adaptive per-row astrometric uncertainties (`skyerr`).
  - 2D Gaussian error ellipses via Mahalanobis distance (`skyellipse`).
  - Proper-motion epoch propagation (`target_epoch`) when epoch and motion metadata are supplied, plus an explicit population drift model (`pm_prior`) for missing motion.
  - Sutherland & Saunders (1992) Likelihood Ratio (`lr`) with magnitude background subtraction.
  - Machine learning classifiers (`ml` Random Forest, `xgb` XGBoost/LightGBM) with self-match pseudo-labeling.
  - AUF-inspired empirical separation/background scores (`auf`) and a flux-augmented heuristic (`macauff`). These are not full reproductions of the published methods and their scores are not calibrated probabilities.
  - Bayesian positional qualification (`probabilistic=True`) or photometric qualification (`prior_columns`) with declared errors or explicit error floors, plus simultaneous $N$-way scores (`nway_match`). These scores use stated assumptions and are not population-calibrated probabilities.
  - Friends-of-Friends transitive closure object bundles (`fof_match`).
- **Complete Relational Joins**:
  - Inner (`1and2`), Left (`all1`), Right (`all2`), Full Outer (`1or2`, `all`), and Anti-joins (`1not2`, `2not1`).
  - Relational ID joins on survey identifiers (`--id-join`).
  - Multi-catalogue sequential chains (`crossmatch_multi`).
- **Local Mirroring & Offline Caching**:
  - `xmatch sync`: Mirror remote TAP and HATS surveys into a durable local HATS cache with keyset paging, rate limiting, and progress checkpoints.
- **Distributed execution**:
  - `engine="ray"` parallelizes pairwise HEALPix candidate searches. The coordinator still holds the input frames and final result in memory.
  - `engine="ray-union"` mirrors inputs into HATS and writes a full-sky, full-outer HATS union. It supports `sky`, `skyerr`, and `skyellipse`; compressed interval planning handles mixed-order/RING layouts and uncertainty or epoch halos in distributed tasks. Planning may scan additional partitions. Output trees add non-null `_union_ra` and `_union_dec` routing columns and declare NESTED ordering.

---

## Installation

Requires Python $\ge 3.13$. The checked Pixi environment and local release gate use Python 3.13.

### Recommended: Pixi

```bash
# Clone the 0.5.0 release tag (the private repository requires access)
git clone --branch v0.5.0 https://github.com/astroai/xmatch.git
cd xmatch

# Install environment and dependencies
pixi install

# Run lint, formatting, bytecode, and offline test checks
pixi run preflight-push
pixi run ci-local
```

### Standard: Pip

The 0.5.0 release is distributed from the private GitHub repository at tag
`v0.5.0`; repository access is required. The PyPI name [`xmatch`](https://pypi.org/project/xmatch/)
belongs to a different project, so this project is not published on PyPI.

```bash
git clone --branch v0.5.0 https://github.com/astroai/xmatch.git
cd xmatch
python -m pip install .

# Optional integrations; use only the extras you need
python -m pip install ".[cds,hats-ray,torchfits,ml]"
```

The tensor-native `torchsky` engine is not a package extra: no `torchsky`
distribution is published on PyPI. To use it, install xmatch alongside a
compatible Torchsky checkout. The integration was tested with the sibling
Torchsky 0.4 development source installed editable and Torchfits 1.0.0 from
PyPI.

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
