# xmatch Public Python API Reference

Stable public surface only. Anything not exported from `xmatch.__init__` is considered internal.

---

## Top-Level Entry Points (`xmatch.CrossMatch`)

`CrossMatch` is the central orchestrator. It parses configuration, manages remote archive discovery, handles coordinate conversions, and dispatches to the optimal local or distributed engine.

```python
from xmatch import CrossMatch

cm = CrossMatch()  # loads bundled xmatch.yaml and user overrides
```

### Methods Summary

| Method | Purpose |
|---|---|
| [`crossmatch(...)`](#crossmatch) | Classic spread-arguments pairwise matching between two catalogues. |
| [`crossmatch_request(req)`](#crossmatch_request) | **Preferred** typed entry point using a `MatchRequest` dataclass. |
| [`crossmatch_multi(...)`](#crossmatch_multi) | $N$-catalogue sequential pairwise crossmatching ($A \times B \times C \dots$). |
| [`union_match(...)`](#union_match) | Master union crossmatch producing a full-outer-join catalogue (in-memory or distributed HATS). |
| [`nway_match(...)`](#nway_match) | Simultaneous Bayesian $N$-way crossmatching (Budavári & Szalay 2008 posterior). |
| [`fof_match(...)`](#fof_match) | Friends-of-Friends transitive closure clustering detections into object bundles. |
| `resolve_source(value, overrides)` | Builds a `CatalogueSource` from a file path, DataFrame, or configured name. |
| `get_catalogue_config(name)` | Returns the merged YAML configuration for a catalogue alias or name. |

---

### `crossmatch_request`

**The preferred, strongly-typed API entry point.**

```python
def crossmatch_request(self, req: MatchRequest) -> pl.DataFrame | pl.LazyFrame | None
```

**Parameters**:
- `req` (`MatchRequest`): Fully validated request dataclass containing input sources, matching specifications, engine choices, and column overrides.

**Returns**:
- `pl.DataFrame` (eager, default) or `pl.LazyFrame` (when `req.lazy=True`). Returns `None` when `req.output_file` is provided and results are streamed to disk.

---

### `crossmatch`

Classic spread-arguments entry point.

```python
def crossmatch(
    self,
    cat1: FrameInput,
    cat2: FrameInput,
    output_file: str | Path | None = None,
    lazy: bool = False,
    radius_arcsec: float = 1.0,
    matcher: str = "sky",
    engine: str = "auto",
    join_type: str = "1and2",
    find: str = "best",
    **params: Any,
) -> pl.DataFrame | pl.LazyFrame | None
```

**Key Parameters**:
- `cat1`, `cat2` (`str | Path | pl.DataFrame | pl.LazyFrame`): Input catalogues. Can be local file paths (`.parquet`, `.csv`, `.tsv`, `.fits`), HATS directory paths, remote TAP catalogue names (e.g. `"gaia"`, `"des"`), or in-memory Polars frames.
- `output_file` (`str | Path | None`): Destination path (`.parquet`, `.csv`, `.tsv`, `.fits`, `.hats`, or `vos:` URI). Streams results via Polars streaming sinks.
- `lazy` (`bool`): If `True`, returns an uncollected `pl.LazyFrame`.
- `radius_arcsec` (`float`): Spatial matching radius in arcseconds.
- `matcher` (`str`): Match algorithm: `"sky"`, `"skyerr"`, `"skyellipse"`, `"lr"`, `"ml"`, `"xgb"`, `"auf"`, `"macauff"`.
- `engine` (`str`): Spatial engine: `"auto"`, `"fast"`, `"zone"`, `"ray"`, `"ray-union"`, `"astropy"`, `"stilts"`, `"torchsky"`.
- `join_type` (`str`): Relational join mode: `"1and2"` (inner), `"all1"` (left), `"all2"` (right), `"1or2"` or `"all"` (full outer), `"1not2"` (left anti), `"2not1"` (right anti).
- `find` (`str`): Candidate selection policy: `"best"` (nearest/top-ranked match) or `"all"` (all pairs within radius).
- `**params`: Per-side coordinate column overrides (`ra_column1`, `dec_column1`), remote cone constraints (`ra`, `dec`, `radius_deg`), and algorithm-specific parameters.

---

### `crossmatch_multi`

Chains sequential pairwise crossmatches across $N \ge 2$ catalogues. Catalogue 1 provides the spatial reference coordinate frame throughout the chain. Overlapping column names automatically receive `_2`, `_3`, `_4`, etc., suffixes.

```python
def crossmatch_multi(
    self,
    catalogues: Sequence[FrameInput],
    output_file: str | Path | None = None,
    lazy: bool = False,
    **params: Any,
) -> pl.DataFrame | pl.LazyFrame | None
```

```python
# 3-way intersection across Parquet, CSV, and remote Gaia
result = cm.crossmatch_multi(
    ["my_sources.parquet", "wise_cutout.csv", "gaia_esa"],
    ra=279.23,
    dec=38.78,
    radius_deg=0.01,
    radius_arcsec=1.5,
)
```

---

### `union_match`

Constructs a master union catalogue via sequential full outer joins or Ray distributed union. Single-source rows retain their original coordinates and attributes while empty counterpart slots are populated with nulls.

```python
def union_match(
    self,
    catalogues: Sequence[FrameInput],
    output_file: str | Path | None = None,
    lazy: bool = False,
    **params: Any,
) -> pl.DataFrame | pl.LazyFrame | None
```

- When full-sky remote surveys are passed without a cone constraint (`ra`, `dec`, `radius_deg`), `union_match` automatically routes to `engine="ray-union"`, mirroring the full tables into HATS and producing a distributed partitioned master HATS catalogue.
- Supports `engine="ray"` across all matchers (`sky`, `skyerr`, `skyellipse`, `lr`, `ml`, `xgb`, `auf`, `macauff`) and output formats, and `engine="ray-union"` for distributed HATS output with `matcher="sky"`, `matcher="skyerr"`, or `matcher="skyellipse"`, `target_epoch`, and `pm_prior`.

---

### `nway_match`

Simultaneous Bayesian $N$-catalogue crossmatching implementing the Budavári & Szalay (2008) multi-catalogue Bayes factor:

```python
def nway_match(
    self,
    catalogues: list[FrameInput],
    output_file: str | Path | None = None,
    *,
    radius_arcsec: float = 1.0,
    prior_columns: list[str] | None = None,
    max_tuples_per_source: int = 10_000,
    chunk_size: int = 50_000,
    hats_threshold: int = 100_000,
    **params: Any,
) -> pl.DataFrame | None
```

Evaluates Cartesian candidate tuples $c_1 \times c_2 \times \dots \times c_N$ inside the spatial radius and scores each joint tuple with positional and photometric KDE Bayes factors, emitting a joint `p_match` posterior in $[0, 1]$. Supports `engine="ray"` for distributed HEALPix candidate discovery and parallel `@ray.remote` chunk tuple scoring, as well as `target_epoch`, `pm_prior`, and HATS catalogue inputs.

---

### `fof_match`

Friends-of-Friends transitive closure algorithm. Groups detections across multiple surveys into connected components (bundles) via union-find graph clustering.

```python
def fof_match(
    self,
    catalogues: list[FrameInput],
    output_file: str | Path | None = None,
    *,
    radius_arcsec: float = 1.0,
    hats_threshold: int = 100_000,
    **params: Any,
) -> pl.DataFrame | None
```

Supports `engine="ray"` for distributed pairwise edge discovery, `target_epoch`, `pm_prior`, and HATS catalogue inputs.

**Output Columns**:
- `bundle_id`: Unique integer assigned to the connected component.
- `n_cats`: Number of distinct catalogues contributing to the bundle.
- `_src_cats`: Membership string (e.g. `"1+2+3"`).

---

## Dataclasses and Request Specifications

### `MatchRequest`

Consolidated request dataclass.

| Field | Type | Default | Description |
|---|---|---|---|
| `cat1`, `cat2` | `FrameInput` | *(required)* | File paths, catalogue names, or Polars frames. |
| `spec` | `MatchSpec` | `MatchSpec()` | Matching parameters and algorithm specification. |
| `output_file` | `str | Path | None` | `None` | Stream output to file. |
| `lazy` | `bool` | `False` | Return an uncollected `pl.LazyFrame`. |
| `engine` | `str` | `"auto"` | Spatial engine selector. |
| `id_join` | `bool` | `False` | Switch from spatial matching to a relational ID join. |
| `id_column_1`, `id_column_2` | `str | None` | `None` | Custom ID column names for ID join. |
| `side1`, `side2` | `SideOverrides` | `SideOverrides()` | Per-side column overrides. |
| `ra`, `dec`, `radius_deg` | `float | None` | `None` | Bounding cone for remote TAP/CDS downloads. |
| `probabilistic` | `bool` | `False` | Append Budavári-style `p_match` posterior column. |

### `MatchSpec`

Defines the algorithm criteria:

| Field | Type | Default | Description |
|---|---|---|---|
| `radius_arcsec` | `float` | `1.0` | Search radius for `matcher="sky"`. |
| `matcher` | `str` | `"sky"` | `"sky"`, `"skyerr"`, `"skyellipse"`, `"lr"`, `"ml"`, `"xgb"`, `"auf"`, `"macauff"`. |
| `max_error` | `float` | `3.0` | $N$-$\sigma$ cap for `skyerr` and Mahalanobis $d^2$ cap for `skyellipse`. |
| `join_type` | `str` | `"1and2"` | `"1and2"`, `"1or2"`, `"all1"`, `"all2"`, `"1not2"`, `"2not1"`, `"all"`. |
| `find` | `str` | `"best"` | `"best"` (nearest/top-ranked) or `"all"` (all pairs within radius). |
| `target_epoch` | `float | None` | `None` | Julian-year epoch for proper-motion propagation. |
| `pm_prior` | `bool` | `False` | Enable probabilistic PM drift prior (Wilson 2023). |
| `pm_prior_magnitude_column` | `str | None` | `None` | Magnitude column for Wilson (2023) distance-proxy scaling. |
| `filter_expr` | `str | None` | `None` | Polars SQL WHERE clause for post-match filtering. |
| `extra_distance_cols` | `dict[str, float]` | `{}` | Extra columns & weights for $N$-dimensional cKDTree ranking. |
| `batch_size` | `int | None` | `None` | HEALPix pixel batch size for out-of-core memory management. |
| `lr_magnitude_column` | `str | None` | `None` | Secondary magnitude column for Likelihood Ratio (`matcher="lr"`). |
| `lr_q` | `float` | `0.8` | Prior probability of counterpart detection in primary survey. |
| `ml_color_columns` | `list[str]` | `[]` | Colour/magnitude feature columns for RF/XGB matchers. |
| `ml_model_path` | `str | None` | `None` | Save/load path for Random Forest model artifact (`.joblib`). |
| `xgb_model_path` | `str | None` | `None` | Save/load path for XGBoost model artifact (`.joblib`). |
| `macauff_flux_columns` | `list[str]` | `[]` | Multi-band flux columns for `matcher="macauff"`. |
| `prior_columns` | `list[str]` | `[]` | Photometric columns for Bayesian KDE prior (`p_match`). |

### `SideOverrides`

```python
@dataclass
class SideOverrides:
    ra_column: str | None = None
    dec_column: str | None = None
    id_column: str | None = None
    columns: list[str] | None = None
```

---

## Execution Engines Matrix

`xmatch` decouples matching algorithms from spatial indexing engines.

| Engine | Primary Backend | Threading / Scaling | Memory Overhead | Supported Algorithms & Features |
|---|---|---|---|---|
| **`fast`** | `scipy.spatial.cKDTree` | OpenMP multi-threaded (`workers=-1`) | Minimal (3D unit vectors) | All 8 matchers (`sky`, `skyerr`, `skyellipse`, `lr`, `ml`, `xgb`, `auf`, `macauff`), `extra_distance_cols`, `probabilistic=True`, `target_epoch`, `pm_prior`, `nway_match`, `fof_match`. |
| **`zone`** | `cdshealpix` + `cKDTree` | Single-machine HEALPix sharding | Bounded per pixel | Full feature parity with `fast`; partitions data into HEALPix pixels with exact boundary neighbour expansion. |
| **`ray`** | Ray cluster | Multi-node / Multi-process Ray tasks | Distributed | Full feature parity with `fast` across flat files, in-memory frames, and HATS inputs (`sky`, `skyerr`, `skyellipse`, `lr`, `ml`, `xgb`, `auf`, `macauff`, `extra_distance_cols`, `probabilistic=True`, `target_epoch`, `pm_prior`, `nway_match`, `fof_match`). |
| **`ray-union`** | Ray cluster | Distributed star-shaped join | Resumable chunk storage | Distributed full-outer-join union producing full HATS partition trees with `matcher="sky"`, `"skyerr"`, or `"skyellipse"`, `target_epoch`, and `pm_prior`. |
| **`astropy`** | Astropy `SkyCoord` | Single-threaded Python | High (object trees) | Pure-Python fallback; used when SciPy or C extensions are unavailable. |
| **`stilts`** | STILTS `tmatch2` | External Java subprocess | Managed by JVM | `sky`, `skyerr`, `skyellipse`; requires Java and `stilts` CLI tool. |
| **`torchsky`** | PyTorch / Torchsky | GPU / Vectorized Tensor ops | Fixed PyTorch RSS | Tensor-native candidate pruning; requires `torchsky`. |

---

### Technical Deep Dive: Astropy vs. SciPy cKDTree

Understanding why `engine="fast"` (`scipy.spatial.cKDTree`) is preferred over `engine="astropy"`:

```mermaid
flowchart TD
    subgraph Astropy Engine
        A1["RA/Dec Arrays"] --> A2["astropy.units.deg"]
        A2 --> A3["SkyCoord Objects"]
        A3 --> A4["astropy internal KDTree<br/>(Single-threaded Python wrapper)"]
        A4 --> A5["Angle / Quantity Seperations"]
    end
    
    subgraph SciPy cKDTree (fast Engine)
        B1["RA/Dec Arrays"] --> B2["Vectorized 3D Unit Sphere<br/>x = cos(δ)cos(α), y = cos(δ)sin(α), z = sin(δ)"]
        B2 --> B3["scipy.spatial.cKDTree<br/>(C++ with OpenMP workers=-1)"]
        B3 --> B4["Chord length d = 2·sin(θ/2)<br/>Zero Python Object Allocations"]
    end
```

1. **Object Allocation & Memory**:
   - **Astropy**: Constructs `SkyCoord` instances containing `UnitSphericalRepresentation` objects, generating high-overhead Python object trees.
   - **SciPy (`fast`)**: Directly computes raw 3D Cartesian coordinates on the unit sphere $[x, y, z]$ as contiguous $N \times 3$ NumPy float64 buffers. Zero Python wrapper objects are allocated per source.
2. **Parallelism & Performance**:
   - **Astropy**: Executes single-threaded Python traversal through `match_to_catalog_sky` or `search_around_sky`.
   - **SciPy (`fast`)**: The C++ implementation of `cKDTree` natively leverages OpenMP parallel search (`workers=-1`), querying all CPU cores concurrently. Benchmarks confirm **3–5× speedups** on modern multi-core systems.
3. **Exact Spherical to Chord Distance Mapping**:
   - The angular separation $\theta$ on the celestial sphere corresponds to the 3D Euclidean chord length $d$:
     $$d = 2 \sin\left(\frac{\theta}{2}\right), \quad \theta = 2 \arcsin\left(\frac{d}{2}\right)$$
   - The `fast` engine transforms search radii via exact, bit-level chord conversions, ensuring bit-level parity with spherical geometry.
4. **N-Dimensional Multimodal Ranking**:
   - `cKDTree` natively supports $N$-dimensional spaces. When `extra_distance_cols` is configured, `fast` can rank spatial candidates using combined spatial distance and z-score normalized photometry or proper motions.

---

## Universal I/O and Storage Helpers

```python
from xmatch import (
    scan_frame,
    write_frame,
    is_hats_dir,
    astropy_table_to_polars,
    polars_to_astropy,
)
```

| Symbol | Signature | Description |
|---|---|---|
| `scan_frame(path)` | `(str | Path) -> pl.LazyFrame` | Lazily scans `.parquet`, `.csv`, `.tsv`, `.tab`. Reads `.fits` via Torchfits/Astropy. |
| `write_frame(frame, path, ...)` | `(FrameLike, str | Path, ...) -> None` | Streams to `.parquet`, `.csv`, `.tsv`, `.fits`, `.hats`, or `vos:`. |
| `is_hats_dir(path)` | `(str | Path) -> bool` | Returns `True` if directory has HATS metadata (`properties`, `_metadata`). |
| `astropy_table_to_polars(tbl)` | `(Table) -> pl.DataFrame` | Zero-copy PyArrow bridge with automatic UTF-8 string decoding. |
| `polars_to_astropy(df)` | `(FrameLike) -> Table` | Zero-copy Arrow bridge to Astropy `Table`. |

---

## Local Mirroring & Caching (`xmatch.mirror`)

The mirroring subsystem backs offline queries and the distributed `ray-union` pipeline:

```python
from xmatch.mirror import mirror_catalogue, TokenBucket
from xmatch.storage import open_storage
```

- **Remote HATS Replication**: Synchronizes remote HATS directories over HTTP or `vos:` into the local cache (`~/.cache/xmatch` or `/arc/projects/hats`). Skips unchanged partitions based on server size probes.
- **TAP Keyset Pagination**: Downloads full TAP catalogues in deterministic keyset/offset pages with local resume manifests (`sync.json`).
- **TokenBucket Rate Limiter**: Thread-safe per-host rate limiting handling HTTP 429/503 responses with exponential backoff.

---

## Association Contract & Provenance Graph (`xmatch.association`)

The `xmatch.association` module provides an audit-grade, content-addressed association contract (`xmatch.association.v1`) for astronomical candidate generation, graph components, and identity continuity. (See [`docs/association-v1.md`](association-v1.md) for normative schema specifications).

```python
from xmatch.association import (
    AssociationRecord,
    AssociationProvenance,
    AssociationScore,
    AssociationComponent,
    AssociationComponentMember,
    AssociationMemberEquivalence,
    write_association_release,
    iter_association_release,
    verify_association_release,
    write_association_component_release,
    verify_association_component_release,
    write_association_member_equivalence_release,
)
```

### Core Data Models

#### `AssociationRecord`
Represents an evaluated candidate pairing with deterministic SHA-256 identity:
- `association_id`: SHA-256 hash of `evidence_id`, `source_id`, `candidate_id`, evaluation epoch, software version, input release IDs, and parameter hash.
- `evidence_id`, `source_id`, `candidate_id`: Opaque, release-scoped identifiers.
- `separation_arcsec`: Non-negative great-circle separation evaluated after epoch propagation.
- `evaluation_epoch_jyear`: Julian year epoch of evaluation (or `epoch_unknown` flag).
- `score`: `AssociationScore` typed with exact semantic interpretation:
  - `ranking_score`: Finite, algorithm-specific heuristic (not a probability).
  - `assumed_prior_posterior`: Constrained to $[0, 1]$, but uncalibrated against real data.
  - `calibrated_probability`: Statistically calibrated probability in $[0, 1]$ with non-empty `calibration_id`.
- `flags`: List of diagnostic flags (e.g. `blended_candidate`, `epoch_unknown`, `pm_extrapolated`).

#### `AssociationComponent` & `AssociationComponentMember`
Sidecar graph representation for multi-survey crossmatch components:
- `input_release_id`: Namespace of the member catalogue (e.g. `"catalog:gaia_dr3:v1"`).
- `member_id`: Source identifier within that catalogue release.
- `parent_component_ids`: Pointers to parent release components, supporting clean split and merge tracking across catalogue versions.

### Release File Operations

```python
# 1. Publish verified association release
manifest = write_association_release(
    sorted_records,
    output_directory="releases/cosmos_v1",
    provenance=provenance_obj,
)

# 2. Verify release checksums and ordering fail-closed
verified_manifest = verify_association_release("releases/cosmos_v1")

# 3. Stream release with bounded memory
for record in iter_association_release("releases/cosmos_v1"):
    process(record)

# 4. Publish immutable component graph sidecar
comp_manifest = write_association_component_release(
    components,
    association_release_directory="releases/cosmos_v1",
    component_release_directory="components/cosmos_v1",
)
```

---

## Exception Hierarchy

All public exceptions inherit from `CrossMatchError`:

```text
CrossMatchError
├── ConfigError     (Malformed YAML configuration or missing keys)
├── InputError      (Unreadable files, unsupported formats, missing columns)
├── TapError        (Remote ADQL syntax, network errors, or timeout)
└── StiltsError     (STILTS CLI execution or Java environment errors)
```

## Catalogue and model-training products

`xmatch.observations` supplies `source_inventory`, `namespaced_source_id`,
`normalize_photometry`, `normalize_property_evidence`,
`required_observation_columns`, `validate_source_inventory`,
`write_observation_release`, and `verify_observation_release`.
`xmatch.candidates` supplies `write_candidate_release`,
`verify_candidate_release`, and `normalize_candidate_hypotheses`.
These functions are also exported from `xmatch`.
See [observation mappings](observations.md) and the
[catalogue-to-model pipeline](catalogue-pipeline.md) for contracts and examples.
