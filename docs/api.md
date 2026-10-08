# xmatcher Public Python API Reference

Stable public surface only. Anything not exported from `xmatcher.__init__` is considered internal.

---

## Top-Level Entry Points (`xmatcher.CrossMatch`)

`CrossMatch` resolves configured catalogues, manages remote archive access, validates declared coordinate-frame compatibility, and dispatches requests to matching engines. It does not transform coordinates between frames.

```python
from xmatcher import CrossMatch

cm = CrossMatch()  # loads bundled xmatcher.yaml and user overrides
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
- `req` (`MatchRequest`): Typed request dataclass containing input sources, matching specifications, engine choices, and column overrides; runtime validation also depends on source metadata and selected engine.

**Returns**:
- `pl.DataFrame` (eager, default) or `pl.LazyFrame` (when `req.lazy=True`). Returns `None` when `req.output_file` is provided and results are streamed to disk.

---

### `crossmatch`

Classic spread-arguments entry point.

```python
def crossmatch(
    self,
    catalogue_1_input: FrameInput,
    catalogue_2_input: FrameInput,
    output_file: str | Path | None = None,
    *,
    lazy: bool = False,
    progress_cb: Callable[[str], None] | None = None,
    **params: Any,
) -> pl.DataFrame | pl.LazyFrame | None
```

This compatibility API takes matching options as keyword parameters; it has no
`spec=` parameter. Use `crossmatch_request(MatchRequest(..., spec=...))` when
you already have a `MatchSpec` or need typed per-side metadata.

```python
request = MatchRequest(
    cat1="cat_a.parquet",
    cat2="cat_b.parquet",
    spec=MatchSpec(radius_arcsec=1.5, matcher="sky", find="best"),
)
result = cm.crossmatch_request(request)
```

**Key Parameters**:
- `cat1`, `cat2` (`str | Path | pl.DataFrame | pl.LazyFrame`): Input catalogues. Can be local file paths (`.parquet`, `.csv`, `.tsv`, `.fits`), HATS directory paths, remote TAP catalogue names (e.g. `"gaia"`, `"des"`), or in-memory Polars frames.
- `output_file` (`str | Path | None`): Destination file/tree supported by the selected writer. Some sinks stream serialization, but that does not mean the matching step is bounded-memory.
- `lazy` (`bool`): If `True`, returns an uncollected `pl.LazyFrame`.
- `radius_arcsec` (`float`): Spatial matching radius in arcseconds.
- `matcher` (`str`): Match algorithm: `"sky"`, `"skyerr"`, `"skyellipse"`, `"lr"`, `"ml"`, `"xgb"`, `"auf"`, `"macauff"`.
- `engine` (`str`): Spatial engine: `"auto"`, `"fast"`, `"zone"`, `"ray"`, `"ray-union"`, `"astropy"`, `"stilts"`, `"torchsky"`.
- `join_type` (`str`): Relational join mode: `"1and2"` (inner), `"all1"` (left), `"all2"` (right), `"1or2"` or `"all"` (full outer), `"1not2"` (left anti), `"2not1"` (right anti).
- `find` (`str`): Candidate selection policy: `"best"` (nearest/top-ranked match) or `"all"` (all pairs within radius).
- `**params`: Matching parameters (`radius_arcsec`, `matcher`, `engine`, `join_type`, `find`, etc.), side overrides such as `ra_column_1` / `dec_column_1`, remote cone constraints (`ra`, `dec`, `radius_deg`), and spill controls.

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

- With all-remote inputs, no cone constraint, `engine="auto"`, and an output path, `union_match` routes to `engine="ray-union"`; this produces a full-sky full-outer HATS tree.
- `ray-union` supports `sky`, `skyerr`, and `skyellipse` and HATS output only. It rejects region constraints and unsupported filters, ID joins, scores, and extra columns. Compressed interval planning handles mixed-order/RING inputs and distributed tasks apply measured per-source uncertainty or epoch-motion halos; planning may scan additional partitions. The separate pairwise native HATS path materializes globally for adaptive, RING, mixed-order, and epoch-aligned cases. `engine="ray"` distributes pairwise candidate search, while its caller still materializes inputs and collects results on the coordinator.
- HATS union outputs include non-null `_union_ra` and `_union_dec` routing coordinates from the lowest-indexed catalogue member in each row; output tree metadata declares NESTED ordering and those columns as the spatial coordinates.

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

Resolves and materializes the input frames, finds primary-to-each-secondary candidate pairs inside the spatial radius, then iterates the per-primary Cartesian product in chunks (optionally capped per source). Positional scores require declared errors or explicit per-catalogue error floors. Each `prior_columns` entry must exist in every catalogue and represent a comparable quantity; its midpoint-KDE ratio is an empirical heuristic. `engine="ray"` distributes candidate discovery, but inputs and resulting tuples are assembled on the coordinator. `p_match` uses equal prior odds and is not calibrated for population prevalence or local sky density. Positional covariance is reduced to an isotropic effective sigma; KDEs are fitted on an unconditional source sample capped at 50,000 values per requested column. See [score assumptions](algorithms.md#8-bayesian-pairwise-qualification-and-n-way-scores).

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

Searches cross-catalogue edges for every pair of catalogues before forming connected components. Only components containing a source from the first catalogue are emitted; isolated first-catalogue sources are retained, while secondary-only components are omitted. `engine="ray"` distributes pairwise edge discovery, but input frames and component output are still assembled on the coordinator. Numeric attributes are averaged, and nonnumeric attributes use the first value.

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
| `ra`, `dec`, `radius_deg` | `float | None` | `None` | Explicit cone for remote TAP/CDS downloads. Required for two-remote `find="best"` and remote `skyerr`, `skyellipse`, or `target_epoch` requests, whose uncertainty/motion halo cannot be safely inferred. |
| `probabilistic` | `bool` | `False` | Enable positional-only `p_match`. A nonempty `MatchSpec.prior_columns` also enables scoring and adds photometric KDE terms. Both catalogues need declared positional errors or an explicit per-side `default_pos_error_arcsec`. |

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
| `pm_prior` | `bool` | `False` | Enable the assumed, Wilson (2023)-inspired PM drift uncertainty model. |
| `pm_prior_magnitude_column` | `str | None` | `None` | Magnitude column for Wilson (2023) distance-proxy scaling. |
| `filter_expr` | `str | None` | `None` | Polars SQL WHERE clause for filtering candidate pairs; malformed expressions and unknown columns raise `CrossMatchError`. |
| `extra_distance_cols` | `dict[str, float]` | `{}` | Extra columns & weights for $N$-dimensional `find="best"` ranking; declared columns must exist on both sides and contain numeric finite values, or `CrossMatchError` is raised. |
| `batch_size` | `int | None` | `None` | Number of HEALPix pixel groups per batch in zone matching; not a general process-memory limit. |
| `lr_magnitude_column` | `str | None` | `None` | Secondary magnitude column for Likelihood Ratio (`matcher="lr"`). |
| `lr_q` | `float` | `0.8` | Prior probability of counterpart detection in primary survey. |
| `ml_color_columns` | `list[str]` | `[]` | Colour/magnitude feature columns for RF/XGB matchers. |
| `ml_model_path` | `str | None` | `None` | Save/load path for Random Forest model artifact (`.joblib`). |
| `xgb_model_path` | `str | None` | `None` | Save/load path for XGBoost model artifact (`.joblib`). |
| `macauff_flux_columns` | `list[str]` | `[]` | Multi-band flux columns for `matcher="macauff"`. |
| `prior_columns` | `list[str]` | `[]` | Nonempty columns enable Bayesian scoring and add heuristic photometric KDE terms; columns must exist in both catalogues and represent comparable quantities. |

### `SideOverrides`

`SideOverrides` includes `ra_column`, `dec_column`, `id_column`, `columns`,
`ra_err_column`, `dec_err_column`, `corr_column`,
`astrometric_covariance_columns`, `pos_err_units`, `default_pos_error_arcsec`,
`epoch`, `epoch_column`, `pm_ra_column`, `pm_dec_column`,
`parallax_column`, `radial_velocity_column`, `endpoint`, and `frame`.

---

## Execution Engines Matrix

`xmatcher` decouples matching algorithms from spatial indexing engines.

| Engine | Primary Backend | Threading / Scaling | Memory Overhead | Supported Algorithms & Features |
|---|---|---|---|---|
| **`fast`** | `scipy.spatial.cKDTree` | Single-process spatial queries (SciPy may use native parallelism where requested) | Holds coordinate arrays and candidate/result data | General local pairwise engine; matcher requirements still apply. |
| **`zone`** | `cdshealpix` + `cKDTree` | HEALPix pixel groups on one machine | Pixel batching; total memory depends on input/result path | Zone-based candidate queries; `batch_size` is not a process-memory cap. |
| **`ray`** | Ray | Distributed HEALPix candidate queries | Coordinator holds input frames and assembles final result | Pairwise work only; it is not an out-of-core flat-file pipeline. |
| **`ray-union`** | Ray + HATS | Full-sky union to HATS output | Distributed interval plan; may read extra partitions for conservative halos | Supports `sky`, `skyerr`, `skyellipse`; handles mixed-order/RING layouts and uncertainty/epoch halos. |
| **`astropy`** | Astropy `SkyCoord` | Astropy sky-coordinate queries | Depends on input and candidate counts | Alternative local geometry backend. |
| **`stilts`** | STILTS `tmatch2` | External Java subprocess | Managed by JVM | Requires Java and a working `stilts` executable; matcher support depends on requested options. |
| **`torchsky`** | PyTorch / Torchsky | Tensor-based candidate search | Depends on tensors and returned candidates | Requires a separately installed compatible Torchsky checkout; no PyPI distribution is available. Tested with the sibling 0.4 development source installed editable. |

---

### Technical Deep Dive: Astropy vs. SciPy cKDTree

`fast` and `astropy` provide alternative local spatial-query implementations. Their relative performance depends on data size, candidate density, hardware, and installed libraries; measure on representative data with [`scripts/bench_engines.py`](../scripts/bench_engines.py).

```mermaid
flowchart TD
    subgraph Astropy Engine
        A1["RA/Dec Arrays"] --> A2["astropy.units.deg"]
        A2 --> A3["SkyCoord Objects"]
        A3 --> A4["Astropy sky-coordinate query"]
        A4 --> A5["Angle / Quantity Separations"]
    end

    subgraph SciPy cKDTree (fast Engine)
        B1["RA/Dec Arrays"] --> B2["Vectorized 3D Unit Sphere<br/>x = cos(δ)cos(α), y = cos(δ)sin(α), z = sin(δ)"]
        B2 --> B3["scipy.spatial.cKDTree"]
        B3 --> B4["Chord length d = 2·sin(θ/2)<br/>Zero Python Object Allocations"]
    end
```

1. **Coordinate representation**: Astropy uses `SkyCoord`; the fast path constructs unit-sphere Cartesian arrays for SciPy's `cKDTree`. Costs depend on input and candidate counts.
2. **Parallelism & Performance**: Runtime and threading depend on the selected query path and native libraries. No fixed speedup or memory-per-row estimate applies across input sizes and candidate densities.
3. **Exact Spherical to Chord Distance Mapping**:
   - The angular separation $\theta$ on the celestial sphere corresponds to the 3D Euclidean chord length $d$:
     $$d = 2 \sin\left(\frac{\theta}{2}\right), \quad \theta = 2 \arcsin\left(\frac{d}{2}\right)$$
   - The `fast` engine maps unit-sphere angular search radii to Euclidean chord radii; returned separations are great-circle angles.
4. **N-Dimensional Multimodal Ranking**:
   - When `extra_distance_cols` is configured, `fast` finds all spatial candidates with `cKDTree`, then ranks them using combined spatial distance and z-score normalized photometry or proper motions.

---

## Universal I/O and Storage Helpers

```python
from xmatcher import (
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

## Local Mirroring & Caching (`xmatcher.mirror`)

The mirroring subsystem backs offline queries and the distributed `ray-union` pipeline:

```python
from xmatcher.mirror import mirror_catalogue, TokenBucket
from xmatcher.storage import open_storage
```

- **Remote HATS Replication**: Synchronizes remote HATS directories over HTTP or `vos:` into the local cache (`~/.cache/xmatcher` or `/arc/projects/hats`). It hashes remote partition content for each sync attempt; this detects same-size edits but requires reading partition bytes and can add substantial network I/O. The documented CANFAR working/output directory is `/arc/projects/hats/xmatcher`.
- **TAP Keyset Pagination**: Downloads TAP catalogues with local resume manifests (`sync.json`) and requires a unique, non-null stable key. Page-window row counts do not detect same-count edits; `--force` requests a full refresh. Append-only growth is detected using a maximum-key probe.
- **TokenBucket Rate Limiter**: Thread-safe per-host rate limiting handling HTTP 429/503 responses with exponential backoff.

---

## Association Contract & Provenance Graph (`xmatcher.association`)

The `xmatcher.association` module provides an audit-grade, content-addressed association contract (`xmatch.association.v1`) for astronomical candidate generation, graph components, and identity continuity. (See [`docs/association-v1.md`](association-v1.md) for normative schema specifications).

```python
from xmatcher.association import (
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

`xmatcher.observations` supplies `source_inventory`, `namespaced_source_id`,
`normalize_photometry`, `normalize_property_evidence`,
`required_observation_columns`, `validate_source_inventory`,
`write_observation_release`, and `verify_observation_release`.
`xmatcher.candidates` supplies `write_candidate_release`,
`verify_candidate_release`, and `normalize_candidate_hypotheses`.
These functions are also exported from `xmatcher`.
See [observation mappings](observations.md) and the
[catalogue-to-model pipeline](catalogue-pipeline.md) for contracts and examples.
