# Source-preserving catalogue and latent-model pipeline

The catalogue foundation publishes measurements and sparse candidate pairs.
Source association, scientific acceptance, and property-label selection remain
explicit. An outer union preserves observations; it does not prove their identity.

## Ingest pinned survey products

Use existing CatalogueSource/config resolution with an explicit
`release_namespace`, `photometry`, `property_evidence`, and `release_metadata`.
[Observation mappings](observations.md) retain native rows, passbands, signed
flux, uncertainties, limit conventions, times, methods, and raw property evidence.
`required_observation_columns(source)` returns the columns to acquire for these
products; matching defaults are not a complete observation selection.

```python
from xmatcher import CrossMatch, write_observation_release, write_candidate_release

cm = CrossMatch()
sources = [
    cm.resolve_source(
        "optical.parquet",
        {
            "id_column": "id",
            "ra_column": "ra",
            "dec_column": "dec",
            "release_namespace": "example/optical/release1",
            "photometry": [
                {
                    "passband": "example:g:v1",
                    "value_column": "g_flux",
                    "error_column": "g_error",
                    "unit": "uJy",
                    "method": "psf",
                }
            ],
        },
    ),
    cm.resolve_source(
        "infrared.parquet",
        {
            "id_column": "id",
            "ra_column": "ra",
            "dec_column": "dec",
            "release_namespace": "example/infrared/release1",
        },
    ),
]
observations = write_observation_release(
    "releases/observations-1",
    sources,
    release_id="example-observations-1",
    provenance={"software_version": "local-development", "acquisition": "pinned local inputs"},
)
candidates = write_candidate_release(
    "releases/candidates-1",
    sources,
    radius_arcsec=3.0,
    memory_budget_bytes=64 * 1024 * 1024,
    provenance={"observation_release_id": observations["release_id"]},
)
```

Both writers refuse to replace existing releases, stage locally, and publish
Parquet files with checksum manifests. Candidate releases require ICRS inputs.
All native sources appear in `sources.parquet`, including zero-candidate sources;
`candidates.parquet` contains all pairs in the requested survey-pair/radius scope.
Namespaced source IDs distinguish identical native IDs in different releases.
`candidate_count` is graph degree, not confidence or the number of physical objects.

## Candidate scope and association alternatives

Candidate generation reuses the local partitioned spill engine. The matching
budget controls its batch memory proxy; it does not guarantee resident memory
for Polars scans, sorting, or graph inference. Pair outputs avoid the N-survey
Cartesian product. `survey_pairs` can explicitly restrict the searched release
pairs; omitted pairs are outside the release's candidate scope.

`target_epoch` aligns coordinates before repartitioning. Rows at a different
reference epoch need known finite proper motion. Unknown epochs or missing
motion fail rather than silently assuming stationary sources. The search radius
still needs to cover the caller's declared astrometric/physical uncertainty.
Raw astrometry remains in the inventory; pair records carry the evaluation epoch.

Target-epoch `skyerr` matching transports a declared five-parameter astrometric
covariance with the existing motion Jacobians; missing covariance fails explicitly.
Physical covariance currently conditions on radial velocity: a source must declare
`release_metadata["radial_velocity_uncertainty"] = "deterministic"` for this path.
It does not propagate an unknown RV variance or promise a full six-parameter
covariance model. Jacobian transport requires a separately installed compatible
Torchsky checkout; no `xmatcher[torchsky]` extra is available. The
integration was tested with the sibling Torchsky 0.4 development source
installed editable and Torchfits 1.0.0 from PyPI.
An explicitly requested missing-motion prior remains an assumed population
model, distinct from measured covariance, and is not supported by spill execution.
The distributed `ray-union` driver supports `target_epoch` when sources declare
reference epochs and either finite proper motions for rows that need
propagation or an explicit missing-motion prior. It rejects unannotated motion
requests before writing output. Distributed union uses compressed interval
planning for mixed-order and RING inputs, then applies measured per-source
uncertainty or epoch-motion halos in distributed tasks. Planner scans can read
additional partitions, so remote or very large inputs can incur substantial
I/O. The separate pairwise native HATS path materializes inputs globally for
adaptive, RING, mixed-order, or epoch-aligned matching. Direct eager and spill entry
points also validate declared frames instead of comparing incompatible coordinates.

```python
import polars as pl
from xmatcher import normalize_candidate_hypotheses

sources = pl.scan_parquet("releases/candidates-1/sources.parquet")
pairs = pl.scan_parquet("releases/candidates-1/candidates.parquet")
# Supply log weights from an explicit likelihood/prior model. Separation or
# cosine similarity alone is not a calibrated Bayes factor.
scored_pairs = pairs.with_columns(pl.lit(0.0).alias("log_weight"))  # illustrative equal weights
hypotheses = normalize_candidate_hypotheses(
    sources,
    scored_pairs,
    candidate_namespace="example/infrared/release1",
    no_match_log_weight=0.0,
)
```

The resulting alternatives compete within the named target survey. Stored pairs
are oriented appropriately, and every source outside that survey receives an
explicit no-match alternative. Log weights must include the declared priors and
selection/coverage assumptions. Output semantics are `assumed_prior_posterior`;
normalization does not establish calibration, global one-to-one consistency,
or deblending. Joint multi-survey/blend hypotheses need a scientific inference
model, rather than pooling all detections into one list of competing candidates.

For namespaced pair tables, both endpoint labels are checked against the
inventory. A nonempty pair table must also contain evidence for the requested
target namespace, in either the inventory or pair labels. An empty pair table
can name a target with no inventory rows; inventoried sources then receive only
the explicit no-match alternative.

Existing association.v1 releases remain authoritative for evaluated pair
records. `construct_association_components(..., additional_members=inventory)`
can retain isolated sources under an explicit included-decision policy. Its
union-find state is O(number of members); components are graph work units,
not automatically adopted celestial objects.
