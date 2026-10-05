# Source and measurement releases

`xmatch.observations` builds lazy, source-preserving measurement tables. It
records supplied conventions and upstream evidence; it does not choose physical
counterparts, adopt conflicting properties, or infer limits from missing data.

Use the existing `CrossMatch.resolve_source` registry and local overrides. Each
input needs an explicit `release_namespace` identifying its upstream release
and product, and a unique string/integer `id_column`. Reordering or partitioning
rows does not change the SHA-256 source identity. Integer `17` and string `"17"`
are different native identifiers. Floats and anonymous row numbers are not
accepted as authoritative IDs.

```python
import polars as pl
from xmatch import CrossMatch
from xmatch.observations import (
    normalize_photometry,
    normalize_property_evidence,
    source_inventory,
    verify_observation_release,
    write_observation_release,
)

source = CrossMatch().resolve_source(
    pl.DataFrame({
        "id": [10, 11], "ra": [20.0, 21.0], "dec": [-5.0, -6.0],
        "g_flux": [-0.2, None], "g_error": [0.1, None],
        "spec_z": [0.3, None],
    }),
    {
        "id_column": "id",
        "release_namespace": "my-survey:release-1:main-product",
        "release_metadata": {"citation": "Provide the actual upstream citation"},
        "photometry": [{
            "passband": "my-survey:g:curve-1",
            "value_column": "g_flux", "error_column": "g_error",
            "unit": "uJy", "method": "psf",
        }],
        "property_evidence": [{
            "property": "redshift.spec", "value_column": "spec_z",
            "unit": "dimensionless", "category": "inferred_spectroscopy",
            "method": "my-spectroscopic-pipeline:v1",
        }],
    },
)

inventory = source_inventory(source)                  # LazyFrame
photometry = normalize_photometry(source)             # LazyFrame
evidence = normalize_property_evidence(source)        # LazyFrame
manifest = write_observation_release(
    "pilot-release", [source], release_id="pilot:1",
    provenance={"software_version": "record the actual version",
                "upstream_checksum": "record the actual acquired-input checksum"},
)
assert verify_observation_release("pilot-release") == manifest
```

The same metadata keys work in a catalogue entry of `xmatch.yaml`; no second
registry is created. Existing catalogue entries without audited photometry
metadata continue to work for matching. They must receive explicit namespaces
and mappings before publication as observation releases. Default matching
column selections may omit these measurement fields. Use
`required_observation_columns(source)` to get the explicit acquisition
selection, including astrometry, motion/error metadata, and all mapped
measurement/evidence columns. Pass it to the acquisition backend before
publication. The helper does not change matching defaults or download data.

## Source inventory

`source_inventory(source)` returns `source_id`, `release_namespace`, string
`native_id`, `native_id_type`, `ra_deg`, `dec_deg`, `epoch_jyear`, and a `native`
struct containing every original column. All sources survive, including sources
without photometry and sources absent from other surveys. Standard coordinate
columns carry the original declared frame's longitude and latitude; the
inventory does not transform frames. The release manifest records that frame.
Consumers doing ICRS matching must transform or reject other frames.
The crossmatch orchestrator rejects unequal declared frames before spatial
matching or remote acquisition. Same-frame spherical matching is allowed;
target-epoch propagation requires ICRS. Relational ID joins do not require a
coordinate frame transform.

`epoch_jyear` comes only from declared astrometric reference-epoch metadata.
An equinox such as J2000 must not be used as an observation/reference epoch.
Photometric times are supplied separately in the measurement mapping.

`validate_source_inventory(parquet_path)` checks null/duplicate source identities
using bounded Arrow batches and a temporary disk-backed SQLite index. It can
also validate a projected candidate-input table containing `source_id`.

## Photometry mappings

Each mapping requires `passband`, `value_column`, and `unit`. Optional fields are:

| Fields | Meaning |
|---|---|
| `error_column` | Native symmetric measurement uncertainty |
| `method`, `calibration_id` | Upstream PSF/aperture/model convention and calibration reference |
| `passband_version`, `passband_uri` | Supplied filter-curve version and reference; no curve is downloaded |
| `zeropoint_jy` | Positive Vega zero-point flux density for this passband/convention |
| `status_column`, `status_values` | Explicit native-code to canonical-status mapping |
| `quality_column` | Native quality code, retained without interpreting it |
| `observation_time_column` or `observation_time` | Native observation time, requiring `time_format` and `time_scale` |
| `limit_sigma`, `limit_confidence`, `limit_convention` | Supplied limit semantics; sigma and probability are never inferred from one another |

Supported flux-density units are `Jy`, `mJy`, `uJy`, and `nanomaggy`; one
nanomaggy is represented as 3.631 microJy. `ABmag` uses the 3631 Jy zero-point
convention. `Vegamag` converts only when `zeropoint_jy` is supplied. Other native
units, including integrated energy-band fluxes and instrument counts, remain
native-only. The complete raw values and dtype remain in the source inventory.

Negative measured fluxes survive. Magnitude uncertainties produce separate
`flux_error_lower_jy` and `flux_error_upper_jy`, representing the transformed
native symmetric interval; `flux_error_jy` is null for magnitude inputs. Those
intervals are not a fitted noise distribution. Invalid native uncertainties and
nonfinite conversions are flagged, and invalid normalized values remain null.
Magnitude interval endpoints are converted directly with the zero point in
logarithmic form to avoid intermediate overflow. A finite native uncertainty
whose converted bound is unrepresentable is flagged
`conversion_uncertainty_nonfinite`; nonpositive converted upper limits are
flagged `invalid_upper_limit` and retain their native value with `invalid` status.

Without a status column, a finite value means `measured`, null means `missing`,
and a nonfinite value means `invalid`. Explicit statuses are `measured`,
`upper_limit`, `missing`, `masked`, `outside_footprint`, `not_detected`,
`ambiguous`, and `invalid`. Unknown native codes become `invalid`. Supplied
limits occupy `upper_limit_jy`, with `flux_jy` null; missing rows never become
limits. A status records upstream measurement semantics, not a calibrated
association decision.

Provider footprint, mask, sensitivity and shared-observation references can be
recorded in `release_metadata`, with their versions and citations. The ledger
does not query those maps or infer source-specific coverage from an absent
catalogue row. Image-based forced photometry requires a separate image pipeline.

## Property and type evidence

Each mapping requires `property`, `value_column`, `category`, and `method`.
Categories are `measured`, `inferred_spectroscopy`,
`inferred_photometry_astrometry`, `literature_assertion`, and `prediction`.
Prediction mappings additionally require `model_id`.

Optional fields are `error_column`, `unit`, `quality_column`, `reference_frame`,
`doppler_convention`, observation-time fields as above, `model_id`,
`input_features`, `association_id_column`, and `upstream_evidence_id_column`.
Values are retained as native strings with their dtype; nested arrays/structs
use JSON so posterior samples or distributions are not collapsed to a scalar.
The original native column remains in the source inventory.

Keep distinct property names for spectroscopic and photometric redshift,
epoch-specific and systemic radial velocity, native type, activity, morphology,
and subtype. Missing values remain unlabelled. The code neither maps native
classifications to a universal ontology nor averages conflicting evidence.
Unknown frames/conventions remain `unknown`; no redshift/velocity conversion
is performed. `input_features` documents known upstream inference inputs so a
training application can audit target leakage.

## Publication and verification

`write_observation_release` streams separate inventory, photometry and evidence
Parquet files for each namespace. This preserves heterogeneous native schemas.
A manifest records paths, row counts, Parquet schemas, SHA-256 checksums,
registry mappings, frame/motion/error metadata and supplied provenance.

Publication uses a sibling temporary directory and exclusive writer lock,
followed by a directory rename. Existing releases are never overwritten.
Duplicate namespaces, native IDs or identical measurement mappings fail before
publication. After an interrupted process, inspect and remove a stale sibling
lock only after confirming no writer is active.

`verify_observation_release` checks each shard's checksum, row count and recorded
schema, rejecting paths outside the release directory. These checks establish
consistency with the local manifest; they do not authenticate the upstream
archive, validate survey calibration, or establish scientific association truth.
