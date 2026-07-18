# `xmatch.association.v1`

This contract records evaluated source/candidate pairs. It does not define a
permanent celestial-object identifier, claim that a selected candidate is true,
or coalesce redshift evidence. Xmatch owns candidate generation and association;
Zensus owns evidence conflict handling; applications own training-view reduction.

Every record contains release-scoped opaque `evidence_id`, `source_id`, and
`candidate_id` values. `association_id` is the SHA-256 identity of those IDs,
the association method/version, evaluation epoch, input release IDs, executable
software version, and parameter hash. Changing any of those inputs creates a new
association ID.

`separation_arcsec` is the non-negative great-circle separation evaluated after
any declared epoch propagation. `evaluation_epoch_jyear` records that epoch; a
missing epoch requires the `epoch_unknown` flag. Flags are an open, preserved
list so producers can add qualified diagnostics without changing existing fields.

Scores must declare one of three semantics:

- `ranking_score`: finite, algorithm-specific, and not a probability;
- `assumed_prior_posterior`: constrained to `[0, 1]` but not empirically calibrated;
- `calibrated_probability`: constrained to `[0, 1]` and paired with a non-empty
  `calibration_id` naming the calibration artifact.

The packaged `fixtures/association_v1.json` file includes a unique selection,
an ambiguous crowded-field candidate pair, and an unpropagated-epoch example.
Consumers must parse every record with `AssociationRecord.from_mapping` and
must preserve unknown flags.

## Release directories

`xmatch.association.release.v1` publishes those records without turning a
mutable graph component into a permanent object ID. A release directory has
exactly two authoritative files:

- `associations.jsonl`: canonical `association.v1` records, strictly ordered by
  `association_id` so duplicate IDs and nondeterministic ordering fail closed;
- `manifest.json`: the record count and SHA-256, shared input/software/parameter
  provenance, optional `parent_release_id`, and a content-derived `release_id`.

Every record in one release must carry the manifest's exact provenance. The
parent is lineage between immutable releases; it does not assert that graph
components or release-scoped object IDs remain stable across releases.

```python
from xmatch.association import (
    iter_association_release,
    verify_association_release,
    write_association_release,
)

records = sorted(candidate_records, key=lambda record: record.association_id)
manifest = write_association_release(
    records,
    "releases/cosmos-v1",
    provenance=records[0].provenance,
)

# Validation uses bounded memory. Consume the iterator to EOF before treating
# its final checksum/count validation as successful.
verified = verify_association_release("releases/cosmos-v1")
for record in iter_association_release("releases/cosmos-v1"):
    consume(record)
```

The writer consumes its input once, holds one record at a time, fsyncs both
files, and renames a temporary sibling directory only after successful
validation. It refuses to replace an existing target. Producers with unsorted
tables should use their table engine's external sort before publication rather
than loading the release into Python memory.
