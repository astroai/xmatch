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

## Component membership and lineage

`xmatch.association.component.release.v1` is an immutable sidecar to one
verified association release. It does not change that release's two-file
contract. Its authoritative files are:

- `memberships.jsonl`: canonical rows ordered by component, input release, and
  member ID;
- `manifest.json`: component/membership counts and checksum, the exact
  `association_release_id` and inherited provenance, optional parent component
  release, and a content-derived `component_release_id`.

Each member is namespaced by `input_release_id`, which must occur in the
association provenance. A component ID covers its association release and exact
canonical member set. The same members therefore receive a different ID in a
new association release. Cross-release continuity is explicit instead:
`parent_component_ids` may name zero, one, or several components in the verified
parent component release, representing new components, continuity/splits, and
merges respectively.

```python
from xmatch.association import (
    AssociationComponent,
    AssociationComponentMember,
    verify_association_release,
    verify_association_component_release,
    write_association_component_release,
)

association_manifest = verify_association_release("releases/cosmos-v1")
component = AssociationComponent(
    association_manifest.release_id,
    (
        AssociationComponentMember("catalog:cosmos2020:v1", "source:101"),
        AssociationComponentMember("catalog:specz:v1", "source:9001"),
    ),
)
components = sorted([component], key=lambda item: item.component_id)
component_manifest = write_association_component_release(
    components,
    "releases/cosmos-v1",
    "components/cosmos-v1",
)
verified = verify_association_component_release(
    "components/cosmos-v1",
    association_release_directory="releases/cosmos-v1",
)
```

For a child release, pass `parent_component_directory` to the writer and list
the applicable parent IDs on each `AssociationComponent`. Publication rejects
dangling parents, association/parent lineage mismatches, duplicate membership,
mixed input releases, nondeterministic component ordering, and overwrite.
Verification rejects byte-level tampering. It streams the file but retains one
global membership set so it can prove that every namespaced member belongs to
at most one component.

## Intentional input-release namespace changes

`xmatch.association.member.equivalence.release.v1` records reviewed identity
assertions when a catalogue is republished under a different
`input_release_id`. Its `equivalences.jsonl` rows map one fully namespaced
parent member to one fully namespaced current member; the manifest binds the
direct parent/current association releases, row count, and canonical byte
checksum. Both sides must be unique, so catalogue splits, merges, or uncertain
matches cannot be mislabeled as identity equivalence. Xmatch never infers these
rows from equal bare IDs.

```python
from xmatch.association import (
    AssociationComponentMember,
    AssociationMemberEquivalence,
    write_association_member_equivalence_release,
)

equivalences = [
    AssociationMemberEquivalence(
        parent_association_release_id,
        current_association_release_id,
        AssociationComponentMember("catalog:cosmos:dr1", "source:101"),
        AssociationComponentMember("catalog:cosmos:dr2", "object:000101"),
    )
]
equivalences.sort(key=lambda item: item.equivalence_id)
write_association_member_equivalence_release(
    equivalences,
    "releases/cosmos-v1",
    "releases/cosmos-v2",
    "member-equivalences/cosmos-v1-to-v2",
)
```

The writer requires direct association-release lineage, validates both member
namespaces against endpoint provenance, rejects non-bijective crosswalks, and
publishes atomically. Verification can rebind the artifact to both association
release directories. Physical identity remains a scientific input that the
producer must justify and review; content addressing proves which assertions
were used, not that the assertions are astrophysically true.

## Deterministic construction and release deltas

`construct_association_components` builds connected components directly from a
verified association release. Callers must supply both endpoint input-release
IDs and the exact `AssociationDecision` values to include; the constructor does
not infer catalogue namespaces or acceptance policy. If a verified parent
component release is supplied, exact namespaced-member overlap becomes each
new component's declared parent lineage. When input-release namespaces changed,
pass the reviewed `member_equivalence_directory` to construction, delta
publication, and endpoint-aware delta verification.

```python
from xmatch.association import (
    AssociationDecision,
    construct_association_components,
    verify_association_component_delta_release,
    write_association_component_delta_release,
)

components = construct_association_components(
    "releases/cosmos-v2",
    source_input_release_id="catalog:cosmos2020:dr1",
    candidate_input_release_id="catalog:specz:dr1",
    included_decisions=(AssociationDecision.SELECTED,),
    parent_component_directory="components/cosmos-v1",
    member_equivalence_directory="member-equivalences/cosmos-v1-to-v2",
)

delta_manifest = write_association_component_delta_release(
    "components/cosmos-v1",
    "components/cosmos-v2",
    "component-deltas/cosmos-v1-to-v2",
    member_equivalence_directory="member-equivalences/cosmos-v1-to-v2",
)
verified = verify_association_component_delta_release(
    "component-deltas/cosmos-v1-to-v2",
    parent_component_directory="components/cosmos-v1",
    current_component_directory="components/cosmos-v2",
    member_equivalence_directory="member-equivalences/cosmos-v1-to-v2",
)
```

`xmatch.association.component.delta.release.v1` contains canonical
`deltas.jsonl` rows plus a content-addressed `manifest.json`. Every row names
the parent and current association-release and component-release endpoints.
Classification uses exact namespaced overlap plus any supplied, verified
member-equivalence crosswalk: zero-degree current or parent components are
`created` or `retired`, isolated one-to-one edges are
`continued`, and many-to-one or one-to-many edges are `merged` or `split`.
Many-to-many changes emit both merge and split events. Publication rejects
declared lineage that differs from exact membership overlap. Verification
checks row IDs, ordering, counts, and byte checksums; when both endpoint
directories are supplied it also recomputes the full classification.
