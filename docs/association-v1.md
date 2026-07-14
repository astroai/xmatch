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
