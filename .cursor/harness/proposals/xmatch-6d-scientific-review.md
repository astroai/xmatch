# Xmatch mixed 6D/4D propagation scientific review (2026-07-21)

status: proposal only; no production code changed

## Decision

The in-memory Xmatch propagation seam can safely use Torchsky's strict
`propagate_space_motion` primitive, but it must remain a **row-wise mixed
model**:

- use 6D propagation only when RA, Dec, both proper-motion components, source
  epoch, parallax, and radial velocity are finite, and parallax is positive;
- use the established angular propagation for every other row;
- never impute distance or radial velocity merely to qualify a row for 6D;
- preserve the present missing-data contract on the angular branch: each NaN
  PM component becomes zero independently, and a NaN source epoch becomes the
  target epoch (therefore no displacement);
- keep `target_epoch=None`, absent PM mappings, and absent epoch metadata on
  their existing no-propagation paths.

This is conservative for Gaia: negative/noisy parallaxes and the majority of
rows without radial velocities still benefit from measured angular proper
motion, while the smaller complete subset gains perspective acceleration and
changing-distance/light-time corrections.

Astropy **must** support the same 6D branch when Torchsky is unavailable. A
user's optional acceleration environment must not change the physical model.
Astropy is already a required Xmatch dependency, so this adds no dependency.

## Exact row classification

Classify from the unmodified input arrays. In particular, do not classify
after the current `nan_to_num` normalization.

```text
complete_6d =
    finite(ra)
  & finite(dec)
  & finite(pm_ra_cosdec)
  & finite(pm_dec)
  & finite(source_epoch)
  & finite(parallax)
  & (parallax > 0)
  & finite(radial_velocity)
```

The target epoch is one finite scalar for the request and should be validated
once. Longitude wrapping is acceptable; declination remains constrained to
[-90, 90] by the propagation backend.

The table below makes the intended split explicit.

| Per-row state | Branch | Reason |
|---|---|---|
| All seven source quantities finite; parallax > 0 | 6D | Complete physical catalogue state |
| Parallax NaN, infinite, zero, or negative | 4D | No positive measured distance; do not infer one |
| Radial velocity NaN or infinite | 4D | No radial component; do not assume zero |
| Either PM component NaN | 4D | Preserve component-wise zero-fill on angular path |
| Source epoch NaN | 4D with epoch set to target | Preserve current no-displacement behavior |
| RA/Dec non-finite | Never 6D; output remains non-finite | Avoid one malformed row aborting the strict vectorized backend |
| PM or epoch mapping absent for the side | No propagation | Existing catalogue-level contract |
| Parallax or RV mapping/column absent | 4D for the entire side | 6D is an optional refinement, not a prerequisite |

`null` values materialized by Polars should follow the same rules as NaN after
conversion to floating NumPy arrays.

## Recommended minimal API shape

Keep the existing `propagate_proper_motion(...) -> (ra, dec)` callable and
backward-compatible positional arguments. Add optional keyword-only
`parallax_mas` and `radial_velocity_km_s` arrays. If either optional array is
absent, execute the exact existing 4D implementation. If both are present,
partition rows by the mask above, propagate the two subsets, and scatter only
the returned RA/Dec into arrays in original row order.

Use a separately loaded optional Torchsky 6D symbol. An installed older
Torchsky that has angular propagation but lacks `propagate_space_motion`
should fall back to Astropy for the 6D subset; it must not disable the working
Torchsky angular path.

The metadata addition remains minimal and additive:

- `CatalogueSource.parallax_column` with a fixed public unit of mas;
- `CatalogueSource.radial_velocity_column` with a fixed public unit of km/s,
  positive for recession;
- matching fields in remote YAML loading and CLI-doctor drift classification;
- `radial_velocity` in the Gaia ESA/NOIRLab requested columns, alongside the
  already requested `parallax`;
- explicit Gaia mappings for both fields.

Local/in-memory catalogues currently cannot override even PM/epoch metadata
through `SideOverrides`. If this milestone is intended to cover arbitrary
local catalogues rather than only configured remote catalogues, add the PM,
epoch, parallax, and RV column names to `SideOverrides`, `_local_source`, and
the legacy parameter plumbing together. Do not auto-detect parallax or radial
velocity names: unit ambiguity makes that unsafe.

## Backend behavior

### Torchsky

Call `torchsky.wcs.propagate_space_motion` only on `complete_6d` rows and use
its first two outputs. Its strict contract rejects non-positive parallax,
non-finite inputs, apparent speeds at or above 0.5c, superluminal inertial
states, and trajectories crossing the observer. Normal Gaia-like states match
ERFA/Astropy at the precision already tested in Torchsky.

A catalogue outlier must not silently force all complete rows back to 4D.
Before vectorized dispatch, an inexpensive physical-admissibility mask may
exclude obviously impossible states using

```text
v_tan_km_s = 4.740470463 * hypot(pm_ra_cosdec, pm_dec) / parallax_mas
v_obs_squared = v_tan_km_s**2 + radial_velocity_km_s**2
```

and send rows at or above Torchsky's 0.5c observed-speed ceiling through the
angular branch with one count-bearing warning. If the remaining strict batch
still fails (for example, an extreme trajectory crosses the observer), raise
a contextual Xmatch error rather than silently changing the physical model
for every valid row. A future Torchsky non-raising validity-mask API would be
the clean high-throughput solution; per-row exception loops are not.

### Astropy fallback

When the 6D Torchsky symbol is unavailable, construct an ICRS `SkyCoord` for
the complete subset with:

- RA/Dec in degrees;
- `pm_ra_cosdec` and `pm_dec` in mas/yr;
- distance `1000 / parallax_mas` pc;
- radial velocity in km/s;
- source and target `Time(..., format="jyear", scale="tcb")`.

Then use `apply_space_motion` and return its RA/Dec. The fallback must be
tested on ordinary physically admissible states, because Astropy/ERFA's
`pmsafe` may warn and substitute a distance for pathological catalogue states
whereas Torchsky deliberately rejects them. Xmatch's admissibility policy
should keep such states out of the 6D subset consistently.

## Output semantics

For this milestone, retain Xmatch's established matching semantics:

- only the in-memory RA/Dec columns used by the matcher are rewritten to the
  requested target epoch;
- PM, parallax, radial-velocity, their errors, and the source `ref_epoch`
  columns remain the original catalogue measurements at their reference
  epoch;
- row order, row count, column names, non-coordinate values, and null masks
  are unchanged;
- longitude is normalized by the backend, and matching continues to consume
  the propagated coordinates exactly as the current 4D path does.

This means a saved match table is not a self-contained propagated astrometric
catalogue: its coordinate columns are at `MatchSpec.target_epoch`, while the
other astrometric columns remain source measurements. That is already true of
the present 4D implementation. Document it now; a later output-schema change
can add an explicit evaluation-epoch field or keep raw and matching
coordinates separately. Do not partly overwrite PM/parallax/RV with the six
Torchsky outputs in this milestone, since mixing propagated values on complete
rows with original values on incomplete rows would make the columns even more
ambiguous.

The positional-error ellipse also remains the source catalogue's existing 2D
error model. Full epoch propagation of Gaia's astrometric covariance requires
PM/parallax uncertainties and correlation terms and is a separate scientific
feature, not part of this mean-position improvement.

## Edge cases that need an explicit policy

1. **Perspective acceleration:** use a nearby, high-PM, non-zero-RV star over
   a long enough baseline to prove the 6D result differs from angular-only
   motion and matches Astropy.
2. **Negative/zero Gaia parallax:** remain 4D; no distance clipping and no
   parallax zero-point correction in the matcher.
3. **Missing RV:** remain 4D; radial velocity zero is a physical assertion, not
   a missing-value default.
4. **One missing PM component:** remain 4D and zero only that component, as
   today.
5. **Missing per-row epoch with catalogue epoch available:** the matcher
   currently uses the per-row array when the column exists and turns NaN into
   target epoch; preserve that precedence rather than falling back row-wise to
   the catalogue scalar.
6. **Zero time baseline:** coordinates should be unchanged to roundoff on both
   branches, including rows with radial velocity.
7. **RA wrap and near-pole positions:** scatter output without linear RA
   subtraction assumptions; logging should use wrapped great-circle motion or
   at least avoid presenting a 359.9-to-0.1 wrap as a 1.3-million-arcsec move.
8. **Empty side / all-invalid coordinates:** do not call `np.nanmax` on an
   empty/all-NaN delta array. Propagation should return the same empty schema.
9. **One pathological complete row in a large batch:** it must not discard 6D
   propagation for every valid row; pre-mask known speed violations and fail
   contextually on residual strict-backend failures.
10. **Remote column planning:** a configured RV mapping is useless unless the
    column is included in `default_columns`/the TAP selection. Test both.
11. **Out-of-core matching:** it already rejects all target-epoch propagation;
    leave that guard unchanged until chunk-wise epoch propagation is designed.

## Precise focused tests

### `tests/test_astro_utils.py`

1. `test_mixed_space_motion_dispatch_preserves_missing_semantics`
   - monkeypatch angular and 6D backends with recorders;
   - pass rows that are respectively complete, parallax-NaN, parallax-zero,
     RV-NaN, one-PM-NaN, and epoch-NaN;
   - assert only the first row reaches 6D;
   - assert angular rows retain original order, the missing PM component is
     received as zero, and missing epoch is received as the target epoch;
   - return distinct sentinel coordinates from each backend and assert scatter
     order exactly.

2. `test_space_motion_astropy_fallback_matches_reference`
   - force the Torchsky 6D loader to return `None`;
   - use a Barnard-like state (`pmra=-801.551`, `pmdec=10362.394`, parallax
     `548.31` mas, RV `-110.6` km/s, 2000 to 2025 TCB Julian years);
   - compare wrapped RA and Dec to an independently constructed Astropy
     `SkyCoord.apply_space_motion` result at `atol=1e-10` degrees;
   - additionally assert it differs measurably from the angular-only result.

3. `test_space_motion_torchsky_and_astropy_fallback_agree`
   - use two realistic complete states, positive and negative RV, vector source
     epochs, and a scalar target epoch;
   - run once through Torchsky and once with the loader disabled;
   - compare wrapped RA/Dec at a tolerance justified by Torchsky's existing
     parity tests (initially `atol=1e-7` degrees).

4. `test_mixed_space_motion_classifies_before_nan_normalization`
   - one row has NaN `pm_dec` but otherwise complete 6D metadata;
   - assert it never reaches the strict 6D fake and moves only in the finite PM
     component on the angular branch.

5. `test_space_motion_absent_optional_arrays_is_exact_legacy_path`
   - call the unchanged signature and compare both output and recorded backend
     arguments with the existing 4D contract.

### `tests/test_matchers.py`

6. `test_apply_proper_motion_uses_6d_only_for_complete_rows`
   - construct three Gaia-like rows: complete 6D, missing RV, and negative
     parallax;
   - monkeypatch the astro-utils wrapper to expose branch-specific sentinel
     coordinates or compare with direct backend references;
   - assert row count/order and every non-coordinate column are unchanged.

7. `test_apply_proper_motion_missing_6d_column_falls_back_for_side`
   - configure parallax/RV mappings but omit the RV data column;
   - assert the side still receives ordinary 4D propagation and does not fail.

8. `test_apply_proper_motion_empty_frame`
   - an empty DataFrame with full astrometric schema returns empty without the
     current `nanmax` failure.

9. `test_space_motion_changes_end_to_end_match`
   - choose a nearby high-PM star with an RV/baseline for which 6D and 4D
     positions straddle the selected match radius;
   - assert `sky_match(..., target_epoch=...)` makes the Astropy-referenced 6D
     decision for the complete row while a neighboring missing-RV row retains
     the 4D decision.

### Configuration/request tests

10. Assert Gaia ESA and NOIRLab resolved sources expose
    `parallax_column="parallax"`,
    `radial_velocity_column="radial_velocity"`, and request both columns.
11. Assert CLI doctor treats changes to either mapping as catalogue drift.
12. If local override support is included, assert `SideOverrides.as_dict()` and
    `_local_source()` carry all PM/epoch/parallax/RV mappings into the source.

The first eight tests are the minimum correctness boundary. The end-to-end and
configuration tests ensure the feature is actually reachable rather than only
unit-tested behind an unused helper.

## Intentionally deferred

- 5x5/6x6 covariance propagation and Gaia correlation-column mappings;
- probabilistic distance inference for noisy or negative parallaxes;
- RV imputation from a learned Galactic prior;
- per-row propagated PM/parallax/RV output columns;
- bounded-memory target-epoch propagation;
- GPU retention through the Polars matching seam (current arrays are NumPy and
  this milestone is about correct model selection first).
