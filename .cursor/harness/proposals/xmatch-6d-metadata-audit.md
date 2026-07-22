# Xmatch 6D source-metadata audit

Status: audit only; no production edits.  Scope is the additive optional
`parallax_column` and `radial_velocity_column` metadata needed before a
complete-row path can call `torchsky.wcs.propagate_space_motion`.

## Required metadata lifecycle changes

| Site | Finding | Required change |
| --- | --- | --- |
| `src/xmatch/sources.py:35-46` | `CatalogueSource` is the sole source-metadata dataclass.  It currently retains epoch and two PM column names only. | Add optional `parallax_column` and `radial_velocity_column` adjacent to the PM metadata.  `with_frame()` at `:60-68` uses `copy.copy`, so it will preserve both automatically; no custom copy/serialization code exists. |
| `src/xmatch/request.py:36-83` | `SideOverrides`/`as_dict()` are the typed per-side construction path; neither new field can currently be expressed for local/HATS inputs. | Add both fields and include them in `as_dict()`. |
| `src/xmatch/request.py:158-235` | Legacy `CrossMatch.crossmatch(**params)` is the compatibility construction path. | Accept `parallax_column_1/_2` and `radial_velocity_column_1/_2`, and carry them into `SideOverrides`.  Add matching CLI flags only if the public CLI is intended to configure local 6D inputs; do not invent a separate override mechanism. |
| `src/xmatch/crossmatch.py:245-270` | `_local_source()` assigns only positional-error/coordinate/id overrides. | Assign the six kinematic fields supplied through overrides (at least the two new fields; PM/epoch overrides are absent today).  Otherwise a `MatchRequest` loses the new metadata before matching. |
| `src/xmatch/crossmatch.py:272-293` | `_remote_source()` is the only config-to-`CatalogueSource` adapter. | Map `cfg.get("parallax_column")` and `cfg.get("radial_velocity_column")`.  Keep values `None` when a catalogue lacks a validated RV field. |
| `src/xmatch/crossmatch.py:295-305` | `_hats_source()` is another explicit constructor. | Forward the optional side overrides here too.  HATS does not need a special transport mechanism after construction. |
| `src/xmatch/crossmatch.py:833` | `dataclasses.replace(first_src, ra_column=..., dec_column=...)` preserves additive fields by construction. | No change required.  Do not substitute aggregate coordinates into a 6D propagation path: its parallax/RV/PM columns no longer refer unambiguously to the coalesced coordinate. |

There is no `asdict`, JSON serializer, pickle adapter, or manual clone beyond
the sites above (`rg` audit).  Therefore no separate serialization change is
required.

## Configuration and remote projection gates

| Site | Finding | Required change |
| --- | --- | --- |
| `src/xmatch/xmatch.yaml:277-297`, `:337-358` | The Gaia NOIRLab and ESA entries include `parallax` but no radial-velocity metadata/column.  The CDS Gaia entry at `:101-125` similarly has `Plx` but no RV. | Add the fields only after validating each archive's actual column spelling and units.  Include them in `default_columns` when configured.  In particular, do **not** assume Gaia CDS's VizieR RV label matches ESA's `radial_velocity`. |
| `src/xmatch/crossmatch.py:115-143` | `_validate_config()` validates structural remote fields but does not validate optional column metadata. | At minimum validate that each supplied optional mapping is a non-empty string.  A schema/remote discovery check belongs in the existing doctor/discovery workflow, not eager initialization. |
| `src/xmatch/cli.py:393-411` | `xmatch doctor` treats PM/epoch/default columns as structural drift, but will not report drift in the two new mappings. | Add both names to `OUTDATED_CATALOGUE_FIELDS`; add doctor fixtures for their missing/changed cases. |
| `src/xmatch/cli.py:1626-1646` | Catalogue display exposes PM availability only. | Show parallax/RV mappings when configured (documentation/UI completeness, not matching correctness). |
| `src/xmatch/remote_tap.py:24-29` | `_select_columns()` force-adds only RA/Dec/ID.  User-supplied `columns_N` can silently omit PM, epoch, parallax, or RV needed later by propagation. | When `target_epoch` selects PM/6D propagation, pass a required-column set into remote planning and force-add all declared state columns.  Do not always fetch RV for ordinary cone matches. |
| `src/xmatch/remote_tap.py:86-96` | `tap_self_join()` selects `default_columns` verbatim and performs server-side matching before the local propagation seam. | A target-epoch request must bypass this optimization, download both sides with their required columns, and run the local propagated matcher.  Merely adding columns here does not make the ADQL spatial predicate epoch aware. |
| `src/xmatch/remote_cds.py:42` | `download_from_cds()` likewise honors `columns` verbatim. | Use the same conditional required-column planning as TAP before downloading for local matching. |
| `src/xmatch/crossmatch.py:1277-1281` and `src/xmatch/remote_cds.py:62-108` | CDS XMatch uploads only RA/Dec and asks the service to match those coordinates. | A target-epoch/6D request must bypass `cds_xmatch_local_remote`; otherwise it silently ignores propagation.  Fetch through the normal CDS download route (or raise an explicit unsupported error if the archive cannot supply a region). |
| `src/xmatch/out_of_core.py:265-268` | Spill mode explicitly rejects every target-epoch request. | Correct as-is.  Keep it rejected until its projected-column and propagation implementation is deliberately added; do not claim 6D support in bounded-memory mode. |

## Matching seam and safety contract

* `src/xmatch/matchers.py:426-497` is the current 4D propagation seam;
  `sky_match()` invokes it only after both frames are fully materialized at
  `:2693-2715`.  Extend this seam, rather than individual engines, so fast,
  Astropy, Torchsky and STILTS see the same propagated positions.
* Select 6D only per row when PM, epoch, positive finite parallax and finite
  RV are all available.  Preserve the existing 4D angular propagation for
  incomplete rows.  Do not impute parallax or radial velocity.
* The output source metadata stays valid after `with_frame()` (shallow copy),
  but the two new numerical columns must remain selected in the downloaded
  Polars frame until propagation has completed.

## Minimum regression tests

1. In `tests/test_crossmatch.py`, resolve `gaia_esa`/a temporary remote config
   and assert both mappings survive config -> `CatalogueSource` ->
   `with_frame()`.  Cover a local frame with explicit `SideOverrides` too.
2. In `tests/test_remote.py`, mock TAP and CDS downloads with restrictive
   `columns_2`; under `target_epoch`, assert the request includes RA, Dec, PM,
   epoch, parallax and RV columns exactly once.  Add a control test proving a
   non-propagated match does not fetch optional 6D fields.
3. In `tests/test_remote.py`, with two same-TAP sources and `target_epoch`,
   assert `tap_self_join()` is not selected; the local propagated route is
   used.  Add the analogous CDS-XMatch bypass/explicit-error test.
4. In `tests/test_matchers.py`, give a mixed-completeness catalogue: one
   complete six-dimensional row and one row lacking RV/parallax.  Assert the
   complete row agrees with Torchsky 6D propagation and the incomplete row
   follows the existing 4D result, with no NaN-induced false match.
5. In `tests/test_cli.py`, verify doctor marks a changed or removed
   `parallax_column` and `radial_velocity_column` as outdated configuration
   drift.

## Deliberate non-changes

* Benchmark/script-only `CatalogueSource(...)` construction sites need no
  update: the fields are optional and they do not serialize or reconstruct a
  source.
* Existing Gaia configuration must not acquire an RV mapping until its
  provider-specific column name and units have been verified.
