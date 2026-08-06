"""CI smoke tests.

These tests pin the project's documentation against its test inventory so the
two cannot silently drift apart.  Each test runs ``pytest --collect-only -q``
and asserts the count matches what the docs claim.

If you intentionally add or remove a test, update **both** the test file and
the corresponding CHANGELOG bullet / doc reference.

Also wired as the ``ci-parity-pre-push-smoke`` hook in
``.pre-commit-config.yaml`` (fast-fail pre-push gate, shorter timeout than the
slow/real-catalogue suite).  When adding a new guard, mirror the existing
pattern: pick a unique keyword the guard asserts against + avoid that keyword
in the test's own name.  See the per-guard ``IMPORTANT NAMING CONSTRAINT``
blocks below for guard-specific details.

!!! Constraints (apply to every guard in this file) — NOTE: this is additive; per-guard admonitions still carry duplicate detail; a future polish pass will trim them to point at this section !!!

1. **Naming — keyword avoidance.**  Each guard's test name is chosen so
   that pytest's ``-k <keyword>`` collection does NOT include the guard
   itself in the returned set.  Pick a wording that does not contain
   the keyword the guard asserts against.  Naming safety in plain terms:
   you can grep ``-k $KEYWORD`` for the guard and see the same count
   whether or not the guard is in the suite.

2. **File-path coupling (static-scan guards only).**  Static-scan guards
   read a specific test file directly to decompose ``def test_*``
   declarations.  Hardcoding the file path is INTENDED: if the
   inventory moves to a new module, the guard fails on purpose — moves
   are a kind of drift we want to catch.  Update the hardcoded path
   as part of the move, not the assertion.

3. **Parser scope.**  All static-scan guards use
   ``re.findall(r"^def\\s+(test_\\w+)", ..., re.MULTILINE)`` to pick up
   top-level sync function definitions.  ``async def``, class-scope
   ``def``, and value-parm-decorated ``def`` are NOT recognised.  The
   codebase does not use those patterns today.  If pattern-breaking
   changes appear, the upgrade path is to replace each guard's regex
   with ``_collect_test_names("<keyword>")`` + ``::test_name`` split;
   NOTE that this upgrade path filters by *name substring*, so it does
   not preserve file-path coupling — see constraint #2.

!!! Remediation Runbook (smoke guard failures) !!!

Work through Steps 0–5 below.  Start at **Step 1** if remediating
a failing guard; start at **Step 0** if authoring a new guard.
Per-guard remediation lives in the failing guard's ``assert ...``
message; this runbook covers the cross-cutting workflow (diagnose →
reconcile → verify) so you only have to scan one place.

**Step 0 — (NEW GUARDS ONLY) Add a smoke guard.**  Skim this step
only when authoring a brand-new guard; if you are fixing a failing
existing guard, jump to Step 1.  To add a guard:

1. **Pick a unique keyword** that isolates the new test inventory
   subset.  The keyword must NOT appear as a substring in the new
   guard's own ``def test_*`` name (Constraint #1) — or the new guard
   silently inflates the ``--collect-only`` count and self-trips.

2. **Write the count assertion** using ``_collect_test_names(keyword)``
   and pin the literal to the count CHANGELOG will claim.  Failure
   message should list the matching CHANGELOG bullet so a reader
   knows where to look — see the three existing count guards for the
   stable shape.

3. **Decide count-only vs static-scan.**  If CHANGELOG names the
   tests verbatim, also write a static-scan guard and a module-level
   ``_<KEYWORD>_TEST_NAMES`` tuple.  Hardcode the ``tests/<file>.py``
   path the regex scans (Constraint #2).  See ``_HATS_TEST_NAMES``,
   ``_DATA_LAB_TEST_NAMES``, ``_PER_ROW_TEST_NAMES`` for the
   established shape.

4. **Update CHANGELOG.md** with the matching bullet — count claim AND
   verbatim function names in backticks (if static-scan).

5. **Verify locally** that the new guard's keyword still produces
   the same ``--collect-only`` count:

       pixi run test tests/test_ci_smoke.py -v

   The count the new guard expects must equal what
   ``-k <keyword> --collect-only -q`` returns.  If they diverge by
   1, the guard's own name collides with its keyword — rename it
   per Constraint #1.

**Step 1 — Read the failing guard.**  The guard name + the failing
assertion pinpoint which of the 6 guards tripped and what kind of
drift occurred.  Read the guard's docstring if it's a static-scan
guard — it lists the specific constant + bullet the guard anchors to.

For a fast triage without re-running the full smoke suite, paste
this one-liner to print the 3 keyword actuals in a single read
(runs the same ``pytest --collect-only`` invocation the guards use,
so the printed counts are the exact ones the guards see):

    for kw in pm_prior hats.py datalab; do printf '%-10s %d\n' "$kw" "$(pixi run test -k "$kw" --collect-only -q 2>/dev/null | grep -c '^tests/')"; done

If any count diverges from the matching ``count == N`` literal in
the corresponding guard's assertion (the pytest failure message
already names the guard AND its expected count), reconcile per
Step 3A.  For static-scan FORWARD/REVERSE failures the pytest
output also names which guard tripped; read that guard's docstring
for the specific constant + bullet.

If all 3 actuals match the corresponding ``count == N`` literals
(no CI failure, or any failure was unrelated to count drift), no
count reconciliation is needed — proceed to Step 4 to verify clean
state.

**Step 2 — Diagnose the drift.**  Four sub-cases:

* **Count mismatch** (``assert count == N`` fires) → Step 3A.
* **Static-scan FORWARD** (CHANGELOG name not in source file) → Step 3B.
* **Static-scan REVERSE** (test name not in CHANGELOG bullet) → Step 3C.
* **Parser-scope miss** (regex can't see the function — ``async def``,
  class-scope ``def``, or value-parm-decorated ``def``) → NOT a content
  bug; treat as a parser deficiency and follow the upgrade path in
  Constraint #3.

**Step 3A — Reconcile count mismatch.**  Inspect your branch's
``tests/`` diff (e.g. ``git log -p`` against the last green commit on
``main``, or your merge-base) against the matching CHANGELOG bullet.
Intentional change: update the ``count == N`` literal in the failing
guard AND the matching bullet's count claim.  Unintentional change:
search the diff for unintended test add/remove and revert.

**Step 3B — Reconcile FORWARD failure** (which also covers test-module
moves — see path 3 below).  Three reconciliation paths:

1. **Revert** rename/delete/file-move on your branch.
2. **Update the constant AND bullet** so they match the new state —
   edit the failing guard's module-level constant (the one named in
   the guard's docstring under Constraint #1) AND the matching
   CHANGELOG bullet's advertised names.
3. **If tests moved to a new module** (Constraint #2's intentional
   failure mode), update the hardcoded ``tests/<file>.py`` path in the
   failing guard's body.  This is by design — file moves are a kind
   of drift we want to catch.

**Step 3C — Reconcile REVERSE failure.**  The test still exists; the
CHANGELOG bullet simply dropped or rewrote the name.  Restore the
missing name verbatim in backticks inside the matching bullet.  Do
NOT also edit the module-level constant — those should match the
actual source.

**Step 4 — Re-run all 6 smoke guards.**

    pixi run test tests/test_ci_smoke.py -v --tb=short

All 6 should pass.  Iterate Steps 2–3 if any still fail.

**Step 5 — Re-run the full non-slow suite.**

    pixi run test -m "not slow and not bench"

The full non-slow suite (~162 tests) must pass before pushing.  This
catches the rare case where a test inventory change coincidentally
breaks an unrelated assertion.

*Note:* when both ``tests/test_ci_smoke.py`` and ``CHANGELOG.md`` are
modified, commit them atomically — split commits leave the guard in
a failing state on intermediate SHAs that CI will flag.

*Note (canonical-pointer / mirror):* this runbook is mirrored in
the top-level `CONTRIBUTING.md` for readers who hit a guard failure
before opening this file.  The **docstring above is canonical** —
edit it here when the workflow changes, then update the mirror
verbatim.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Pinned test-name tuples used by the static-scan guards below.  Each guard
# asserts its tuple (a) exists in the corresponding test file and (b) appears
# verbatim in CHANGELOG.md.  Underscore-prefixed to signal they are internal
# to this smoke-test module — not part of the project's public API — but
# these are **explicitly intended to be edited by contributors** as the test
# inventory changes, so updating a list here is the correct workflow for any
# rename/add/delete.  Order matches the order of the corresponding
# ``test_changelog_smoke_*_named_in_bullets`` guards below (HATS first,
# Data Lab second, per-row last).
_HATS_TEST_NAMES: tuple[str, ...] = (
    "test_lsdb_available_false",
    "test_require_lsdb_raises_with_helpful_message",
    "test_read_hats_no_path",
    "test_hats_crossmatch_unsupported_join_type_raises",
    "test_hats_target_epoch_routes_to_native_not_lsdb",
    "test_hats_outer_join_and_engine_prefer_native",
    "test_hats_crossmatch_passes_n_neighbors_best",
    "test_hats_crossmatch_passes_n_neighbors_all",
    "test_hats_crossmatch_find_all_multi_row_rename",
    "test_hats_crossmatch_custom_right_suffix",
    "test_hats_crossmatch_warns_on_missing_dist_arcsec",
    "test_hats_crossmatch_with_local_left_frame",
    "test_hats_crossmatch_warns_on_prior_columns",
    "test_resolve_source_hats_dir",
    "test_resolve_source_hats_dir_with_overrides",
    "test_crossmatch_multi_hats_at_position_3_routes_via_hats_crossmatch",
    "test_crossmatch_two_hats_via_dispatch",
    "test_is_hats_dir_multiple_markers",
)
_DATA_LAB_TEST_NAMES: tuple[str, ...] = (
    "test_resolve_source_datalab_alias",
    "test_resolve_source_datalab_aliases_point_to_tractor_not_object",
)
_PER_ROW_TEST_NAMES: tuple[str, ...] = (
    "test_pm_prior_per_row_drift_added_to_astrometric_errors",
    "test_pm_prior_per_row_gaia_realistic_error_budgets",
    "test_pm_prior_both_sides_drift_inflation",
)


def _collect_test_names(keyword: str) -> list[str]:
    """Run ``pytest --collect-only -q -k <keyword>`` and return collected IDs.

    Each collected test is emitted on its own line as ``path::test_name``.
    Lines that don't start with ``tests/`` are ignored — they include
    version banners, warnings, and (when present) the final summary line.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-k",
            keyword,
            "--collect-only",
            "-q",
            "--no-header",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    # Exit code 5 means "no tests collected" — perfectly valid for some keywords.
    # Anything else is a real failure.
    if proc.returncode not in (0, 5):
        raise RuntimeError(
            f"`pytest --collect-only -k {keyword!r}` failed with exit "
            f"{proc.returncode}.\n"
            f"--- STDOUT ---\n{proc.stdout}\n"
            f"--- STDERR ---\n{proc.stderr}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("tests/")]


def test_changelog_smoke_pm_drift_test_count() -> None:
    """Pin the count of PM drift prior tests documented in CHANGELOG.md.

    The v0.5 ``[Unreleased]`` section claims:

        * 5 PM drift prior baseline tests  (skyerr inflation, no-epoch skip,
          skyerr matcher, magnitude scaling, missing magnitude column)
        * 3 new per-row PM drift tests     (per-row quadrature, Gaia-class
          realistic budgets, two-old-survey case)

    Total: 8.  This guard fails if the count diverges — forcing the test
    suite and the CHANGELOG to be updated together.

    !!! IMPORTANT NAMING CONSTRAINT !!!
    This test name deliberately does NOT contain the substring ``pm_prior``
    so that ``pytest -k pm_prior --collect-only`` returns 8 (not 9,
    which would silently include this guard).  Do not rename this test to
    contain ``pm_prior`` — that would inflate the count to 9 and the
    guard would fail wrongly.  Use ``pm_drift`` (as above) or unrelated
    wording instead.

    Verify locally from the project root:

        pixi run test -k pm_prior --collect-only -q

    (Internally this test invokes ``python -m pytest --collect-only -q``
    via subprocess, using the current ``sys.executable``.  When invoked
    under pixi this matches ``pixi run test``; if you ever call this
    file with a different Python, re-run under pixi to keep parity.)

    To find a divergence, inspect what was collected and reconcile with
    CHANGELOG.md sections:

        * ``### Added — PM Drift Prior (pm_prior=True)``     (5 baseline)
        * ``### Added — Per-Row PM Drift Mode (when pm_prior=True)`` (3 new)
    """
    collected = _collect_test_names("pm_prior")
    count = len(collected)

    assert count == 8, (
        f"Expected 8 PM drift prior tests in CHANGELOG (5 baseline + 3 per-row), "
        f"but pytest collected {count}. "
        f"This usually means a test was added, removed, or renamed without "
        f"updating CHANGELOG.md (or vice versa).\n\n"
        f"Collected {count} test(s):\n  " + "\n  ".join(collected) + "\n\n"
        "Sections to update in CHANGELOG.md:\n"
        "  * 'Added — PM Drift Prior (`pm_prior=True`)': 5 baseline tests\n"
        "  * 'Added — Per-Row PM Drift Mode (when `pm_prior=True`)': 3 new tests\n"
        "If you intentionally changed the inventory, update both the test "
        "file and CHANGELOG.md, then update the count here."
    )


def test_changelog_smoke_hats_test_count() -> None:
    """Pin the count of HATS tests documented in CHANGELOG.md.

    The v0.5 ``[Unreleased]`` section claims "**18 HATS tests**
    (``tests/test_hats.py``) — fully mocked (no LSDB required),
    covering alias resolution, crossmatch parameter passthrough, multi-row
    ``find='all'`` results, join type validation, local-frame conversion,
    column renaming, and ``crossmatch_multi`` routing."

    This guard uses the keyword ``hats.py`` (a substring of every test ID
    in that file) so it matches all 18 ``tests/test_hats.py`` functions
    regardless of whether their function name contains ``hats`` (e.g.
    ``test_lsdb_available_false`` and
    ``test_require_lsdb_raises_with_helpful_message`` do not).

    !!! Match-scope limitation !!!
    The keyword ``hats.py`` matches only tests under
    ``tests/test_hats.py``.  If a contributor adds a *new* HATS-related
    test to a different file (e.g.
    ``tests/test_matchers.py::test_hats_crossmatch_xyz``), the guard
    still returns 18 and silently passes while CHANGELOG.md should
    claim 19.  Such cross-file additions are out of scope; the guard
    tracks the invariant "all HATS tests live in tests/test_hats.py"
    implied by the current CHANGELOG wording.
    """
    collected = _collect_test_names("hats.py")
    count = len(collected)

    assert count == 18, (
        f"Expected 18 HATS tests in CHANGELOG (all in tests/test_hats.py), "
        f"but pytest collected {count}. "
        f"Update tests/test_hats.py and the CHANGELOG.md bullet:\n\n"
        f"  * 'Added — HATS / LSDB Integration': 18 HATS tests\n\n"
        f"Collected {count} test(s):\n  " + "\n  ".join(collected) + "\n\n"
        "If you intentionally changed the inventory, update both the test "
        "file and CHANGELOG.md, then update the count here."
    )


def test_changelog_smoke_hats_named_in_bullets() -> None:
    """Static complement to the count-based HATS guard.

    While ``test_changelog_smoke_hats_test_count`` only checks the *count*
    of HATS tests (18), this guard asserts the 18 specific test names
    listed in CHANGELOG's "Added — HATS / LSDB Integration" bullet
    exist verbatim in **both** the source file and the CHANGELOG text.
    Together they form a bidirectional exact-match check that catches
    renames AND file moves even when the count stays at 18.

    Two failure modes are caught here, both invisible to the count guard:

    1. **Rename drift.**  CHANGELOG still references the OLD test name; the
       renamed test has a different name (e.g. ``_v2`` suffix added).
       Forward check fails.
    2. **Edit drift.**  A contributor removes a name from CHANGELOG but
       the corresponding test still exists.  Reverse check fails.

    The 18 test names are listed inline so a contributor reading this
    guard can audit them alongside the inventory without consulting
    CHANGELOG.md separately.

    !!! Naming — see module-level Constraint #1 !!!
    This test name deliberately does NOT contain the literal substring
    ``hats.py``; the count guard uses ``-k 'hats.py'`` (with the dot) so
    this guard is not silently included in the count.  Renaming this
    test to ``test_changelog_smoke_hats.py_names_listed`` would inflate
    the count to 19.

    !!! File-path coupling (intentional) — see module-level Constraint #2 !!!
    Reads ``tests/test_hats.py`` directly.  If HATS tests reorganise to
    a new file, this guard fails on purpose.

    !!! Parser scope — see module-level Constraint #3 !!!
    """
    # Names pinned by module-level `_HATS_TEST_NAMES`.
    expected_names: tuple[str, ...] = _HATS_TEST_NAMES

    # Forward: scan tests/test_hats.py for `def test_*` definitions
    test_hats_path = PROJECT_ROOT / "tests" / "test_hats.py"
    test_funcs_in_suite = set(
        re.findall(r"^def\s+(test_\w+)", test_hats_path.read_text(), re.MULTILINE)
    )
    missing_from_suite = set(expected_names) - test_funcs_in_suite

    # Reverse: scan CHANGELOG.md for verbatim mention of each expected name
    changelog_text = (PROJECT_ROOT / "CHANGELOG.md").read_text()
    missing_from_changelog = {n for n in expected_names if n not in changelog_text}

    assert not missing_from_suite, (
        f"\n--- FORWARD CHECK FAILED ---\n"
        f"CHANGELOG.md's 'Added — HATS / LSDB Integration' bullet advertises "
        f"{len(expected_names)} test names; one or more are missing from "
        f"tests/test_hats.py (likely renamed or deleted without updating "
        f"CHANGELOG).  The count-based guard "
        f"(`test_changelog_smoke_hats_test_count`) still passes because the "
        f"total count is unchanged, but the SPECIFIC NAMES the CHANGELOG "
        f"promises no longer exist.\n\n"
        f"Missing from test suite:\n  " + "\n  ".join(sorted(missing_from_suite)) + "\n\nEither:\n"
        "  1. Revert the rename/deletion to restore the advertised names.\n"
        "  2. Update CHANGELOG.md to advertise the new names.\n"
    )
    assert not missing_from_changelog, (
        "\n--- REVERSE CHECK FAILED ---\n"
        "These HATS test names exist in tests/test_hats.py but are no "
        "longer mentioned in CHANGELOG.md (a contributor likely edited the "
        "bullet text without touching the test file):\n\n"
        "Missing from CHANGELOG:\n  "
        + "\n  ".join(sorted(missing_from_changelog))
        + "\n\nUpdate CHANGELOG.md's 'Added — HATS / LSDB Integration' bullet "
        "to mention these names verbatim.\n"
    )


def test_changelog_smoke_data_lab_test_count() -> None:
    """Pin the count of Data Lab / NOAO source-resolution tests documented
    in CHANGELOG.md.

    The v0.5 ``[Unreleased]`` section (under "Added — NOAO Data Lab
    Catalogues") claims "**2 Data Lab source resolution tests**
    (``tests/test_crossmatch.py``) — verify aliases resolve to TAP-backed
    ``CatalogueSource`` with correct table names, archive, and column
    metadata.  Tractor vs object table routing verified."

    The two tests:

        * ``test_resolve_source_datalab_alias``
        * ``test_resolve_source_datalab_aliases_point_to_tractor_not_object``

    The keyword ``datalab`` matches both function names (lowercase
    substring, no separator underscore) and matches nothing else in the
    suite.

    !!! IMPORTANT NAMING CONSTRAINT !!!
    This test name deliberately avoids ``datalab`` so pytest doesn't
    include this guard in the collected set.  Do not rename it.

    Verify locally:

        pixi run test -k datalab --collect-only -q
    """
    collected = _collect_test_names("datalab")
    count = len(collected)

    assert count == 2, (
        f"Expected 2 Data Lab source-resolution tests in CHANGELOG, "
        f"but pytest collected {count}. "
        f"Update tests/test_crossmatch.py and the CHANGELOG.md bullet:\n\n"
        f"  * 'Added — NOAO Data Lab Catalogues': 2 Data Lab tests\n\n"
        f"Collected {count} test(s):\n  " + "\n  ".join(collected) + "\n\n"
        "If you intentionally changed the inventory, update both the test "
        "file and CHANGELOG.md, then update the count here."
    )


def test_changelog_smoke_data_lab_named_in_bullets() -> None:
    """Static complement to the count-based Data Lab guard.

    While ``test_changelog_smoke_data_lab_test_count`` only checks the
    *count* of Data Lab tests (2), this guard asserts the 2 specific
    test names listed in CHANGELOG's "Added — NOAO Data Lab Catalogues"
    bullet exist verbatim in **both** the source file and the CHANGELOG
    text.  Together they form a bidirectional exact-match check that
    catches renames AND file moves even when the count stays at 2.

    Two failure modes are caught here, both invisible to the count guard:

    1. **Rename drift.**  CHANGELOG still references the OLD test name;
       the renamed test has a different name.  Forward check fails.
    2. **Edit drift.**  A contributor removes a name from CHANGELOG but
       the corresponding test still exists.  Reverse check fails.

    The 2 test names are listed inline so a contributor reading this
    guard can audit them alongside the inventory without consulting
    CHANGELOG.md separately.

    !!! Naming — see module-level Constraint #1 !!!
    This test name (which contains ``data_lab`` WITH underscore, not
    ``datalab`` without) deliberately avoids the literal substring
    ``datalab``; the count guard uses ``-k datalab`` (no underscore) so
    this guard is not silently included in the count.  Renaming this
    test to ``test_changelog_smoke_datalab_named_in_bullets`` (no
    underscore between data and lab) would inflate the count to 3.

    !!! File-path coupling (intentional) — see module-level Constraint #2 !!!
    Reads ``tests/test_crossmatch.py`` directly.  If Data Lab tests
    reorganise to a new file, this guard fails on purpose.

    !!! Parser scope — see module-level Constraint #3 !!!
    """
    # Names pinned by module-level `_DATA_LAB_TEST_NAMES`.
    expected_names: tuple[str, ...] = _DATA_LAB_TEST_NAMES

    # Forward: scan tests/test_crossmatch.py for `def test_*` definitions
    test_crossmatch_path = PROJECT_ROOT / "tests" / "test_crossmatch.py"
    test_funcs_in_suite = set(
        re.findall(r"^def\s+(test_\w+)", test_crossmatch_path.read_text(), re.MULTILINE)
    )
    missing_from_suite = set(expected_names) - test_funcs_in_suite

    # Reverse: scan CHANGELOG.md for verbatim mention of each expected name
    changelog_text = (PROJECT_ROOT / "CHANGELOG.md").read_text()
    missing_from_changelog = {n for n in expected_names if n not in changelog_text}

    assert not missing_from_suite, (
        f"\n--- FORWARD CHECK FAILED ---\n"
        f"CHANGELOG.md's 'Added — NOAO Data Lab Catalogues' bullet "
        f"advertises {len(expected_names)} test names; one or more are "
        f"missing from tests/test_crossmatch.py (likely renamed or "
        f"deleted without updating CHANGELOG).  The count-based guard "
        f"(`test_changelog_smoke_data_lab_test_count`) still passes "
        f"because the total count is unchanged, but the SPECIFIC NAMES "
        f"the CHANGELOG promises no longer exist.\n\n"
        f"Missing from test suite:\n  " + "\n  ".join(sorted(missing_from_suite)) + "\n\nEither:\n"
        "  1. Revert the rename/deletion to restore the advertised names.\n"
        "  2. Update CHANGELOG.md to advertise the new names.\n"
    )
    assert not missing_from_changelog, (
        "\n--- REVERSE CHECK FAILED ---\n"
        "These Data Lab test names exist in tests/test_crossmatch.py but "
        "are no longer mentioned in CHANGELOG.md (a contributor likely "
        "edited the bullet text without touching the test file):\n\n"
        "Missing from CHANGELOG:\n  "
        + "\n  ".join(sorted(missing_from_changelog))
        + "\n\nUpdate CHANGELOG.md's 'Added — NOAO Data Lab Catalogues' "
        "bullet to mention these names verbatim.\n"
    )


def test_changelog_smoke_per_row_names_listed() -> None:
    """Static complement to the count-based pm_prior guard.

    While ``test_changelog_smoke_pm_drift_test_count`` only checks the
    *count* of pm_prior tests (8), this guard asserts that the 3 test
    names explicitly advertised in CHANGELOG's "Added — Per-Row PM
    Drift Mode" bullet exist verbatim in **both** the test file and the
    CHANGELOG text.  Together they form a bidirectional exact-match
    check.

    The 5 baseline tests are NOT asserted at the function-name level
    because CHANGELOG describes them by topic ("skyerr inflation,
    no-epoch skip, ..."), not by exact identifier — matching by topic
    description would be brittle.

    Two failure modes are caught here, both invisible to the count guard:

    1. **Rename drift.**  CHANGELOG still references the OLD test name;
       the renamed test now has a different name.  Forward check fails.
    2. **Edit drift.**  A contributor removes a name from CHANGELOG but
       the corresponding test still exists.  Reverse check fails.

    Bidirectional implementation:

    * **Forward**: each name in ``EXPECTED_NAMES`` must be defined in
      ``tests/test_matchers.py``.
    * **Reverse**: each name in ``EXPECTED_NAMES`` must appear verbatim
      in ``CHANGELOG.md`` text.

    The reverse check is somewhat redundant with the count guard for
    additions/deletions, but it precisely catches the case where a
    contributor silently removes a name from CHANGELOG without
    touching the test file.

    !!! IMPORTANT NAMING CONSTRAINT !!!
    This test name deliberately avoids ``pm_prior`` (so it isn't silently
    included in the ``-k pm_prior`` collected set that ``pytest --collect-only
    -k pm_prior`` returns to the companion count guard,
    ``test_changelog_smoke_pm_drift_test_count``).  It also avoids
    ``hats.py`` and ``datalab`` so it isn't matched by the other two
    collection-keyword guards.

    The test name *does* contain ``per_row`` and ``drift`` because those
    are the identifying word-roots for the per-row PM drift feature this
    guard is about; no other guard uses those substrings today, but if a
    future guard does, this test name should be renamed to a
    non-overlapping wording.

    Rename with care.

    !!! File-path coupling (intentional) !!!
    This guard reads ``tests/test_matchers.py`` directly because that's
    where pm_prior tests live today.  If they are reorganised into a
    separate file (e.g. ``tests/test_pm_drift.py``), this guard fails
    *on purpose* — moves are a kind of drift we want to catch.  Update
    the path here as part of the move.

    !!! Regex scope (assumed) !!!
    The parser below assumes top-level, synchronous ``def test_*``
    statements.  ``async def``, class-scope ``def``, or value-parm
    decorators are not handled.  The codebase does not use those
    patterns today; if it does appear, replace this guard's regex with
    ``_collect_test_names("pm_prior")`` + ``::test_name`` split.  NOTE:
    that helper-based upgrade does NOT cover file moves between modules
    — see the File-path coupling block above for that scenario.
    """
    # Names pinned by module-level `_PER_ROW_TEST_NAMES`.
    expected_names: tuple[str, ...] = _PER_ROW_TEST_NAMES

    # Forward: scan tests/test_matchers.py for `def test_*` definitions
    test_matchers_path = PROJECT_ROOT / "tests" / "test_matchers.py"
    test_funcs_in_suite = set(
        re.findall(r"^def\s+(test_\w+)", test_matchers_path.read_text(), re.MULTILINE)
    )
    missing_from_suite = set(expected_names) - test_funcs_in_suite

    # Reverse: scan CHANGELOG.md for verbatim mention of each expected name
    changelog_text = (PROJECT_ROOT / "CHANGELOG.md").read_text()
    missing_from_changelog = {n for n in expected_names if n not in changelog_text}

    assert not missing_from_suite, (
        f"\n--- FORWARD CHECK FAILED ---\n"
        f"CHANGELOG.md's 'Added — Per-Row PM Drift Mode' bullet advertises "
        f"{len(expected_names)} test names; one or more are missing from "
        f"tests/test_matchers.py (likely renamed or deleted without updating "
        f"CHANGELOG).  The count-based guard (\n"
        f"    test_changelog_smoke_pm_drift_test_count\n"
        f") still passes because total count is unchanged, but the SPECIFIC "
        f"NAMES the CHANGELOG promises no longer exist.\n\n"
        f"Missing from test suite:\n  " + "\n  ".join(sorted(missing_from_suite)) + "\n\nEither:\n"
        "  1. Revert the rename/deletion to restore the advertised names.\n"
        "  2. Update CHANGELOG.md to advertise the new names.\n"
    )
    assert not missing_from_changelog, (
        "\n--- REVERSE CHECK FAILED ---\n"
        "These per-row test names exist in tests/test_matchers.py but are "
        "no longer mentioned in CHANGELOG.md (a contributor likely edited "
        "the bullet text without touching the test file):\n\n"
        "Missing from CHANGELOG:\n  "
        + "\n  ".join(sorted(missing_from_changelog))
        + "\n\nUpdate CHANGELOG.md's 'Added — Per-Row PM Drift Mode' bullet "
        "to mention these names verbatim.\n"
    )
