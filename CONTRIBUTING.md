# Contributing

Operating procedures and conventions for contributors to `xmatch`.

## Tests & CI smoke guards

The project pins its test inventory against its `CHANGELOG.md` via six
bi-directional smoke guards in
[`tests/test_ci_smoke.py`](tests/test_ci_smoke.py) — three count
guards and three name-list guards for the HATS / LSDB Integration,
NOAO Data Lab Catalogues, and Per-Row PM Drift Mode sections.  When
you add, rename, or remove a test, expect one of those guards to fail
in CI.  This section walks you through fixing the failure.

> ### Canonical-source relationship
>
> This section is a **mirror** of the "Remediation Runbook" section
> inside the module docstring of
> [`tests/test_ci_smoke.py`](tests/test_ci_smoke.py).  The **docstring
> is canonical** — edits made there propagate here by hand.  When
> changing the workflow:
>
> 1. Edit the docstring first (canonical).
> 2. Update this mirror verbatim.
> 3. Commit them atomically; on intermediate SHAs the docstring
>    wins if the two diverges.
>
> Editing only this file risks the two drifting out of sync and
> readers of either copy following stale guidance.

### Remediation runbook (mirrored from `tests/test_ci_smoke.py`)

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
