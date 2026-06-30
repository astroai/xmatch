# Quarterly AI-agent PR cleanup — maintainer checklist

> Reusable checklist for the quarterly stale-`bolt-`/`palette-`/`jules-`/
> `integrate-` PR cleanup pass. **The destructive close step is always
> run manually by a maintainer**, never unattended. The GitHub Actions
> workflow at `.github/workflows/q3-review-reminder.yml` automates only
> the dry-run scan + triage issue creation.

## When

* **Scheduled:** `0 6 28 3,6,9,12 *` — 06:00 UTC on the 28th of the
  last month of each prior quarter (Mar / Jun / Sep / Dec). This gives
  maintainers ~3 days of heads-up before the quarter turns.
* **Snapshot tag:** `q3-2026-review` (annotated) marks the AUDIT Tier 2
  closure for Q3-2026.

## Step-by-step

1. **Confirm `--help` reads cleanly.**
   ```bash
   pixi run python scripts/cleanup-stale-prs.py --help
   ```
2. **Run the dry-run scan** (reads remote state; no destructive ops).
   ```bash
   pixi run python scripts/cleanup-stale-prs.py --json | jq
   ```
   Expected: a JSON array on `stdout`; status text on `stderr` (the
   `--apply` + `--json` composition contract pinned in
   `tests/test_cleanup_stale_prs.py`). Empty array means: skip step 3
   and you're done for the quarter.
3. **Triage candidates.** For each row:
   * `keep` if a maintainer has commented in the last 14 days, or if
     the PR is otherwise still in-flight; the script leaves those rows
     on the JSON and the `--apply` step skips them.
   * `close` for everything else.
   Add a per-row decision to the GitHub triage issue the workflow
   opens (when matches are non-empty).
4. **Apply destructive close + branch delete** (manual, never
   unattended). Per `scripts/cleanup-stale-prs.py`'s `--apply` + `--json`
   composition contract (pinned by `tests/test_cleanup_stale_prs.py`),
   `--apply` runs FIRST and `--json` then emits the post-close state.
   All status messages route to `stderr` so `jq` sees pure JSON.
   ```bash
   pixi run python scripts/cleanup-stale-prs.py --apply --json \
     | tee q$(date -u +%-m | awk '{q=int(($1-1)/3)+1; print q}')-stale-dump.json \
     | jq
   ```
   The quarter math is in `awk` instead of GNU `date +%q` so the
   command is portable to macOS / BSD `date`.
5. **Verify the workflow would be clean next time.**
   Re-run step 2. If the result is still non-empty, repeat.

## Reference

* `AUDIT.md` — Tier 3 ("remote cleanup") section.
* `scripts/cleanup-stale-prs.py` — the script itself.
* `scripts/cleanup-stale-prs.sh` — thin wrapper.
* `tests/test_cleanup_stale_prs.py` — 7 regression tests covering the
  `--apply` + `--json` composition, stdout purity, and the
  destructive-progress-to-stderr routing.
* `.github/workflows/q3-review-reminder.yml` — schedule + workflow
  dispatch that runs step 2 + posts a triage issue when non-empty.
* `CONTRIBUTING.md` — "Quarterly maintenance: stale AI-agent PR
  cleanup" (links back to this file).
