#!/usr/bin/env bash
# scripts/cleanup-stale-prs.sh
#
# Closes and de-branches the stale `bolt-*`/`palette-*`/`ux-*`/`jules-*` PRs
# from earlier AI-agent flows. **Default is dry-run** to avoid accidental
# destruction. Pass `--apply` once you've reviewed the printed plan.
#
# Rollback: every closed PR is recorded in $PR_LIST; reopen with
#   while read N; do gh pr reopen "$N" && echo "reopened $N"; done < $PR_LIST
#
# Remote branch deletion is also recorded in $BR_LIST; re-create with
#   while read B SHA; do git push origin "$SHA:refs/heads/$B"; done < $BR_LIST
#
# Requirements: `gh` authed as a maintainer and `git` with push access.

set -euo pipefail

APPLY=""
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

PR_LIST="$(mktemp -t cleanup-prs.XXXXXX)"
BR_LIST="$(mktemp -t cleanup-branches.XXXXXX)"
trap 'rm -f "$PR_LIST" "$BR_LIST"' EXIT

echo "==> Fetching every open PR with a bolt/palette/ux/jules/integrate/perf prefix"
mapfile -t PRS < <(
  gh pr list --state open --limit 200 \
    --json number,title,state,headRefName,createdAt,updatedAt,author,additions,deletions \
    | python3 -c '
import json, sys
data = json.load(sys.stdin)
PREFIXES = ("bolt","palette","ux","jules","integrate","perf")
for p in data:
    head = p["headRefName"] or ""
    if any(head.split("-")[0].startswith(pfx) for pfx in PREFIXES):
        add = p.get("additions") or 0
        dels = p.get("deletions") or 0
        author = p.get("author", {}).get("login", "?")
        print(f"{p[\"number\"]}|{head}|{author}|{add}|{dels}|{(p.get(\"updatedAt\") or \"\")[:10]}|{p[\"title\"][:80]}")
'
)

printf "%-5s %-32s %-14s %6s %6s %-12s %s\n" "PR" "head" "author" "+add" "-del" "updated" "title"
printf "%-5s %-32s %-14s %6s %6s %-12s %s\n" "----" "----" "----" "----" "----" "----" "----"
for line in "${PRS[@]:-}"; do
  [[ -z "$line" ]] && continue
  IFS='|' read -r NUM HEAD AUTHOR ADD DEL UPDATED TITLE <<<"$line"
  printf "%-5s %-32s %-14s %6s %6s %-12s %s\n" "#$NUM" "$HEAD" "$AUTHOR" "+$ADD" "-$DEL" "$UPDATED" "$TITLE"
done

if [[ -z "${PRS[*]:-}" ]]; then
  echo "No matching open PRs. Nothing to do."
  exit 0
fi

if [[ -n "$APPLY" ]]; then
  echo "==> --apply: closing open PRs in the stale prefixes"
  for line in "${PRS[@]:-}"; do
    [[ -z "$line" ]] && continue
    IFS='|' read -r NUM HEAD _ _ _ _ _ <<<"$line"
    echo "  closing #$NUM (head=$HEAD)"
    gh pr close "$NUM" --delete-branch --comment \
      "Closing as superseded by v0.3 audit batch (see AUDIT.md, CHANGELOG.md). \
       Reopen if still relevant: gh pr reopen $NUM."
    echo "$NUM" >> "$PR_LIST"
  done

  echo "==> --apply: deleting remote stale branches"
  for line in "${PRS[@]:-}"; do
    [[ -z "$line" ]] && continue
    IFS='|' read -r NUM HEAD _ _ _ _ _ <<<"$line"
    if git ls-remote --heads origin "$HEAD" | grep -q "$HEAD"; then
      sha=$(git ls-remote --heads origin "$HEAD" | awk '{print $1}')
      echo "  deleting origin/$HEAD (was $sha)"
      git push origin --delete "$HEAD" || true
      echo "$HEAD $sha" >> "$BR_LIST"
    fi
  done

  echo
  echo "==> Done. Rollback list (re-open and re-create branches):"
  echo "   PR list:      $PR_LIST"
  echo "   branch list:  $BR_LIST"
else
  echo
  echo "==> Dry-run only. Re-run with --apply to actually close + delete."
  echo "  PRs that would close: ${#PRS[@]}"
fi
