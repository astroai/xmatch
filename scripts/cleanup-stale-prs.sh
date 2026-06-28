#!/usr/bin/env bash
# scripts/cleanup-stale-prs.sh
#
# Thin bash wrapper around scripts/cleanup-stale-prs.py.
#
# Why two files?
#   The original one-file design put a `python3 -c '...'` heredoc inside the
#   bash script, which rounds trips f-strings with escaped double quotes very
#   poorly. Splitting the logic into a sibling .py file keeps quoting
#   straightforward in both languages (you write python one way, bash the
#   other), and adds argparse to the .py so behaviour is self-documenting.
#
# Usage:
#   scripts/cleanup-stale-prs.sh            # dry-run, prints table
#   scripts/cleanup-stale-prs.sh --apply    # destructive
#   scripts/cleanup-stale-prs.sh --json     # machine-readable
#
# See cleanup-stale-prs.py for full documentation.

set -euo pipefail

# Locate the python sibling relative to this script, regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/cleanup-stale-prs.py"

# Preflight: python3, gh, git must all be on PATH. We refuse to execute the
# python script if any of these are missing so git-push mishaps don't happen.
for tool in python3 gh git; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: '$tool' not on PATH; install it before running this script." >&2
    exit 1
  fi
done

exec python3 "$PY_SCRIPT" "$@"
