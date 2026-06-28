#!/usr/bin/env python3
"""scripts/cleanup-stale-prs.py

Closes and de-branches the stale ``bolt-*`` / ``palette-*`` / ``ux-*`` /
``jules-*`` / ``integrate-*`` / ``perf-*`` PRs authored by AI-agent flows
that were superseded by the v0.3 audit batch.

The cleanest invocation is::

    python3 scripts/cleanup-stale-prs.py            # dry-run, prints table
    python3 scripts/cleanup-stale-prs.py --apply   # destructive
    python3 scripts/cleanup-stale-prs.py --json     # machine-readable

The bash wrapper ``scripts/cleanup-stale-prs.sh`` simply exec's this script.

Rollback
--------
When ``--apply`` is used, two mktemp files are written next to the terminal
output:

* PR-list (each line: PR number) -- reopen with::

      while read N; do gh pr reopen "$N"; done < /tmp/cleanup-prs.XXXXXX

* branch-list (each line: ``<head> <sha>``) -- re-create with::

      while read line; do
        set -- $line; git push origin "$2:refs/heads/$1"
      done < /tmp/cleanup-branches.XXXXXX

Requirements
------------
* ``gh`` authenticated as a maintainer (``gh auth status``).
* ``git`` with push access to the upstream remote.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass

PREFIXES: tuple[str, ...] = ("bolt", "palette", "ux", "jules", "integrate", "perf")
PR_FIELDS: tuple[str, ...] = (
    "number",
    "title",
    "headRefName",
    "updatedAt",
    "author",
    "additions",
    "deletions",
)


@dataclass
class StalePR:
    number: int
    head: str
    title: str
    author: str
    additions: int
    deletions: int
    updated: str
    group: str

    @classmethod
    def from_gh(cls, payload: dict[str, object]) -> "StalePR | None":
        head = str(payload.get("headRefName") or "")
        # Branch names come in two flavours: ``bolt/vectorize-…`` (slash) and
        # ``bolt-vectorize-…`` (dash). Pick the first segment for matching.
        if "/" in head:
            first = head.split("/", 1)[0]
        else:
            first = head.split("-", 1)[0]
        if first not in PREFIXES:
            return None
        author_obj = payload.get("author") or {}
        author = author_obj.get("login", "?") if isinstance(author_obj, dict) else "?"
        return cls(
            number=int(payload["number"]),  # type: ignore[arg-type]
            head=head,
            title=str(payload.get("title") or "")[:80],
            author=str(author),
            additions=int(payload.get("additions") or 0),
            deletions=int(payload.get("deletions") or 0),
            updated=str(payload.get("updatedAt") or "")[:10],
            group=first,
        )


def fetch_open_prs() -> list[dict[str, object]]:
    """Return every open PR (≤200) on the configured GitHub repo."""
    if shutil.which("gh") is None:
        sys.exit("gh not on PATH; install from https://cli.github.com")
    result = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "200",
            "--json",
            ",".join(PR_FIELDS),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout or "[]")


def filter_stale(prs: list[dict[str, object]]) -> list[StalePR]:
    out: list[StalePR] = []
    for payload in prs:
        row = StalePR.from_gh(payload)
        if row is not None:
            out.append(row)
    out.sort(key=lambda r: r.number)
    return out


def render_table(rows: list[StalePR]) -> str:
    cols = ("PR", "head", "title", "author", "+add", "-del", "updated", "group")
    widths = (5, 33, 48, 14, 6, 6, 12, 10)
    head = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    body_lines = [
        "  ".join(
            [
                f"#{row.number}".ljust(widths[0]),
                (row.head[:31] + (".." if len(row.head) > 31 else "")).ljust(widths[1]),
                row.title[:46].ljust(widths[2]),
                ("@" + row.author).ljust(widths[3]),
                ("+" + str(row.additions)).ljust(widths[4]),
                ("-" + str(row.deletions)).ljust(widths[5]),
                row.updated.ljust(widths[6]),
                row.group.ljust(widths[7]),
            ]
        )
        for row in rows
    ]
    return "\n".join([head, sep, *body_lines])


def render_json(rows: list[StalePR]) -> str:
    return json.dumps([row.__dict__ for row in rows], indent=2)


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def close_and_debranch(rows: list[StalePR]) -> tuple[str, str]:
    """Close each PR (with --delete-branch) and capture rollback info."""
    pr_list = tempfile.NamedTemporaryFile(  # noqa: SIM115 - using context for clarity
        prefix="cleanup-prs-", suffix=".txt", mode="w", delete=False
    )
    br_list = tempfile.NamedTemporaryFile(
        prefix="cleanup-branches-", suffix=".txt", mode="w", delete=False
    )
    pr_list.close()
    br_list.close()

    for row in rows:
        print(f"  closing #{row.number} (head={row.head})", flush=True)
        subprocess.run(
            [
                "gh",
                "pr",
                "close",
                str(row.number),
                "--delete-branch",
                "--comment",
                "Closing as superseded by v0.3 audit batch (see AUDIT.md, "
                "CHANGELOG.md). Reopen if still relevant: gh pr reopen "
                f"{row.number}.",
            ],
            check=True,
        )
        with open(pr_list.name, "a") as fh:
            fh.write(f"{row.number}\n")

        # Capture the branch's last SHA so a future `git push <sha>:refs/heads/<head>`
        # can fully restore it.
        ls_remote = subprocess.run(
            ["git", "ls-remote", "--heads", "origin", row.head],
            capture_output=True,
            text=True,
        )
        sha = ""
        if ls_remote.returncode == 0 and ls_remote.stdout.strip():
            sha = ls_remote.stdout.split()[0]
        with open(br_list.name, "a") as fh:
            fh.write(f"{row.head} {sha}\n")

    return pr_list.name, br_list.name


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cleanup-stale-prs",
        description="Close/delete stale bolt-/palette-/ux-/jules-prefixed AI PRs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually close PRs + delete remote branches (default: dry-run only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output matching PRs as JSON instead of a tabular summary.",
    )
    args = parser.parse_args(argv)

    print("==> Fetching every open PR with a bolt/palette/ux/jules/integrate/perf prefix")
    all_prs = fetch_open_prs()
    rows = filter_stale(all_prs)

    if args.json:
        print(render_json(rows))
        return 0

    print()
    print(render_table(rows))
    print()
    print(f"==> Total matching stale open PRs: {len(rows)}")
    groups = Counter(r.group for r in rows)
    print("==> Group breakdown:")
    for g, c in sorted(groups.items()):
        print(f"    {g:<10} {c}")
    print()

    if not rows:
        return 0

    if not args.apply:
        print("==> Dry-run only. Re-run with --apply to actually close + delete branches.")
        return 0

    print("==> --apply: closing PRs (with --delete-branch) and recording rollback lists")
    pr_list, br_list = close_and_debranch(rows)
    print()
    print("==> Done. Rollback lists:")
    print(f"    PR list (re-open via `gh pr reopen`):      {pr_list}")
    print(f"    branch list (re-push via `git push <sha>:refs/heads/<head>`):")
    print(f"      {br_list}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
