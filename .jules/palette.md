## 2026-06-08 - CLI error messages to stderr (LANDED — do not reopen)
**Learning:** CLI applications should write normal output to stdout and errors/diagnostics to stderr so users can pipe stdout (`xmatch ... > out.csv`) without mixing error text into the pipe.
**Action:** **Already merged on main** (PR #79, commit `8a37fd2`). Error paths in `cli.py` and `auth.py` use `print(..., file=sys.stderr)`. Do not open duplicate Palette PRs for stderr routing.

## 2026-06-16 - Keep connected UI logic streams united
**Learning:** In CLI apps, mixing `stderr` and `stdout` for logically connected UI sequences (e.g. `stderr` for a list heading and `stdout` for the items) causes interleaving artifacts and buffering issues, leading to poor visual output and breaking tools that parse the standard output or standard error individually.
**Action:** Always maintain related visual content blocks (e.g. list headers and body items) inside the same stream, typically standard output.
