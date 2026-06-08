## 2026-06-08 - CLI error messages to stderr (LANDED — do not reopen)
**Learning:** CLI applications should write normal output to stdout and errors/diagnostics to stderr so users can pipe stdout (`xmatch ... > out.csv`) without mixing error text into the pipe.
**Action:** **Already merged on main** (PR #79, commit `8a37fd2`). Error paths in `cli.py` and `auth.py` use `print(..., file=sys.stderr)`. Do not open duplicate Palette PRs for stderr routing.
