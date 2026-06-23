## 2026-06-08 - CLI error messages to stderr (LANDED — do not reopen)
**Learning:** CLI applications should write normal output to stdout and errors/diagnostics to stderr so users can pipe stdout (`xmatch ... > out.csv`) without mixing error text into the pipe.
**Action:** **Already merged on main** (PR #79, commit `8a37fd2`). Error paths in `cli.py` and `auth.py` use `print(..., file=sys.stderr)`. Do not open duplicate Palette PRs for stderr routing.

## 2024-06-18 - CLI standard output routing and logical error exit codes are critical for shell script UX
**Learning:** When users chain CLI commands in shell scripts, they expect data arrays (like list output) and their headers to group in `stdout` to allow clean piping, while application errors and warnings should specifically route to `stderr`. Additionally, non-zero error codes must be issued when requested data is not found; silent `0` exits on "not found" cases break logic branching in bash files.
**Action:** In `src/xmatch/cli.py`, updated `list_catalogues` so its header outputs to `stdout` to maintain stream consistency. Additionally, updated `describe` logic to properly signal logical failures with boolean returns and subsequently output an error `1` exit code in `main()`.

## 2024-11-20 - CLI error UX - "Did you mean?" suggestions
**Learning:** For a much better user experience in CLI tools where typoes are common (e.g. catalogue names like `gaia` vs `gaja`), leveraging `difflib.get_close_matches` against the known configuration keys provides highly actionable error messages ("Did you mean: gaia?") instead of plain "Not found" errors.
**Action:** Implemented `difflib` suggestions into `xmatch.crossmatch.CrossMatch.get_catalogue_config()` and `xmatch.cli.describe()` so that users receive hints when they mistype an archive or catalogue name.
