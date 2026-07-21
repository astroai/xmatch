## 2026-06-08 - CLI error messages to stderr (LANDED — do not reopen)
**Learning:** CLI applications should write normal output to stdout and errors/diagnostics to stderr so users can pipe stdout (`xmatch ... > out.csv`) without mixing error text into the pipe.
**Action:** **Already merged on main** (PR #79, commit `8a37fd2`). Error paths in `cli.py` and `auth.py` use `print(..., file=sys.stderr)`. Do not open duplicate Palette PRs for stderr routing.

## 2024-06-18 - CLI standard output routing and logical error exit codes are critical for shell script UX
**Learning:** When users chain CLI commands in shell scripts, they expect data arrays (like list output) and their headers to group in `stdout` to allow clean piping, while application errors and warnings should specifically route to `stderr`. Additionally, non-zero error codes must be issued when requested data is not found; silent `0` exits on "not found" cases break logic branching in bash files.
**Action:** In `src/xmatch/cli.py`, updated `list_catalogues` so its header outputs to `stdout` to maintain stream consistency. Additionally, updated `describe` logic to properly signal logical failures with boolean returns and subsequently output an error `1` exit code in `main()`.

## 2026-07-07 - Endpoint typo suggestions on discover (LANDED)
**Learning:** Unknown endpoint errors should suggest close matches via `difflib.get_close_matches` on a lowercased candidate pool, then map back to original casing for display.
**Action:** Added `_suggest_endpoint` and wired it into `handle_discover` when `_resolve_discovery_endpoint` fails.

## 2024-07-21 - Conditional display of CLI typo suggestions
**Learning:** When using `difflib.get_close_matches` to suggest alternatives for unknown CLI inputs, the returned suggestion string might be empty. Blindly concatenating this string into error messages or printing it as a hint leads to dangling whitespace or blank lines in the terminal, harming the perceived polish of the CLI UX.
**Action:** Always conditionally check if a generated suggestion (e.g., from `_suggest()`) is non-empty before interpolating it into error logs or printing it as a hint.
