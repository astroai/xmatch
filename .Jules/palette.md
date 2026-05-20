## 2024-05-20 - Error Routing in CLI
**Learning:** This Python CLI app was printing error messages, suggestions, and warnings directly to `sys.stdout` via basic `print()` statements, which makes standard UNIX composition (e.g. `xmatch ... > out.csv`) break when errors occur or invalid options are passed. Proper error routing to `sys.stderr` is critical for command line tool UX.
**Action:** When adding or modifying error handling in the CLI components (`src/xmatch/cli.py`), always specify `file=sys.stderr` for all print statements that are not expected data output.
