## 2024-05-26 - Routing CLI errors to stderr
**Learning:** Python CLI applications should route error messages (e.g. "Error:", "Note:", "Did you mean:") to `sys.stderr` rather than standard output (`sys.stdout`) to allow proper shell pipe usage and better error tracking in shell scripts.
**Action:** Replace `print("Error: ...")` with `print("Error: ...", file=sys.stderr)` for error cases.
