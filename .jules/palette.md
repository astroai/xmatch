## 2024-05-30 - Routing Application Errors to Standard Error
**Learning:** Found that error messages in the CLI (e.g., missing catalog arguments) were being printed directly to stdout instead of stderr, which can break downstream pipelines where users expect only data to arrive on stdout. When improving CLI user experience and error handling, ensure that application error messages are explicitly routed to `sys.stderr`.
**Action:** Replace `print("Error: ...")` with `print("Error: ...", file=sys.stderr)` for CLI error states.
