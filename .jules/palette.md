## 2024-05-25 - CLI Error Messaging Routing
**Learning:** For command-line tools, application error messages (such as missing catalogues, configuration failures, or missing credentials) should be routed to `sys.stderr` rather than standard output (`sys.stdout`). This ensures that error messages do not interfere with piped output, allowing users to safely chain commands.
**Action:** When printing application error messages, missing item alerts, or authentication failures in CLI scripts, use `print(..., file=sys.stderr)` to explicitly route them to standard error.
