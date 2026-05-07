## 2024-05-15 - Improve CLI error message output
**Learning:** Application error messages should be explicitly routed to `sys.stderr` rather than standard output to improve CLI user experience and error handling, especially when outputs might be piped or parsed.
**Action:** Use `print(..., file=sys.stderr)` for CLI application error messages, as suggested by my memory directive.
