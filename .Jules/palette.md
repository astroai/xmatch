## 2026-05-11 - Route error messages to sys.stderr
**Learning:** For CLI tools, printing error messages to standard output breaks the user experience when piping results or redirecting output to files. Users expect the main output (like matched datasets) on stdout and errors/warnings on stderr.
**Action:** Always verify that application error messages in CLI tools are routed to `sys.stderr` using `print(..., file=sys.stderr)` instead of standard `print()`.
