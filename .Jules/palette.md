## 2026-05-04 - Route CLI errors to stderr
**Learning:** Application error messages and help outputs in the CLI shouldn't pollute `stdout`. Users often pipe the output of CLI commands into other processes or files. If error messages are printed to `stdout`, they can corrupt the resulting data stream or get ignored silently.
**Action:** Always route CLI error messages, warnings, and error-related help usage text to `sys.stderr` using `print(..., file=sys.stderr)` or equivalent logging configuration.
