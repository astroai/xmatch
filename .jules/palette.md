## 2024-05-18 - Route application errors to sys.stderr
**Learning:** For a backend python tool CLI error messages, using `print()` routes it to standard output which isn't useful for users wanting to pipe the outputs or parsing standard output for non-error behavior.
**Action:** Always route application error messages to `sys.stderr` when improving CLI UX by modifying existing `print()` functions to `print(file=sys.stderr)`.
