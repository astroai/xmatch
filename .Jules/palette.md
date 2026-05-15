## 2024-05-19 - Redirecting CLI Error Messages to stderr
**Learning:** For a backend Python command-line tool, users expect error messages to go to `sys.stderr` so that `sys.stdout` can be safely piped or redirected.
**Action:** When printing error messages, explicitly set `file=sys.stderr` in the `print()` calls.
