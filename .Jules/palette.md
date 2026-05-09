## 2024-05-09 - CLI Error Messaging
**Learning:** Routing CLI application error messages to `sys.stderr` is a critical UX improvement that prevents error strings from breaking downstream processes when users pipe stdout to files or other tools.
**Action:** Always ensure terminal-facing applications separate user-facing errors (stderr) from pure data output (stdout).
