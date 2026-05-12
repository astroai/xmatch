## 2024-05-12 - CLI Error Message Routing
**Learning:** This is a pure Python CLI app, not a web app. The most crucial 'UX' for CLI tools is ensuring standard output can be cleanly piped.
**Action:** Route error and informational 'not found' messages to sys.stderr so they don't break pipeline commands (e.g. `xmatch ... | grep ...`).
