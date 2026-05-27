
## 2024-05-27 - CLI Error Stream Routing
**Learning:** In command-line applications, routing error and warning messages to standard output (`stdout`) instead of standard error (`stderr`) creates a poor user experience. Users expecting to pipe valid output data to files or other tools end up with corrupted data streams if errors occur, and they may miss the errors entirely.
**Action:** Always route CLI errors, warnings, and fallback messages to `sys.stderr` to keep standard output pure for actual application data.
