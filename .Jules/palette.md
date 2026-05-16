## 2026-05-16 - [Route CLI Errors to Stderr]
**Learning:** CLI error messages outputted to standard output can break pipelining and create confusing experiences for users writing scripts. Errors should always go to `stderr`.
**Action:** Modified CLI error `print` statements in `src/xmatch/cli.py` to route to `sys.stderr`.
