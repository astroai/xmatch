## 2026-06-02 - Route Application Error Messages to sys.stderr
**Learning:** In CLI tools without UI components, users expect standard UNIX behaviors, which includes outputting error messages and warnings directly to standard error (`sys.stderr`) rather than standard out. Failing to do so makes programmatic parsing and pipelining output difficult.
**Action:** When printing application error messages to console in python applications, ensure you route the output to stderr (`print("Error...", file=sys.stderr)`).
