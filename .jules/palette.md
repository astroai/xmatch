## 2024-05-14 - Error Stream Routing
**Learning:** Application error messages routing to stdout can break pipelines and confuse users who expect standard error patterns.
**Action:** Always ensure CLI application error messages are explicitly routed to sys.stderr.
