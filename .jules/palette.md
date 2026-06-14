## 2024-05-18 - CLI standard output streams and error codes
**Learning:** Returning explicit exit codes (like `1` on failure instead of `0`) and grouping logically connected output messages to the same stream (`sys.stdout`) rather than scattering across `sys.stderr` are crucial micro-UX elements for shell scripting. Failing silently with exit code `0` causes script failures down the line.
**Action:** Always ensure CLI utilities return an error exit code when resources are not found or logical failures occur, and ensure multi-line lists keep their heading and body in the same standard stream.
