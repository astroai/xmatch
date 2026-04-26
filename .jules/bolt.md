## 2024-05-19 - Avoid unnecessary DataFrame copying for numeric catalogs

**Learning:** Astronomical catalogs are typically huge and purely numeric. Calling `df.copy()` unnecessarily is a massive performance bottleneck, particularly inside data preparation methods like `_prepare_input_table` and `_save_output`. Operations like iterating over `select_dtypes(include=["object"])` can also trigger Pandas warnings if "string" is not explicitly included.

**Action:** Whenever operating on pandas DataFrames, evaluate whether column types can be checked *before* performing an expensive `.copy()` operation. For large datasets, always prefer inspecting columns with `df.select_dtypes(...)` on the original DataFrame to determine if a copy is strictly necessary before performing one.
