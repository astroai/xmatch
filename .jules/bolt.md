## 2024-06-04 - Pandas iteration overhead
**Learning:** Pandas `.iterrows()` and `.iloc[]` lookups within loops are extremely slow for performance-critical matching processes (e.g. `skymatch` and chunked chunk generation) since `.iterrows()` yields Series objects with enormous overhead.
**Action:** Extract the DataFrame's columns into a dictionary of NumPy arrays (e.g. `{col: df[col].to_numpy() for col in df.columns}`) before the loop and perform standard positional indexing. This yields O(1) time complexity per access without massive memory allocation.
