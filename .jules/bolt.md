## 2026-04-27 - Unnecessary DataFrame Copies
**Learning:** In astronomical data processing with large datasets, deep copying pandas DataFrames `df.copy()` unconditionally before data validation or writing can cause massive memory and CPU overhead. The codebase had an instance in `src/xmatch/stilts.py` where a copy was made just to convert potential `object` columns to strings. Often these datasets are purely numeric and have no `object` columns.
**Action:** When working with `pandas` DataFrames, only deep copy when mutation is strictly required (e.g., after checking if `object` columns even exist). Check condition first, copy later.

## 2026-04-28 - Unnecessary DataFrame Copies in FITS Preparation
**Learning:** Astronomical catalogues are typically purely numeric and very large. Unconditionally copying a DataFrame (`df.copy()`) before checking if `object` column string conversions are needed creates a massive, unnecessary memory overhead and slows down data preparation significantly.
**Action:** Always check `df.select_dtypes(include=["object"]).columns` first. Only create a `.copy()` if string conversions are actually required; otherwise, pass the original DataFrame reference.

## 2026-04-29 - Avoid DataFrame copy for numeric datasets
**Learning:** Unconditionally copying a Pandas DataFrame (`df.copy()`) before checking if any column type conversion is needed (e.g., for object type columns) creates significant memory and execution overhead, especially since astronomical datasets are typically very large and purely numeric.
**Action:** Always check if a transformation (e.g., string conversion) is actually necessary before making a defensive copy of a DataFrame. For example, conditionally create `df.copy()` only when `df.select_dtypes(include=["object", "string"])` returns non-empty.

## 2024-05-24 - Pandas Shallow Copy Optimization
**Learning:** In pandas, `df.copy()` creates a deep copy by default, duplicating both the metadata and the underlying data arrays. In `xmatch`, which processes large astronomical datasets (DataFrames), making deep copies inside internal propagation and local matching functions incurs heavy memory and CPU overhead. Also, `dict.copy()` does not take any arguments, so applying the same optimization to dictionaries blindly causes `TypeError`.
**Action:** When working with pandas DataFrames, use `df.copy(deep=False)` when you only need to assign new columns (like `ra_propagated`) without modifying existing data. This creates a new object referencing the same data buffer, yielding identical functionality with significantly less overhead. Only apply this to DataFrames and not dictionaries.

## 2024-05-24 - Avoid `.iterrows()` for Large Astronomics Datasets
**Learning:** Found multiple instances of pandas `.iterrows()` and `.iloc[]` used inside tight loops over dataframe rows in `skymatch` and `_process_coordinate_chunk`. While iterating over a small chunk is fine, `.iterrows()` adds massive memory and CPU overhead for larger astronomical catalogs. In profiling, it's roughly 100x slower than direct NumPy access.
**Action:** Always extract columns into a dictionary of NumPy arrays (e.g. `{col: df[col].to_numpy() for col in df.columns}`) and iterate over the loop by index instead for fast O(1) row access and memory efficiency.
