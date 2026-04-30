## 2026-04-27 - Unnecessary DataFrame Copies
**Learning:** In astronomical data processing with large datasets, deep copying pandas DataFrames `df.copy()` unconditionally before data validation or writing can cause massive memory and CPU overhead. The codebase had an instance in `src/xmatch/stilts.py` where a copy was made just to convert potential `object` columns to strings. Often these datasets are purely numeric and have no `object` columns.
**Action:** When working with `pandas` DataFrames, only deep copy when mutation is strictly required (e.g., after checking if `object` columns even exist). Check condition first, copy later.

## 2026-04-28 - Unnecessary DataFrame Copies in FITS Preparation
**Learning:** Astronomical catalogues are typically purely numeric and very large. Unconditionally copying a DataFrame (`df.copy()`) before checking if `object` column string conversions are needed creates a massive, unnecessary memory overhead and slows down data preparation significantly.
**Action:** Always check `df.select_dtypes(include=["object"]).columns` first. Only create a `.copy()` if string conversions are actually required; otherwise, pass the original DataFrame reference.

## 2026-04-29 - Avoid DataFrame copy for numeric datasets
**Learning:** Unconditionally copying a Pandas DataFrame (`df.copy()`) before checking if any column type conversion is needed (e.g., for object type columns) creates significant memory and execution overhead, especially since astronomical datasets are typically very large and purely numeric.
**Action:** Always check if a transformation (e.g., string conversion) is actually necessary before making a defensive copy of a DataFrame. For example, conditionally create `df.copy()` only when `df.select_dtypes(include=["object", "string"])` returns non-empty.

## 2026-04-30 - Shallow DataFrame Copies
**Learning:** For extremely large numeric datasets typical in astronomical data processing, deep copying Pandas DataFrames (`df.copy()`) before applying transformations or generating FITS files can cause a massive spike in memory footprint and latency. Because modifying columns in Pandas with a new Series object does not mutate the original data block underlying a shallow copy, deep copies are usually entirely unnecessary even when type conversions are performed on a column-by-column basis.
**Action:** When you just need to append new columns (like propagated coordinates) or cast specific columns to strings (like FITS table string conversion), unconditionally use shallow copying: `df.copy(deep=False)`. This shares all unmodified numeric arrays and preserves memory while allowing safe local modifications to the copy.
