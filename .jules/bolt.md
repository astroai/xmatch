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

## 2026-06-08 - Pandas iterrows / iloc loop optimization (LANDED — do not reopen)
**Learning:** `DataFrame.iterrows()` and per-row `.iloc[]` in hot loops (`skymatch`, chunked orchestration) are very slow on astronomical catalog sizes. The fix is to pre-extract columns to `{col: ndarray}` once, then index by integer position in the loop.
**Action:** **Already merged on main** (PR #79, commit `8a37fd2`). Do not open new PRs for this pattern in `orchestration.py` / `stilts.py`.

## 2024-06-18 - NumPy Unique Vectorization Over Python Sets
**Learning:** When performing duplicate filtering in large array operations (such as finding the best astropy matches), using a native Python `for` loop with a `set` for lookup incurs significant overhead due to Python's variable tracking and dynamic typing.
**Action:** Prefer vectorized NumPy operations like `np.unique(array, return_index=True)` followed by `np.sort` on the indices. This pushes the loop down to the fast C backend, providing a substantial speedup for massive astronomical arrays.

## 2024-05-18 - Fast Boolean Array Construction from Polars Series Indices
**Learning:** In astronomical data processing with massive arrays, using `set(int(r) for r in series.to_list())` and list comprehensions to construct a boolean mask array introduces immense Python variable tracking overhead, turning an O(1) vectorized operation into a slow O(N) bottleneck.
**Action:** Always construct boolean arrays by instantiating empty arrays with `np.zeros(size, dtype=bool)` and use NumPy vectorized indexing (e.g. `keep[series.to_numpy()] = True`) to populate truthy values, yielding orders of magnitude speedups.

## 2024-05-18 - Vectorized Pandas Column Identification
**Learning:** `df.select_dtypes(include=['object'])` and `df['col'].iloc[0]` add surprisingly large overhead and trigger deprecation warnings in newer Pandas versions.
**Action:** Use fast boolean indexing over column attributes: `df.columns[df.dtypes == "object"]`, and `df[col].to_numpy()[0]` instead of `.iloc[0]` for quick element peeks.
