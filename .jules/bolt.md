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

## 2026-07-07 - Vectorized boolean mask from Polars row indices (LANDED)
**Learning:** Building `keep = np.array([i in keep_rows for i in range(n)])` from a Python `set` of row IDs is O(n) with heavy Python overhead on large match tables.
**Action:** Use `keep = np.zeros(n, dtype=bool); keep[filtered["_row_id"].to_numpy()] = True` in `_apply_match_filter`.

## 2026-07-07 - Fast pandas object-column filtering (LANDED)
**Learning:** `select_dtypes(include="object")` and `.iloc[0]` add overhead and trigger pandas deprecation warnings on wide catalog frames.
**Action:** Use `pdf.columns[pdf.dtypes == "object"]` and `pdf[col].to_numpy()[0]` for bytes-decode gating in `_decode_pandas_bytes_columns`.

## 2026-07-08 - Optimize pseudo-label generation (LANDED)
**Learning:** A Python loop over `np.unique(left_idx)` with per-group boolean masks and `np.argmin` is O(N × unique_left) and dominates ML feature prep on large match tables.
**Action:** Use `np.lexsort((X[:, 0], left_idx))` then `np.unique(..., return_index=True)` on the sorted left indices to pick the minimum-separation pair per primary in O(N log N).

## 2024-06-25 - Lexsort vectorization for boolean masks
**Learning:** In astronomical data processing with large datasets, finding the best or nearest candidate per primary source (e.g. `_pick_best_per_primary`, `_ml_fallback_best_by_sep`) using a `for` loop over unique sources to create boolean masks (`inverse == idx`) and apply `np.argmax`/`np.argmin` incurs an O(N^2) overhead, significantly bottlenecking performance.
**Action:** Replace `O(N^2)` boolean mask loops with `np.lexsort` to sort by group identifier and target value, followed by `np.unique(..., return_index=True)`. Crucially, to maintain identical row order behavior as the original boolean mask, explicitly sort the final `best_positions` array.

## 2026-07-19 - Unnecessary DataFrame Copies in Torchsky Match
**Learning:** Polars `to_numpy()` without `copy` returns a read-only numpy array. When passing DataFrame columns to external matching libraries like `torchsky` that do not mutate the input array, calling `.to_numpy().copy()` explicitly performs a deep copy which incurs massive unnecessary CPU and memory overhead for large astronomical datasets.
**Action:** Avoid calling `.copy()` on numpy arrays obtained from Polars or Pandas when the consumer does not mutate the array. Use the read-only output natively.

## 2026-07-29 - Unnecessary Deep Copies from astype()
**Learning:** When converting data types on NumPy arrays (e.g. `.astype(float)`), NumPy natively allocates and returns a *new, writeable array copy*. Chaining `.copy()` after `.astype(float)` as in `.astype(float).copy()` causes a completely redundant deep copy, resulting in massive unnecessary CPU and memory overhead when processing large astronomical datasets.
**Action:** Do not chain `.copy()` on the output of `.astype()`. Use the natively returned, writeable copy directly.

## 2026-08-05 - Avoid O(N) identity matrix copies in valid-only eigvalsh
**Learning:** When performing row-wise eigenvalue calculations on massive stacked matrices via `np.linalg.eigvalsh`, deep copying the entire N x 5 x 5 block and filling invalid rows with identity matrices causes enormous memory allocations and wastes CPU time solving trivial matrices. NumPy boolean masking natively copies only the required valid elements for calculation.
**Action:** Instead of `safe = cov.copy(); safe[~valid] = eye; res = eigvalsh(safe)`, use `res = eigvalsh(cov[valid])` and update the condition array inplace. This eliminates `O(N)` copies and irrelevant eigendecompositions.

## 2026-08-01 - Lexsort vectorization for astropy boolean masks
**Learning:** Building on the lexsort vectorization pattern learned earlier, when picking the best match by Mahalanobis distance `d2` or fallback `score` in `_astropy_match`, using `np.lexsort((value, group_id))` combined with `np.unique` replaces the slow `np.argsort` approach. The previous `argsort` approach did not guarantee the group ID was the primary sorting key, causing issues.
**Action:** Use `np.lexsort((d2, left_idx))` and `np.unique(..., return_index=True)` in `_astropy_match` for `skyellipse` match and error-based fallback. Ensure to call `sel.sort()` to preserve original array row order implicitly done by boolean masking.
## 2023-10-27 - Fast first-occurrence indices in NumPy
**Learning:** `np.unique(..., return_index=True)` incurs massive overhead (up to 6-7x slower) even when the input array is already guaranteed to be sorted (e.g. immediately after `np.lexsort`). This is because it lacks a fast-path for pre-sorted inputs in standard numpy and performs a redundant sort internally.
**Action:** When filtering to the best matching pairs using a sorted array (e.g. `order = np.lexsort((scores, group_ids))`, `sorted_groups = group_ids[order]`), replace `np.unique(sorted_groups, return_index=True)` with a manual diff mask: `split_points = np.nonzero(sorted_groups[1:] != sorted_groups[:-1])[0] + 1` prepended with `[0]`. This correctly identifies the first index of each group boundary ~6x faster on large astronomical data arrays.

## 2026-08-10 - Avoid redundant deep copies after astype conversion
**Learning:** When pulling data out of a Polars DataFrame as a NumPy array (e.g. `df["col"].to_numpy().astype(float)`), `.astype()` natively returns a new writeable array if the data isn't natively float. Chaining `.copy()` or doing `new_arr = arr.copy()` immediately afterwards causes a redundant deep copy of the arrays, consuming twice the required memory and significant CPU cycles over large catalogs.
**Action:** Do not chain `.copy()` on the output of `.astype()`. Rely on the writeable reference it returns, and use raw `.to_numpy()` zero-copy reads from the DataFrame when referencing the pre-mutation state of an array to calculate deltas.

## 2026-08-08 - O(N log N) vectorized grouping instead of native loop
**Learning:** When grouping large arrays (like HEALPix indices) into dictionaries by unique values, using a native Python loop with `setdefault` or iterating over unique values with `np.where(arr == k)` results in extremely slow O(N) or O(N*K) performance due to Python overhead or repeated full-array boolean scans.
**Action:** Replace slow grouping loops with O(N log N) vectorized sorting and splitting: `sort_idx = np.argsort(arr)`, then `unique, start_idx = np.unique(arr[sort_idx], return_index=True)`, followed by `splits = np.split(sort_idx, start_idx[1:])` and a quick dictionary comprehension `dict(zip(unique, splits))`.

## 2026-08-20 - Fast unique values extraction on sorted arrays
**Learning:** `np.unique(..., return_index=True)` incurs a massive `O(N log N)` sorting overhead even when passed an array that was *already* sorted on the immediately preceding line (e.g. `l_sorted_pix`). This causes significant unnecessary CPU time consumption in tight inner loops like HEALPix pixel batching.
**Action:** When working with pre-sorted arrays where both the unique values and their first occurrence indices are needed (e.g., for `np.split`), replace `np.unique` with the manual linear diff-mask function `_first_occurrence_indices(sorted_array)` to get the indices, then extract the unique values via simple indexing: `unique_vals = sorted_array[indices]`. This drops the complexity to O(N) and provides an ~8-12x speedup on this operation.

## 2024-08-25 - Replace np.unique + loop + boolean mask with Polars partition_by
**Learning:** When grouping large NumPy arrays (e.g., HEALPix indices) into dictionaries by unique values and extracting subsets of a DataFrame, using native Python loops with `np.unique` combined with sequential boolean masks (`sub.filter(pl.Series("_mask", child == cpix))`) causes massive O(N*K) performance overhead. In a Polars context, this is extremely inefficient.
**Action:** Use Polars' native O(N) vectorized approach: `df.with_columns(pl.Series('_k', arr)).partition_by('_k', as_dict=True)` to retrieve group subsets as a dictionary. This single optimization reduced execution time from ~1.37s to ~0.09s for large (1M+) arrays.
