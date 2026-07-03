import time
import numpy as np
import polars as pl

# Mock data
N = 10_000_000
np.random.seed(42)
keep_indices_raw = np.random.choice(N, size=N//2, replace=False)
filtered = pl.DataFrame({"_row_id": keep_indices_raw})
height = N

# Old approach
t0 = time.time()
keep_rows = set(int(r) for r in filtered["_row_id"].to_list())
keep_old = np.array([i in keep_rows for i in range(height)], dtype=bool)
t1 = time.time()
print(f"Old approach: {t1 - t0:.4f} seconds")

# New approach
t0 = time.time()
keep_indices = filtered["_row_id"].to_numpy()
keep_new = np.zeros(height, dtype=bool)
keep_new[keep_indices] = True
t1 = time.time()
print(f"New approach: {t1 - t0:.4f} seconds")

assert np.array_equal(keep_old, keep_new)
print("Results match!")
