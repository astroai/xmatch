import numpy as np
import time

def _first_occurrence_indices(sorted_array: np.ndarray) -> np.ndarray:
    if len(sorted_array) == 0:
        return np.array([], dtype=int)
    split_points = np.nonzero(sorted_array[1:] != sorted_array[:-1])[0] + 1
    return np.concatenate(([0], split_points))

# Simulate l_sorted_pix
N = 1000000
unique_vals = 10000
arr = np.sort(np.random.randint(0, unique_vals, size=N))

t0 = time.perf_counter()
res1_pix, res1_idx = np.unique(arr, return_index=True)
t1 = time.perf_counter()

t2 = time.perf_counter()
res2_idx = _first_occurrence_indices(arr)
res2_pix = arr[res2_idx]
t3 = time.perf_counter()

print(f"np.unique: {t1-t0:.6f}s")
print(f"_first_occurrence_indices: {t3-t2:.6f}s")
print(f"Speedup: {(t1-t0)/(t3-t2):.2f}x")
print(f"Equal pix: {np.array_equal(res1_pix, res2_pix)}")
print(f"Equal idx: {np.array_equal(res1_idx, res2_idx)}")
