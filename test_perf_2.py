import pandas as pd
import numpy as np
import time

# Create dummy data
np.random.seed(42)
n_chunks = 10
chunk_size = 500

chunks = []
for i in range(n_chunks):
    chunks.append(pd.DataFrame({
        'ra': np.random.uniform(0, 360, chunk_size),
        'dec': np.random.uniform(-90, 90, chunk_size),
        'mag1': np.random.uniform(15, 20, chunk_size),
    }))

def orch_orig(chunks):
    for local_chunk in chunks:
        for idx, row in local_chunk.iterrows():
            ra = row['ra']
            dec = row['dec']
            pass

def orch_opt(chunks):
    for local_chunk in chunks:
        in1_cols = {c: local_chunk[c].to_numpy() for c in local_chunk.columns}
        in1_len = len(local_chunk)
        for i in range(in1_len):
            ra = in1_cols['ra'][i]
            dec = in1_cols['dec'][i]
            pass

t0 = time.time()
orch_orig(chunks)
t1 = time.time()
print(f"Original: {t1-t0:.4f}s")

t0 = time.time()
orch_opt(chunks)
t1 = time.time()
print(f"Optimized: {t1-t0:.4f}s")
