import pandas as pd
import time
import numpy as np
from astropy.table import Table

df = pd.DataFrame({
    'ra': np.random.rand(1000000),
    'dec': np.random.rand(1000000),
    'id': ['abc' + str(i) for i in range(1000000)],
    'type': ['STAR' for i in range(1000000)],
})

def slow(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["object"]).columns:
        if not pd.api.types.is_numeric_dtype(df_copy[col].dropna()):
            df_copy[col] = df_copy[col].astype(str)

def fast(df):
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        for col in object_cols:
            col_data = df[col]
            if not pd.api.types.is_numeric_dtype(col_data):
                # We need to fillna if there are nans, but here we just cast
                df[col] = col_data.astype(str)

t0 = time.time()
df_copy1 = df.copy()
slow(df_copy1)
print("Slow save output:", time.time() - t0)

t0 = time.time()
df_copy2 = df.copy()
fast(df_copy2)
print("Fast save output:", time.time() - t0)
