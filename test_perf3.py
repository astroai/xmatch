import pandas as pd
import numpy as np
from astropy.table import Table

df = pd.DataFrame({
    'ra': np.random.rand(1000000),
    'dec': np.random.rand(1000000),
    'id': ['abc' + str(i) for i in range(1000000)]
})

import time

def slow(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["object"]).columns:
        try:
            pd.to_numeric(df_copy[col].dropna())
        except (ValueError, TypeError):
            df_copy[col] = df_copy[col].fillna("").astype(str)
    return Table.from_pandas(df_copy)

def fast(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["object"]).columns:
        if not pd.api.types.is_numeric_dtype(df_copy[col].dropna()):
            df_copy[col] = df_copy[col].fillna("").astype(str)
    return Table.from_pandas(df_copy)

t0 = time.time()
slow(df)
print("Slow:", time.time() - t0)

t0 = time.time()
fast(df)
print("Fast:", time.time() - t0)
