import pandas as pd
import numpy as np
from astropy.table import Table

df = pd.DataFrame({
    'ra': np.random.rand(1000000),
    'dec': np.random.rand(1000000),
    'id': np.arange(1000000)
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
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        df_copy = df.copy()
        for col in object_cols:
            try:
                pd.to_numeric(df_copy[col].dropna())
            except (ValueError, TypeError):
                df_copy[col] = df_copy[col].fillna("").astype(str)
        return Table.from_pandas(df_copy)
    return Table.from_pandas(df)

t0 = time.time()
slow(df)
print("Slow:", time.time() - t0)

t0 = time.time()
fast(df)
print("Fast:", time.time() - t0)
