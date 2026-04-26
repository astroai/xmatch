import pandas as pd
import numpy as np
import time

def fast(df):
    # Only copy dataframe if it has object columns, this avoids copying the whole dataframe which is memory/time intensive
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        df_copy = df.copy()
        for col in object_cols:
            col_data = df_copy[col]
            try:
                pd.to_numeric(col_data.dropna())
            except (ValueError, TypeError):
                df_copy[col] = df_copy[col].fillna("").astype(str)
        return df_copy
    return df

def slow(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["object"]).columns:
        try:
            pd.to_numeric(df_copy[col].dropna())
        except (ValueError, TypeError):
            df_copy[col] = df_copy[col].fillna("").astype(str)
    return df_copy

df = pd.DataFrame({
    'ra': np.random.rand(10000000),
    'dec': np.random.rand(10000000),
    'id': np.arange(10000000)
})

t0 = time.time()
slow(df)
print("Slow purely numeric:", time.time() - t0)

t0 = time.time()
fast(df)
print("Fast purely numeric:", time.time() - t0)
