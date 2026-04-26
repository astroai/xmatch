import pandas as pd
import numpy as np
import time

def slow(df):
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            pd.to_numeric(df[col].dropna())
        except (ValueError, TypeError):
            df[col] = df[col].fillna("").astype(str)

def fast(df):
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        for col in object_cols:
            col_data = df[col]
            if not pd.api.types.is_numeric_dtype(col_data):
                try:
                    pd.to_numeric(col_data.dropna())
                except (ValueError, TypeError):
                    df[col] = df[col].fillna("").astype(str)

df = pd.DataFrame({
    'ra': np.random.rand(1000000),
    'dec': np.random.rand(1000000),
    'id': ['abc' + str(i) for i in range(1000000)],
    'type': ['STAR' for i in range(1000000)],
    'class': ['O' for i in range(1000000)],
})

t0 = time.time()
df_copy1 = df.copy()
slow(df_copy1)
print("Slow mixed:", time.time() - t0)

t0 = time.time()
df_copy2 = df.copy()
fast(df_copy2)
print("Fast mixed:", time.time() - t0)
