import pandas as pd
import numpy as np
import time
from astropy.table import Table

df = pd.DataFrame({
    'ra': np.random.rand(1000000),
    'dec': np.random.rand(1000000),
    'id': np.arange(1000000)
})

def prepare_input_table_orig(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["object"]).columns:
        try:
            pd.to_numeric(df_copy[col].dropna())
        except (ValueError, TypeError):
            df_copy[col] = df_copy[col].fillna("").astype(str)

    return Table.from_pandas(df_copy)

def prepare_input_table_fast(df):
    # Only copy if we need to modify columns
    object_cols = df.select_dtypes(include=["object"]).columns

    if len(object_cols) == 0:
        return Table.from_pandas(df)

    df_copy = df.copy()
    for col in object_cols:
        col_data = df_copy[col].dropna()
        try:
            pd.to_numeric(col_data)
        except (ValueError, TypeError):
            df_copy[col] = df_copy[col].fillna("").astype(str)

    return Table.from_pandas(df_copy)

t0 = time.time()
prepare_input_table_orig(df)
print("Original:", time.time() - t0)

t0 = time.time()
prepare_input_table_fast(df)
print("Fast:", time.time() - t0)
