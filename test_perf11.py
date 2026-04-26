import pandas as pd
import numpy as np
import time

def fast(df):
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        df_copy = df.copy()
        for col in object_cols:
            col_data = df_copy[col]
            # Avoid the slow pd.to_numeric if it's mostly strings, we can check if it's numeric first
            if not pd.api.types.is_numeric_dtype(col_data):
                try:
                    pd.to_numeric(col_data.dropna())
                except (ValueError, TypeError):
                    df_copy[col] = col_data.fillna("").astype(str)
        return df_copy
    return df

df = pd.DataFrame({
    'ra': np.random.rand(1000000),
    'dec': np.random.rand(1000000),
    'id': ['abc' + str(i) for i in range(1000000)],
    'type': ['STAR' for i in range(1000000)],
    'class': ['O' for i in range(1000000)],
})

t0 = time.time()
fast(df)
print("Fast mixed:", time.time() - t0)
