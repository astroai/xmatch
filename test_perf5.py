import pandas as pd
import numpy as np
import time

def fast_convert(df_copy):
    object_cols = df_copy.select_dtypes(include=["object", "string"]).columns
    if len(object_cols) > 0:
        for col in object_cols:
            col_data = df_copy[col]
            if not pd.api.types.is_numeric_dtype(col_data):
                try:
                    # try to see if it's all numeric strings
                    pd.to_numeric(col_data.dropna())
                except (ValueError, TypeError):
                    df_copy[col] = df_copy[col].fillna("").astype(str)

# large dataframe with string columns
df = pd.DataFrame({
    'ra': np.random.rand(1000000),
    'dec': np.random.rand(1000000),
    'id': ['abc' + str(i) for i in range(1000000)],
    'type': ['STAR' for i in range(1000000)],
    'class': ['O' for i in range(1000000)],
})

t0 = time.time()
df_copy1 = df.copy()
for col in df_copy1.select_dtypes(include=["object"]).columns:
    try:
        pd.to_numeric(df_copy1[col].dropna())
    except (ValueError, TypeError):
        df_copy1[col] = df_copy1[col].fillna("").astype(str)
print("Current approach:", time.time() - t0)

t0 = time.time()
df_copy2 = df.copy()
fast_convert(df_copy2)
print("Fast approach:", time.time() - t0)
