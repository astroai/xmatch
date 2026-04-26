import pandas as pd
import numpy as np
import time

# Create a dataframe with numeric columns only
df_num = pd.DataFrame({
    'id': np.arange(100000),
    'num': np.random.rand(100000)
})

# slow version calls select_dtypes which loops
t0 = time.time()
df_copy = df_num.copy()
for col in df_copy.select_dtypes(include=["object"]).columns:
    try:
        pd.to_numeric(df_copy[col].dropna())
    except (ValueError, TypeError):
        df_copy[col] = df_copy[col].fillna("").astype(str)
print("Numeric only with select_dtypes:", time.time() - t0)

# Create a dataframe with some string columns
df_mixed = pd.DataFrame({
    'id': ['abc' + str(i) for i in range(100000)],
    'num': np.random.rand(100000)
})

t0 = time.time()
df_copy = df_mixed.copy()
for col in df_copy.select_dtypes(include=["object"]).columns:
    try:
        pd.to_numeric(df_copy[col].dropna())
    except (ValueError, TypeError):
        df_copy[col] = df_copy[col].fillna("").astype(str)
print("Mixed with to_numeric (current):", time.time() - t0)

t0 = time.time()
df_copy = df_mixed.copy()
# avoid select_dtypes copying by getting columns first
object_cols = df_copy.select_dtypes(include=["object", "string"]).columns
if len(object_cols) > 0:
    for col in object_cols:
        col_data = df_copy[col].dropna()
        if not pd.api.types.is_numeric_dtype(col_data):
            # check if it can be converted to numeric
            try:
                # Attempt to convert to numeric without assigning back to catch strings that are actually numbers
                pd.to_numeric(col_data)
            except (ValueError, TypeError):
                df_copy[col] = df_copy[col].fillna("").astype(str)
print("Mixed with is_numeric_dtype & to_numeric:", time.time() - t0)
