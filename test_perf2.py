import pandas as pd
import numpy as np
import time

# Create a dataframe with a string column
df = pd.DataFrame({
    'id': ['abc' + str(i) for i in range(100000)],
    'num': np.random.rand(100000)
})

t0 = time.time()
try:
    pd.to_numeric(df['id'].dropna())
except (ValueError, TypeError):
    pass
print("to_numeric with try:", time.time() - t0)

t0 = time.time()
if not pd.api.types.is_numeric_dtype(df['id']):
    pass
print("is_numeric_dtype:", time.time() - t0)
