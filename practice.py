import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series([1,3,4,5,np.nan,7,8])
dates = pd.date_range('20260101', periods=10)
df = pd.DataFrame(np.random.randn(10,4), index=dates, columns=list('ABCD'))
# print(df)

print(df.values)
print(df.index)