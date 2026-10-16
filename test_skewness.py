import pandas as pd
from src.eda import analyze_skewness

df = pd.DataFrame({
    "symmetric": [1, 2, 3, 4, 5],
    "right_skewed": [1, 2, 2, 3, 20],
})

print(analyze_skewness(df))