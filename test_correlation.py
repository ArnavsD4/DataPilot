import pandas as pd
from src.eda import find_high_correlations

df = pd.DataFrame({
    "age": [10, 20, 30, 40, 50],
    "age_scaled": [100, 200, 300, 400, 500],
    "score": [50, 10, 40, 20, 30],
})

print(find_high_correlations(df))