
import pandas as pd

from src.eda import plot_histogram

# Load the real dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Convert TotalCharges from text to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Generate histogram
path = plot_histogram(df, "MonthlyCharges")

print("Histogram saved at:", path)
