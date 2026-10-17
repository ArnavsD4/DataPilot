
from pathlib import Path

import pandas as pd

from src.cleaning import clean_total_charges
from src.profiler import profile_dataset
from src.quality import generate_quality_report
from src.eda import run_eda



def run_pipeline(csv_path: str | Path) -> dict:
    """
    Load a CSV and generate profiling, quality,
    and cleaning reports.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Cannot run the pipeline on an empty dataset.")

    # Step 1: Analyze the original dataset.
    profile = profile_dataset(df)
    quality_before = generate_quality_report(df)

    # Step 2: Apply only explicitly defined cleaning rules.
    cleaned_df = df.copy()
    cleaning_actions = []

    if {"TotalCharges", "tenure"}.issubset(cleaned_df.columns):
        cleaned_df = clean_total_charges(cleaned_df)
        cleaning_actions.append("clean_total_charges")

    # Step 3: Check the cleaned dataset.
    quality_after = generate_quality_report(cleaned_df)

        # Step 4: Run automated EDA on the cleaned dataset.
    eda_report = run_eda(
        cleaned_df,
        output_path="reports/eda_report.json",
        output_dir="reports/plots",
    )


    return {
        "dataset": {
            "file_name": path.name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "profile": profile,
        "quality_before": quality_before,
        "quality_after": quality_after,
        "cleaning": {
            "actions": cleaning_actions,
            "rows_after_cleaning": int(cleaned_df.shape[0]),
            "columns_after_cleaning": int(cleaned_df.shape[1]),
        },
        "eda": eda_report,
    }
