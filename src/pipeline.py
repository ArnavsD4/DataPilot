
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

from src.cleaning import auto_clean_dataset
from src.profiler import profile_dataset
from src.quality import generate_quality_report
from src.eda import run_eda
from src.ml import prepare_features, compare_models



def run_pipeline(
    csv_path: str | Path,
    target_column: str | None = None,
) -> dict:
    """
    Load a CSV and generate profiling, quality,
    cleaning, EDA, and optional ML reports.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    try:
        df = pd.read_csv(path)
    except EmptyDataError as exc:
        raise ValueError(
            "Cannot run the pipeline on an empty dataset."
        ) from exc

    if df.empty:
        raise ValueError("Cannot run the pipeline on an empty dataset.")

    if target_column is not None:
        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError("target_column must be a non-empty string.")

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' does not exist."
            )

    # Step 1: Analyze the original dataset.
    profile = profile_dataset(df)
    quality_before = generate_quality_report(df)

    # Step 2: Automatically clean supported data-quality issues.
    cleaned_df, cleaning_report = auto_clean_dataset(df)

    # Step 3: Check the cleaned dataset.
    quality_after = generate_quality_report(cleaned_df)

    # Step 4: Run automated EDA on the cleaned dataset.
    eda_report = run_eda(
        cleaned_df,
        output_path="reports/eda_report.json",
        output_dir="reports/plots",
    )

    # Step 5: Run ML only when a target column is supplied.
    ml_report = None

    if target_column is not None:
        if cleaned_df[target_column].isna().any():
            raise ValueError(
                f"Target column '{target_column}' still contains "
                "missing values after cleaning. ML training cannot proceed."
            )

        X, y = prepare_features(cleaned_df, target_column)
        ml_report = compare_models(X, y)

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
            **cleaning_report,
            "rows_after_cleaning": int(cleaned_df.shape[0]),
            "columns_after_cleaning": int(cleaned_df.shape[1]),
        },
        "eda": eda_report,
        "ml": ml_report,
    }
