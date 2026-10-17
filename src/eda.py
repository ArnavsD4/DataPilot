
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate a DataFrame before performing EDA."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")



def summarize_numeric(df: pd.DataFrame) -> dict:
    """Return descriptive statistics for numeric columns."""
    _validate_dataframe(df)

    numeric_df = df.select_dtypes(include="number")

    # No numeric columns means there is nothing to summarize.
    if numeric_df.empty or numeric_df.shape[1] == 0:
        return {}

    summary = numeric_df.describe().T
    report = {}

    for column, statistics in summary.iterrows():
        report[str(column)] = {
            str(statistic): (
                float(value) if pd.notna(value) else None
            )
            for statistic, value in statistics.items()
        }

    return report

def analyze_skewness(df: pd.DataFrame) -> dict:
    """Calculate skewness for each numeric column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    numeric_df = df.select_dtypes(include="number")
    results = {}

    for column in numeric_df.columns:
        values = numeric_df[column].dropna()

        if len(values) < 3 or values.nunique() < 2:
            skewness = None
            interpretation = "insufficient_variation"
        else:
            skewness = round(float(values.skew()), 4)

            if skewness > 0.5:
                interpretation = "right_skewed"
            elif skewness < -0.5:
                interpretation = "left_skewed"
            else:
                interpretation = "approximately_symmetric"

        results[column] = {
            "skewness": skewness,
            "interpretation": interpretation,
        }

    return results



def summarize_categorical(df: pd.DataFrame) -> dict:
    """Summarize categorical columns, including missing values."""
    _validate_dataframe(df)

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    report = {}

    for column in categorical_columns:
        counts = df[column].value_counts(dropna=False)

        report[column] = {
            "unique_values": int(df[column].nunique(dropna=True)),
            "missing_count": int(df[column].isna().sum()),
            "value_counts": {
                "nan" if pd.isna(value) else str(value): int(count)
                for value, count in counts.items()
            },
        }

    return report


def calculate_correlations(df: pd.DataFrame) -> dict:
    """Calculate Pearson correlations between numeric columns."""
    _validate_dataframe(df)

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return {}

    correlation_df = numeric_df.corr()

    report = {}

    for column, values in correlation_df.iterrows():
        report[column] = {
            str(other_column): (
                float(value) if pd.notna(value) else None
            )
            for other_column, value in values.items()
        }

    return report


def analyze_target(df: pd.DataFrame, target_column: str) -> dict:
    """Summarize a target column, whether categorical or numeric."""
    _validate_dataframe(df)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    target = df[target_column]

    if (
        pd.api.types.is_object_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(target)
        or pd.api.types.is_bool_dtype(target)
    ):
        counts = target.value_counts(dropna=False)

        return {
            "column": target_column,
            "type": "categorical",
            "value_counts": {
                "nan" if pd.isna(value) else str(value): int(count)
                for value, count in counts.items()
            },
        }

    statistics = target.describe()

    return {
        "column": target_column,
        "type": "numeric",
        "statistics": {
            str(key): float(value) if pd.notna(value) else None
            for key, value in statistics.items()
        },
    }


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    output_dir: str = "reports/plots",
) -> str:
    """Generate and save a histogram for one numeric column."""
    _validate_dataframe(df)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric.")

    values = df[column].dropna()

    if values.empty:
        raise ValueError(
            f"Column '{column}' contains no numeric values."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Avoid unsafe path characters in generated filenames.
    safe_column = "".join(
        character if character.isalnum() or character in "-_"
        else "_"
        for character in str(column)
    )
    file_path = output_path / f"{safe_column}_histogram.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        ax.hist(values, bins=30, edgecolor="black")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
    finally:
        plt.close(fig)

    return str(file_path)


def plot_numeric_histograms(
    df: pd.DataFrame,
    output_dir: str = "reports/plots",
) -> dict:
    """Generate a histogram for every numeric column."""
    _validate_dataframe(df)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    plots = {}

    for column in numeric_columns:
        # Skip columns with no usable numeric observations.
        if df[column].notna().sum() == 0:
            continue

        plots[column] = plot_histogram(
            df,
            column,
            output_dir=output_dir,
        )

    return plots

def find_high_correlations(
    df: pd.DataFrame,
    threshold: float = 0.85,
) -> list[dict]:
    """Find numeric feature pairs with strong absolute correlation."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1.")

    numeric_df = df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr()

    high_correlations = []

    for i, column_a in enumerate(correlation_matrix.columns):
        for j in range(i + 1, len(correlation_matrix.columns)):
            column_b = correlation_matrix.columns[j]
            correlation = correlation_matrix.iloc[i, j]

            if pd.notna(correlation) and abs(correlation) >= threshold:
                high_correlations.append({
                    "feature_1": column_a,
                    "feature_2": column_b,
                    "correlation": round(float(correlation), 4),
                })

    return high_correlations


def run_eda(
    df: pd.DataFrame,
    target_column: str | None = None,
    output_path: str = "reports/eda_report.json",
    output_dir: str = "reports/plots",
) -> dict:
    """Run EDA and save a JSON report plus numeric histograms."""
    _validate_dataframe(df)

    numeric_summary = summarize_numeric(df)
    categorical_summary = summarize_categorical(df)
    correlations = calculate_correlations(df)

    report = {
        "dataset_overview": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": [str(column) for column in df.columns],
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": {
                str(column): int(count)
                for column, count in df.isna().sum().items()
            },
        },
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "target_analysis": None,
        "visualizations": {},
    }

    if target_column is not None:
        report["target_analysis"] = analyze_target(
            df, target_column
        )

    report["visualizations"] = plot_numeric_histograms(
        df,
        output_dir=output_dir,
    )

    report_file = Path(output_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    report["skewness_analysis"] = analyze_skewness(df)
    report["high_correlations"] = find_high_correlations(df)

    with report_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return report
