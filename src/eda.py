
import pandas as pd


def summarize_numeric(df: pd.DataFrame) -> dict:
    """Generate descriptive statistics for numeric columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {}

    summary = numeric_df.describe().T

    return {
        column: {
            stat: float(value)
            for stat, value in stats.items()
        }
        for column, stats in summary.to_dict(orient="index").items()
    }


def summarize_categorical(df: pd.DataFrame) -> dict:
    """Summarize categorical columns using value counts."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    )

    report = {}

    for column in categorical_df.columns:
        counts = categorical_df[column].value_counts(
            dropna=False
        )

        report[column] = {
            "unique_values": int(categorical_df[column].nunique()),
            "value_counts": {
                str(value): int(count)
                for value, count in counts.items()
            },
            "missing_count": int(categorical_df[column].isna().sum()),
        }

    return report


def calculate_correlations(df: pd.DataFrame) -> dict:
    """Calculate pairwise Pearson correlations between numeric columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return {}

    correlation_df = numeric_df.corr()

    return {
        column: {
            other_column: (
                float(value) if pd.notna(value) else None
            )
            for other_column, value in correlations.items()
        }
        for column, correlations in correlation_df.to_dict().items()
    }


def run_eda(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> dict:
    """Run the complete automated EDA workflow."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if target_column is not None and target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    report = {
        "dataset_overview": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
        },
        "numeric_summary": summarize_numeric(df),
        "categorical_summary": summarize_categorical(df),
        "correlations": calculate_correlations(df),
    }

    if target_column is not None:
        target = df[target_column]

        is_categorical = (
            pd.api.types.is_object_dtype(target.dtype)
            or isinstance(target.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(target.dtype)
            or pd.api.types.is_string_dtype(target.dtype)
        )

        if is_categorical:
            target_counts = target.value_counts(dropna=False)

            report["target_analysis"] = {
                "column": target_column,
                "type": "categorical",
                "value_counts": {
                    str(value): int(count)
                    for value, count in target_counts.items()
                },
            }
        else:
            report["target_analysis"] = {
                "column": target_column,
                "type": "numeric",
                "statistics": {
                    str(key): (
                        float(value) if pd.notna(value) else None
                    )
                    for key, value in target.describe().items()
                },
            }

    return report
