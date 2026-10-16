import pandas as pd

def check_missing_values(df: pd.DataFrame) -> dict:
    """Report missing values in each column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot check an empty dataset.")

    missing_counts = df.isnull().sum()
    missing_percentages = (df.isnull().mean() * 100).round(2)

    report = {
        "missing_values": missing_counts.to_dict(),
        "missing_percentage": missing_percentages.to_dict(),
    }

    return report


def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """Report duplicate rows in a dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot check an empty dataset.")

    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = round(
        (duplicate_count / len(df)) * 100, 2
    )

    return {
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
    }


def generate_quality_report(df: pd.DataFrame) -> dict:
    """Combine data-quality checks into one report."""

    missing_report = check_missing_values(df)
    duplicate_report = check_duplicate_rows(df)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        **missing_report,
        **duplicate_report,
    }
