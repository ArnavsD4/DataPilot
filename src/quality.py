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

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot generate a report for an empty dataset.")
    missing_report = check_missing_values(df)
    duplicate_report = check_duplicate_rows(df)
    type_report = inspect_column_types(df)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": missing_report["missing_values"],
        "missing_percentage": missing_report["missing_percentage"],
        "duplicate_rows": duplicate_report["duplicate_rows"],
        "data_types": type_report["data_types"]
    }


def inspect_column_types(df: pd.DataFrame) -> dict:
    """Report the stored data type of each column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot inspect an empty dataset.")

    return {
        "data_types": df.dtypes.astype(str).to_dict()
    }
