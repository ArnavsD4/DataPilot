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


def check_numeric_ranges(
    df: pd.DataFrame,
    rules: dict[str, dict[str, float]],
) -> dict:
    """Flag numeric values outside configured minimum and maximum bounds."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot validate an empty dataset.")

    violations = {}

    for column, limits in rules.items():
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' must be numeric.")

        minimum = limits.get("min")
        maximum = limits.get("max")

        invalid_mask = pd.Series(False, index=df.index)

        if minimum is not None:
            invalid_mask |= df[column] < minimum

        if maximum is not None:
            invalid_mask |= df[column] > maximum

        violations[column] = {
            "invalid_count": int(invalid_mask.sum()),
            "invalid_indices": df.index[invalid_mask].tolist(),
        }

    return {"range_violations": violations}


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """Detect potential outliers in numeric columns using IQR."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot detect outliers in an empty dataset.")

    numerical_columns = df.select_dtypes(include="number").columns
    outliers = {}

    for column in numerical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        # IQR is not useful for binary or constant columns.
        if series.nunique() <= 2:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        )

        outliers[column] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(mask.sum()),
            "outlier_indices": df.index[mask].tolist(),
        }

    return {"outliers": outliers}