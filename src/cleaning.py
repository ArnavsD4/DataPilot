
import pandas as pd
from src.quality import check_duplicate_rows
from src.quality import detect_outliers_iqr


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TotalCharges to numeric values.

    Blank charges for customers with tenure == 0 are set to 0.
    The input DataFrame is not modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    required_columns = {"TotalCharges", "tenure"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    cleaned = df.copy()

    # Convert blank strings to missing values, then parse numbers.
    charges = pd.to_numeric(
        cleaned["TotalCharges"].astype("string").str.strip().replace("", pd.NA),
        errors="coerce",
    )

    # Apply the domain-specific rule only to zero-tenure customers.
    zero_tenure = cleaned["tenure"].eq(0)
    charges.loc[zero_tenure & charges.isna()] = 0.0

    cleaned["TotalCharges"] = charges

    return cleaned

def handle_missing_values(
    df: pd.DataFrame,
    strategies: dict,
) -> pd.DataFrame:
    """Fill missing values using column-specific strategies."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    cleaned = df.copy()

    for column, strategy in strategies.items():
        if column not in cleaned.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        if strategy == "median":
            if not pd.api.types.is_numeric_dtype(cleaned[column]):
                raise TypeError(f"Column '{column}' must be numeric.")

            cleaned[column] = cleaned[column].fillna(
                cleaned[column].median()
            )

        elif strategy == "mean":
            if not pd.api.types.is_numeric_dtype(cleaned[column]):
                raise TypeError(f"Column '{column}' must be numeric.")

            cleaned[column] = cleaned[column].fillna(
                cleaned[column].mean()
            )

        elif strategy == "mode":
            modes = cleaned[column].mode(dropna=True)

            if not modes.empty:
                cleaned[column] = cleaned[column].fillna(modes.iloc[0])

        elif isinstance(strategy, dict) and "value" in strategy:
            cleaned[column] = cleaned[column].fillna(strategy["value"])

        else:
            raise ValueError(
                f"Unsupported strategy for column '{column}': {strategy}"
            )

    return cleaned



def suggest_cleaning_strategies(
    df: pd.DataFrame,
) -> dict:
    """
    Suggest strategies for columns containing missing values.

    Columns containing only missing values are flagged for review.
    The input DataFrame is not modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot inspect an empty dataset.")

    suggestions = {}

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count == 0:
            continue

        # No valid values are available for imputation.
        if df[column].isna().all():
            suggestions[column] = {
                "strategy": None,
                "missing_count": missing_count,
                "reason": "Column contains only missing values; review required.",
            }
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            suggestions[column] = {
                "strategy": "median",
                "missing_count": missing_count,
                "reason": "Numerical column with missing values.",
            }
        else:
            suggestions[column] = {
                "strategy": "mode",
                "missing_count": missing_count,
                "reason": "Non-numerical column with missing values.",
            }

    return suggestions


def auto_clean_missing_values(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Apply suggested missing-value strategies."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    suggestions = suggest_cleaning_strategies(df)
    cleaned = df.copy(deep=True)

    report = {
        "columns_processed": [],
        "unresolved_columns": [],
        "values_filled": {},
    }

    for column, suggestion in suggestions.items():
        strategy = suggestion["strategy"]

        if strategy is None:
            report["unresolved_columns"].append(column)
            continue

        missing_before = int(cleaned[column].isna().sum())

        if strategy == "median":
            fill_value = cleaned[column].median()

        elif strategy == "mode":
            modes = cleaned[column].mode(dropna=True)

            if modes.empty:
                report["unresolved_columns"].append(column)
                continue

            fill_value = modes.iloc[0]

        else:
            report["unresolved_columns"].append(column)
            continue

        if pd.isna(fill_value):
            report["unresolved_columns"].append(column)
            continue

        cleaned[column] = cleaned[column].fillna(fill_value)

        filled_count = (
            missing_before - int(cleaned[column].isna().sum())
        )

        report["columns_processed"].append(column)
        report["values_filled"][column] = filled_count

    return cleaned, report


def remove_duplicate_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove exact duplicate rows from a copy of the dataset.

    Keeps the first occurrence of each row and reports
    how many rows were removed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    duplicate_count = int(df.duplicated().sum())

    cleaned = df.drop_duplicates(keep="first").copy()

    report = {
        "duplicate_rows_found": duplicate_count,
        "duplicate_rows_removed": int(len(df) - len(cleaned)),
        "rows_before": int(len(df)),
        "rows_after": int(len(cleaned)),
    }

    return cleaned, report



def auto_clean_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Clean supported data-quality issues and return an audit report."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    cleaned = df.copy(deep=True)

    report = {
        "rows_before": int(len(cleaned)),
        "actions": [],
        "unresolved_columns": [],
    }

    # Step 1: Apply the known TotalCharges cleaning rule.
    if {"TotalCharges", "tenure"}.issubset(cleaned.columns):
        cleaned = clean_total_charges(cleaned)
        report["actions"].append({
            "action": "clean_total_charges",
            "description": (
                "Converted TotalCharges to numeric and handled "
                "blank charges for zero-tenure customers."
            ),
        })

    # Step 2: Handle missing values.
    cleaned, missing_report = auto_clean_missing_values(cleaned)
    report["missing_values"] = missing_report
    report["unresolved_columns"].extend(
        missing_report["unresolved_columns"]
    )

    # Step 3: Remove exact duplicate rows.
    cleaned, duplicate_report = remove_duplicate_rows(cleaned)
    report["duplicates"] = duplicate_report

    if duplicate_report["duplicate_rows_removed"] > 0:
        report["actions"].append({
            "action": "remove_duplicate_rows",
            "description": (
                "Removed exact duplicate rows, keeping the first occurrence."
            ),
        })

    # Step 4: Detect outliers without deleting them.
    outlier_report = detect_outliers_iqr(cleaned)
    report["outliers"] = outlier_report["outliers"]

    # Step 5: Record final dimensions.
    report["rows_after"] = int(len(cleaned))
    report["columns_after"] = int(len(cleaned.columns))

    return cleaned, report
