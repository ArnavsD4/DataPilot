
import pandas as pd


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