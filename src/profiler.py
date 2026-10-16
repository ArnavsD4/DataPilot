import pandas as pd

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate basic structural information about a dataset
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot profile an empty dataset.")

    numerical_columns = df.select_dtypes(
    include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
    ).columns.tolist()

    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (
            df.isnull().mean().mul(100).round(2).to_dict()
        ),
        "unique_values": df.nunique().to_dict(),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "numerical_summary": df[numerical_columns].describe().T.to_dict(orient = "index"),
        "duplicate_rows": int(df.duplicated().sum())
    }

    return report