import pandas as pd
import pytest

from src.cleaning import handle_missing_values


def test_median_fills_numeric_missing_values():
    df = pd.DataFrame({"age": [20, None, 40]})

    cleaned = handle_missing_values(df, {"age": "median"})

    assert cleaned["age"].isna().sum() == 0
    assert cleaned["age"].iloc[1] == 30


def test_mode_fills_categorical_missing_values():
    df = pd.DataFrame({"city": ["Delhi", None, "Delhi", "Mumbai"]})

    cleaned = handle_missing_values(df, {"city": "mode"})

    assert cleaned["city"].isna().sum() == 0
    assert cleaned["city"].iloc[1] == "Delhi"


def test_custom_value_fills_missing_values():
    df = pd.DataFrame({"city": ["Delhi", None]})

    cleaned = handle_missing_values(
        df,
        {"city": {"value": "Unknown"}},
    )

    assert cleaned["city"].iloc[1] == "Unknown"


def test_original_dataframe_is_unchanged():
    df = pd.DataFrame({"age": [20, None, 40]})

    handle_missing_values(df, {"age": "median"})

    assert df["age"].isna().sum() == 1


def test_missing_column_is_rejected():
    df = pd.DataFrame({"age": [20, 30]})

    with pytest.raises(ValueError, match="does not exist"):
        handle_missing_values(df, {"salary": "median"})