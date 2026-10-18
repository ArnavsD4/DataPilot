
import pandas as pd

from src.cleaning import (
    clean_total_charges,
    suggest_cleaning_strategies,
    auto_clean_missing_values,
    remove_duplicate_rows,
    auto_clean_dataset,
)


def test_clean_total_charges_converts_strings_and_fills_zero_tenure():
    df = pd.DataFrame({
        "tenure": [0, 2, 5],
        "TotalCharges": [" ", "120.50", "300"],
    })

    cleaned = clean_total_charges(df)

    assert cleaned["TotalCharges"].tolist() == [0.0, 120.50, 300.0]
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_clean_total_charges_does_not_modify_original():
    df = pd.DataFrame({
        "tenure": [0],
        "TotalCharges": [" "],
    })

    clean_total_charges(df)

    assert df["TotalCharges"].iloc[0] == " "


def test_clean_total_charges_preserves_other_missing_values():
    df = pd.DataFrame({
        "tenure": [5],
        "TotalCharges": ["invalid"],
    })

    cleaned = clean_total_charges(df)

    assert pd.isna(cleaned["TotalCharges"].iloc[0])


def test_suggest_cleaning_strategies_for_numeric_and_categorical_columns():
    df = pd.DataFrame({
        "Age": [20, None, 30],
        "Salary": [40000, None, 60000],
        "City": ["Delhi", None, "Pune"],
        "Churn": ["No", "Yes", "No"],
    })

    suggestions = suggest_cleaning_strategies(df)

    assert suggestions["Age"]["strategy"] == "median"
    assert suggestions["Salary"]["strategy"] == "median"
    assert suggestions["City"]["strategy"] == "mode"
    assert "Churn" not in suggestions


def test_suggest_cleaning_strategies_reports_missing_counts():
    df = pd.DataFrame({
        "Age": [None, 30, None],
        "City": ["Delhi", None, "Pune"],
    })

    suggestions = suggest_cleaning_strategies(df)

    assert suggestions["Age"]["missing_count"] == 2
    assert suggestions["City"]["missing_count"] == 1


def test_suggest_cleaning_strategies_does_not_modify_original():
    df = pd.DataFrame({
        "Age": [20, None, 30],
        "City": ["Delhi", None, "Pune"],
    })

    original = df.copy(deep=True)

    suggest_cleaning_strategies(df)

    pd.testing.assert_frame_equal(df, original)


def test_suggest_cleaning_strategies_rejects_empty_dataframe():
    df = pd.DataFrame()

    try:
        suggest_cleaning_strategies(df)
    except ValueError as exc:
        assert str(exc) == "Cannot inspect an empty dataset."
    else:
        raise AssertionError("Expected ValueError for an empty DataFrame.")

def test_suggest_cleaning_strategies_flags_all_missing_columns():
    df = pd.DataFrame({
        "Age": [None, None, None],
        "City": [None, None, None],
    })

    suggestions = suggest_cleaning_strategies(df)

    assert suggestions["Age"]["strategy"] is None
    assert suggestions["City"]["strategy"] is None
    assert "review required" in suggestions["Age"]["reason"]


def test_auto_clean_missing_values_fills_missing_values():
    df = pd.DataFrame({
        "Age": [20, None, 30],
        "City": ["Delhi", None, "Pune"],
    })

    cleaned, report = auto_clean_missing_values(df)

    assert cleaned["Age"].tolist() == [20.0, 25.0, 30.0]
    assert cleaned["City"].isna().sum() == 0
    assert report["values_filled"] == {"Age": 1, "City": 1}


def test_auto_clean_missing_values_preserves_original():
    df = pd.DataFrame({"Age": [20, None, 30]})
    original = df.copy(deep=True)

    auto_clean_missing_values(df)

    pd.testing.assert_frame_equal(df, original)


def test_auto_clean_missing_values_reports_all_missing_columns():
    df = pd.DataFrame({"Age": [None, None, None]})

    cleaned, report = auto_clean_missing_values(df)

    assert cleaned["Age"].isna().sum() == 3
    assert report["unresolved_columns"] == ["Age"]
    assert report["values_filled"] == {}


def test_remove_duplicate_rows_removes_exact_duplicates():
    df = pd.DataFrame({
        "Age": [20, 30, 20],
        "City": ["Delhi", "Pune", "Delhi"],
    })

    cleaned, report = remove_duplicate_rows(df)

    assert len(cleaned) == 2
    assert report["duplicate_rows_found"] == 1
    assert report["duplicate_rows_removed"] == 1
    assert report["rows_before"] == 3
    assert report["rows_after"] == 2


def test_remove_duplicate_rows_preserves_original():
    df = pd.DataFrame({
        "Age": [20, 20],
        "City": ["Delhi", "Delhi"],
    })

    original = df.copy(deep=True)

    remove_duplicate_rows(df)

    pd.testing.assert_frame_equal(df, original)


def test_remove_duplicate_rows_keeps_unique_rows():
    df = pd.DataFrame({
        "Age": [20, 30],
        "City": ["Delhi", "Pune"],
    })

    cleaned, report = remove_duplicate_rows(df)

    assert len(cleaned) == 2
    assert report["duplicate_rows_removed"] == 0


def test_auto_clean_dataset_combines_cleaning_steps():
    df = pd.DataFrame({
        "Age": [20, None, 30, 20],
        "City": ["Delhi", "Pune", "Pune", "Delhi"],
    })

    cleaned, report = auto_clean_dataset(df)

    assert len(cleaned) == 3
    assert cleaned["Age"].isna().sum() == 0
    assert report["rows_before"] == 4
    assert report["rows_after"] == 3
    assert report["duplicates"]["duplicate_rows_removed"] == 1
    assert report["missing_values"]["values_filled"]["Age"] == 1



