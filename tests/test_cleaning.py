
import pandas as pd

from src.cleaning import clean_total_charges


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
