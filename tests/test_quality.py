
import pandas as pd
import pytest

from src.quality import check_missing_values
from src.quality import check_duplicate_rows
from src.quality import generate_quality_report
from src.quality import inspect_column_types


def test_check_missing_values():
    df = pd.DataFrame({
        "age": [20, None, 25],
        "city": ["Delhi", "Mumbai", None],
    })

    report = check_missing_values(df)

    assert report["missing_values"]["age"] == 1
    assert report["missing_values"]["city"] == 1

    assert report["missing_percentage"]["age"] == 33.33
    assert report["missing_percentage"]["city"] == 33.33


def test_rejects_empty_dataframe():
    df = pd.DataFrame(columns=["age", "city"])

    with pytest.raises(ValueError):
        check_missing_values(df)


def test_rejects_invalid_input():
    with pytest.raises(TypeError):
        check_missing_values([1, 2, 3])


def test_check_duplicate_rows():
    df = pd.DataFrame({
        "age": [20, 30, 20],
        "city": ["Delhi", "Mumbai", "Delhi"],
    })

    report = check_duplicate_rows(df)

    assert report["duplicate_rows"] == 1
    assert report["duplicate_percentage"] == 33.33


def test_generate_quality_report():
    df = pd.DataFrame({
        "age": [20, None, 20],
        "city": ["Delhi", "Mumbai", "Delhi"],
    })

    report = generate_quality_report(df)

    assert report["rows"] == 3
    assert report["columns"] == 2
    assert report["missing_values"]["age"] == 1
    assert report["missing_values"]["city"] == 0
    assert report["duplicate_rows"] == 1


def test_inspect_column_types():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "city": ["Delhi", "Mumbai", "Kolkata"],
    })

    report = inspect_column_types(df)

    assert report["data_types"]["age"] == str(df["age"].dtype)
    assert report["data_types"]["city"] == str(df["city"].dtype)


def test_quality_report_includes_data_types():
    df = pd.DataFrame({
        "age": [20, 30],
        "city": ["Delhi", "Mumbai"],
    })

    report = generate_quality_report(df)

    
    assert "data_types" in report
    assert report["data_types"]["age"] == str(df["age"].dtype)
    assert report["data_types"]["city"] == str(df["city"].dtype)

