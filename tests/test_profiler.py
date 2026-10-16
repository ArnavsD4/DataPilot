import json
import pandas as pd
import pytest

from src.profiler import profile_dataset


def test_profile_dataset_basic():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["Delhi", "Mumbai", "Delhi"],
    })

    report = profile_dataset(df)

    assert report["rows"] == 3
    assert report["columns"] == 2
    assert report["column_names"] == ["age", "city"]
    assert report["missing_values"]["age"] == 0
    assert report["unique_values"]["city"] == 2


def test_profile_dataset_with_missing_values():
    df = pd.DataFrame({
        "age": [20, None, 40],
        "city": ["Delhi", "Mumbai", None],
    })

    report = profile_dataset(df)

    assert report["rows"] == 3
    assert report["missing_values"]["age"] == 1
    assert report["missing_values"]["city"] == 1
    assert report["missing_percentage"]["age"] == 33.33
    assert report["missing_percentage"]["city"] == 33.33

def test_profile_dataset_detects_duplicates():
    df = pd.DataFrame({
        "age": [20, 30, 20],
        "city": ["Delhi", "Mumbai", "Delhi"],
    })

    report = profile_dataset(df)

    assert report["duplicate_rows"] == 1

def test_profile_dataset_rejects_empty_dataframe():
    df = pd.DataFrame(columns=["age", "city"])

    with pytest.raises(
        ValueError,
        match="Cannot profile an empty dataset."
    ):
        profile_dataset(df)

def test_profile_dataset_rejects_invalid_input():
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        profile_dataset([1, 2, 3])

def test_profile_report_is_json_serializable():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["Delhi", "Mumbai", "Delhi"]
    })

    report = profile_dataset(df)

    json_report = json.dumps(report)
    loaded_report = json.loads(json_report)

    assert loaded_report["rows"] == 3
    assert loaded_report["columns"] == 2