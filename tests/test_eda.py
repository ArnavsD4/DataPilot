
import pandas as pd
import pytest

from src.eda import (
    summarize_numeric,
    summarize_categorical,
    calculate_correlations,
    run_eda
)


def test_summarize_numeric_returns_statistics():
    df = pd.DataFrame({
        "age": [10, 20, 30],
        "city": ["A", "B", "A"]
    })

    report = summarize_numeric(df)

    assert report["age"]["count"] == 3.0
    assert report["age"]["mean"] == 20.0
    assert report["age"]["min"] == 10.0
    assert report["age"]["max"] == 30.0
    assert "city" not in report


def test_summarize_numeric_returns_empty_dict_without_numeric_columns():
    df = pd.DataFrame({
        "city": ["Delhi", "Kolkata"]
    })

    assert summarize_numeric(df) == {}


def test_summarize_numeric_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        summarize_numeric(pd.DataFrame())


def test_summarize_categorical_counts_values():
    df = pd.DataFrame({
        "city": ["Delhi", "Kolkata", "Delhi", None],
        "age": [20, 21, 22, 23]
    })

    report = summarize_categorical(df)

    assert report["city"]["unique_values"] == 2
    assert report["city"]["value_counts"]["Delhi"] == 2
    assert report["city"]["value_counts"]["Kolkata"] == 1
    assert report["city"]["value_counts"]["nan"] == 1
    assert report["city"]["missing_count"] == 1
    assert "age" not in report


def test_calculate_correlations():
    df = pd.DataFrame({
        "age": [1, 2, 3],
        "salary": [10, 20, 30],
        "score": [30, 20, 10],
    })

    report = calculate_correlations(df)

    assert report["age"]["salary"] == pytest.approx(1.0)
    assert report["age"]["score"] == pytest.approx(-1.0)
    assert report["salary"]["score"] == pytest.approx(-1.0)


def test_run_eda_combines_reports():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "salary": [100, 200, 300],
        "city": ["A", "B", "A"],
        "churn": ["No", "Yes", "No"],
    })

    report = run_eda(df, target_column="churn")

    assert report["dataset_overview"]["rows"] == 3
    assert "age" in report["numeric_summary"]
    assert "city" in report["categorical_summary"]
    assert "age" in report["correlations"]
    assert report["target_analysis"]["type"] == "categorical"
    assert report["target_analysis"]["value_counts"]["No"] == 2


def test_run_eda_rejects_missing_target_column():
    df = pd.DataFrame({"age": [20, 30]})

    with pytest.raises(ValueError, match="does not exist"):
        run_eda(df, target_column="churn")
