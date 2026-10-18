
import pandas as pd
import pytest

from src.pipeline import run_pipeline


def test_run_pipeline_returns_expected_structure(tmp_path):
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["Delhi", "Mumbai", "Delhi"],
    })

    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    result = run_pipeline(csv_path)

    assert result["dataset"]["rows"] == 3
    assert result["dataset"]["columns"] == 2
    assert result["profile"]["rows"] == 3
    assert result["quality_before"]["rows"] == 3
    assert result["quality_after"]["rows"] == 3
    assert result["cleaning"]["rows_after_cleaning"] == 3
    assert "eda" in result

    expected_eda_sections = {
        "dataset_overview",
        "numeric_summary",
        "categorical_summary",
        "correlations",
        "visualizations",
        "skewness_analysis",
        "high_correlations",
    }

    assert expected_eda_sections.issubset(result["eda"].keys())
    assert result["ml"] is None


def test_run_pipeline_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        run_pipeline(missing_path)


def test_run_pipeline_rejects_non_csv_file(tmp_path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Only CSV"):
        run_pipeline(text_path)

def test_run_pipeline_rejects_empty_dataset(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot run the pipeline"):
        run_pipeline(empty_path)

def test_run_pipeline_runs_ml_when_target_is_provided(tmp_path):
    df = pd.DataFrame({
        "age": list(range(20)),
        "city": ["Delhi", "Mumbai"] * 10,
        "churn": ["Yes", "No"] * 10,
    })

    csv_path = tmp_path / "ml_sample.csv"
    df.to_csv(csv_path, index=False)

    result = run_pipeline(csv_path, target_column="churn")

    assert result["ml"] is not None
    assert set(result["ml"]["models"].keys()) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }
    assert result["ml"]["best_model"] in result["ml"]["models"]

def test_run_pipeline_rejects_invalid_target(tmp_path):
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "city": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })

    csv_path = tmp_path / "invalid_target.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="does not exist"):
        run_pipeline(csv_path, target_column="unknown_column")


def test_pipeline_cleaning_report_includes_outliers(tmp_path):
    import pandas as pd

    from src.pipeline import run_pipeline

    csv_path = tmp_path / "outlier_demo.csv"
    pd.DataFrame({
        "Age": [20, 21, 22, 23, 24, 100],
        "City": ["A", "B", "A", "B", "A", "B"],
    }).to_csv(csv_path, index=False)

    result = run_pipeline(csv_path)

    assert "outliers" in result["cleaning"]
    assert result["cleaning"]["outliers"]["Age"]["outlier_count"] == 1
    assert result["cleaning"]["rows_before"] == 6
    assert result["cleaning"]["rows_after"] == 6
