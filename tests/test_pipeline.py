
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


def test_run_pipeline_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        run_pipeline(missing_path)


def test_run_pipeline_rejects_non_csv_file(tmp_path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Only CSV"):
        run_pipeline(text_path)
