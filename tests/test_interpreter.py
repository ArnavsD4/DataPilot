
from src.interpreter import interpret_results


def test_interprets_dataset_dimensions_and_model_metrics():
    results = {
        "eda": {
            "dataset_overview": {"rows": 7043, "columns": 21},
            "target_analysis": None,
        },
        "model_comparison": {
            "test_rows": 1409,
            "best_model": "Logistic Regression",
            "models": {
                "Logistic Regression": {"accuracy": 0.8055},
                "Decision Tree": {"accuracy": 0.7736},
            },
        },
    }

    report = interpret_results("How well can we predict churn?", results)

    assert "7,043 rows" in report["answer"]
    assert "1,409 test rows" in report["answer"]
    assert "Logistic Regression" in report["answer"]
    assert "80.5%" in report["answer"]
    assert report["evidence_based"] is True


def test_interprets_binary_target_distribution():
    results = {
        "eda": {
            "dataset_overview": {"rows": 7043, "columns": 21},
            "target_analysis": {
                "column": "Churn",
                "type": "categorical",
                "value_counts": {"No": 5174, "Yes": 1869},
            },
        }
    }

    report = interpret_results("Explain customer churn", results)

    assert "73.5%" in report["answer"]
    assert "26.5%" in report["answer"]
    assert "'Yes' target class" in report["answer"]


def test_interprets_positive_class_metrics():
    results = {
        "model_comparison": {
            "test_rows": 1409,
            "best_model": "Logistic Regression",
            "models": {
                "Logistic Regression": {
                    "accuracy": 0.8055,
                    "classification_report": {
                        "No": {
                            "precision": 0.8494,
                            "recall": 0.8937,
                            "f1-score": 0.8710,
                            "support": 1035,
                        },
                        "Yes": {
                            "precision": 0.6563,
                            "recall": 0.5615,
                            "f1-score": 0.6052,
                            "support": 374,
                        },
                    },
                }
            },
        }
    }

    report = interpret_results("Predict customer churn", results)

    assert "precision 65.6%" in report["answer"]
    assert "recall 56.1%" in report["answer"]
    assert "F1-score 60.5%" in report["answer"]


def test_does_not_assume_arbitrary_positive_class():
    results = {
        "model_comparison": {
            "best_model": "Model A",
            "models": {
                "Model A": {
                    "accuracy": 0.9,
                    "classification_report": {
                        "Class A": {"precision": 0.9},
                        "Class B": {"precision": 0.8},
                    },
                }
            },
        }
    }

    report = interpret_results("Analyze classes", results)

    assert any(
        "Could not unambiguously identify a positive class"
        in item
        for item in report["limitations"]
    )


def test_handles_missing_results():
    report = interpret_results("Explain the data", {})

    assert report["findings"] == []
    assert report["limitations"]
    assert "couldn't extract enough verified findings" in report["answer"]


def test_rejects_empty_question():
    import pytest

    with pytest.raises(ValueError):
        interpret_results("  ", {})


def test_rejects_non_dictionary_results():
    import pytest

    with pytest.raises(TypeError):
        interpret_results("Explain churn", [])
