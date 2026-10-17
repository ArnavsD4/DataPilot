
import pandas as pd
import pytest

from src.ml import (
    compare_models,
    prepare_features,
    cross_validate_models,
)


@pytest.fixture
def prepared_titanic_data():
    df = pd.read_csv("data/testing/titanic.csv")

    X, y = prepare_features(df, "Survived")
    X = X.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"]
    )

    return X, y


def test_compare_models_returns_expected_models(
    prepared_titanic_data,
):
    X, y = prepared_titanic_data

    result = compare_models(X, y)

    assert set(result["models"]) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }


def test_compare_models_returns_valid_accuracies(
    prepared_titanic_data,
):
    X, y = prepared_titanic_data

    result = compare_models(X, y)

    for metrics in result["models"].values():
        assert 0 <= metrics["accuracy"] <= 1


def test_compare_models_selects_best_accuracy(
    prepared_titanic_data,
):
    X, y = prepared_titanic_data

    result = compare_models(X, y)

    best_accuracy = max(
        metrics["accuracy"]
        for metrics in result["models"].values()
    )

    assert (
        result["models"][result["best_model"]]["accuracy"]
        == best_accuracy
    )


def test_compare_models_rejects_invalid_X():
    with pytest.raises(TypeError):
        compare_models([1, 2, 3], pd.Series([0, 1, 0]))


def test_cross_validate_models_returns_expected_structure(prepared_titanic_data):
    X, y = prepared_titanic_data

    result = cross_validate_models(X, y, cv=3)

    assert result["folds"] == 3
    assert set(result["models"]) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }
    assert result["best_model"] in result["models"]


def test_cross_validate_models_returns_valid_scores(prepared_titanic_data):
    X, y = prepared_titanic_data

    result = cross_validate_models(X, y, cv=3)

    for model_result in result["models"].values():
        assert len(model_result["fold_scores"]) == 3
        assert 0 <= model_result["mean_accuracy"] <= 1
        assert model_result["std_accuracy"] >= 0

