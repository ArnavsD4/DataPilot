
import pandas as pd
import pytest

from src.agent import execute_plan


def make_plan(**overrides):
    plan = {
        "intent": "Explore the dataset",
        "analysis_types": ["eda"],
        "tools": ["run_eda"],
        "target_column": None,
        "needs_clarification": False,
        "clarification_question": None,
        "reasoning": "Explore distributions and relationships.",
    }
    plan.update(overrides)
    return plan


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="non-empty"):
        execute_plan(pd.DataFrame(), make_plan())


def test_rejects_non_dataframe():
    with pytest.raises(ValueError, match="non-empty"):
        execute_plan([], make_plan())


def test_rejects_non_dictionary_plan():
    df = pd.DataFrame({"Age": [20, 30, 40]})

    with pytest.raises(TypeError, match="dictionary"):
        execute_plan(df, [])


def test_rejects_unsupported_tool():
    df = pd.DataFrame({"Age": [20, 30, 40]})
    plan = make_plan(tools=["delete_database"])

    with pytest.raises(ValueError, match="unsupported"):
        execute_plan(df, plan)


def test_rejects_non_list_tools():
    df = pd.DataFrame({"Age": [20, 30, 40]})
    plan = make_plan(tools="run_eda")

    with pytest.raises(ValueError, match="tools list"):
        execute_plan(df, plan)


def test_returns_clarification_without_running_tools():
    df = pd.DataFrame({"Age": [20, 30, 40]})
    plan = make_plan(
        needs_clarification=True,
        clarification_question="Which outcome do you want to investigate?",
    )

    result = execute_plan(df, plan)

    assert result["status"] == "needs_clarification"
    assert result["question"] == (
        "Which outcome do you want to investigate?"
    )
    assert result["results"] == {}


def test_rejects_missing_classification_target():
    df = pd.DataFrame({"Age": [20, 30, 40]})
    plan = make_plan(
        tools=["compare_models"],
        target_column="Churn",
    )

    with pytest.raises(ValueError, match="valid target column"):
        execute_plan(df, plan)
