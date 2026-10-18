
import pandas as pd

from src.cleaning import auto_clean_dataset
from src.eda import run_eda
from src.ml import prepare_features, compare_models
from src.profiler import profile_dataset
from src.quality import generate_quality_report
from src.interpreter import interpret_results


SUPPORTED_TOOLS = {
    "profile_dataset",
    "generate_quality_report",
    "run_eda",
    "compare_models",
}



def execute_plan(
    df: pd.DataFrame,
    plan: dict,
    question: str | None = None,
) -> dict:
    """Execute a validated analysis plan and interpret its results."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Provide a non-empty pandas DataFrame.")

    if not isinstance(plan, dict):
        raise TypeError("Plan must be a dictionary.")

    tools = plan.get("tools")
    if not isinstance(tools, list):
        raise ValueError("Plan must contain a tools list.")

    if not all(
        isinstance(tool, str) and tool in SUPPORTED_TOOLS
        for tool in tools
    ):
        raise ValueError("Plan contains unsupported tools.")

    if plan.get("needs_clarification", False):
        return {
            "status": "needs_clarification",
            "question": plan.get("clarification_question"),
            "results": {},
        }

    cleaned_df, cleaning_report = auto_clean_dataset(df)
    results = {"cleaning": cleaning_report}

    for tool in dict.fromkeys(tools):
        if tool == "profile_dataset":
            results["profile"] = profile_dataset(cleaned_df)

        elif tool == "generate_quality_report":
            results["quality"] = generate_quality_report(cleaned_df)

        elif tool == "run_eda":
            results["eda"] = run_eda(
                cleaned_df,
                target_column=plan.get("target_column"),
            )

        elif tool == "compare_models":
            target = plan.get("target_column")

            if (
                not isinstance(target, str)
                or target not in cleaned_df.columns
            ):
                raise ValueError(
                    "Classification requires a valid target column."
                )

            if cleaned_df[target].isna().any():
                raise ValueError(
                    f"Target column '{target}' contains missing values."
                )

            if cleaned_df[target].nunique() < 2:
                raise ValueError(
                    f"Target column '{target}' must contain at least "
                    "two distinct classes."
                )

            X, y = prepare_features(cleaned_df, target)
            results["model_comparison"] = compare_models(X, y)

    # Interpret only after the requested tools have finished.
    interpretation_question = (
        question
        or plan.get("intent")
        or "Summarize the dataset analysis."
    )

    interpretation = interpret_results(
        interpretation_question,
        results,
    )

    return {
        "status": "completed",
        "tools_executed": list(dict.fromkeys(tools)),
        "results": results,
        "interpretation": interpretation,
    }
