
import json

from src.llm import generate_text


def plan_analysis(question: str, columns: list[str]) -> dict:
    """Convert a business question into a structured analysis plan."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not columns or not all(isinstance(c, str) for c in columns):
        raise ValueError("Provide a non-empty list of column names.")

    prompt = f"""
You are the analysis planner for DataPilot, a data-science agent.

Interpret the user's business question and propose a plan using
only the available dataset columns and supported tools.

Available tools:
- profile_dataset: inspect dataset structure and data types
- generate_quality_report: inspect data-quality issues
- run_eda: explore distributions, correlations, and patterns
- compare_models: compare classification models when a valid
  target column and suitable data are available

Rules:
- Do not invent columns or claim that analysis has already happened.
- Do not claim correlation proves causation.
- If the question cannot be answered with the available information,
  set needs_clarification to true or explain the limitation.
- Do not select a target column unless it is justified by the question.
- Return only valid JSON, with no Markdown fences.

Return this exact JSON structure:
{{
  "intent": "short description of the user's goal",
  "analysis_types": ["eda"],
  "tools": ["run_eda"],
  "target_column": null,
  "needs_clarification": false,
  "clarification_question": null,
  "reasoning": "brief explanation of the proposed plan"
}}

The analysis_types list may contain: profiling, data_quality, eda,
classification, forecasting, or optimization.

The tools list may contain only the four tools listed above.
Use classification only when justified by the question and data.
Do not select forecasting or optimization tools because none are
currently available in the supported tools list.

Dataset columns:
{json.dumps(columns)}

User's business question:
{question}
"""

    response_text = generate_text(prompt)

    try:
        plan = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The model did not return valid JSON. Try again."
        ) from exc

    required_keys = {
        "intent",
        "analysis_types",
        "tools",
        "target_column",
        "needs_clarification",
        "clarification_question",
        "reasoning",
    }

    if not isinstance(plan, dict) or not required_keys.issubset(plan):
        raise ValueError("The model returned an invalid plan structure.")

    allowed_tools = {
        "profile_dataset",
        "generate_quality_report",
        "run_eda",
        "compare_models",
    }

    if (
        not isinstance(plan["tools"], list)
        or not all(
            isinstance(tool, str) and tool in allowed_tools
            for tool in plan["tools"]
        )
    ):
        raise ValueError("The plan contains unsupported tools.")

    if not isinstance(plan["needs_clarification"], bool):
        raise ValueError("Invalid needs_clarification value.")

    if not isinstance(plan["analysis_types"], list):
        raise ValueError("Invalid analysis_types value.")

    target = plan["target_column"]
    if target is not None and target not in columns:
        raise ValueError("The plan selected a column not in the dataset.")

    return plan
