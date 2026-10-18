
from typing import Any


def interpret_results(
    question: str,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Turn verified DataPilot analysis results into readable findings."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(results, dict):
        raise TypeError("Results must be a dictionary.")

    findings: list[str] = []
    limitations: list[str] = []

    # 1. Dataset dimensions and target distribution
    eda = results.get("eda", {})
    overview = eda.get("dataset_overview", {}) if isinstance(eda, dict) else {}

    if isinstance(overview, dict):
        rows = overview.get("rows")
        columns = overview.get("columns")

        if (
            isinstance(rows, (int, float))
            and not isinstance(rows, bool)
            and isinstance(columns, (int, float))
            and not isinstance(columns, bool)
        ):
            findings.append(
                f"The analyzed dataset contains {int(rows):,} rows "
                f"and {int(columns):,} columns."
            )
        else:
            limitations.append("Dataset dimensions were not available.")

    else:
        limitations.append("EDA results were not available.")

    target_analysis = eda.get("target_analysis") if isinstance(eda, dict) else None

    if isinstance(target_analysis, dict):
        value_counts = target_analysis.get("value_counts")

        if isinstance(value_counts, dict) and value_counts:
            valid_counts = {
                str(label): count
                for label, count in value_counts.items()
                if isinstance(count, (int, float))
                and not isinstance(count, bool)
                and count >= 0
            }

            total = sum(valid_counts.values())

            if total > 0 and valid_counts:
                majority_label = max(valid_counts, key=valid_counts.get)
                majority_count = valid_counts[majority_label]
                findings.append(
                    f"The most common target class is '{majority_label}' "
                    f"({majority_count / total:.1%} of records)."
                )

                # Report a positive-class rate only when a binary target
                # has an unambiguous positive label.
                positive_labels = {"yes", "true", "1", "churn", "positive"}

                positive_matches = [
                    label for label in valid_counts
                    if label.strip().lower() in positive_labels
                ]

                if len(valid_counts) == 2 and len(positive_matches) == 1:
                    positive_label = positive_matches[0]
                    rate = valid_counts[positive_label] / total
                    findings.append(
                        f"The '{positive_label}' target class represents "
                        f"{rate:.1%} of records."
                    )
            else:
                limitations.append(
                    "Target-class counts were empty or invalid."
                )
        else:
            limitations.append(
                "Target-class counts were not available in the EDA results."
            )
    else:
        limitations.append("Target-analysis details were not available.")

    # 2. Model comparison and performance metrics
    model_report = results.get("model_comparison")

    if not isinstance(model_report, dict):
        # Also support the result key used by the agent.
        model_report = results.get("model_comparison", {})

    if isinstance(model_report, dict) and model_report:
        models = model_report.get("models", {})
        best_model = model_report.get("best_model")
        test_rows = model_report.get("test_rows")

        if isinstance(test_rows, int) and not isinstance(test_rows, bool):
            findings.append(
                f"Model evaluation used {test_rows:,} test rows."
            )

        if isinstance(models, dict) and models:
            if isinstance(best_model, str) and best_model in models:
                metrics = models[best_model]

                if isinstance(metrics, dict):
                    accuracy = metrics.get("accuracy")

                    if (
                        isinstance(accuracy, (int, float))
                        and not isinstance(accuracy, bool)
                        and 0 <= accuracy <= 1
                    ):
                        findings.append(
                            f"{best_model} was the reported best model, "
                            f"with test accuracy of {accuracy:.1%}."
                        )

                    classification_report = metrics.get(
                        "classification_report", {}
                    )

                    if isinstance(classification_report, dict):
                        # Prefer a clearly identified positive class.
                        positive_labels = {
                            "yes", "true", "1", "churn", "positive"
                        }

                        positive_classes = [
                            (label, values)
                            for label, values in classification_report.items()
                            if label.strip().lower() in positive_labels
                            and isinstance(values, dict)
                        ]

                        if len(positive_classes) == 1:
                            label, class_metrics = positive_classes[0]

                            metric_names = (
                                ("precision", "precision"),
                                ("recall", "recall"),
                                ("f1-score", "F1-score"),
                            )

                            reported_metrics = []

                            for key, display_name in metric_names:
                                value = class_metrics.get(key)

                                if (
                                    isinstance(value, (int, float))
                                    and not isinstance(value, bool)
                                    and 0 <= value <= 1
                                ):
                                    reported_metrics.append(
                                        f"{display_name} {value:.1%}"
                                    )

                            if reported_metrics:
                                findings.append(
                                    f"For the '{label}' class, {best_model} "
                                    f"has " + ", ".join(reported_metrics) + "."
                                )
                            else:
                                limitations.append(
                                    "Positive-class metrics were unavailable."
                                )
                        else:
                            limitations.append(
                                "Could not unambiguously identify a "
                                "positive class for detailed model metrics."
                            )
                    else:
                        limitations.append(
                            "Classification metrics were not available."
                        )
                else:
                    limitations.append(
                        "Metrics for the reported best model were invalid."
                    )
            else:
                limitations.append(
                    "The best model was not identified in the model results."
                )
        else:
            limitations.append("Model-level metrics were not available.")
    else:
        limitations.append("Model-comparison results were not supplied.")

    # 3. Compose the answer using only available evidence
    if findings:
        answer = (
            "Here is what the available analysis supports:\n- "
            + "\n- ".join(findings)
        )
    else:
        answer = (
            "I couldn't extract enough verified findings to answer "
            "the question from the supplied results."
        )

    return {
        "question": question.strip(),
        "answer": answer,
        "findings": findings,
        "limitations": limitations,
        "evidence_based": True,
    }
