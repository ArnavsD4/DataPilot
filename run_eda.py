
import json
from pathlib import Path

import pandas as pd

from src.eda import run_eda


def main():
    project_root = Path(__file__).resolve().parent

    dataset_path = (
        project_root / "data" / "raw" / "Telco-Customer-Churn.csv"
    )
    report_path = project_root / "reports" / "telco_eda.json"
    plots_dir = project_root / "reports" / "plots"

    print(f"Loading dataset from: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Convert TotalCharges to numeric because the CSV may contain blanks.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

    report = run_eda(
        df,
        target_column="Churn",
        output_dir=str(plots_dir),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nEDA completed successfully!")
    print(f"Rows: {report['dataset_overview']['rows']}")
    print(f"Columns: {report['dataset_overview']['columns']}")
    print(f"Numeric summaries: {len(report['numeric_summary'])}")
    print(
        "Categorical summaries: "
        f"{len(report['categorical_summary'])}"
    )
    print(f"Histograms generated: {len(report['visualizations'])}")
    print(f"Report saved to: {report_path}")
    print(f"Plots saved to: {plots_dir}")
    print(f"Target analysis: {report['target_analysis']}")


if __name__ == "__main__":
    main()
