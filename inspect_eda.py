
import json
from pathlib import Path

report_path = Path("reports/telco_eda.json")

with report_path.open("r", encoding="utf-8") as file:
    report = json.load(file)

print("DATASET OVERVIEW")
print(json.dumps(report["dataset_overview"], indent=2))

print("\nNUMERIC SUMMARY")
print(json.dumps(report["numeric_summary"], indent=2))

print("\nCORRELATIONS")
print(json.dumps(report["correlations"], indent=2))

print("\nCHURN DISTRIBUTION")
print(json.dumps(report["target_analysis"], indent=2))
