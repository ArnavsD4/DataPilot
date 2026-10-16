import json

with open("reports/telco_eda.json", "r", encoding="utf-8") as file:
    report = json.load(file)

print("Report sections:", list(report.keys()))

visualizations = report.get("visualizations", {})
print("Number of visualizations:", len(visualizations))
print(json.dumps(visualizations, indent=2))