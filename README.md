# DataPilot — Automated Data Analysis & Science Agent

DataPilot is a Python-based data science project that profiles tabular datasets, identifies data-quality issues, applies configurable cleaning strategies, performs automated exploratory data analysis (EDA), and supports machine learning model evaluation.

The long-term goal is to evolve DataPilot into an autonomous data science agent that can investigate datasets, formulate hypotheses, evaluate them using data-driven evidence, and produce actionable analytical insights.

## Current Features

### 1. Dataset Profiling
- Dataset dimensions and column names
- Data types and numerical/categorical column classification
- Missing-value counts and percentages
- Unique-value counts
- Numerical descriptive statistics
- Duplicate-row detection
- JSON-serializable profiling reports

### 2. Data Quality Checks
- Missing-value analysis
- Duplicate-row detection
- Column data-type inspection
- Configurable numerical range validation
- IQR-based outlier detection

### 3. Data Cleaning
- Configurable missing-value handling strategies
- Mean, median, and mode imputation
- Custom-value imputation
- Domain-specific cleaning for the Telco Customer Churn dataset
- Non-destructive cleaning functions that preserve the original DataFrame

### 4. Automated Exploratory Data Analysis
- Numerical descriptive summaries
- Categorical value counts and cardinality
- Numerical correlation analysis
- Detection of highly correlated numerical feature pairs
- Skewness analysis with basic interpretation
- Optional target-column analysis
- Histograms for numerical columns
- Structured JSON reports and saved visualizations

### 5. Machine Learning Baseline
- Feature and target separation
- Numerical and categorical preprocessing
- Missing-value imputation
- Feature scaling and categorical encoding
- Train/test splitting
- Classification model training and evaluation
- Logistic Regression, Decision Tree, and Random Forest comparison
- Accuracy, confusion matrix, and classification report
- Cross-validation for more reliable model comparison

### 6. Integrated Data Pipeline
- Load and validate CSV datasets
- Generate profiling and data-quality reports
- Apply supported cleaning operations
- Run automated EDA
- Return structured results for downstream use

## Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** scikit-learn
- **Visualization:** Matplotlib
- **Testing:** pytest
- **Input format:** CSV
- **Output formats:** JSON reports and image files

## Project Structure

```text
DataPilot/
├── data/
│   ├── raw/
│   └── testing/
├── notebooks/
├── reports/
│   ├── plots/
│   ├── eda_report.json
│   ├── telco_eda.json
│   └── telco_profile.json
├── src/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── eda.py
│   ├── ml.py
│   ├── pipeline.py
│   ├── profiler.py
│   └── quality.py
├── tests/
├── run_eda.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ArnavsD4/DataPilot.git
cd DataPilot
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add a dataset

Place the Telco Customer Churn CSV file at:

```text
data/raw/Telco-Customer-Churn.csv
```

The dataset is not included in this repository. Obtain it separately and place it at the path expected by the runner.

### 5. Run automated EDA

```bash
python run_eda.py
```

This runs the EDA workflow and saves a structured report and generated visualizations using the configured output paths.

### 6. Run the integrated pipeline

The pipeline can also be called from Python:

```python
from src.pipeline import run_pipeline

result = run_pipeline("data/raw/Telco-Customer-Churn.csv")

print(result["dataset"])
print(result["quality"])
print(result["eda"]["dataset_overview"])
```

The returned structure contains dataset metadata, profiling and quality results, cleaning information, and EDA results according to the pipeline implementation.

### 7. Run the test suite

```bash
python -m pytest -v
```

The latest verified local test run completed with **44 passing tests**. Run the suite again after making changes to confirm the current result.

## Example Dataset

The Telco Customer Churn dataset used during development contains:

- **Rows:** 7,043
- **Columns:** 21
- **Numerical columns:** 4
- **Categorical columns:** 17

EDA results and visualization counts depend on the dataset, the selected target column, and the configured output settings.

## Development Roadmap

- [x] Dataset profiling
- [x] Data-quality checks
- [x] Configurable data cleaning
- [x] Automated EDA
- [x] Numerical histogram generation
- [x] Skewness and high-correlation analysis
- [x] Integrated CSV analysis pipeline
- [x] ML preprocessing and baseline model training
- [x] Model comparison and cross-validation
- [x] Automated tests for implemented modules
- [ ] More robust model explainability
- [ ] LangGraph-based agent orchestration
- [ ] MCP tool integration
- [ ] FastAPI service
- [ ] Interactive frontend
- [ ] End-to-end autonomous analysis and insight generation

## Testing

The test suite covers dataset profiling, data-quality checks, cleaning behavior, EDA functionality, machine learning utilities, and pipeline integration.

Run:

```bash
python -m pytest -v
```

## Author

**Arnav Dasmal**

GitHub: [@ArnavsD4](https://github.com/ArnavsD4)
