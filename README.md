# DataPilot — Automated Data Analysis & Science Agent

DataPilot is a Python-based data science project that profiles tabular datasets, identifies data-quality issues, performs automated exploratory data analysis (EDA), and generates structured reports with visualizations.

The long-term goal is to evolve DataPilot into an autonomous data science agent that can investigate datasets, formulate hypotheses, evaluate them using data-driven evidence, and produce analytical insights.

## Current Features

### 1. Dataset Profiling

- Dataset dimensions and column names
- Data types and column classification
- Missing-value counts and percentages
- Unique-value counts
- Numerical descriptive statistics
- Duplicate-row detection
- JSON-serializable profiling reports

### 2. Data Quality Checks

- Missing-value analysis
- Duplicate-row detection
- Column data-type inspection
- Configurable numeric range validation
- IQR-based outlier detection

### 3. Automated Exploratory Data Analysis

- Numerical summaries
- Categorical value counts and cardinality
- Correlation analysis
- Detection of highly correlated numerical feature pairs
- Skewness analysis with basic interpretation
- Optional target-column analysis
- Histograms for numerical columns
- JSON reports and saved plot images

## Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Testing:** pytest
- **Dataset:** Telco Customer Churn dataset

## Project Structure

```text
DataPilot/
├── data/
│   └── raw/
├── notebooks/
├── reports/
│   ├── plots/
│   ├── telco_profile.json
│   └── telco_eda.json
├── src/
│   ├── profiler.py
│   ├── quality.py
│   └── eda.py
├── tests/
│   ├── test_profiler.py
│   ├── test_quality.py
│   ├── test_eda.py
│   └── test_histograms.py
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

### 4. Add the dataset

Place the Telco Customer Churn CSV file at:

```text
data/raw/Telco-Customer-Churn.csv
```

The dataset is not included in this repository, so obtain it separately.

### 5. Run automated EDA

```bash
python run_eda.py
```

The pipeline generates a structured EDA report and saves numerical histograms under `reports/plots/`.

### 6. Run the test suite

```bash
python -m pytest -v
```

## Example Results

The current Telco Customer Churn dataset contains:

- **Rows:** 7,043
- **Columns:** 21
- **Numerical summaries:** 4
- **Categorical summaries:** 17
- **Numerical histograms:** 4

The EDA report includes dataset overview, numerical and categorical summaries, correlations, target analysis, skewness analysis, and visualization paths.

These results describe the current implementation and this dataset; results will vary for other datasets.

## Development Roadmap

- [x] Dataset profiling
- [x] Data-quality checks
- [x] Automated EDA
- [x] Numerical histogram generation
- [x] Skewness and high-correlation analysis
- [x] Automated tests for the implemented modules
- [ ] Automated data cleaning and preprocessing pipeline
- [ ] ML model training and evaluation
- [ ] Model comparison and explainability
- [ ] LangGraph-based agent orchestration
- [ ] FastAPI service
- [ ] MCP tool integration
- [ ] Interactive frontend

## Testing

The current test suite has **27 passing tests** covering profiling, data-quality checks, EDA, and histogram generation.

Run the tests locally to verify the current state of the code.

## Author

**Arnav Dasmal**

GitHub: [@ArnavsD4](https://github.com/ArnavsD4)
