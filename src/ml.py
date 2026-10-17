import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features (X) and target (y).

    The original DataFrame remains unchanged.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot prepare features from an empty dataset.")

    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("Target column must be a non-empty string.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    if df[target_column].isna().any():
        raise ValueError("Target column contains missing values.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    if X.shape[1] == 0:
        raise ValueError("Dataset must contain at least one feature column.")

    return X, y

def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """Build preprocessing pipelines for numerical and categorical features."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if X.empty:
        raise ValueError("Cannot preprocess an empty dataset.")

    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")

    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train a logistic regression model using a leakage-safe pipeline."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if X.empty or X.shape[1] == 0:
        raise ValueError("X must contain rows and feature columns.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if X.columns.duplicated().any():
        raise ValueError("X contains duplicate column names.")

    # Split before fitting preprocessing to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    return {
        "model": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a trained model on held-out test data."""

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")

    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    if X_test.empty:
        raise ValueError("Test data cannot be empty.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have equal lengths.")

    predictions = model.predict(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "test_rows": len(y_test),
    }


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train and compare multiple classification models."""

    from sklearn.model_selection import train_test_split

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if X.empty or y.empty:
        raise ValueError("X and y cannot be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must have equal lengths.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
    }

    results = {}

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        results[name] = {
            "accuracy": metrics["accuracy"],
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
        }

    return {
        "test_rows": len(y_test),
        "models": results,
        "best_model": max(
            results,
            key=lambda name: results[name]["accuracy"],
        ),
    }


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Evaluate classifiers across stratified cross-validation folds."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if X.empty or y.empty:
        raise ValueError("X and y cannot be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must have equal lengths.")

    if not isinstance(cv, int) or isinstance(cv, bool) or cv < 2:
        raise ValueError("cv must be an integer of at least 2.")

    if y.value_counts().min() < cv:
        raise ValueError(
            "Each target class must have at least cv observations."
        )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
    }

    splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    results = {}

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("classifier", classifier),
            ]
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=splitter,
            scoring="accuracy",
            error_score="raise",
        )

        results[name] = {
            "fold_scores": scores.round(4).tolist(),
            "mean_accuracy": round(float(scores.mean()), 4),
            "std_accuracy": round(float(scores.std()), 4),
        }

    return {
        "folds": cv,
        "models": results,
        "best_model": max(
            results,
            key=lambda name: results[name]["mean_accuracy"],
        ),
    }
