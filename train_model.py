"""Train and evaluate the diabetes prediction model."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/gopiashokan/dataset/main/"
    "diabetes_prediction_dataset.csv"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "diabetes_model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"

CATEGORICAL_FEATURES = ["gender", "smoking_history"]
NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "diabetes"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def get_candidate_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=18,
                        min_samples_split=4,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.08,
                        max_depth=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(name: str, pipeline: Pipeline, x_test, y_test) -> dict:
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    return {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }


def train_and_save() -> dict:
    df = pd.read_csv(DATA_URL)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_models = get_candidate_models()
    comparison = []

    best_name = None
    best_pipeline = None
    best_f1 = -1.0

    for name, pipeline in candidate_models.items():
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(name, pipeline, x_test, y_test)
        cv_scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        metrics["cv_f1_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_f1_std"] = round(float(cv_scores.std()), 4)
        comparison.append(metrics)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name
            best_pipeline = pipeline

    feature_importance = []
    classifier = best_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        feature_importance = [
            {"feature": feature, "importance": round(float(score), 4)}
            for feature, score in sorted(
                zip(FEATURE_COLUMNS, importances),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

    artifact = {
        "dataset_rows": int(len(df)),
        "diabetes_positive_rate": round(float(y.mean()), 4),
        "feature_columns": FEATURE_COLUMNS,
        "best_model": best_name,
        "model_comparison": comparison,
        "selected_model_metrics": next(
            item for item in comparison if item["model_name"] == best_name
        ),
        "feature_importance": feature_importance,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"Best model: {best_name}")
    print(json.dumps(artifact["selected_model_metrics"], indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")

    return artifact


if __name__ == "__main__":
    train_and_save()
