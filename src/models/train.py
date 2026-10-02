import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineering import add_mechanical_power


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "predictive_maintenance_rf_reproduced.joblib"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)

RANDOM_STATE = 42


RAW_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

MODEL_FEATURES = RAW_FEATURES + [
    "Mechanical power [W]",
]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Mechanical power [W]",
]

TARGET = "Machine failure"


def load_data() -> pd.DataFrame:
    """Load the AI4I predictive-maintenance dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def build_pipeline() -> Pipeline:
    """Build the preprocessing and Random Forest pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["Type"],
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=700,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    print("Loading dataset...")
    df = load_data()

    print(f"Dataset shape: {df.shape}")

    # Physics-informed feature engineering
    df = add_mechanical_power(df)

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    model = build_pipeline()

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print("\nHeld-out test metrics")
    print("---------------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, OUTPUT_PATH)
    metadata = {
        "model": {
            "name": "Industrial Predictive Maintenance Random Forest",
            "version": "1.0.0",
            "algorithm": "RandomForestClassifier",
            "artifact": OUTPUT_PATH.name,
        },
        "training": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "failure_rate": float(y.mean()),
        },
        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "rows": int(len(df)),
            "target": TARGET,
        },
        "features": {
            "raw_features": RAW_FEATURES,
            "engineered_features": [
                "Mechanical power [W]"
            ],
            "model_features": MODEL_FEATURES,
            "excluded_leakage_features": [
                "TWF",
                "HDF",
                "PWF",
                "OSF",
                "RNF",
            ],
        },
        "physics": {
            "mechanical_power_formula":
                "P = torque * (2 * pi * RPM / 60)"
        },
        "hyperparameters": {
            "n_estimators": 700,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "log2",
            "class_weight": "balanced",
        },
        "decision": {
            "threshold": 0.50
        },
        "test_metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print(f"\nModel metadata saved to:\n{METADATA_PATH}")

    print(
        f"\nReproduced model saved to:\n{OUTPUT_PATH}"
    )
    

if __name__ == "__main__":
    main()