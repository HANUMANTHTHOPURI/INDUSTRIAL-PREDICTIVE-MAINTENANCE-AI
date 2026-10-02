from pathlib import Path

import joblib
import pandas as pd

from src.features.engineering import add_mechanical_power


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "predictive_maintenance_rf.joblib"
)

MODEL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Mechanical power [W]",
]


def load_model():
    """Load the trained predictive-maintenance pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


def predict_failure(
    input_data: pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Predict machine-failure probability from raw machine measurements.

    Mechanical power is calculated internally from RPM and torque.
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    engineered_data = add_mechanical_power(input_data)

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in engineered_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    model_input = engineered_data[MODEL_FEATURES]

    failure_probability = model.predict_proba(model_input)[:, 1]

    prediction = (
        failure_probability >= threshold
    ).astype(int)

    return pd.DataFrame(
        {
            "failure_probability": failure_probability,
            "predicted_failure": prediction,
        },
        index=input_data.index,
    )