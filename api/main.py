from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from src.models.predict import predict_failure


app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description=(
        "Machine-learning API for predicting industrial "
        "equipment failure from operating measurements."
    ),
    version="1.0.0",
)


class MachineInput(BaseModel):
    type: str = Field(
        ...,
        pattern="^[LMH]$",
        description="Product quality type: L, M, or H",
    )

    air_temperature: float = Field(
        ...,
        description="Air temperature in Kelvin",
    )

    process_temperature: float = Field(
        ...,
        description="Process temperature in Kelvin",
    )

    rotational_speed: float = Field(
        ...,
        gt=0,
        description="Rotational speed in RPM",
    )

    torque: float = Field(
        ...,
        ge=0,
        description="Torque in Nm",
    )

    tool_wear: float = Field(
        ...,
        ge=0,
        description="Tool wear in minutes",
    )


@app.get("/")
def root():
    return {
        "message": "Industrial Predictive Maintenance API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(machine: MachineInput):
    input_df = pd.DataFrame(
        [{
            "Type": machine.type,
            "Air temperature [K]": machine.air_temperature,
            "Process temperature [K]": machine.process_temperature,
            "Rotational speed [rpm]": machine.rotational_speed,
            "Torque [Nm]": machine.torque,
            "Tool wear [min]": machine.tool_wear,
        }]
    )

    result = predict_failure(input_df)

    probability = float(
        result["failure_probability"].iloc[0]
    )

    prediction = int(
        result["predicted_failure"].iloc[0]
    )

    return {
        "failure_probability": round(probability, 6),
        "predicted_failure": prediction,
    }

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)