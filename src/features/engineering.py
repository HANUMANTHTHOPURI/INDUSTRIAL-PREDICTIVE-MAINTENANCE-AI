import numpy as np
import pandas as pd


def add_mechanical_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mechanical power calculated from rotational speed and torque.

    Formula:
        angular_velocity = 2 * pi * RPM / 60
        mechanical_power = torque * angular_velocity
    """

    required_columns = [
        "Rotational speed [rpm]",
        "Torque [Nm]",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = df.copy()

    angular_velocity = (
        2
        * np.pi
        * result["Rotational speed [rpm]"]
        / 60
    )

    result["Mechanical power [W]"] = (
        result["Torque [Nm]"] * angular_velocity
    )

    return result