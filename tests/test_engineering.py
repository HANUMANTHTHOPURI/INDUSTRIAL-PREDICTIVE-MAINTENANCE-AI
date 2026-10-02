import numpy as np
import pandas as pd
import pytest

from src.features.engineering import add_mechanical_power


def test_mechanical_power_calculation():
    df = pd.DataFrame({
        "Rotational speed [rpm]": [2737],
        "Torque [Nm]": [8.8],
    })

    result = add_mechanical_power(df)

    expected_power = 8.8 * (2 * np.pi * 2737 / 60)

    assert result["Mechanical power [W]"].iloc[0] == pytest.approx(
        expected_power
    )


def test_missing_required_column():
    df = pd.DataFrame({
        "Torque [Nm]": [8.8]
    })

    with pytest.raises(ValueError):
        add_mechanical_power(df)