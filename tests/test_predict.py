import pandas as pd

from src.models.predict import predict_failure


def test_known_failure_prediction():
    machine = pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [298.9],
        "Process temperature [K]": [310.2],
        "Rotational speed [rpm]": [2737],
        "Torque [Nm]": [8.8],
        "Tool wear [min]": [142],
    })

    result = predict_failure(machine)

    probability = result["failure_probability"].iloc[0]
    prediction = result["predicted_failure"].iloc[0]

    assert 0 <= probability <= 1
    assert probability > 0.90
    assert prediction == 1


def test_invalid_threshold():
    machine = pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [298.9],
        "Process temperature [K]": [310.2],
        "Rotational speed [rpm]": [2737],
        "Torque [Nm]": [8.8],
        "Tool wear [min]": [142],
    })

    try:
        predict_failure(machine, threshold=1.5)
        assert False
    except ValueError:
        assert True