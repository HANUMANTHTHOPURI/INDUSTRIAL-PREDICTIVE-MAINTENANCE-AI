from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_failure():
    payload = {
        "type": "L",
        "air_temperature": 298.9,
        "process_temperature": 310.2,
        "rotational_speed": 2737,
        "torque": 8.8,
        "tool_wear": 142,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["failure_probability"] <= 1
    assert result["failure_probability"] > 0.90
    assert result["predicted_failure"] == 1


def test_invalid_product_type():
    payload = {
        "type": "X",
        "air_temperature": 298.9,
        "process_temperature": 310.2,
        "rotational_speed": 2737,
        "torque": 8.8,
        "tool_wear": 142,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_rotational_speed():
    payload = {
        "type": "L",
        "air_temperature": 298.9,
        "process_temperature": 310.2,
        "rotational_speed": -100,
        "torque": 8.8,
        "tool_wear": 142,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422