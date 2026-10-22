
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Fraud Detection API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    transaction = {
        "Time": 0,
        "Amount": 100,
        **{f"V{i}": 0 for i in range(1, 29)},
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["label"] in ["fraud", "legitimate"]
    assert 0 <= result["fraud_probability"] <= 1


def test_predict_rejects_negative_amount():
    transaction = {
        "Time": 0,
        "Amount": -100,
        **{f"V{i}": 0 for i in range(1, 29)},
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422
