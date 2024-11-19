from fastapi.testclient import TestClient
from src.serve import app

# Initialize TestClient
client = TestClient(app)

def test_predict_claim_invalid():
    response = client.post("/claim/v1/predict", json={})
    assert response.status_code == 422 

def test_predict_claim():
    response = client.post("/claim/v1/predict", json={"claim": "Wheat Protien causes liver damage."})
    assert response.status_code == 200
    assert "veracity" in response.json()
    assert "confidence" in response.json()
    assert response.json()["veracity"] in ["False", "Mixture", "True", "Unproven"]
    assert 0 <= response.json()["confidence"] <= 1
