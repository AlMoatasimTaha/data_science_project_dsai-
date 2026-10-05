import pytest
from fastapi.testclient import TestClient
from api.main import app
from jobmarket.config import ROOT, TASKS
from jobmarket.service import load_model

client = TestClient(app)
POSTING = {"job_title": "Data Scientist", "company_type": "Startup", "industry": "Tech", "country": "USA", "company_size": "Medium", "posted_year": 2026, "experience_level": "Mid", "remote_type": "Hybrid"}

def test_health():
    assert client.get("/health").status_code == 200

@pytest.mark.parametrize("patch", [{"posted_year": 2030}, {"posted_year": "2026"}, {"country": "Unknown"}, {"salary_min_usd": 50000}, {"city": "Remote"}])
def test_invalid_and_leaky_inputs_rejected(patch):
    assert client.post("/predict/salary", json={**POSTING, **patch}).status_code == 422

def test_missing_task_features_rejected():
    assert client.post("/predict/salary", json={k: v for k, v in POSTING.items() if k not in ["experience_level", "remote_type"]}).status_code == 422

def test_unknown_task():
    assert client.post("/predict/path_traversal", json=POSTING).status_code == 422

def test_missing_model_returns_503(monkeypatch):
    def missing(*args):
        raise FileNotFoundError("Model not trained")
    monkeypatch.setattr("jobmarket.service.load_model", missing)
    assert client.post("/predict/salary", json=POSTING).status_code == 503

@pytest.mark.parametrize("name", list(TASKS))
def test_actual_inference(name):
    if not (ROOT / f"models/{name}/temporal.joblib").exists():
        pytest.skip("Integration requires training")
    response = client.post(f"/predict/{name}", json=POSTING)
    assert response.status_code == 200
    result = response.json()
    assert result["warning"] and result["validation"]
    if name in ["salary", "salary_width"]:
        assert result["interval_90_usd"][0] <= result["prediction_usd"] <= result["interval_90_usd"][1]
    else:
        assert all(0 <= x <= 1 for x in result["probabilities"].values())
