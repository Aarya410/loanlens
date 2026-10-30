from fastapi.testclient import TestClient

from app.main import app

GOOD = dict(age=34, annual_income_lakh=18, loan_amount_lakh=20, term_years=15,
            credit_score=790, employment_years=9, existing_emi_k=4, dependents=1)
BAD = dict(age=58, annual_income_lakh=3, loan_amount_lakh=40, term_years=25,
           credit_score=480, employment_years=1, existing_emi_k=12, dependents=5)


def test_health():
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"]["accuracy"] > 0.8


def test_good_applicant_scores_higher_than_bad():
    with TestClient(app) as client:
        good = client.post("/api/predict", json=GOOD).json()
        bad = client.post("/api/predict", json=BAD).json()
    assert good["probability"] > 0.7
    assert bad["probability"] < 0.3
    assert len(good["factors"]) > 0


def test_validation_rejects_bad_input():
    with TestClient(app) as client:
        r = client.post("/api/predict", json={**GOOD, "credit_score": 50})
    assert r.status_code == 422


def test_rates_endpoint_returns_all_loan_types():
    with TestClient(app) as client:
        r = client.get("/api/rates")
    assert r.status_code == 200
    data = r.json()
    assert len(data["rates"]) == 6
    assert all(rate["rate"] > 0 for rate in data["rates"])
