"""Loan-approval ML model: data generation, training, loading and inference."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# The 8 inputs a user fills in on the web form
INPUT_FEATURES = [
    "age",
    "annual_income_lakh",
    "loan_amount_lakh",
    "term_years",
    "credit_score",
    "employment_years",
    "existing_emi_k",
    "dependents",
]

LABELS = {
    "age": "Age",
    "annual_income_lakh": "Annual income",
    "loan_amount_lakh": "Loan amount",
    "term_years": "Loan term",
    "credit_score": "Credit score",
    "employment_years": "Work experience",
    "existing_emi_k": "Existing EMIs",
    "dependents": "Dependents",
}


def emi_per_month(loan_lakh: float, years: float, annual_rate: float = 0.095) -> float:
    """Standard EMI formula, returns rupees per month."""
    p = loan_lakh * 1e5
    r = annual_rate / 12
    n = max(years, 0.5) * 12
    return p * r * (1 + r) ** n / ((1 + r) ** n - 1)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    monthly_income = out["annual_income_lakh"] * 1e5 / 12
    new_emi = out.apply(lambda r: emi_per_month(r["loan_amount_lakh"], r["term_years"]), axis=1)
    out["loan_to_income"] = out["loan_amount_lakh"] / out["annual_income_lakh"]
    out["debt_burden"] = (new_emi + out["existing_emi_k"] * 1000) / monthly_income
    return out


def generate_data(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Synthetic but realistic applicant data (swap with a real dataset if you have one)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    income = np.clip(rng.lognormal(2.2, 0.55, n), 2, 60)
    credit = np.clip(rng.normal(690, 85, n), 300, 900)
    exp = np.clip(age - 22 - rng.integers(0, 6, n), 0, 40)
    term = rng.integers(1, 31, n)
    loan = np.clip(income * rng.uniform(0.3, 6, n), 1, 150)
    emi_k = np.clip(income * 1e5 / 12 / 1000 * rng.uniform(0, 0.45, n), 0, 100)
    dep = rng.integers(0, 7, n)

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income_lakh": income.round(1),
            "loan_amount_lakh": loan.round(1),
            "term_years": term,
            "credit_score": credit.round(),
            "employment_years": exp,
            "existing_emi_k": emi_k.round(1),
            "dependents": dep,
        }
    )
    d = add_derived(df)
    logit = (
        0.9 * (df["credit_score"] - 650) / 100
        - 5.5 * (d["debt_burden"] - 0.38)
        - 0.25 * d["loan_to_income"]
        + 0.05 * np.minimum(df["employment_years"], 10)
        - 0.12 * df["dependents"]
        - 0.04 * np.maximum(df["age"] + df["term_years"] - 62, 0)
        + 1.9
        + rng.normal(0, 0.6, n)
    )
    df["approved"] = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    return df


def train(save: bool = True) -> dict:
    df = generate_data()
    X = add_derived(df[INPUT_FEATURES])
    y = df["approved"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

    clf = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.08, random_state=7)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)[:, 1]

    metrics = {
        "algorithm": "Gradient Boosting Classifier",
        "samples": int(len(df)),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "approval_rate": round(float(y.mean()), 4),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    bundle = {"model": clf, "medians": df[INPUT_FEATURES].median().to_dict(), "columns": list(X.columns)}
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


_cache: dict = {}


def load():
    if "bundle" not in _cache:
        if not MODEL_PATH.exists():
            train()
        _cache["bundle"] = joblib.load(MODEL_PATH)
        _cache["metrics"] = json.loads(METRICS_PATH.read_text())
    return _cache["bundle"], _cache["metrics"]


def _proba(bundle, rows: pd.DataFrame) -> np.ndarray:
    X = add_derived(rows[INPUT_FEATURES])[bundle["columns"]]
    return bundle["model"].predict_proba(X)[:, 1]


def predict(inputs: dict) -> dict:
    bundle, _ = load()
    row = pd.DataFrame([inputs])
    p = float(_proba(bundle, row)[0])

    # Explain: swap each feature with the typical (median) applicant's value and
    # see how the approval probability moves. Positive = this input helps you.
    factors = []
    for f in INPUT_FEATURES:
        alt = row.copy()
        alt[f] = bundle["medians"][f]
        delta = p - float(_proba(bundle, alt)[0])
        factors.append({"feature": f, "label": LABELS[f], "impact": round(delta, 4)})
    factors.sort(key=lambda x: abs(x["impact"]), reverse=True)

    emi = emi_per_month(inputs["loan_amount_lakh"], inputs["term_years"])
    income_m = inputs["annual_income_lakh"] * 1e5 / 12
    burden = (emi + inputs["existing_emi_k"] * 1000) / income_m

    if p >= 0.7:
        verdict, tier = "Likely to be approved", "high"
    elif p >= 0.4:
        verdict, tier = "Borderline, could go either way", "mid"
    else:
        verdict, tier = "Likely to be declined", "low"

    return {
        "probability": round(p, 4),
        "verdict": verdict,
        "tier": tier,
        "monthly_emi": round(emi),
        "debt_burden_pct": round(burden * 100, 1),
        "factors": factors[:6],
    }
