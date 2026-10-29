import os
import time
import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import model

STATIC = Path(__file__).parent / "static"
VERSION = os.getenv("GIT_SHA") or os.getenv("RENDER_GIT_COMMIT") or "dev"
BUILT_AT = os.getenv("BUILD_TIME", "local")

# In-memory user store (replace with a real DB in production)
USERS: dict[str, dict] = {}

# In-memory prediction history per user (keyed by username)
HISTORY: dict[str, list] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model.load()
    yield


app = FastAPI(
    title="LoanLens",
    description="ML loan-approval predictor with a full CI/CD pipeline",
    version=VERSION[:7],
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ── Models ────────────────────────────────────────────────────────────────────

class Applicant(BaseModel):
    age: int = Field(ge=18, le=70)
    annual_income_lakh: float = Field(gt=0, le=200)
    loan_amount_lakh: float = Field(gt=0, le=500)
    term_years: int = Field(ge=1, le=30)
    credit_score: int = Field(ge=300, le=900)
    employment_years: float = Field(ge=0, le=50)
    existing_emi_k: float = Field(ge=0, le=500)
    dependents: int = Field(ge=0, le=10)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    password: str = Field(min_length=6)
    full_name: str = Field(min_length=2, max_length=60)
    email: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


# ── Auth API ──────────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    if req.username in USERS:
        raise HTTPException(status_code=409, detail="Username already taken")
    USERS[req.username] = {
        "username": req.username,
        "password": req.password,   # hash in production!
        "full_name": req.full_name,
        "email": req.email,
        "joined": time.strftime("%Y-%m-%d", time.gmtime()),
        "avatar": f"https://api.dicebear.com/7.x/initials/svg?seed={req.full_name}",
    }
    HISTORY[req.username] = []
    return {"ok": True, "username": req.username, "full_name": req.full_name,
            "email": req.email, "joined": USERS[req.username]["joined"],
            "avatar": USERS[req.username]["avatar"]}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = USERS.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"ok": True, "username": user["username"], "full_name": user["full_name"],
            "email": user["email"], "joined": user["joined"], "avatar": user["avatar"]}


# ── Prediction API ────────────────────────────────────────────────────────────

@app.post("/api/predict")
def predict(a: Applicant, username: str = ""):
    result = model.predict(a.model_dump())
    # Save to history if user is logged in
    if username and username in HISTORY:
        HISTORY[username].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            "inputs": a.model_dump(),
            "probability": result["probability"],
            "verdict": result["verdict"],
            "tier": result["tier"],
            "monthly_emi": result["monthly_emi"],
        })
        if len(HISTORY[username]) > 20:
            HISTORY[username] = HISTORY[username][-20:]
    return result


@app.get("/api/history/{username}")
def get_history(username: str):
    if username not in USERS:
        raise HTTPException(status_code=404, detail="User not found")
    return {"history": list(reversed(HISTORY.get(username, [])))}


# ── Loan Rates API ────────────────────────────────────────────────────────────

@app.get("/api/rates")
def loan_rates():
    """Current indicative loan interest rates (simulated market data)."""
    base = {
        "home_loan": 8.50,
        "personal_loan": 10.75,
        "car_loan": 9.10,
        "education_loan": 8.15,
        "business_loan": 11.50,
        "gold_loan": 7.80,
    }
    # Simulate slight market fluctuation (±0.15%) each call
    rng = random.Random(int(time.time() // 3600))   # changes hourly
    rates = []
    icons = {"home_loan": "🏠", "personal_loan": "👤", "car_loan": "🚗",
             "education_loan": "🎓", "business_loan": "💼", "gold_loan": "🥇"}
    labels = {"home_loan": "Home Loan", "personal_loan": "Personal Loan",
              "car_loan": "Car Loan", "education_loan": "Education Loan",
              "business_loan": "Business Loan", "gold_loan": "Gold Loan"}
    for k, v in base.items():
        fluctuation = round(rng.uniform(-0.15, 0.15), 2)
        current = round(v + fluctuation, 2)
        change = round(fluctuation, 2)
        rates.append({
            "type": k,
            "label": labels[k],
            "icon": icons[k],
            "rate": current,
            "change": change,
            "trend": "up" if change > 0 else "down" if change < 0 else "flat",
        })
    return {
        "rates": rates,
        "as_of": time.strftime("%d %b %Y, %H:%M IST", time.gmtime()),
        "note": "Indicative rates. Actual rates may vary by bank and profile.",
    }


# ── Insights API ──────────────────────────────────────────────────────────────

@app.get("/api/insights")
def insights():
    """Aggregate model + market insights for the Insights page."""
    _, metrics = model.load()

    # Simulated approval trend over last 7 days
    rng = random.Random(42)
    trend = []
    import datetime
    for i in range(7, 0, -1):
        day = (datetime.date.today() - datetime.timedelta(days=i)).strftime("%d %b")
        trend.append({"day": day, "approval_rate": round(rng.uniform(0.52, 0.71), 3)})

    # Credit score buckets (simulated distribution)
    score_buckets = [
        {"range": "300–500", "approved": 8, "declined": 92},
        {"range": "500–600", "approved": 22, "declined": 78},
        {"range": "600–700", "approved": 51, "declined": 49},
        {"range": "700–800", "approved": 78, "declined": 22},
        {"range": "800–900", "approved": 94, "declined": 6},
    ]

    return {
        "model": metrics,
        "approval_trend": trend,
        "score_buckets": score_buckets,
        "top_factors": [
            {"label": "Debt burden", "importance": 0.38},
            {"label": "Credit score", "importance": 0.31},
            {"label": "Loan-to-income", "importance": 0.14},
            {"label": "Employment", "importance": 0.09},
            {"label": "Dependents", "importance": 0.05},
            {"label": "Age + term", "importance": 0.03},
        ],
    }


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    _, metrics = model.load()
    return {"status": "ok", "version": VERSION[:7], "built_at": BUILT_AT, "model": metrics}
