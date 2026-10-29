import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import model

STATIC = Path(__file__).parent / "static"
VERSION = os.getenv("GIT_SHA") or os.getenv("RENDER_GIT_COMMIT") or "dev"
BUILT_AT = os.getenv("BUILD_TIME", "local")


@asynccontextmanager
async def lifespan(_: FastAPI):
    model.load()  # warm up so the first request is fast
    yield


app = FastAPI(
    title="LoanLens",
    description="ML loan-approval predictor with a full CI/CD pipeline",
    version=VERSION[:7],
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Applicant(BaseModel):
    age: int = Field(ge=18, le=70)
    annual_income_lakh: float = Field(gt=0, le=200)
    loan_amount_lakh: float = Field(gt=0, le=500)
    term_years: int = Field(ge=1, le=30)
    credit_score: int = Field(ge=300, le=900)
    employment_years: float = Field(ge=0, le=50)
    existing_emi_k: float = Field(ge=0, le=500)
    dependents: int = Field(ge=0, le=10)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    _, metrics = model.load()
    return {"status": "ok", "version": VERSION[:7], "built_at": BUILT_AT, "model": metrics}


@app.post("/api/predict")
def predict(a: Applicant):
    return model.predict(a.model_dump())
