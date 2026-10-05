import json
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from jobmarket.config import ROOT, TASKS
from jobmarket.service import predict

app = FastAPI(title="AI Job Market Lab", version="1.0.0", description="Educational, leakage-audited predictions and validation evidence.")

class Posting(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_title: Literal["Data Scientist", "Machine Learning Engineer", "AI Researcher", "Data Analyst", "Applied Scientist", "MLOps Engineer"]
    company_type: Literal["Startup", "MNC", "Research Lab"]
    industry: Literal["Tech", "Finance", "Healthcare", "Education", "Retail"]
    country: Literal["USA", "India", "Germany", "UK", "Canada", "Australia"]
    company_size: Literal["Small", "Medium", "Large"]
    posted_year: int = Field(ge=2020, le=2026, strict=True)
    experience_level: Literal["Entry", "Mid", "Senior"] | None = None
    remote_type: Literal["Remote", "Hybrid", "Onsite"] | None = None

@app.get("/health")
def health():
    available = [name for name in TASKS if (ROOT / f"models/{name}/temporal.joblib").exists()]
    return {"status": "ok", "models_available": available, "ready": len(available) == len(TASKS)}

@app.get("/models")
def models():
    path = ROOT / "models/registry.json"
    if not path.exists():
        raise HTTPException(503, "Models not trained")
    return json.loads(path.read_text(encoding="utf-8"))

@app.post("/predict/{task_name}")
def prediction(task_name: Literal["salary", "experience", "remote", "skills", "salary_width"], posting: Posting):
    try:
        return predict(task_name, posting.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
