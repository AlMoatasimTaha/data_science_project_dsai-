from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
BASE = ["job_title", "company_type", "industry", "country", "company_size", "posted_year"]

@dataclass(frozen=True)
class Task:
    name: str
    target: str
    kind: str
    features: tuple[str, ...]
    excluded: tuple[str, ...]
    description: str

TASKS = {
    "salary": Task("salary", "salary_mid_usd", "regression", tuple(BASE + ["experience_level", "remote_type"]),
                   ("salary_min_usd", "salary_max_usd", "salary_width_usd", "job_id", "city", "min_experience_years"),
                   "Estimate advertised annual salary midpoint from known job attributes."),
    "experience": Task("experience", "experience_level", "classification", tuple(BASE + ["remote_type"]),
                   ("experience_level", "min_experience_years", "salary_min_usd", "salary_max_usd", "salary_mid_usd", "salary_width_usd", "job_id", "city"),
                   "Infer advertised seniority before experience years or salary are known."),
    "remote": Task("remote", "remote_type", "classification", tuple(BASE + ["experience_level"]),
                   ("remote_type", "city", "job_id", "salary_min_usd", "salary_max_usd", "salary_mid_usd", "salary_width_usd", "min_experience_years"),
                   "Infer Remote/Hybrid/Onsite from non-location posting attributes."),
    "skills": Task("skills", "skills", "multilabel", tuple(BASE + ["experience_level", "remote_type"]),
                   ("job_id", "skill", "skill_category", "skill_level", "city", "min_experience_years", "salary_min_usd", "salary_max_usd"),
                   "Estimate skill demand on the exact-key matched subset; exploratory only."),
    "salary_width": Task("salary_width", "salary_width_usd", "regression", tuple(BASE + ["experience_level", "remote_type"]),
                   ("salary_min_usd", "salary_max_usd", "salary_mid_usd", "salary_width_usd", "job_id", "city", "min_experience_years"),
                   "Discover whether salary range width is predictable before endpoints are known."),
}

def ensure_dirs():
    for p in ["data/raw", "data/interim", "data/processed", "models", "reports/figures", "reports/experiments", "docs", "notebooks", "app", "api", "tests", "scripts", ".github/workflows"]:
        (ROOT / p).mkdir(parents=True, exist_ok=True)
