import hashlib
import json
import sqlite3
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd
from .config import ROOT, ensure_dirs

DATASET = "mann14/global-ai-and-data-science-job-market-20202026"
FILES = ["ai_jobs.csv", "skills_demand.csv", "country_ai_trends.csv", "job_title_mapping.csv", "data_dictionary.csv"]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")

def download():
    ensure_dirs()
    if all((ROOT / "data/raw" / f).exists() for f in FILES):
        return
    archive = ROOT / "data/raw/kaggle_download.zip"
    try:
        urllib.request.urlretrieve(f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}", archive)
    except Exception as exc:
        raise RuntimeError("Public Kaggle download failed. Authenticate the Kaggle CLI and run: kaggle datasets download -d " + DATASET + " -p data/raw --unzip") from exc
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (ROOT / "data/raw" / member.filename).resolve()
            if not target.is_relative_to((ROOT / "data/raw").resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(ROOT / "data/raw")

def load_raw():
    return {f.removesuffix(".csv"): pd.read_csv(ROOT / "data/raw" / f) for f in FILES}

def validate_jobs(j):
    required = {"job_id", "job_title", "country", "posted_year", "experience_level", "remote_type", "salary_min_usd", "salary_max_usd"}
    if not required.issubset(j.columns):
        raise ValueError(f"Missing columns: {required - set(j.columns)}")
    if j.job_id.isna().any() or j.job_id.duplicated().any():
        raise ValueError("Job primary key must be unique and non-null")
    if (j.salary_min_usd <= 0).any() or (j.salary_max_usd < j.salary_min_usd).any():
        raise ValueError("Invalid salary endpoints")
    if not j.posted_year.between(2020, 2026).all():
        raise ValueError("Unexpected posting year")
    for col, values in [("experience_level", {"Entry", "Mid", "Senior"}), ("remote_type", {"Remote", "Hybrid", "Onsite"})]:
        if not set(j[col].dropna()).issubset(values):
            raise ValueError(f"Unexpected {col} labels")

def prepare():
    ensure_dirs()
    download()
    raw = load_raw()
    j, s, t, m = [raw[k] for k in ["ai_jobs", "skills_demand", "country_ai_trends", "job_title_mapping"]]
    validate_jobs(j)
    if m.job_title.duplicated().any() or t.duplicated(["country", "year"]).any():
        raise ValueError("Dimension keys must be unique")
    matched = s[s.job_id.isin(j.job_id)].copy()
    orphans = s[~s.job_id.isin(j.job_id)].copy()
    if s.duplicated(["job_id", "skill"]).any():
        raise ValueError("Duplicate job-skill keys")
    j = j.merge(m, on="job_title", how="left", validate="many_to_one")
    if j.standardized_title.isna().any():
        raise ValueError("Unmapped job titles")
    j["salary_mid_usd"] = (j.salary_min_usd + j.salary_max_usd) / 2
    j["salary_width_usd"] = j.salary_max_usd - j.salary_min_usd
    j.to_csv(ROOT / "data/processed/jobs.csv", index=False)
    matched.to_csv(ROOT / "data/processed/job_skills.csv", index=False)
    orphans.to_csv(ROOT / "data/interim/orphan_skills.csv", index=False)
    with sqlite3.connect(ROOT / "data/processed/jobmarket.sqlite") as con:
        con.execute("PRAGMA foreign_keys=ON")
        for table in ["job_skills", "jobs", "role_mapping", "country_trends", "orphan_skills"]:
            con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute("CREATE TABLE role_mapping(job_title TEXT PRIMARY KEY, standardized_title TEXT NOT NULL, role_category TEXT NOT NULL)")
        m.to_sql("role_mapping", con, if_exists="append", index=False)
        columns = ", ".join(f'"{c}" {"REAL" if pd.api.types.is_numeric_dtype(j[c]) else "TEXT"}' for c in j.columns if c != "job_id")
        con.execute(f'CREATE TABLE jobs(job_id TEXT PRIMARY KEY, {columns}, FOREIGN KEY(job_title) REFERENCES role_mapping(job_title))')
        j.to_sql("jobs", con, if_exists="append", index=False)
        con.execute("CREATE TABLE job_skills(job_id TEXT NOT NULL, skill TEXT NOT NULL, skill_category TEXT, skill_level TEXT, PRIMARY KEY(job_id,skill), FOREIGN KEY(job_id) REFERENCES jobs(job_id))")
        matched.to_sql("job_skills", con, if_exists="append", index=False)
        con.execute("CREATE TABLE country_trends(country TEXT, year INTEGER, total_ai_jobs INTEGER, avg_salary_usd REAL, remote_percentage REAL, top_skill TEXT, PRIMARY KEY(country,year))")
        t.to_sql("country_trends", con, if_exists="append", index=False)
        orphans.to_sql("orphan_skills", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX idx_jobs_country_year ON jobs(country,posted_year)")
        assert not con.execute("PRAGMA foreign_key_check").fetchall()
    audit = {
        "jobs": len(j), "skills_rows": len(s), "unique_skills_job_ids": int(s.job_id.nunique()),
        "matched_skill_rows": len(matched), "matched_jobs": int(matched.job_id.nunique()),
        "matched_job_fraction": float(matched.job_id.nunique() / len(j)), "orphan_skill_rows": len(orphans),
        "missing_values": {k: v.isna().sum().to_dict() for k, v in raw.items()},
        "duplicates": {k: int(v.duplicated().sum()) for k, v in raw.items()},
        "experience_years_mapping": pd.crosstab(j.experience_level, j.min_experience_years).to_dict(),
        "city_remote_proxy": pd.crosstab(j.remote_type, j.city.eq("Remote")).to_dict(),
        "constant_columns": [c for c in j if j[c].nunique() <= 1],
        "year_counts": j.posted_year.value_counts().sort_index().to_dict(),
        "joined_skill_coverage_by_year": j.assign(has_skills=j.job_id.isin(matched.job_id)).groupby("posted_year").has_skills.mean().to_dict(),
    }
    observed = j.groupby(["country", "posted_year"]).agg(observed_jobs=("job_id", "size"), observed_salary=("salary_mid_usd", "mean"), observed_remote=("remote_type", lambda x: x.eq("Remote").mean()*100)).reset_index()
    reconciled = t.merge(observed, left_on=["country", "year"], right_on=["country", "posted_year"], validate="one_to_one")
    reconciled.to_csv(ROOT / "reports/trends_reconciliation.csv", index=False)
    audit["trends_count_matches"] = int(reconciled.total_ai_jobs.eq(reconciled.observed_jobs).sum())
    write_json(ROOT / "reports/data_quality.json", audit)
    manifest = {"dataset": DATASET, "version": 3, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "license": "CC BY 4.0", "files": {f: {"sha256": hashlib.sha256((ROOT / "data/raw" / f).read_bytes()).hexdigest(), "bytes": (ROOT / "data/raw" / f).stat().st_size, "rows": len(raw[f.removesuffix('.csv')])} for f in FILES}}
    write_json(ROOT / "data/manifest.json", manifest)
    dictionary = []
    source_desc = raw["data_dictionary"].set_index(["file", "column"]).description.to_dict()
    for name, frame in raw.items():
        for c in frame:
            dictionary.append({"table": name, "column": c, "dtype": str(frame[c].dtype), "nulls": int(frame[c].isna().sum()), "unique": int(frame[c].nunique()), "description": source_desc.get((name + ".csv", c), "See docs/data_dictionary.md for semantics"), "examples": "; ".join(map(str, frame[c].dropna().unique()[:5]))})
    pd.DataFrame(dictionary).to_csv(ROOT / "reports/data_dictionary.csv", index=False)
    return audit

def load_jobs():
    path = ROOT / "data/processed/jobs.csv"
    if not path.exists():
        raise FileNotFoundError("Run job-market prepare first")
    return pd.read_csv(path)
