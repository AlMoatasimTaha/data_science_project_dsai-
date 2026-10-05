import json
import sqlite3
import pandas as pd
import pytest
from jobmarket.config import ROOT
from jobmarket.data import load_jobs, validate_jobs

@pytest.fixture
def jobs():
    return load_jobs()

def test_real_data_integrity(jobs):
    validate_jobs(jobs)
    assert len(jobs) == 50000
    assert jobs.job_id.is_unique
    assert jobs.salary_mid_usd.eq((jobs.salary_min_usd + jobs.salary_max_usd)/2).all()
    assert jobs.salary_width_usd.eq(jobs.salary_max_usd - jobs.salary_min_usd).all()

def test_invalid_endpoints_rejected(jobs):
    bad = jobs.head(5).copy()
    bad.loc[bad.index[0], "salary_max_usd"] = -1
    with pytest.raises(ValueError, match="salary endpoints"):
        validate_jobs(bad)

def test_duplicate_job_rejected(jobs):
    with pytest.raises(ValueError, match="primary key"):
        validate_jobs(pd.concat([jobs.head(2), jobs.head(1)]))

def test_skill_relationship_is_not_fabricated(jobs):
    skills = pd.read_csv(ROOT / "data/processed/job_skills.csv")
    orphan = pd.read_csv(ROOT / "data/interim/orphan_skills.csv")
    assert skills.job_id.isin(jobs.job_id).all()
    assert not orphan.job_id.isin(jobs.job_id).any()
    assert skills.job_id.nunique() == 2050
    assert len(skills) + len(orphan) == 224605

def test_database_constraints():
    with sqlite3.connect(ROOT / "data/processed/jobmarket.sqlite") as con:
        con.execute("PRAGMA foreign_keys=ON")
        assert not con.execute("PRAGMA foreign_key_check").fetchall()
        assert con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 50000
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO job_skills VALUES ('nonexistent','SQL','Programming','Basic')")

def test_raw_hash_manifest():
    import hashlib
    m = json.loads((ROOT / "data/manifest.json").read_text())
    for name, entry in m["files"].items():
        assert hashlib.sha256((ROOT / "data/raw" / name).read_bytes()).hexdigest() == entry["sha256"]
