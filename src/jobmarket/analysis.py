import numpy as np
import pandas as pd
import os
from .config import ROOT, TASKS
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports/.mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from .data import load_jobs, write_json

def analyze():
    j = load_jobs()
    sns.set_theme(style="whitegrid", palette="deep")
    def save(name):
        plt.tight_layout()
        plt.savefig(ROOT / f"reports/figures/{name}.png", dpi=160)
        plt.close()
    plt.figure(figsize=(9, 5))
    sns.histplot(j, x="salary_mid_usd", hue="experience_level", bins=50, element="step", stat="density", common_norm=False)
    plt.title("Advertised salary midpoint by experience — dataset only")
    save("salary_distribution")
    plt.figure(figsize=(9, 5))
    sns.boxplot(j, x="country", y="salary_mid_usd", hue="experience_level", showfliers=False)
    plt.title("Salary structure is dominated by experience")
    save("country_salary")
    annual = j.groupby(["posted_year", "remote_type"]).size().unstack(fill_value=0)
    annual.div(annual.sum(axis=1), axis=0).plot(figsize=(9, 5), marker="o")
    plt.title("Work arrangement share by posting year")
    plt.ylabel("Fraction of records")
    save("remote_trends")
    skills = pd.read_csv(ROOT / "data/raw/skills_demand.csv")
    skills.skill.value_counts().sort_values().plot.barh(figsize=(9, 5))
    plt.title("Skills demand: all supplied skill rows, including unmatched IDs")
    save("skills_demand")
    plt.figure(figsize=(8, 5))
    sns.heatmap(j[["salary_min_usd", "salary_max_usd", "salary_mid_usd", "salary_width_usd", "min_experience_years", "posted_year"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Numeric correlations include known target proxies")
    save("correlations")
    for cols in [["posted_year"], ["country"], ["job_title"], ["experience_level"], ["country", "experience_level"]]:
        j.groupby(cols).agg(jobs=("job_id", "size"), median_salary=("salary_mid_usd", "median"), mean_salary=("salary_mid_usd", "mean"), remote_share=("remote_type", lambda x: x.eq("Remote").mean())).to_csv(ROOT / f"reports/eda_{'_'.join(cols)}.csv")
    reference, current = j[j.posted_year.le(2024)], j[j.posted_year.eq(2026)]
    drift = []
    for c in j:
        if c in ["job_id", "posted_year"]:
            continue
        if pd.api.types.is_numeric_dtype(j[c]):
            stat = ks_2samp(reference[c], current[c])
            edges = np.unique(np.quantile(reference[c], np.linspace(0, 1, 11)))
            edges[0], edges[-1] = -np.inf, np.inf
            p, q = np.histogram(reference[c], edges)[0], np.histogram(current[c], edges)[0]
            p, q = np.maximum(p/p.sum(), 1e-6), np.maximum(q/q.sum(), 1e-6)
            psi = float(np.sum((q-p)*np.log(q/p)))
            drift.append({"feature": c, "kind": "numeric", "ks_stat": float(stat.statistic), "p_value": float(stat.pvalue), "psi": psi})
        else:
            levels = sorted(j[c].unique())
            p = reference[c].value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
            q = current[c].value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
            drift.append({"feature": c, "kind": "categorical", "total_variation": float(.5*np.abs(p-q).sum()), "unseen_levels": [x for x in current[c].unique() if x not in reference[c].unique()]})
    write_json(ROOT / "reports/drift.json", {"reference": "2020–2024", "current": "2026", "features": drift, "warning": "Year labels alone do not prove real historical collection. Distribution stability is not proof of external validity."})
    write_json(ROOT / "reports/leakage_audit.json", {name: {"target": t.target, "features": list(t.features), "excluded": list(t.excluded), "preprocessing": "Fit inside each training fold only", "aggregates": "Country trends excluded", "id_policy": "job_id used for split integrity only"} for name, t in TASKS.items()})

def report():
    quality = __import__("json").loads((ROOT / "reports/data_quality.json").read_text())
    registry = __import__("json").loads((ROOT / "models/registry.json").read_text())
    lines = ["# Final project report", "", "Dataset: Mann Raval (mann14), Kaggle version 3, CC BY 4.0. Educational findings; external market validity is unverified.", "", "## Data quality", "", f"{quality['jobs']:,} jobs; {quality['skills_rows']:,} skill rows. Only {quality['matched_jobs']:,} jobs ({quality['matched_job_fraction']:.1%}) have exact matched skills. {quality['orphan_skill_rows']:,} skill rows are quarantined. All joins enforce their cardinality. Raw files are unchanged.", "", "`min_experience_years` perfectly identifies experience; `city == Remote` perfectly identifies Remote postings. Both are excluded from the corresponding tasks. Salary endpoints never enter salary or seniority predictors. `employment_type` is constant. Country aggregate counts match none of the 42 observed country/year groups.", "", "## Temporal results", "", "| Task | CV-selected model | Main holdout metric | 95% bootstrap interval | Assessment |", "|---|---|---:|---|---|"]
    for name, modes in registry.items():
        e = modes["temporal"]
        ci = e["bootstrap_95"]
        lines.append(f"| {name} | {e['family']} | {ci['metric']} = {e['metrics'][ci['metric']]:.4f} | {ci['low']:.4f}–{ci['high']:.4f} | {e['status']} |")
    lines += ["", "## Strong salary baselines", "", "| Task | Learned/selected MAE USD | Train-only experience median MAE USD |", "|---|---:|---:|"]
    for name in ["salary", "salary_width"]:
        e = registry[name]["temporal"]
        lines.append(f"| {name} | {e['metrics']['mae']:.2f} | {e['domain_baseline']['metrics']['mae']:.2f} |")
    lines += ["", "Salary midpoint's apparent success against a global median is almost entirely explained by known seniority. The learned model provides no material gain over the experience-group median. None of the five tasks passes the final improvement gate; negative results are retained rather than presented as useful production ML."]
    lines += ["", "## Validation and model selection", "", "Random holdout measures IID performance; temporal holdout trains through 2024 and tests 2026. Regression intervals calibrate on 2025; random intervals use a separate 16% calibration partition. Search uses at most 6,000 training records with three folds; temporal folds always train on earlier years. Models refit on the full training partition. Test records never select model families, hyperparameters, thresholds, or preprocessing. Promotion requires >1% relative regression improvement over both global and experience-group median baselines, or >0.02 classification/multilabel F1 improvement in CV and frozen holdout; skills are always exploratory due to join coverage. The stronger grouped baseline audit leaves learned models frozen and does not retune them. These are educational gates, not production certification.", "", "Optuna performs a bounded AutoML search across linear, random forest, and (regression) gradient boosting families plus family-specific hyperparameters. Trial histories and a persistent SQLite study store are included. This is a lightweight transparent AutoML implementation, not a claim that AutoGluon/FLAML was run. Search uncertainty and IID bootstrap assumptions limit inference.", "", "## NLP and extra targets", "", "Job title is the only text field and has six distinct values. Word/bigram TF-IDF is compared to categorical title encoding; rich description embeddings, LLM-generated descriptions and skill-name text inputs are rejected as unsupported or leaky. Salary width is an additional valid derived target, with both salary endpoints withheld. Role category is a deterministic title mapping; employment type is constant; country-trend forecasting has only seven years per country and unreconciled aggregates. Those targets are rejected.", "", "## Explainability, error analysis and drift", "", "Permutation importance, worst-error records, errors by country/title/experience, confusion matrices, probability calibration bins and per-skill F1 are stored under reports/experiments. Feature importance describes fitted association, not causal effects. reports/drift.json compares 2020–2024 with 2026 using KS, train-quantile PSI and categorical total variation; inspect effect sizes rather than p-values alone. Distribution stability cannot validate synthetic-looking or otherwise unverified provenance.", "", "## Limits and next steps", "", "Do not use these outputs for hiring decisions, salary negotiation or claims about actual global demand. Experience-driven salary separation and uniform-looking marginal counts are consistent with constructed data, but do not by themselves prove synthetic origin. Obtain verified posting dates and job descriptions, repair source relationships with the publisher, then validate on an independent corpus. Skills availability is unknown for 95.9% of jobs; absent skill rows cannot be interpreted as no skill requirements. Intervals rely on exchangeability and observed temporal coverage, not guaranteed future coverage.", "", "## Reproduction", "", "See README.md. Automated tests cover FK integrity, leakage contracts, chronological folds, model artifact round trips, malformed API requests, missing models and actual prediction endpoints. Run scripts/quality_audit.py to inspect project containment and artifact integrity. Raw data, processed tables and model binaries stay local by default; source, reports, notebooks and metadata are versioned."]
    (ROOT / "reports/final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
