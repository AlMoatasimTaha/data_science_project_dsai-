"""Run from project root: streamlit run app/streamlit_app.py"""
import json
import pandas as pd
import streamlit as st
from jobmarket.config import ROOT, TASKS
from jobmarket.data import load_jobs
from jobmarket.service import predict

st.set_page_config(page_title="AI Job Market Lab", page_icon="📊", layout="wide")
st.title("AI Job Market Lab")
st.caption("Explore the data, inspect validation evidence, and try five audited prediction tasks.")
st.warning("Educational dataset • Market provenance is unverified • Skills link to only 4.1% of jobs")

@st.cache_data
def jobs():
    return load_jobs()

if not (ROOT / "data/processed/jobs.csv").exists():
    st.info("Prepare the dataset first: job-market prepare")
    st.stop()

j = jobs()
explore, predict_tab, evidence = st.tabs(["Explore", "Predict", "Validation & quality"])
with explore:
    countries = st.multiselect("Countries", sorted(j.country.unique()), default=sorted(j.country.unique()))
    years = st.slider("Posting years", 2020, 2026, (2020, 2026))
    subset = j[j.country.isin(countries) & j.posted_year.between(*years)]
    a, b, c = st.columns(3)
    a.metric("Job records", f"{len(subset):,}")
    b.metric("Median salary midpoint", f"${subset.salary_mid_usd.median():,.0f}" if len(subset) else "—")
    c.metric("Remote share", f"{subset.remote_type.eq('Remote').mean():.1%}" if len(subset) else "—")
    left, right = st.columns(2)
    with left:
        st.subheader("Salary by experience")
        st.bar_chart(subset.groupby("experience_level").salary_mid_usd.median())
    with right:
        st.subheader("Work arrangement by year")
        counts = subset.groupby(["posted_year", "remote_type"]).size().unstack(fill_value=0)
        st.line_chart(counts.div(counts.sum(axis=1), axis=0))
    st.subheader("Posting sample")
    st.dataframe(subset.head(200), hide_index=True, width="stretch")
    st.download_button("Download filtered records", subset.to_csv(index=False), "filtered_jobs.csv", "text/csv")
    st.caption("Country-trend totals are supplied independently and do not reconcile with these job records.")
with predict_tab:
    name = st.selectbox("Prediction task", list(TASKS), format_func=lambda x: x.replace("_", " ").title())
    task = TASKS[name]
    st.write(task.description)
    with st.form("posting"):
        posting = {}
        cols = st.columns(2)
        for i, field in enumerate(task.features):
            with cols[i % 2]:
                if field == "posted_year":
                    posting[field] = st.number_input("Posting year", 2020, 2026, 2026)
                else:
                    posting[field] = st.selectbox(field.replace("_", " ").title(), sorted(j[field].unique()))
        submitted = st.form_submit_button("Run prediction")
    if submitted:
        try:
            result = predict(name, posting)
            st.warning(result["warning"])
            if "prediction_usd" in result:
                st.metric("Estimated advertised value", f"${result['prediction_usd']:,.0f}")
                interval = result["interval_90_usd"]
                st.write(f"Nominal 90% interval: \\${interval[0]:,.0f} – \\${interval[1]:,.0f}")
                st.caption(result["interval_note"])
            elif "prediction" in result:
                st.metric("Predicted class", result["prediction"])
            else:
                st.write("Predicted skills:", ", ".join(result["skills"]) or "None above the decision threshold")
            if "probabilities" in result:
                st.bar_chart(pd.Series(result["probabilities"], name="Probability"))
            with st.expander("Model and holdout evidence"):
                st.json({k: v for k, v in result.items() if k not in ["probabilities", "warning"]})
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
with evidence:
    st.subheader("Baseline comparisons")
    comparison = ROOT / "reports/model_comparison.csv"
    if comparison.exists():
        st.dataframe(pd.read_csv(comparison), hide_index=True, width="stretch")
    st.write("Models are selected inside training folds. Temporal training ends in 2024; 2026 is held out. Salary intervals use 2025 calibration records.")
    for filename, title in [("data_quality.json", "Relational and data quality audit"), ("leakage_audit.json", "Feature contracts"), ("drift.json", "Distribution drift")]:
        path = ROOT / "reports" / filename
        if path.exists():
            with st.expander(title):
                st.json(json.loads(path.read_text(encoding="utf-8")))
