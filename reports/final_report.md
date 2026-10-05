# Final project report

Dataset: Mann Raval (mann14), Kaggle version 3, CC BY 4.0. Educational findings; external market validity is unverified.

## Data quality

50,000 jobs; 224,605 skill rows. Only 2,050 jobs (4.1%) have exact matched skills. 215,345 skill rows are quarantined. All joins enforce their cardinality. Raw files are unchanged.

`min_experience_years` perfectly identifies experience; `city == Remote` perfectly identifies Remote postings. Both are excluded from the corresponding tasks. Salary endpoints never enter salary or seniority predictors. `employment_type` is constant. Country aggregate counts match none of the 42 observed country/year groups.

## Temporal results

| Task | CV-selected model | Main holdout metric | 95% bootstrap interval | Assessment |
|---|---|---:|---|---|
| salary | automl_forest | mae = 3563.9551 | 3514.3073–3614.9279 | experimental_no_reliable_lift |
| experience | automl_forest | f1_macro = 0.3330 | 0.3209–0.3436 | experimental_no_reliable_lift |
| remote | forest | f1_macro = 0.3401 | 0.3281–0.3494 | experimental_no_reliable_lift |
| skills | baseline | f1_micro = 0.4473 | 0.4255–0.4667 | experimental_no_reliable_lift |
| salary_width | baseline | mae = 7019.0237 | 6907.6460–7132.6094 | experimental_no_reliable_lift |

## Strong salary baselines

| Task | Learned/selected MAE USD | Train-only experience median MAE USD |
|---|---:|---:|
| salary | 3563.96 | 3564.88 |
| salary_width | 7019.02 | 7020.76 |

Salary midpoint's apparent success against a global median is almost entirely explained by known seniority. The learned model provides no material gain over the experience-group median. None of the five tasks passes the final improvement gate; negative results are retained rather than presented as useful production ML.

## Validation and model selection

Random holdout measures IID performance; temporal holdout trains through 2024 and tests 2026. Regression intervals calibrate on 2025; random intervals use a separate 16% calibration partition. Search uses at most 6,000 training records with three folds; temporal folds always train on earlier years. Models refit on the full training partition. Test records never select model families, hyperparameters, thresholds, or preprocessing. Promotion requires >1% relative regression improvement over both global and experience-group median baselines, or >0.02 classification/multilabel F1 improvement in CV and frozen holdout; skills are always exploratory due to join coverage. The stronger grouped baseline audit leaves learned models frozen and does not retune them. These are educational gates, not production certification.

Optuna performs a bounded AutoML search across linear, random forest, and (regression) gradient boosting families plus family-specific hyperparameters. Trial histories and a persistent SQLite study store are included. This is a lightweight transparent AutoML implementation, not a claim that AutoGluon/FLAML was run. Search uncertainty and IID bootstrap assumptions limit inference.

## NLP and extra targets

Job title is the only text field and has six distinct values. Word/bigram TF-IDF is compared to categorical title encoding; rich description embeddings, LLM-generated descriptions and skill-name text inputs are rejected as unsupported or leaky. Salary width is an additional valid derived target, with both salary endpoints withheld. Role category is a deterministic title mapping; employment type is constant; country-trend forecasting has only seven years per country and unreconciled aggregates. Those targets are rejected.

## Explainability, error analysis and drift

Permutation importance, worst-error records, errors by country/title/experience, confusion matrices, probability calibration bins and per-skill F1 are stored under reports/experiments. Feature importance describes fitted association, not causal effects. reports/drift.json compares 2020–2024 with 2026 using KS, train-quantile PSI and categorical total variation; inspect effect sizes rather than p-values alone. Distribution stability cannot validate synthetic-looking or otherwise unverified provenance.

## Limits and next steps

Do not use these outputs for hiring decisions, salary negotiation or claims about actual global demand. Experience-driven salary separation and uniform-looking marginal counts are consistent with constructed data, but do not by themselves prove synthetic origin. Obtain verified posting dates and job descriptions, repair source relationships with the publisher, then validate on an independent corpus. Skills availability is unknown for 95.9% of jobs; absent skill rows cannot be interpreted as no skill requirements. Intervals rely on exchangeability and observed temporal coverage, not guaranteed future coverage.

## Reproduction

See README.md. Automated tests cover FK integrity, leakage contracts, chronological folds, model artifact round trips, malformed API requests, missing models and actual prediction endpoints. Run scripts/quality_audit.py to inspect project containment and artifact integrity. Raw data, processed tables and model binaries stay local by default; source, reports, notebooks and metadata are versioned.
