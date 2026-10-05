# Project decision log

2026-10-05 — All project work stays inside `data_science_project_dsai/`. Runtime executables supplied by the desktop app are external tooling; the environment, cached wheels and all outputs are inside the root.

2026-10-05 — Public Kaggle API download succeeded through Python urllib without credentials. Preserve five original CSV files and generate a hash manifest. Kaggle metadata identifies version 3 and CC BY 4.0. Avoid redistributing raw data or large binary artifacts in Git; provide attribution, download code and exact training commands.

2026-10-05 — Reject an unconstrained merge of skills and jobs: only 2,050 of 50,000 jobs match. Quarantine 215,345 orphan skill rows. Do not align records by row order or infer labels for unmatched jobs.

2026-10-05 — Exclude exact experience-years and remote-city proxies. Keep country aggregates separate: none of 42 total counts matches job-level grouping. Reject employment type, deterministic role-category prediction and unsupported description NLP.

2026-10-05 — Train salary midpoint, experience, remote, multilabel skills and salary width. Baselines are median salary, seeded stratified classification and train-prevalence top-k skills. A stratified baseline prevents flattering random classifiers by comparing only with a majority baseline whose macro F1 is artificially low.

2026-10-05 — Use scikit-learn pipelines and bounded Optuna model-family search. Preprocessing is fit inside each fold. Search at most 6,000 train rows to keep execution practical; all selected estimators fit on the full training partition. Preserve trials, model comparisons and split assignments.

2026-10-05 — Select by training CV, freeze selection, then evaluate all candidates on holdout for transparent comparison. Temporal train years end in 2024, 2025 calibrates regression intervals, 2026 is untouched during selection. Random regression uses independent training/calibration/test partitions.

2026-10-05 — Do not label constructed-looking data as definitively synthetic without publisher proof. Unverified provenance, simple deterministic structure and broken relationships remain prominent in results and UI.

2026-10-05 — Publishing authorized by user to `AlMoatasimTaha/data_science_project_dsai-`; preserve the trailing hyphen. Initialize Git only at the project root. Use a public GitHub noreply email for commits.

2026-10-05 — Audit salary performance against a stronger train-only experience-group median after the initial global-median comparisons. Keep the fitted models and hyperparameters frozen. Require lift over this domain baseline for promotion; a strong global-median score alone does not establish value from sophisticated ML. Retain both baseline comparisons in the published evidence.
