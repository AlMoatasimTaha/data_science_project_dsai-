# Reproducibility

Python 3.12 is the validated interpreter. Install `requirements.lock.txt` for exact package versions, then install this source as an editable package without downloading further dependencies. `requirements.txt` defines broad compatible ranges; the lock records the executed environment. `scripts/bootstrap_wheels.py` is a fallback for broken pip HTTP transport: it resolves compatible wheels from official PyPI, verifies their SHA-256 hashes, installs into the current interpreter, and writes the lock. It is not a complete general-purpose resolver and stops on conflicts.

Seed 42 is shared by random partitions, estimators, Optuna and bootstrap sampling. Three training-only folds and eight Optuna trials per task/split are the default. Search limits computation to 6,000 randomly sampled training rows; final models fit all training rows. Parallel forests use two workers. Small numeric differences across BLAS libraries or platforms remain possible.

`data/manifest.json` records immutable input checksums, row counts, byte lengths, dataset version and retrieval time. Split CSVs retain original IDs under `reports/experiments`. All preprocessing and estimators serialize together. Model metadata records the precise feature contract, fitted years, baseline results, search budget and promotion decision.

SQLite Optuna studies accumulate reruns under unique study names; task-specific CSVs, reports and model artifacts replace the previous run. Back up `reports/` and `models/` inside the root before comparing separate executions. Applications cache model files; restart them after training.

Generated data/model binaries, per-record errors and partition files are ignored by Git. Summary evidence, trial CSVs, plots, notebooks, source, metadata and documentation are versioned. The full-training workflow exports models and reports as a GitHub Actions artifact for 14 days. Do not load untrusted joblib files.
