"""Strong, train-only domain baseline audit for advertised salary tasks.

Models remain frozen. This comparator cannot retune or select a learned model.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from .config import ROOT, TASKS, SEED
from .data import load_jobs, write_json
from .modeling import task_data, split_indices, cv_folds, metric_values

def group_prediction(frame_train, y_train, frame_test):
    medians = pd.DataFrame({"experience_level": frame_train.experience_level.to_numpy(), "target": y_train}).groupby("experience_level").target.median()
    return frame_test.experience_level.map(medians).fillna(float(np.median(y_train))).to_numpy()

def audit_domain_baselines():
    path = ROOT / "models/registry.json"
    registry = json.loads(path.read_text())
    jobs = load_jobs()
    comparison = pd.read_csv(ROOT / "reports/model_comparison.csv")
    comparison = comparison[comparison.family.ne("experience_median_baseline")]
    extra = []
    for name in ["salary", "salary_width"]:
        t = TASKS[name]
        frame, y, _ = task_data(jobs, t)
        for mode, meta in registry[name].items():
            train, cal, test = split_indices(frame, y, t, mode)
            rng = np.random.default_rng(SEED)
            ix = np.sort(rng.choice(train, meta["search_rows"], replace=False))
            sf = frame.iloc[ix].reset_index(drop=True)
            sy = y[ix]
            scores = [-mean_absolute_error(sy[b], group_prediction(sf.iloc[a], sy[a], sf.iloc[b])) for a, b in cv_folds(sf, sy, t, mode)]
            cv_score = float(np.mean(scores))
            pred = group_prediction(frame.iloc[train], y[train], frame.iloc[test])
            metrics = metric_values(t, y[test], pred)
            meta["domain_baseline"] = {"family": "experience_median_baseline", "cv_score": cv_score, "metrics": metrics, "method": "Training-only median advertised target by known experience; no test feedback to model selection"}
            meta.setdefault("global_baseline_promotion_passed", meta["promotion_passed"])
            strong_cv = max(meta["baseline_cv"], cv_score)
            strong_mae = min(meta["baseline_test"]["mae"], metrics["mae"])
            meta["promotion_passed"] = bool(meta["cv_score"]-strong_cv > abs(strong_cv)*.01 and meta["metrics"]["mae"] < strong_mae*.99)
            meta["status"] = "educational_demo" if meta["promotion_passed"] else "experimental_no_reliable_lift"
            model_path = ROOT / f"models/{name}/{mode}.joblib"
            bundle = joblib.load(model_path)
            bundle["metadata"] = meta
            joblib.dump(bundle, model_path)
            write_json(model_path.with_suffix(".json"), meta)
            extra.append({"task": name, "split": mode, "family": "experience_median_baseline", "selected_by_cv": False, "cv_score": cv_score, "cv_std": float(np.std(scores)), "train_rows": len(train), "calibration_rows": len(cal), "test_rows": len(test), **metrics})
    pd.concat([comparison, pd.DataFrame(extra)], ignore_index=True).to_csv(ROOT / "reports/model_comparison.csv", index=False)
    write_json(path, registry)
    return registry
