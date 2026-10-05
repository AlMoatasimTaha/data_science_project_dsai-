"""Train-only preprocessing, model selection and honest holdout evaluation."""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                            balanced_accuracy_score, f1_score, hamming_loss, average_precision_score,
                            classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer

from .config import ROOT, TASKS, SEED
from .data import load_jobs, write_json

class SkillPrior(ClassifierMixin, BaseEstimator):
    """Predict train-derived top-k skills, k = rounded train mean cardinality."""
    def fit(self, X, y):
        self.prevalence_ = np.asarray(y).mean(axis=0)
        self.k_ = max(1, int(round(np.asarray(y).sum(axis=1).mean())))
        self.classes_ = [np.array([0, 1]) for _ in self.prevalence_]
        return self

    def predict(self, X):
        pred = np.zeros((len(X), len(self.prevalence_)), dtype=int)
        pred[:, np.argsort(self.prevalence_)[-self.k_:]] = 1
        return pred

    def predict_proba(self, X):
        return [np.tile([1-p, p], (len(X), 1)) for p in self.prevalence_]

def pipeline(task, family, params=None, nlp=False):
    params = params or {}
    categorical = [c for c in task.features if c != "posted_year" and not (nlp and c == "job_title")]
    transformers = [
        ("categories", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ("year", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["posted_year"]),
    ]
    if nlp:
        transformers.append(("title_text", TfidfVectorizer(ngram_range=(1, 2), max_features=100), "job_title"))
    prep = ColumnTransformer(transformers, sparse_threshold=0)
    if task.kind == "regression":
        estimator = {"baseline": lambda: DummyRegressor(strategy="median"),
                     "linear": lambda: Ridge(alpha=params.get("alpha", 10)),
                     "forest": lambda: RandomForestRegressor(n_estimators=80, max_depth=params.get("max_depth", 10), min_samples_leaf=params.get("min_samples_leaf", 10), n_jobs=2, random_state=SEED),
                     "boosting": lambda: HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=params.get("max_leaf_nodes", 15), learning_rate=params.get("learning_rate", .1), random_state=SEED)}[family]()
    else:
        estimator = {"baseline": lambda: SkillPrior() if task.kind == "multilabel" else DummyClassifier(strategy="stratified", random_state=SEED),
                     "linear": lambda: LogisticRegression(C=params.get("C", 1), max_iter=500, random_state=SEED),
                     "forest": lambda: RandomForestClassifier(n_estimators=80, max_depth=params.get("max_depth", 10), min_samples_leaf=params.get("min_samples_leaf", 10), n_jobs=2, random_state=SEED)}[family]()
        if task.kind == "multilabel" and family == "linear":
            estimator = MultiOutputClassifier(estimator, n_jobs=1)
    return Pipeline([("preprocess", prep), ("model", estimator)])

def task_data(j, task):
    labels = None
    if task.kind == "multilabel":
        s = pd.read_csv(ROOT / "data/processed/job_skills.csv")
        grouped = s.groupby("job_id").skill.agg(list)
        frame = j[j.job_id.isin(grouped.index)].copy().reset_index(drop=True)
        # Fixed vocabulary is the 11 skill names in the source dictionary; no test frequencies enter training.
        vocabulary = sorted(pd.read_csv(ROOT / "data/raw/skills_demand.csv").skill.unique())
        mlb = MultiLabelBinarizer(classes=vocabulary)
        y = mlb.fit_transform(grouped.loc[frame.job_id])
        labels = vocabulary
    else:
        frame = j.copy().reset_index(drop=True)
        y = frame[task.target].to_numpy()
    return frame, y, labels

def split_indices(frame, y, task, mode):
    idx = np.arange(len(frame))
    if mode == "temporal":
        train = idx[frame.posted_year.le(2024)]
        calibration = idx[frame.posted_year.eq(2025)] if task.kind == "regression" else np.array([], dtype=int)
        test = idx[frame.posted_year.eq(2026)]
        # 2025 is held out for all temporal tasks; not folded into selection.
    else:
        train, test = train_test_split(idx, test_size=.2, random_state=SEED, stratify=y if task.kind == "classification" else None)
        calibration = np.array([], dtype=int)
        if task.kind == "regression":
            train, calibration = train_test_split(train, test_size=.2, random_state=SEED)
    assert not (set(train) & set(test) or set(train) & set(calibration) or set(test) & set(calibration))
    return np.sort(train), np.sort(calibration), np.sort(test)

def cv_folds(frame, y, task, mode):
    if mode == "temporal":
        years = frame.posted_year.to_numpy()
        return [(np.flatnonzero(years < year), np.flatnonzero(years == year)) for year in [2022, 2023, 2024]]
    if task.kind == "classification":
        return list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(frame, y))
    return list(KFold(3, shuffle=True, random_state=SEED).split(frame))

def metric_values(task, y, pred):
    if task.kind == "regression":
        return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred))}
    if task.kind == "classification":
        return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0))}
    return {"f1_micro": float(f1_score(y, pred, average="micro", zero_division=0)), "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)), "hamming_loss": float(hamming_loss(y, pred)), "subset_accuracy": float(accuracy_score(y, pred))}

def probabilities(model, X, kind):
    values = model.predict_proba(X)
    if kind == "multilabel":
        classes = model.named_steps["model"].classes_
        return np.column_stack([p[:, list(c).index(1)] if 1 in c else np.zeros(len(X)) for p, c in zip(values, classes)])
    return values

def bootstrap_interval(task, y, pred):
    rng = np.random.default_rng(SEED)
    metric = "mae" if task.kind == "regression" else "f1_micro" if task.kind == "multilabel" else "f1_macro"
    values = []
    for _ in range(200):
        ix = rng.integers(0, len(y), size=len(y))
        values.append(metric_values(task, y[ix], pred[ix])[metric])
    return {"metric": metric, "low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975)), "replicates": 200, "method": "IID holdout row bootstrap; not uncertainty of market provenance"}

def run_training(trials=8, search_rows=6000, resume=False):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    j = load_jobs()
    comparison_path = ROOT / "reports/model_comparison.csv"
    registry_path = ROOT / "models/registry.json"
    all_rows = pd.read_csv(comparison_path).to_dict("records") if resume and comparison_path.exists() else []
    # Missing metrics are written as empty CSV cells; JSON metadata remains strict.
    registry = json.loads(registry_path.read_text()) if resume and registry_path.exists() else {}
    for name, task in TASKS.items():
        frame, y, labels = task_data(j, task)
        registry.setdefault(name, {})
        for mode in ["random", "temporal"]:
            if resume and mode in registry[name] and (ROOT / f"models/{name}/{mode}.joblib").exists():
                print(f"Reusing completed {name}/{mode}", flush=True)
                continue
            started = time.time()
            print(f"Training {name}/{mode}: {len(frame)} records", flush=True)
            train, cal, test = split_indices(frame, y, task, mode)
            X = frame[list(task.features)]
            # Limit search compute using train-only sample; final fit uses all available training rows.
            rng = np.random.default_rng(SEED)
            search = np.sort(rng.choice(train, min(search_rows, len(train)), replace=False))
            search_frame = frame.iloc[search].reset_index(drop=True)
            search_X = X.iloc[search].reset_index(drop=True)
            search_y = y[search]
            folds = cv_folds(search_frame, search_y, task, mode)
            scoring = "neg_mean_absolute_error" if task.kind == "regression" else "f1_micro" if task.kind == "multilabel" else "f1_macro"
            families = ["baseline", "linear", "forest"] + (["boosting"] if task.kind == "regression" else [])
            candidates = []
            for family in families + ["title_tfidf"]:
                model = pipeline(task, "linear" if family == "title_tfidf" else family, nlp=family == "title_tfidf")
                scores = cross_val_score(model, search_X, search_y, cv=folds, scoring=scoring, n_jobs=1)
                candidates.append({"family": family, "cv_score": float(scores.mean()), "cv_std": float(scores.std()), "params": {}, "nlp": family == "title_tfidf"})
            def objective(trial):
                family = trial.suggest_categorical("family", families[1:])
                params = {}
                if family == "linear":
                    key = "alpha" if task.kind == "regression" else "C"
                    params[key] = trial.suggest_float(key, .01, 100, log=True)
                elif family == "forest":
                    params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
                    params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 50)
                else:
                    params["learning_rate"] = trial.suggest_float("learning_rate", .03, .2, log=True)
                    params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 7, 31)
                model = pipeline(task, family, params)
                return float(cross_val_score(model, search_X, search_y, cv=folds, scoring=scoring, n_jobs=1).mean())
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED), storage=f"sqlite:///{(ROOT / 'reports/experiments/automl.sqlite').as_posix()}", study_name=f"{name}_{mode}_{time.time_ns()}")
            study.optimize(objective, n_trials=trials)
            study.trials_dataframe().to_csv(ROOT / f"reports/experiments/{name}_{mode}_automl.csv", index=False)
            best = dict(study.best_params)
            family = best.pop("family")
            candidates.append({"family": "automl_" + family, "cv_score": float(study.best_value), "cv_std": None, "params": best, "nlp": False})
            selected = max(candidates, key=lambda x: x["cv_score"])
            baseline_cv = candidates[0]["cv_score"]
            # Fit and evaluate every model family after selection is frozen. Never select by holdout metric.
            fitted = {}
            for candidate in candidates:
                family = candidate["family"]
                estimator_family = "linear" if candidate["nlp"] else family.removeprefix("automl_")
                model = pipeline(task, estimator_family, candidate["params"], candidate["nlp"])
                model.fit(X.iloc[train], y[train])
                pred = model.predict(X.iloc[test])
                metrics = metric_values(task, y[test], pred)
                all_rows.append({"task": name, "split": mode, "family": family, "selected_by_cv": family == selected["family"], "cv_score": candidate["cv_score"], "cv_std": candidate["cv_std"], "train_rows": len(train), "calibration_rows": len(cal), "test_rows": len(test), **metrics})
                fitted[family] = (model, pred, metrics)
            model, pred, metrics = fitted[selected["family"]]
            baseline_metrics = fitted["baseline"][2]
            metric = "mae" if task.kind == "regression" else "f1_micro" if task.kind == "multilabel" else "f1_macro"
            if task.kind == "regression":
                cv_pass = selected["cv_score"] - baseline_cv > abs(baseline_cv)*.01
                test_pass = metrics[metric] < baseline_metrics[metric]*.99
            else:
                cv_pass = selected["cv_score"] - baseline_cv > .02
                test_pass = metrics[metric] - baseline_metrics[metric] > .02
            release = bool(cv_pass and test_pass and name != "skills")
            # Promotion is an assessment after frozen selection, never an invitation to retune the holdout.
            interval = None
            if task.kind == "regression":
                residuals = np.abs(y[cal] - model.predict(X.iloc[cal]))
                rank = min(1, np.ceil((len(residuals)+1)*.9)/len(residuals))
                interval = float(np.quantile(residuals, rank, method="higher"))
                metrics["interval_coverage_90"] = float((np.abs(y[test]-pred) <= interval).mean())
                metrics["interval_half_width"] = interval
            if task.kind == "multilabel":
                probs = probabilities(model, X.iloc[test], task.kind)
                metrics["average_precision_micro"] = float(average_precision_score(y[test], probs, average="micro"))
                pd.DataFrame({"skill": labels, "f1": [f1_score(y[test, i], pred[:, i], zero_division=0) for i in range(len(labels))], "test_prevalence": y[test].mean(axis=0)}).to_csv(ROOT / f"reports/experiments/{name}_{mode}_labels.csv", index=False)
            elif task.kind == "classification":
                write_json(ROOT / f"reports/experiments/{name}_{mode}_classification.json", classification_report(y[test], pred, output_dict=True, zero_division=0))
                classes = sorted(np.unique(y))
                pd.DataFrame(confusion_matrix(y[test], pred, labels=classes), index=classes, columns=classes).to_csv(ROOT / f"reports/experiments/{name}_{mode}_confusion.csv")
                probs = probabilities(model, X.iloc[test], task.kind)
                clf_classes = list(model.named_steps["model"].classes_)
                onehot = np.column_stack([y[test] == c for c in clf_classes]).astype(int)
                metrics["multiclass_brier"] = float(np.mean(np.sum((probs-onehot)**2, axis=1)))
                calibration = pd.DataFrame({"confidence": probs.max(axis=1), "correct": pred == y[test]})
                calibration["bin"] = pd.cut(calibration.confidence, bins=np.linspace(0, 1, 11), include_lowest=True)
                calibration.groupby("bin", observed=True).agg(rows=("correct", "size"), accuracy=("correct", "mean"), confidence=("confidence", "mean")).to_csv(ROOT / f"reports/experiments/{name}_{mode}_calibration.csv")
            # Permutation explanation on held-out rows; no feedback into selection.
            explain_idx = test[:min(1000, len(test))]
            imp = permutation_importance(model, X.iloc[explain_idx], y[explain_idx], scoring=scoring, n_repeats=3, random_state=SEED, n_jobs=1)
            pd.DataFrame({"feature": task.features, "importance": imp.importances_mean, "std": imp.importances_std}).sort_values("importance", ascending=False).to_csv(ROOT / f"reports/experiments/{name}_{mode}_importance.csv", index=False)
            errors = frame.iloc[test][["job_id", "country", "job_title", "experience_level", "posted_year"]].copy()
            errors["loss"] = np.abs(y[test]-pred) if task.kind == "regression" else np.mean(y[test] != pred, axis=1) if task.kind == "multilabel" else (y[test] != pred).astype(float)
            if task.kind != "multilabel":
                errors["actual"], errors["prediction"] = y[test], pred
            errors.sort_values("loss", ascending=False).to_csv(ROOT / f"reports/experiments/{name}_{mode}_errors.csv", index=False)
            for group in ["country", "job_title", "experience_level"]:
                errors.groupby(group).loss.agg(["count", "mean", "median"]).to_csv(ROOT / f"reports/experiments/{name}_{mode}_error_by_{group}.csv")
            split = pd.DataFrame({"job_id": frame.job_id, "partition": "unused"})
            split.loc[train, "partition"] = "train"
            split.loc[cal, "partition"] = "calibration"
            split.loc[test, "partition"] = "test"
            split.to_csv(ROOT / f"reports/experiments/{name}_{mode}_split.csv", index=False)
            entry = {"task": name, "split": mode, "family": selected["family"], "features": list(task.features), "excluded": list(task.excluded), "labels": labels,
                     "metrics": metrics, "bootstrap_95": bootstrap_interval(task, y[test], pred), "cv_score": selected["cv_score"], "baseline_cv": baseline_cv,
                     "baseline_test": baseline_metrics, "promotion_passed": release, "status": "educational_demo" if release else "experimental_no_reliable_lift",
                     "interval_half_width": interval, "train_rows": len(train), "test_rows": len(test), "search_rows": len(search), "trials": trials,
                     "fit_years": sorted(map(int, frame.iloc[train].posted_year.unique())), "threshold": .5, "elapsed_seconds": round(time.time()-started, 2)}
            path = ROOT / f"models/{name}/{mode}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"pipeline": model, "metadata": entry}, path)
            write_json(path.with_suffix(".json"), entry)
            registry[name][mode] = entry
            pd.DataFrame(all_rows).to_csv(ROOT / "reports/model_comparison.csv", index=False)
            write_json(ROOT / "models/registry.json", registry)
            print(f"Finished {name}/{mode}: {selected['family']}, {metric}={metrics[metric]:.4f}, promotion={release}", flush=True)
    return registry
