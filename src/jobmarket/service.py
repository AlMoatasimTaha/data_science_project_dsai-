from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from .config import ROOT, TASKS
from .modeling import probabilities

@lru_cache(maxsize=10)
def load_model(task_name, split="temporal"):
    if task_name not in TASKS or split not in ["temporal", "random"]:
        raise ValueError("Unknown task or split")
    path = ROOT / f"models/{task_name}/{split}.joblib"
    if not path.exists():
        raise FileNotFoundError("Model not trained. Run job-market train.")
    return joblib.load(path)

def predict(task_name, posting):
    task = TASKS.get(task_name)
    if task is None:
        raise ValueError("Unknown prediction task")
    missing = [f for f in task.features if posting.get(f) is None]
    if missing:
        raise ValueError("Required posting fields: " + ", ".join(missing))
    bundle = load_model(task_name)
    model, meta = bundle["pipeline"], bundle["metadata"]
    X = pd.DataFrame([{f: posting[f] for f in task.features}])
    result = {"task": task_name, "model": meta["family"], "status": meta["status"], "warning": "Educational dataset with unverified market provenance; predictions describe this dataset only.", "validation": meta["metrics"]}
    pred = model.predict(X)
    if task.kind == "regression":
        result["prediction_usd"] = float(pred[0])
        q = meta.get("interval_half_width")
        result["interval_90_usd"] = [max(0, float(pred[0]-q)), float(pred[0]+q)] if q is not None else None
        result["interval_note"] = "Calibrated on 2025; nominal 90% requires exchangeability. This estimates advertised values, not personal compensation."
    elif task.kind == "classification":
        result["prediction"] = str(pred[0])
        probs = probabilities(model, X, task.kind)[0]
        result["probabilities"] = {str(c): float(p) for c, p in zip(model.named_steps["model"].classes_, probs)}
    else:
        probs = probabilities(model, X, task.kind)[0]
        result["skills"] = [label for label, chosen in zip(meta["labels"], pred[0]) if chosen]
        result["probabilities"] = {label: float(p) for label, p in zip(meta["labels"], probs)}
        result["warning"] += " Only 4.1% of jobs have matched skill records; predictions are exploratory."
    if not meta["promotion_passed"]:
        result["warning"] += " Model did not pass the baseline improvement gate."
    return result
