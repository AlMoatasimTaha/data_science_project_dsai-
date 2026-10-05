import joblib
import numpy as np
import pytest
from jobmarket.config import ROOT, TASKS
from jobmarket.data import load_jobs
from jobmarket.modeling import pipeline, task_data, split_indices, cv_folds

@pytest.mark.parametrize("name", list(TASKS))
def test_leakage_allowlists(name):
    t = TASKS[name]
    assert not set(t.features) & set(t.excluded)
    assert t.target not in t.features
    assert "job_id" not in t.features
    assert "salary_min_usd" not in t.features and "salary_max_usd" not in t.features
    if name == "experience":
        assert "min_experience_years" not in t.features
    if name == "remote":
        assert "city" not in t.features

@pytest.mark.parametrize("name", list(TASKS))
def test_partitions_and_chronological_folds(name):
    t = TASKS[name]
    frame, y, _ = task_data(load_jobs(), t)
    for mode in ["random", "temporal"]:
        train, cal, test = split_indices(frame, y, t, mode)
        assert not set(train) & set(test)
        assert not set(cal) & (set(train) | set(test))
        assert frame.iloc[train].job_id.is_unique
        if mode == "temporal":
            assert frame.iloc[train].posted_year.max() <= 2024
            assert frame.iloc[test].posted_year.eq(2026).all()
            sub = frame.iloc[train].reset_index(drop=True)
            for a, b in cv_folds(sub, y[train], t, mode):
                assert sub.iloc[a].posted_year.max() < sub.iloc[b].posted_year.min()

def test_preprocessing_fits_train_only_and_unknown_category(tmp_path):
    t = TASKS["salary"]
    j = load_jobs().head(400)
    X = j[list(t.features)].copy()
    X.loc[X.index[-1], "country"] = "NeverSeen"
    model = pipeline(t, "linear").fit(X.iloc[:-1], j.salary_mid_usd.iloc[:-1])
    categories = model.named_steps["preprocess"].named_transformers_["categories"].named_steps["encode"].categories_
    assert not any("NeverSeen" in group for group in categories)
    pred = model.predict(X.iloc[-1:])
    assert np.isfinite(pred).all()
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(pred, joblib.load(path).predict(X.iloc[-1:]))

@pytest.mark.parametrize("name", list(TASKS))
def test_trained_artifact_shapes(name):
    path = ROOT / f"models/{name}/temporal.joblib"
    if not path.exists():
        pytest.skip("Integration requires model training")
    bundle = joblib.load(path)
    t = TASKS[name]
    frame, y, labels = task_data(load_jobs(), t)
    prediction = bundle["pipeline"].predict(frame[list(t.features)].head(3))
    assert len(prediction) == 3
    assert bundle["metadata"]["features"] == list(t.features)
    if name == "skills":
        assert prediction.shape == (3, 11)
