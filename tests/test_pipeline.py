"""Fast tests on synthetic data (no real dataset needed)."""

import numpy as np
import pandas as pd
import pytest

from brca_classifier.config import load_config
from brca_classifier.data import select_feature_set, split
from brca_classifier.models import make_sampler, with_sampler
from brca_classifier.training import run_experiment


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n = 300
    y = pd.Series((rng.random(n) < 0.15).astype(int), name="target")
    meth = {f"{c}:{a}:i{i}": rng.normal(y * 1.5, 1) for c in (1, 2) for a in "pq" for i in (1, 2)}
    aber = {f"{c}p_del": rng.integers(0, 2, n) for c in range(1, 5)}
    return pd.DataFrame({**meth, **aber}), y


@pytest.fixture
def cfg():
    cfg = load_config("configs/smoke.yaml")
    cfg["n_jobs"] = 1
    return cfg


def test_feature_sets_partition_columns(synthetic):
    X, _ = synthetic
    m = select_feature_set(X, "methylation")
    a = select_feature_set(X, "aberration")
    assert len(m.columns) == 8 and len(a.columns) == 4
    assert set(m.columns) | set(a.columns) == set(X.columns)
    assert not set(m.columns) & set(a.columns)


def test_split_is_disjoint_and_stratified(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = split(X, y, test_size=0.2, seed=42)
    assert not set(X_tr.index) & set(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03


def test_sampler_not_applied_at_prediction(synthetic):
    """Oversampling must change the training data but never the data we predict on."""
    from sklearn.dummy import DummyClassifier

    X, y = synthetic
    pipe = with_sampler(DummyClassifier(), make_sampler("smote", 0)).fit(X, y)
    assert len(pipe.predict(X)) == len(X)


def test_stacking_experiment_runs(synthetic, cfg):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = split(X, y, test_size=0.3, seed=42)
    results = run_experiment(
        "methylation_aberration", "smote", "stacking", X_tr, y_tr, X_te, y_te, cfg
    )
    assert [r.model for r in results] == ["svc", "xgb", "stacking"]
    for r in results:
        assert 0 <= r.test_metrics["balanced_accuracy"] <= 1
        assert r.test_metrics["tp"] + r.test_metrics["fn"] == y_te.sum()
