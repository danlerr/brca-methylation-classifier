"""Test-set metrics with bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_scores(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """Continuous score for the positive class (probability if available)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric=balanced_accuracy_score,
    n_iterations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``metric`` on the test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def evaluate(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    y_true = np.asarray(y_test)
    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ci_lo, ci_hi = bootstrap_ci(
        y_true, y_pred, n_iterations=cfg["evaluation"]["bootstrap_iterations"], seed=seed
    )
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "balanced_accuracy_ci_low": ci_lo,
        "balanced_accuracy_ci_high": ci_hi,
        "roc_auc": roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
