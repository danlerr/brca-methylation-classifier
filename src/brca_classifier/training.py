"""Hyperparameter tuning and experiment execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, cross_validate

from .evaluation import evaluate
from .models import (
    build_base_model,
    build_stacking,
    make_cv,
    make_sampler,
    search_space,
    with_sampler,
)

log = logging.getLogger(__name__)

# Metrics computed on every cross-validation fold.
CV_SCORING = {
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


@dataclass
class ExperimentResult:
    experiment_id: str
    feature_set: str
    sampler: str
    model: str
    best_params: dict[str, Any]
    cv_scores: dict[str, float]
    test_metrics: dict[str, float]
    fitted_model: BaseEstimator = field(repr=False)
    fit_seconds: float = 0.0

    def to_row(self) -> dict[str, Any]:
        row = {
            "experiment_id": self.experiment_id,
            "feature_set": self.feature_set,
            "sampler": self.sampler,
            "model": self.model,
        }
        row.update({f"cv_{k}": v for k, v in self.cv_scores.items()})
        row.update({f"test_{k}": v for k, v in self.test_metrics.items()})
        row["fit_seconds"] = round(self.fit_seconds, 1)
        return row


def _cv_summary(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean and std of each CV metric from a cross_validate / cv_results_-style dict."""
    out = {}
    for metric in CV_SCORING:
        values = np.asarray(scores[metric])
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_std"] = float(values.std())
    return out


def tune_base_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    sampler,
    cfg: dict[str, Any],
) -> tuple[BaseEstimator, dict[str, Any], dict[str, float]]:
    """Randomized search for one base learner.

    The search optimises the metric configured for the model (e.g. recall for the SVC),
    while all metrics in ``CV_SCORING`` are recorded for the selected candidate.

    Returns the refitted pipeline, its best hyperparameters and its CV scores.
    """
    model_cfg = cfg["models"][name]
    seed = cfg["seed"]
    pipeline = with_sampler(build_base_model(name, model_cfg.get("fixed", {}), seed), sampler)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space(name, model_cfg["param_distributions"]),
        n_iter=model_cfg["n_iter"],
        scoring=CV_SCORING,
        refit=model_cfg["scoring"],
        cv=make_cv(cfg["cv"], seed),
        n_jobs=cfg["n_jobs"],
        random_state=seed,
    )
    search.fit(X, y)

    i = search.best_index_
    fold_scores = {
        m: [search.cv_results_[f"split{k}_test_{m}"][i] for k in range(cfg["cv"]["n_splits"])]
        for m in CV_SCORING
    }
    best_params = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    log.info("  %s best params: %s", name, best_params)
    return search.best_estimator_, best_params, _cv_summary(fold_scores)


def run_experiment(
    feature_set: str,
    sampler_name: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict[str, Any],
) -> list[ExperimentResult]:
    """Train, cross-validate and test one (feature set, sampler, model) combination.

    For ``stacking`` the tuned base learners are returned as separate results as well,
    since they come for free.
    """
    seed = cfg["seed"]
    resampling = cfg["resampling"]
    sampler = make_sampler(sampler_name, seed)

    if resampling == "before_cv" and sampler is not None:
        # Thesis behaviour: oversample the whole training set once, then cross-validate.
        X_fit, y_fit = sampler.fit_resample(X_train, y_train)
        pipeline_sampler = None
    elif resampling in ("inside_cv", "before_cv"):
        X_fit, y_fit = X_train, y_train
        pipeline_sampler = sampler
    else:
        raise ValueError(f"Unknown resampling mode '{resampling}'.")

    def result(model: str, fitted, params, cv_scores, seconds) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=f"{feature_set}__{sampler_name}__{model}",
            feature_set=feature_set,
            sampler=sampler_name,
            model=model,
            best_params=params,
            cv_scores=cv_scores,
            test_metrics=evaluate(fitted, X_test, y_test, cfg, seed),
            fitted_model=fitted,
            fit_seconds=seconds,
        )

    if model_name == "stacking":
        base_names = cfg["models"]["stacking"]["base_models"]
    else:
        base_names = [model_name]

    results, tuned = [], []
    for name in base_names:
        start = time.perf_counter()
        fitted, params, cv_scores = tune_base_model(name, X_fit, y_fit, pipeline_sampler, cfg)
        results.append(result(name, fitted, params, cv_scores, time.perf_counter() - start))
        # Stacking re-fits its base learners, so pass the bare tuned model (no sampler).
        tuned.append((name, fitted.named_steps["model"]))

    if model_name != "stacking":
        return results

    start = time.perf_counter()
    cv = make_cv(cfg["cv"], seed)
    stacking = with_sampler(
        build_stacking(tuned, cfg["models"]["stacking"], cv, cfg["n_jobs"]), pipeline_sampler
    )
    log.info("  cross-validating stacking...")
    # Outer folds run sequentially; parallelism happens inside the stacking, which fits its
    # internal CV folds and base learners in parallel (measured to be faster than the opposite).
    cv_raw = cross_validate(stacking, X_fit, y_fit, cv=cv, scoring=CV_SCORING, n_jobs=1)
    cv_scores = _cv_summary({m: cv_raw[f"test_{m}"] for m in CV_SCORING})
    log.info("  fitting stacking on the full training set...")
    stacking.fit(X_fit, y_fit)
    params = {f"{r.model}__{k}": v for r in results for k, v in r.best_params.items()}
    results.append(result("stacking", stacking, params, cv_scores, time.perf_counter() - start))
    return results
