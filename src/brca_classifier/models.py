"""Model, sampler and pipeline factories."""

from __future__ import annotations

from typing import Any

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

SAMPLERS = ("none", "smote", "adasyn")
BASE_MODELS = ("svc", "xgb", "rf")

# Prefix that tunable hyperparameters need inside the estimator returned by build_base_model.
_PARAM_PREFIX = {"svc": "clf__", "xgb": "", "rf": ""}


def make_sampler(name: str, seed: int):
    if name == "none":
        return None
    if name == "smote":
        return SMOTE(random_state=seed)
    if name == "adasyn":
        return ADASYN(random_state=seed)
    raise ValueError(f"Unknown sampler '{name}'. Choose from {SAMPLERS}.")


def make_cv(cv_cfg: dict[str, Any], seed: int) -> StratifiedKFold:
    shuffle = cv_cfg.get("shuffle", False)
    return StratifiedKFold(
        n_splits=cv_cfg["n_splits"], shuffle=shuffle, random_state=seed if shuffle else None
    )


def build_base_model(name: str, params: dict[str, Any], seed: int) -> BaseEstimator:
    """Instantiate a base learner with fixed parameters ``params``."""
    if name == "svc":
        # Linear SVMs need standardised inputs; tree models do not.
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", LinearSVC(random_state=seed, **params))]
        )
    if name == "xgb":
        return XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1, **params)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    raise ValueError(f"Unknown model '{name}'. Choose from {BASE_MODELS}.")


def search_space(name: str, param_distributions: dict[str, list]) -> dict[str, list]:
    """Map config hyperparameter names to the names used by ``with_sampler`` pipelines."""
    prefix = "model__" + _PARAM_PREFIX[name]
    return {prefix + k: v for k, v in param_distributions.items()}


def build_stacking(
    base_estimators: list[tuple[str, BaseEstimator]],
    stacking_cfg: dict[str, Any],
    cv: StratifiedKFold,
    n_jobs: int,
) -> StackingClassifier:
    meta = XGBClassifier(eval_metric="logloss", n_jobs=1, **stacking_cfg["meta_model"])
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=cv,
        passthrough=stacking_cfg.get("passthrough", True),
        n_jobs=n_jobs,
    )


def with_sampler(estimator: BaseEstimator, sampler) -> Pipeline:
    """Wrap ``estimator`` in an imbalanced-learn pipeline.

    Samplers in an imblearn ``Pipeline`` run only during ``fit``: during cross-validation
    synthetic samples are generated from the training folds only, and prediction is
    always performed on real data.
    """
    steps = [("sampler", sampler)] if sampler is not None else []
    return Pipeline([*steps, ("model", estimator)])
