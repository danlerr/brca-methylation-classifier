"""Model-agnostic SHAP explanations (permutation explainer)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from .evaluation import predict_scores  # noqa: E402

log = logging.getLogger(__name__)


def compute_shap(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    seed: int,
) -> shap.Explanation:
    """SHAP values of the positive-class score.

    The permutation explainer only needs a prediction function, so it works with any
    model (including the stacking ensemble). Its cost grows with
    ``len(X_background) * len(X_explain) * n_features``.
    """
    columns = X_background.columns

    def score(a: np.ndarray) -> np.ndarray:
        return predict_scores(model, pd.DataFrame(a, columns=columns))

    explainer = shap.PermutationExplainer(
        score, shap.maskers.Independent(X_background, max_samples=len(X_background)), seed=seed
    )
    return explainer(X_explain, max_evals=2 * X_explain.shape[1] + 1, silent=False)


def importance_table(explanation: shap.Explanation) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": explanation.feature_names,
                "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def save_plots(explanation: shap.Explanation, top_n: int, title: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(explanation, max_display=top_n, show=False)
    plt.title(title)
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    shap.plots.bar(explanation, max_display=top_n, show=False)
    plt.title(title)
    plt.savefig(out_dir / "shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close("all")


def explain_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: dict[str, Any],
    title: str,
    out_dir: Path,
) -> pd.DataFrame:
    shap_cfg = cfg["shap"]
    seed = cfg["seed"]
    background = X_train.sample(n=min(shap_cfg["background_size"], len(X_train)), random_state=seed)
    n_explain = shap_cfg.get("explain_size") or len(X_test)
    X_explain = X_test.sample(n=min(n_explain, len(X_test)), random_state=seed)

    log.info(
        "Computing SHAP: %d background rows, %d explained rows, %d features",
        len(background),
        len(X_explain),
        X_explain.shape[1],
    )
    explanation = compute_shap(model, background, X_explain, seed)

    table = importance_table(explanation)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "shap_importance.csv", index=False)
    save_plots(explanation, shap_cfg["top_n"], title, out_dir)
    return table
