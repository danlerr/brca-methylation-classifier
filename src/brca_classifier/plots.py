"""Figures saved to disk (no interactive display)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .evaluation import predict_scores  # noqa: E402

DPI = 200
CLASS_NAMES = ["Other", "BRCA"]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def class_distribution(y: pd.Series, path: Path) -> Path:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(CLASS_NAMES, counts.values, color=["#8c8c8c", "#d6604d"])
    ax.bar_label(bars, labels=[f"{v} ({v / counts.sum():.1%})" for v in counts.values])
    ax.set_ylabel("Patients")
    ax.set_title("Class distribution")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def pca_projection(feature_sets: dict[str, pd.DataFrame], y: pd.Series, path: Path) -> Path:
    """2-D PCA of each (standardised) feature set, coloured by class."""
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(5 * len(feature_sets), 4.5))
    for ax, (name, X) in zip(np.atleast_1d(axes), feature_sets.items(), strict=True):
        pcs = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
        for label, color in [(0, "#8c8c8c"), (1, "#d6604d")]:
            mask = y.to_numpy() == label
            ax.scatter(*pcs[mask].T, s=6, alpha=0.5, c=color, label=CLASS_NAMES[label])
        ax.set_title(name.replace("_", " + "))
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    np.atleast_1d(axes)[0].legend(markerscale=3)
    return _save(fig, path)


def confusion(model, X_test, y_test, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=CLASS_NAMES, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    return _save(fig, path)


def roc_curves(models: dict[str, object], X_tests: dict[str, pd.DataFrame], y_test, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, model in models.items():
        RocCurveDisplay.from_predictions(
            y_test, predict_scores(model, X_tests[name]), name=name, ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC curves (test set)")
    return _save(fig, path)
