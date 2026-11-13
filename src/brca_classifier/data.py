"""Dataset loading, target encoding and feature-set selection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

# Methylation features are named "<chromosome>:<arm>:i<index>", e.g. "17:q:i3".
METHYLATION_PATTERN = re.compile(r"^\d+:[pq]:i\d+$")

FEATURE_SETS = ("methylation_aberration", "methylation", "aberration")


def load_dataset(data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Read the raw CSV and return features ``X`` and binary target ``y`` (1 = BRCA)."""
    path = Path(data_cfg["raw_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. See data/README.md for how to obtain it."
        )

    df = pd.read_csv(path)
    df = df.drop(columns=data_cfg.get("drop_columns", []), errors="ignore")

    target = data_cfg["target_column"]
    y = (df[target] == data_cfg["positive_label"]).astype(int).rename("target")
    X = df.drop(columns=[data_cfg["id_column"], target])
    return X, y


def methylation_columns(columns: pd.Index) -> list[str]:
    return [c for c in columns if METHYLATION_PATTERN.match(c)]


def select_feature_set(X: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return the columns of ``X`` belonging to the requested feature set."""
    meth = methylation_columns(X.columns)
    if name == "methylation_aberration":
        return X
    if name == "methylation":
        return X[meth]
    if name == "aberration":
        return X.drop(columns=meth)
    raise ValueError(f"Unknown feature set '{name}'. Choose from {FEATURE_SETS}.")


def split(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    The split is done once on the full feature matrix, and feature sets are selected
    afterwards, so every experiment is evaluated on exactly the same test patients.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
