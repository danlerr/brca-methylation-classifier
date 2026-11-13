"""Loading of YAML configuration files.

A config may declare ``extends: other.yaml`` (path relative to itself); the parent is
loaded first and the child is deep-merged on top of it.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a config file, resolving ``extends`` recursively."""
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("extends", None)
    if parent is not None:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg
