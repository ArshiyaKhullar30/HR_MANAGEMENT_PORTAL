"""Config access (Phase 0.5).

One source of truth: ``conf/config.yaml``. Nothing under ``src/`` or ``app/``
hard-codes a path, threshold or seed — they are read from here.

Usage::

    from hrai.utils.config import get_config, raw_path

    cfg = get_config()
    df = pd.read_csv(raw_path("employee_attrition"))
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = "conf/config.yaml"


def project_root() -> Path:
    """Repo root, resolved from this file rather than the working directory.

    Notebooks run from ``notebooks/`` and the API runs from the root; neither
    should need to know where it is.
    """
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def get_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load and cache ``conf/config.yaml``.

    Override the location with the ``HRAI_CONFIG`` environment variable.
    """
    cfg_path = Path(path or os.getenv("HRAI_CONFIG", _DEFAULT_CONFIG))
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get(dotted_key: str, default: Any = None) -> Any:
    """Read a nested config value by dotted path, e.g. ``"model.cv_folds"``."""
    node: Any = get_config()
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def data_dir(kind: str) -> Path:
    """Absolute path to a configured data directory (raw/interim/processed/...)."""
    rel = get(f"paths.data.{kind}")
    if rel is None:
        raise KeyError(f"Unknown data directory: {kind!r}")
    return project_root() / rel


def raw_path(dataset: str) -> Path:
    """Absolute path to one of the five permitted source datasets.

    Raises for any name not in the config allow-list, which is how the brief's
    scope rule ("only the datasets in enterprise_hr_ai/data/") is enforced in
    code rather than by convention.
    """
    datasets = get("datasets", {})
    if dataset not in datasets:
        raise KeyError(
            f"{dataset!r} is not a permitted source dataset. " f"Allowed: {sorted(datasets)}"
        )
    return data_dir("raw") / datasets[dataset]["file"]


def dataset_meta(dataset: str) -> dict[str, Any]:
    """Declared metadata for a source dataset (key, grain, encoding, label...)."""
    datasets = get("datasets", {})
    if dataset not in datasets:
        raise KeyError(f"Unknown dataset: {dataset!r}")
    return datasets[dataset]


def seed() -> int:
    """The single project seed (Phase 0.6 determinism contract)."""
    return int(get("random_seed", 42))
