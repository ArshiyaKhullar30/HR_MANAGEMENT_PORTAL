"""SHAP explainability (Step 08).

"Employee 101 — 82% attrition risk" is useless to an HR person without "why".
SHAP answers it at two levels: globally (what drives attrition across the
company) and locally (why *this* person is flagged).

Two production details the Build Notes' snippet does not cover:

* **Which model to explain.** The deployed model is calibrated, and calibration
  is a monotonic transform of the score. It changes the number but not which
  features drove it, so SHAP runs against the uncalibrated base pipeline —
  which also avoids needing a slow model-agnostic explainer.
* **Where the cost goes.** Explaining inside a request would make the "why is
  this person flagged" endpoint unusably slow, so the explainer and its
  background sample are built once and cached per model version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from hrai.features.pipeline import fitted_feature_names
from hrai.ml.registry import load_model, models_dir
from hrai.utils.logger import get_logger

log = get_logger(__name__)

# One-hot names arrive as "cat__JobRole_Sales Executive"; strip the machinery so
# an HR user sees "JobRole: Sales Executive".
_PREFIXES = ("num__", "cat__", "remainder__")


def strip_prefix(feature_name: str) -> str:
    for prefix in _PREFIXES:
        if feature_name.startswith(prefix):
            return feature_name[len(prefix) :]
    return feature_name


def humanise(feature_name: str) -> str:
    name = strip_prefix(feature_name)
    if "_" in name:
        head, _, tail = name.partition("_")
        if tail and not tail[0].isdigit():
            return f"{head}: {tail}"
    return name


@dataclass
class Contribution:
    feature: str
    label: str
    value: Any
    shap_value: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": self.value,
            "shap_value": round(self.shap_value, 5),
            "direction": self.direction,
        }


class AttritionExplainer:
    """Wraps a fitted pipeline with a SHAP explainer over its encoded matrix."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame) -> None:
        self.pipeline = pipeline
        self.preprocess = Pipeline(pipeline.steps[:-1])
        self.model = pipeline.named_steps["clf"]
        self.feature_names = fitted_feature_names(pipeline)
        self.background_encoded = self.preprocess.transform(background)
        self.source_features = self._source_feature_map()
        self.explainer = self._build()

    def _source_feature_map(self) -> dict[str, str]:
        """Map each encoded column back to the original column it came from.

        One-hot encoding splits `OverTime` into `OverTime_Yes` and `OverTime_No`,
        which are two halves of one fact. Showing them as separate drivers is
        confusing to an HR reader and double-counts the feature's importance, so
        attributions are summed back onto the source column.
        """
        prep = self.pipeline.named_steps["prep"]
        originals: list[str] = []
        self.categorical_sources: set[str] = set()
        for name, _transformer, columns in prep.transformers_:
            if isinstance(columns, list):
                originals.extend(columns)
                if name == "cat":
                    self.categorical_sources.update(columns)
        # Longest match first, so `Years` never captures `YearsAtCompany`.
        originals.sort(key=len, reverse=True)

        mapping: dict[str, str] = {}
        for encoded in self.feature_names:
            bare = strip_prefix(encoded)
            mapping[encoded] = next(
                (col for col in originals if bare == col or bare.startswith(f"{col}_")),
                bare,
            )
        return mapping

    def _build(self):
        name = type(self.model).__name__
        if name in {
            "XGBClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "DecisionTreeClassifier",
            "LGBMClassifier",
        }:
            log.info("using TreeExplainer", extra={"model": name})
            return shap.TreeExplainer(self.model)
        if name in {"LogisticRegression", "LinearSVC", "RidgeClassifier"}:
            log.info("using LinearExplainer", extra={"model": name})
            return shap.LinearExplainer(self.model, self.background_encoded)
        log.warning("falling back to model-agnostic explainer (slow)", extra={"model": name})
        return shap.Explainer(self.model.predict_proba, self.background_encoded)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values for the positive class, shaped (n_rows, n_features)."""
        encoded = self.preprocess.transform(X)
        values = self.explainer.shap_values(encoded)
        if isinstance(values, list):  # some explainers return one array per class
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:  # (rows, features, classes)
            values = values[:, :, -1]
        return values

    def global_importance(self, X: pd.DataFrame, top_n: int = 20) -> list[dict[str, Any]]:
        """Mean |SHAP| per source feature — what drives attrition company-wide.

        Direction comes from the correlation between a feature's value and its
        SHAP value, not from the mean signed SHAP. Mean signed SHAP is ~0 for
        every feature by construction (SHAP values are centred), so using it
        would produce confident, meaningless directions.
        """
        values = self.shap_values(X)
        encoded = np.asarray(self.preprocess.transform(X), dtype=float)

        per_source: dict[str, dict[str, Any]] = {}
        for i, encoded_name in enumerate(self.feature_names):
            source = self.source_features.get(encoded_name, encoded_name)
            bucket = per_source.setdefault(
                source, {"importance": 0.0, "weighted_corr": 0.0, "weight": 0.0}
            )
            magnitude = float(np.abs(values[:, i]).mean())
            bucket["importance"] += magnitude

            column, shap_column = encoded[:, i], values[:, i]
            if column.std() > 1e-12 and shap_column.std() > 1e-12:
                corr = float(np.corrcoef(column, shap_column)[0, 1])
                bucket["weighted_corr"] += corr * magnitude
                bucket["weight"] += magnitude

        rows = []
        for source, bucket in per_source.items():
            corr = bucket["weighted_corr"] / bucket["weight"] if bucket["weight"] else 0.0
            rows.append(
                {
                    "feature": source,
                    "label": humanise(source),
                    "importance": round(bucket["importance"], 5),
                    "direction_correlation": round(corr, 4),
                    "is_categorical": source in self.categorical_sources,
                    # A category has no "higher value", so a correlation-based
                    # direction would be an artefact of one-hot column ordering.
                    "direction": (
                        "varies by category"
                        if source in self.categorical_sources
                        else (
                            "higher value increases risk"
                            if corr > 0.05
                            else (
                                "higher value reduces risk"
                                if corr < -0.05
                                else "mixed / non-monotonic"
                            )
                        )
                    ),
                }
            )
        rows.sort(key=lambda r: -r["importance"])
        return rows[:top_n]

    def explain_row(self, row: pd.DataFrame, top_n: int = 5) -> list[Contribution]:
        """The personal top-N contributing factors for one employee.

        One-hot columns are summed back onto their source feature, so the reader
        sees "OverTime" once rather than "OverTime: Yes" and "OverTime: No" as
        two competing drivers.
        """
        values = self.shap_values(row)[0]
        # Engineered features (PromotionGap, TenureInRoleRatio, ...) do not exist
        # on the raw row, so read values from the engineered frame instead.
        engineered = self.pipeline.named_steps["engineer"].fit(row).transform(row)
        raw_row = engineered.iloc[0]

        totals: dict[str, float] = {}
        for i, encoded_name in enumerate(self.feature_names):
            source = self.source_features.get(encoded_name, encoded_name)
            totals[source] = totals.get(source, 0.0) + float(values[i])

        ranked = sorted(totals.items(), key=lambda kv: -abs(kv[1]))[:top_n]
        return [
            Contribution(
                feature=source,
                label=humanise(source),
                value=_jsonable(raw_row[source]) if source in raw_row.index else None,
                shap_value=total,
                direction="increases risk" if total > 0 else "reduces risk",
            )
            for source, total in ranked
        ]


@lru_cache(maxsize=4)
def get_explainer(version: str | None = None) -> AttritionExplainer:
    """Load (and cache) the explainer for a model version."""
    import joblib

    from hrai.ml.registry import latest_version

    version = version or latest_version()
    version_dir = models_dir() / version
    base_path = version_dir / "attrition_base.joblib"
    background_path = version_dir / "shap_background.joblib"

    if base_path.exists() and background_path.exists():
        pipeline = joblib.load(base_path)
        background = joblib.load(background_path)
    else:  # fall back to the deployed model if the base was not persisted
        pipeline, _ = load_model(version)
        from hrai.utils.io import load_processed

        background = load_processed("employee_attrition_processed").sample(n=200, random_state=42)
    return AttritionExplainer(pipeline, background)


def write_global_report(version: str | None = None, top_n: int = 20) -> dict[str, Any]:
    """Persist the global explanation so the dashboard never computes it live."""
    from hrai.utils.config import project_root
    from hrai.utils.io import load_processed

    explainer = get_explainer(version)
    df = load_processed("employee_attrition_processed")
    importance = explainer.global_importance(df, top_n=top_n)

    payload = {"version": version or "latest", "top_features": importance}
    (project_root() / "docs" / "shap_global_importance.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    log.info(
        "global SHAP report written",
        extra={"features": len(importance), "top": importance[0]["label"] if importance else None},
    )
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    return value
