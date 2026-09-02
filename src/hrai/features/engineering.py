"""Feature engineering (Step 05).

Every engineered feature carries a stated reason — statistical or business —
per the Build Notes' own rule. Features that only "seemed interesting" are
noise, and noise costs recall on a dataset with 237 positive cases.

The transformer is a scikit-learn estimator so it lives *inside* the pipeline.
Nothing is computed outside the fitted object, which is what makes train/serve
skew structurally impossible rather than merely unlikely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from hrai.utils.config import get
from hrai.utils.logger import get_logger

log = get_logger(__name__)

# Columns that must never reach the model: the target itself, identity keys,
# and anything on the leakage register (finding F7).
IDENTITY_COLUMNS = {"employee_id", "EmployeeNumber", "Employee ID", "population"}
TARGET_COLUMNS = {"Attrition", "attrition_flag", "is_voluntary_exit", "is_terminated"}


def excluded_columns() -> set[str]:
    """The full exclusion set, read from the leakage register in config."""
    leakage = get("leakage", {}) or {}
    blocked: set[str] = set(IDENTITY_COLUMNS) | set(TARGET_COLUMNS)
    for key, columns in leakage.items():
        if key == "never_features" or isinstance(columns, list):
            blocked |= set(columns)
    return blocked


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model-eligible columns: everything not excluded, not a datetime, not free text."""
    blocked = excluded_columns()
    out: list[str] = []
    for col in df.columns:
        if col in blocked or col.startswith("_"):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        # High-cardinality free text would explode one-hot encoding.
        is_text = df[col].dtype == object or str(df[col].dtype) == "string"
        if is_text and df[col].nunique(dropna=True) > 50:
            continue
        out.append(col)
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add the engineered features described in `conf/config.yaml`.

    Ratios use a ``+1`` denominator so a zero-tenure employee produces 0 rather
    than infinity — a division-by-zero here would surface as a NaN deep inside
    the model, which is far harder to diagnose than a slightly damped ratio.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - sklearn convention
        self.feature_names_in_ = list(X.columns)
        self.added_features_: list[str] = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        out = X.copy()
        if not self.enabled:
            return out
        added: list[str] = []

        def has(*cols: str) -> bool:
            return all(c in out.columns for c in cols)

        def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
            return (
                (numerator / (denominator.astype(float) + 1.0))
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
                .round(4)
            )

        # Pay progression relative to tenure: flat pay over long tenure is a
        # well-documented driver of voluntary exit.
        if has("MonthlyIncome", "YearsAtCompany"):
            out["IncomePerYearAtCompany"] = safe_ratio(out["MonthlyIncome"], out["YearsAtCompany"])
            added.append("IncomePerYearAtCompany")

        # Stagnation, normalised by tenure so a 3-year wait means something
        # different for a 4-year employee than for a 20-year one.
        if has("YearsSinceLastPromotion", "YearsAtCompany"):
            out["PromotionGap"] = safe_ratio(out["YearsSinceLastPromotion"], out["YearsAtCompany"])
            added.append("PromotionGap")

        # One stable morale construct instead of three correlated 1-4 scales.
        satisfaction = [
            c
            for c in ("JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction")
            if c in out.columns
        ]
        if len(satisfaction) >= 2:
            out["SatisfactionIndex"] = out[satisfaction].mean(axis=1).round(4)
            added.append("SatisfactionIndex")

        # Loyalty vs overall career mobility.
        if has("YearsAtCompany", "TotalWorkingYears"):
            out["ExperienceRatio"] = safe_ratio(out["YearsAtCompany"], out["TotalWorkingYears"])
            added.append("ExperienceRatio")

        # Internal mobility: low movement within a long tenure drives exits.
        if has("YearsInCurrentRole", "YearsAtCompany"):
            out["TenureInRoleRatio"] = safe_ratio(out["YearsInCurrentRole"], out["YearsAtCompany"])
            added.append("TenureInRoleRatio")

        # Manager churn is a documented attrition driver.
        if has("YearsWithCurrManager", "YearsAtCompany"):
            out["ManagerStability"] = safe_ratio(out["YearsWithCurrManager"], out["YearsAtCompany"])
            added.append("ManagerStability")

        self.added_features_ = added
        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.array(base + getattr(self, "added_features_", []), dtype=object)
