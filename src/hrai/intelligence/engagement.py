"""Engagement analytics (Step 10).

No ML, as the Build Notes specify — aggregation and ranking. Two corrections
carried in from the audit:

* **Grain (F5).** The source file is employee x event, so aggregating it
  directly double-counts the 150 employees who appear twice. Everything here
  runs on the employee-grain table.
* **Scale (F6).** Engagement is a 1-5 Likert scale, not 0-100. The Build Notes'
  0-100 range check would pass every row and catch nothing.
"""

from __future__ import annotations

import pandas as pd

from hrai.utils.io import load_processed
from hrai.utils.logger import get_logger

log = get_logger(__name__)

LIKERT_MAX = 5
SCORES = [
    "engagement_score",
    "satisfaction_score",
    "work_life_balance_score",
    "current_employee_rating",
]


def engagement_summary() -> dict:
    df = load_processed("engagement_processed")
    return {
        "employees": int(len(df)),
        "average_engagement": round(float(df["engagement_score"].mean()), 2),
        "average_engagement_pct": round(100 * float(df["engagement_score"].mean()) / LIKERT_MAX, 1),
        "average_satisfaction": round(float(df["satisfaction_score"].mean()), 2),
        "average_work_life_balance": round(float(df["work_life_balance_score"].mean()), 2),
        "scale": "1-5 Likert",
        "voluntary_exit_rate": round(float(df["is_voluntary_exit"].mean()), 4),
        "low_engagement_employees": int((df["engagement_score"] <= 2).sum()),
    }


def engagement_by(dimension: str = "DepartmentType") -> pd.DataFrame:
    """Engagement broken down by department, division, title or tenure band."""
    df = load_processed("engagement_processed")
    if dimension == "tenure_band":
        df = df.copy()
        df["tenure_band"] = pd.cut(
            df["tenure_years"],
            bins=[-0.01, 1, 3, 5, 10, 100],
            labels=["<1y", "1-3y", "3-5y", "5-10y", "10y+"],
        ).astype(str)
    if dimension not in df.columns:
        raise KeyError(f"Unknown engagement dimension: {dimension!r}")

    out = (
        df.groupby(dimension, as_index=False)
        .agg(
            employees=("employee_id", "nunique"),
            avg_engagement=("engagement_score", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_work_life_balance=("work_life_balance_score", "mean"),
            voluntary_exit_rate=("is_voluntary_exit", "mean"),
            low_engagement=("engagement_score", lambda s: int((s <= 2).sum())),
        )
        .round(3)
        .sort_values("avg_engagement")
    )
    return out.reset_index(drop=True)


def lowest_engagement(n: int = 25) -> pd.DataFrame:
    """The employees HR should look at directly, worst first."""
    df = load_processed("engagement_processed")
    columns = [
        "employee_id",
        "Title",
        "DepartmentType",
        "Division",
        "tenure_years",
        *SCORES,
        "training_pass_rate",
        "EmployeeStatus",
    ]
    available = [c for c in columns if c in df.columns]
    return (
        df.sort_values(["engagement_score", "satisfaction_score"])
        .head(n)[available]
        .reset_index(drop=True)
    )


def training_effectiveness() -> pd.DataFrame:
    """Does training actually move engagement? Aggregated over real records."""
    events = load_processed("engagement_events_processed")
    return (
        events.groupby(["Training Program Name", "Training Outcome"], as_index=False)
        .agg(
            events=("employee_id", "size"),
            avg_engagement=("engagement_score", "mean"),
            avg_cost=("Training Cost", "mean"),
            avg_days=("Training Duration(Days)", "mean"),
        )
        .round(2)
        .sort_values(["Training Program Name", "Training Outcome"])
        .reset_index(drop=True)
    )
