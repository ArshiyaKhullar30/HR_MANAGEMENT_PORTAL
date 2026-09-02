"""Cleaning for `hr_performance_engagement.csv` — Population B (Step 03).

Three decisions are made here, each on evidence from the profiler:

**1. Grain (F5).** The file is *employee x survey/training event*, not one row
per employee: 3,150 rows over 3,000 employees. We emit both — an event-level
table for training analytics, and an employee-level table (latest survey per
employee) for everything that needs one row per person.

**2. Label authority.** 1,198 rows carry an `ExitDate` while `EmployeeStatus`
says Active/Future Start/Leave of Absence (F6). Cross-tabulating the columns
settles which to trust: `TerminationType` is distributed almost uniformly
*within* every `EmployeeStatus` value — `Voluntarily Terminated` employees split
91/77/81/90 across Involuntary/Resignation/Retirement/Voluntary — so it carries
no signal. `ExitDate` is likewise populated for `Future Start` employees, who by
definition cannot have left. **`EmployeeStatus` is authoritative**; `ExitDate`,
`TerminationType` and `TerminationDescription` are dropped as unreliable, which
also satisfies the leakage register (F7).

**3. PII.** `FirstName`, `LastName`, `ADEmail` and `Supervisor` are dropped
outright rather than hashed — nothing downstream needs to re-identify a person.
"""

from __future__ import annotations

import pandas as pd

from hrai.cleaning.text import clean_category, parse_dates
from hrai.utils.config import get
from hrai.utils.logger import get_logger

log = get_logger(__name__)

_DATE_FORMATS = ["%d-%b-%y", "%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y"]

_LIKERT_RENAME = {
    "Engagement Score": "engagement_score",
    "Satisfaction Score": "satisfaction_score",
    "Work-Life Balance Score": "work_life_balance_score",
    "Current Employee Rating": "current_employee_rating",
}

# Unreliable per the evidence above; also on the leakage register.
_UNRELIABLE = ["ExitDate", "TerminationType", "TerminationDescription"]


def _reference_year(df: pd.DataFrame) -> int:
    """Data vintage, taken from the latest survey date rather than `today`.

    Using the wall clock would make ages drift every time the pipeline runs,
    breaking the determinism contract.
    """
    surveys = parse_dates(df["Survey Date"], _DATE_FORMATS)
    return int(surveys.dt.year.max()) if surveys.notna().any() else 2023


def clean_engagement_events(df: pd.DataFrame) -> pd.DataFrame:
    """Event-grain cleaning: one row per survey/training event (3,150 rows)."""
    out = df.copy()
    out = out.drop(columns=[c for c in out.columns if c.startswith("Unnamed")], errors="ignore")

    pii = get("pii.drop_columns.hr_performance_engagement", [])
    dropped_pii = [c for c in pii if c in out.columns]
    out = out.drop(columns=dropped_pii)

    dropped_unreliable = [c for c in _UNRELIABLE if c in out.columns]
    out = out.drop(columns=dropped_unreliable)

    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = clean_category(out[col])

    reference_year = _reference_year(df)
    dob = parse_dates(df["DOB"], _DATE_FORMATS)
    start = parse_dates(df["StartDate"], _DATE_FORMATS)
    survey = parse_dates(df["Survey Date"], _DATE_FORMATS)
    training = parse_dates(df["Training Date"], _DATE_FORMATS)

    out["dob"] = dob
    out["start_date"] = start
    out["survey_date"] = survey
    out["training_date"] = training

    age = reference_year - dob.dt.year
    # Two-digit years parse a century forward; roll them back.
    out["age"] = age.where(age > 0, age + 100).astype("Int64")
    out["tenure_years"] = (
        (pd.Timestamp(year=reference_year, month=12, day=31) - start).dt.days / 365.25
    ).round(2)

    out = out.rename(columns=_LIKERT_RENAME)
    for col in _LIKERT_RENAME.values():
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    terminated = set(get("validation.engagement.terminated_statuses", []))
    out["employee_id"] = pd.to_numeric(out["Employee ID"], errors="coerce").astype("Int64")
    out["population"] = "B"
    # EmployeeStatus is the authoritative label; see the module docstring.
    out["is_voluntary_exit"] = (out["EmployeeStatus"] == "Voluntarily Terminated").astype(int)
    out["is_terminated"] = out["EmployeeStatus"].isin(terminated).astype(int)

    out = out.drop(
        columns=["Employee ID", "DOB", "StartDate", "Survey Date", "Training Date"], errors="ignore"
    )

    log.info(
        "engagement events cleaned",
        extra={
            "rows": len(out),
            "dropped_pii": dropped_pii,
            "dropped_unreliable": dropped_unreliable,
            "reference_year": reference_year,
            "voluntary_exit_rate": round(float(out["is_voluntary_exit"].mean()), 4),
        },
    )
    return out.reset_index(drop=True)


def to_employee_grain(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse the event table to one row per employee (F5).

    The most recent survey wins for point-in-time attributes; training activity
    is aggregated across every event, because a person's training history is
    cumulative and taking only the latest would discard most of it.
    """
    events = events.copy()
    events["_order"] = events["survey_date"].fillna(pd.Timestamp.min)

    latest = (
        events.sort_values(["employee_id", "_order"])
        .groupby("employee_id", as_index=False)
        .tail(1)
        .drop(columns="_order")
    )

    training = (
        events.groupby("employee_id")
        .agg(
            training_events=("Training Program Name", "size"),
            training_days_total=("Training Duration(Days)", "sum"),
            training_cost_total=("Training Cost", "sum"),
            trainings_passed=(
                "Training Outcome",
                lambda s: int(s.isin(["Passed", "Completed"]).sum()),
            ),
            trainings_failed=(
                "Training Outcome",
                lambda s: int(s.isin(["Failed", "Incomplete"]).sum()),
            ),
            training_programs=(
                "Training Program Name",
                lambda s: "|".join(sorted(set(s.dropna().astype(str)))),
            ),
        )
        .reset_index()
    )

    out = latest.merge(training, on="employee_id", how="left", validate="one_to_one")
    out["training_pass_rate"] = (
        (out["trainings_passed"] / out["training_events"].replace(0, pd.NA)).astype(float).round(3)
    )

    log.info(
        "engagement collapsed to employee grain",
        extra={
            "event_rows": len(events),
            "employee_rows": len(out),
            "collapsed": len(events) - len(out),
        },
    )
    return out.reset_index(drop=True)
