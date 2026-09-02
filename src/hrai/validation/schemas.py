"""Pandera schemas — the data contract (Step 02).

The Build Notes suggest plain pandas asserts for the MVP and Pandera later.
We go straight to Pandera because the document already names it as the
destination: the rules then live in one place instead of scattered across
notebooks, and the same objects are imported by the pipeline, the tests and the
API.

Two layers:

* **Structural** — columns, dtypes, keys. These must always hold; a failure
  means the file is not the file we think it is.
* **Quality** — ranges, categories, cross-field consistency. These *fail on the
  raw data by design* (finding F6), which is exactly what makes them worth
  having: the cleaning step is judged by whether it makes them pass.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:  # pandera >= 0.20 namespaces the pandas backend
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:  # pragma: no cover - older pandera
    from pandera import Check, Column, DataFrameSchema

from hrai.utils.config import get

# --------------------------------------------------------------------------
# employee_attrition — Population A
# --------------------------------------------------------------------------

_AGE_MIN, _AGE_MAX = get("validation.attrition.age_range", [18, 100])
_ATTRITION_VALUES = get("validation.attrition.attrition_values", ["Yes", "No"])
_LIKERT_MIN, _LIKERT_MAX = get("validation.engagement.likert_range", [1, 5])
_TERMINATED = get("validation.engagement.terminated_statuses", [])


def attrition_raw_schema() -> DataFrameSchema:
    """Contract for `employee_attrition.csv` as it arrives."""
    return DataFrameSchema(
        {
            "EmployeeNumber": Column(
                int, unique=True, nullable=False, description="Primary key for Population A."
            ),
            "Age": Column(int, Check.in_range(_AGE_MIN, _AGE_MAX), nullable=False),
            "Attrition": Column(
                str,
                Check.isin(_ATTRITION_VALUES),
                nullable=False,
                description="Target. Never a feature (leakage register).",
            ),
            "Department": Column(str, nullable=False),
            "JobRole": Column(str, nullable=False),
            "MonthlyIncome": Column(int, Check.gt(0), nullable=False),
            "YearsAtCompany": Column(int, Check.ge(0), nullable=False),
            "TotalWorkingYears": Column(int, Check.ge(0), nullable=False),
            "YearsInCurrentRole": Column(int, Check.ge(0), nullable=False),
            "YearsSinceLastPromotion": Column(int, Check.ge(0), nullable=False),
            "YearsWithCurrManager": Column(int, Check.ge(0), nullable=False),
            "JobSatisfaction": Column(int, Check.in_range(1, 4), nullable=False),
            "EnvironmentSatisfaction": Column(int, Check.in_range(1, 4), nullable=False),
            "RelationshipSatisfaction": Column(int, Check.in_range(1, 4), nullable=False),
            "WorkLifeBalance": Column(int, Check.in_range(1, 4), nullable=False),
            "JobInvolvement": Column(int, Check.in_range(1, 4), nullable=False),
            "PerformanceRating": Column(int, Check.in_range(1, 4), nullable=False),
            "JobLevel": Column(int, Check.in_range(1, 5), nullable=False),
            "StockOptionLevel": Column(int, Check.in_range(0, 3), nullable=False),
            "TrainingTimesLastYear": Column(int, Check.ge(0), nullable=False),
            "NumCompaniesWorked": Column(int, Check.ge(0), nullable=False),
            "DistanceFromHome": Column(int, Check.gt(0), nullable=False),
            "PercentSalaryHike": Column(int, Check.ge(0), nullable=False),
            "OverTime": Column(str, Check.isin(["Yes", "No"]), nullable=False),
            "Gender": Column(str, Check.isin(["Male", "Female"]), nullable=False),
            "MaritalStatus": Column(str, nullable=False),
            "BusinessTravel": Column(str, nullable=False),
            "EducationField": Column(str, nullable=False),
            "Education": Column(int, Check.in_range(1, 5), nullable=False),
        },
        checks=[
            Check(
                lambda df: df["YearsAtCompany"] <= df["TotalWorkingYears"],
                element_wise=False,
                error="YearsAtCompany exceeds TotalWorkingYears",
                name="tenure_not_greater_than_career",
            ),
            Check(
                lambda df: df["YearsInCurrentRole"] <= df["YearsAtCompany"],
                element_wise=False,
                error="YearsInCurrentRole exceeds YearsAtCompany",
                name="role_tenure_within_company_tenure",
            ),
        ],
        strict=False,
        coerce=True,
        name="employee_attrition_raw",
    )


def attrition_processed_schema() -> DataFrameSchema:
    """Contract for the cleaned frame: constants gone, engineered keys present."""
    schema = attrition_raw_schema()
    dropped = set(get("validation.attrition.constant_columns_to_drop", []))
    return schema.remove_columns([c for c in dropped if c in schema.columns]).add_columns(
        {
            "employee_id": Column(int, unique=True, nullable=False),
            "population": Column(str, Check.eq("A"), nullable=False),
        }
    )


# --------------------------------------------------------------------------
# hr_performance_engagement — Population B
# --------------------------------------------------------------------------


def _likert(nullable: bool = False) -> Column:
    return Column(int, Check.in_range(_LIKERT_MIN, _LIKERT_MAX), nullable=nullable)


def engagement_raw_schema() -> DataFrameSchema:
    """Contract for `hr_performance_engagement.csv` as it arrives.

    Note the Likert range is 1-5. The Build Notes specify a 0-100 range check
    for engagement; applied to this file it would pass every row and catch
    nothing (finding F6).
    """
    return DataFrameSchema(
        {
            "Employee ID": Column(
                int, nullable=False, description="Not unique — grain is employee x event (F5)."
            ),
            "Title": Column(str, nullable=False),
            "DepartmentType": Column(str, nullable=False),
            "EmployeeStatus": Column(
                str, nullable=False, description="Label source. Leakage register."
            ),
            "Engagement Score": _likert(),
            "Satisfaction Score": _likert(),
            "Work-Life Balance Score": _likert(),
            "Current Employee Rating": _likert(),
            "Performance Score": Column(str, nullable=False),
            "GenderCode": Column(str, nullable=False),
            "DOB": Column(str, nullable=False),
            "StartDate": Column(str, nullable=False),
            "ExitDate": Column(str, nullable=True),
            "Training Program Name": Column(str, nullable=False),
            "Training Outcome": Column(str, nullable=False),
            "Training Duration(Days)": Column(int, Check.ge(0), nullable=False),
            "Training Cost": Column(float, Check.ge(0), nullable=False),
        },
        checks=[exit_date_consistency_check()],
        strict=False,
        coerce=True,
        name="hr_performance_engagement_raw",
    )


def exit_date_consistency_check() -> Check:
    """F6: an ExitDate must imply a terminated status.

    1,198 raw rows violate this. The check exists precisely so the violation is
    surfaced and then provably fixed, rather than quietly carried forward.
    """

    def _rule(df: pd.DataFrame) -> pd.Series:
        has_exit = df["ExitDate"].notna() & df["ExitDate"].astype(str).str.strip().ne("")
        return ~has_exit | df["EmployeeStatus"].isin(_TERMINATED)

    return Check(
        _rule,
        element_wise=False,
        error="ExitDate is populated but EmployeeStatus is not a terminated status",
        name="exit_date_implies_terminated",
    )


def engagement_processed_schema() -> DataFrameSchema:
    """Contract for the cleaned engagement frame — one row per employee."""
    return DataFrameSchema(
        {
            "employee_id": Column(int, unique=True, nullable=False),
            "population": Column(str, Check.eq("B"), nullable=False),
            "Title": Column(str, nullable=False),
            "DepartmentType": Column(str, nullable=False),
            "engagement_score": _likert(),
            "satisfaction_score": _likert(),
            "work_life_balance_score": _likert(),
            "current_employee_rating": _likert(),
            "age": Column(int, Check.in_range(_AGE_MIN, _AGE_MAX), nullable=True),
            "tenure_years": Column(float, Check.ge(0), nullable=True),
            "is_voluntary_exit": Column(int, Check.isin([0, 1]), nullable=False),
        },
        checks=[
            Check(
                lambda df: ~df["DepartmentType"].astype(str).str.contains(r"^\s|\s$", regex=True),
                element_wise=False,
                error="DepartmentType still carries leading/trailing whitespace",
                name="department_whitespace_normalised",
            ),
        ],
        strict=False,
        coerce=True,
        name="hr_performance_engagement_processed",
    )


# --------------------------------------------------------------------------
# O*NET reference tables
# --------------------------------------------------------------------------


def occupation_schema() -> DataFrameSchema:
    return DataFrameSchema(
        {
            "O*NET-SOC Code": Column(str, unique=True, nullable=False),
            "Title": Column(str, nullable=False),
            "Description": Column(str, nullable=False),
        },
        strict=False,
        coerce=True,
        name="occupation_data",
    )


def essential_skills_schema() -> DataFrameSchema:
    """O*NET Basic Skills — 10 foundational competencies, IM 1-5 / LV 0-7 (F2)."""
    return DataFrameSchema(
        {
            "O*NET-SOC Code": Column(str, nullable=False),
            "Element Name": Column(str, nullable=False),
            "Scale ID": Column(str, Check.isin(["IM", "LV"]), nullable=False),
            "Data Value": Column(float, Check.in_range(0, 7), nullable=False),
            "Recommend Suppress": Column(str, nullable=True),
        },
        strict=False,
        coerce=True,
        name="essential_skills",
    )


def software_skills_schema() -> DataFrameSchema:
    """O*NET Technology Skills — 8,753 tools across 134 categories (F2)."""
    return DataFrameSchema(
        {
            "O*NET-SOC Code": Column(str, nullable=False),
            "Element Name": Column(str, nullable=False, description="Technology category."),
            "Workplace Example": Column(
                str, nullable=False, description="The concrete tool, e.g. 'Microsoft Excel'."
            ),
            "Hot Technology": Column(str, Check.isin(["Y", "N"]), nullable=False),
            "In Demand": Column(str, Check.isin(["Y", "N"]), nullable=False),
        },
        strict=False,
        coerce=True,
        name="software_skills",
    )


RAW_SCHEMAS: dict[str, Any] = {
    "employee_attrition": attrition_raw_schema,
    "hr_performance_engagement": engagement_raw_schema,
    "occupation_data": occupation_schema,
    "essential_skills": essential_skills_schema,
    "software_skills": software_skills_schema,
}

PROCESSED_SCHEMAS: dict[str, Any] = {
    "employee_attrition_processed": attrition_processed_schema,
    "engagement_processed": engagement_processed_schema,
}
