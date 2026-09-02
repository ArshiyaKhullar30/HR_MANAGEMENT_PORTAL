"""Schema contract tests (Step 22).

Covers the Build Notes' first two named cases — a missing required column is
caught, and an invalid engagement score is rejected — plus the specific defects
the audit found.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hrai.validation.runner import ValidationFailed, validate
from hrai.validation.schemas import (
    attrition_raw_schema,
    engagement_processed_schema,
    engagement_raw_schema,
)


@pytest.mark.unit
def test_missing_required_column_is_caught():
    df = pd.DataFrame({"Age": [30, 40]})  # no EmployeeNumber, no Attrition
    report = validate(df, attrition_raw_schema(), "broken", mode="report")
    assert not report.passed
    missing = {f["column"] for f in report.failures if f["check"] == "required column is missing"}
    # The report must name the columns, not just count them.
    assert {"EmployeeNumber", "Attrition", "JobRole"} <= missing


@pytest.mark.unit
def test_strict_mode_raises_where_report_mode_collects():
    df = pd.DataFrame({"Age": [30]})
    assert not validate(df, attrition_raw_schema(), "broken", mode="report").passed
    with pytest.raises(ValidationFailed):
        validate(df, attrition_raw_schema(), "broken", mode="strict")


@pytest.mark.unit
def test_invalid_engagement_score_is_rejected():
    """Finding F6: engagement is a 1-5 Likert scale, not 0-100."""
    df = pd.DataFrame(
        {
            "Employee ID": [1],
            "Title": ["Data Analyst"],
            "DepartmentType": ["IT/IS"],
            "EmployeeStatus": ["Active"],
            "Engagement Score": [250],
            "Satisfaction Score": [3],
            "Work-Life Balance Score": [3],
            "Current Employee Rating": [3],
            "Performance Score": ["Fully Meets"],
            "GenderCode": ["Female"],
            "DOB": ["01-01-1990"],
            "StartDate": ["01-Jan-20"],
            "ExitDate": [None],
            "Training Program Name": ["Technical Skills"],
            "Training Outcome": ["Passed"],
            "Training Duration(Days)": [2],
            "Training Cost": [500.0],
        }
    )
    report = validate(df, engagement_raw_schema(), "engagement", mode="report")
    assert not report.passed
    assert any("Engagement Score" in str(f["column"]) for f in report.failures)


@pytest.mark.unit
def test_an_engagement_score_of_80_would_pass_a_0_to_100_rule_but_fails_ours():
    """The Build Notes' 0-100 range check would let this through unnoticed."""
    df = pd.DataFrame(
        {
            "Employee ID": [1],
            "Title": ["Data Analyst"],
            "DepartmentType": ["IT/IS"],
            "EmployeeStatus": ["Active"],
            "Engagement Score": [80],
            "Satisfaction Score": [3],
            "Work-Life Balance Score": [3],
            "Current Employee Rating": [3],
            "Performance Score": ["Fully Meets"],
            "GenderCode": ["Female"],
            "DOB": ["01-01-1990"],
            "StartDate": ["01-Jan-20"],
            "ExitDate": [None],
            "Training Program Name": ["Technical Skills"],
            "Training Outcome": ["Passed"],
            "Training Duration(Days)": [2],
            "Training Cost": [500.0],
        }
    )
    assert 0 <= 80 <= 100  # would satisfy the documented rule
    assert not validate(df, engagement_raw_schema(), "engagement", mode="report").passed


@pytest.mark.unit
def test_exit_date_without_a_terminated_status_is_flagged():
    """Finding F6: 1,198 raw rows violate this."""
    df = pd.DataFrame(
        {
            "Employee ID": [1],
            "Title": ["Data Analyst"],
            "DepartmentType": ["IT/IS"],
            "EmployeeStatus": ["Active"],
            "Engagement Score": [3],
            "Satisfaction Score": [3],
            "Work-Life Balance Score": [3],
            "Current Employee Rating": [3],
            "Performance Score": ["Fully Meets"],
            "GenderCode": ["Female"],
            "DOB": ["01-01-1990"],
            "StartDate": ["01-Jan-20"],
            "ExitDate": ["01-Jun-23"],
            "Training Program Name": ["Technical Skills"],
            "Training Outcome": ["Passed"],
            "Training Duration(Days)": [2],
            "Training Cost": [500.0],
        }
    )
    report = validate(df, engagement_raw_schema(), "engagement", mode="report")
    assert not report.passed
    assert any("ExitDate" in f["check"] for f in report.failures)


@pytest.mark.unit
def test_raw_data_passes_its_structural_contract(attrition_processed):
    from hrai.validation.schemas import attrition_processed_schema

    report = validate(
        attrition_processed, attrition_processed_schema(), "attrition_processed", mode="report"
    )
    assert report.passed, report.failures


@pytest.mark.unit
def test_cleaning_fixed_every_defect_the_raw_data_had(engagement_processed):
    """The cleaning step is judged by whether it makes strict validation pass."""
    report = validate(
        engagement_processed, engagement_processed_schema(), "engagement_processed", mode="strict"
    )
    assert report.passed


@pytest.mark.unit
def test_department_whitespace_was_normalised(engagement_processed):
    values = engagement_processed["DepartmentType"].dropna().astype(str)
    assert (values == values.str.strip()).all()
    assert "Production" in set(values)
