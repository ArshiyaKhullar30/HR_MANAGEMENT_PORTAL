"""Shared fixtures.

Session-scoped where the object is expensive (the model, the API client) so the
suite stays fast enough to run on every commit.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

logging.disable(logging.INFO)


@pytest.fixture(scope="session")
def attrition_processed() -> pd.DataFrame:
    from hrai.utils.io import load_processed

    return load_processed("employee_attrition_processed")


@pytest.fixture(scope="session")
def engagement_processed() -> pd.DataFrame:
    from hrai.utils.io import load_processed

    return load_processed("engagement_processed")


@pytest.fixture(scope="session")
def intelligence_table() -> pd.DataFrame:
    from hrai.utils.io import load_processed

    return load_processed("employee_intelligence")


@pytest.fixture(scope="session")
def model_bundle():
    from app.ml.model_loader import load_bundle

    return load_bundle()


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def valid_employee() -> dict:
    """A well-formed request body for the prediction endpoint."""
    return {
        "Age": 32,
        "Department": "Sales",
        "JobRole": "Sales Executive",
        "MonthlyIncome": 5200,
        "OverTime": "No",
        "JobSatisfaction": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "TotalWorkingYears": 8,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
        "NumCompaniesWorked": 2,
        "employee_id": 424242,
    }
