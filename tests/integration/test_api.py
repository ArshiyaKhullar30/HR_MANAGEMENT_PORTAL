"""API integration tests (Step 22).

Covers the Build Notes' named case — "API returns the expected status codes" —
plus the contract guarantees that would be expensive to discover in production:
that validation rejects bad input before it reaches the model, that the ID
collision cannot silently return the wrong person, and that Population B never
receives a fabricated risk score.
"""

from __future__ import annotations

import pytest

PREFIX = "/api/v1"


@pytest.mark.integration
def test_health_and_readiness(api_client):
    health = api_client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert api_client.get("/ready").status_code == 200


@pytest.mark.integration
def test_openapi_schema_is_served(api_client):
    response = api_client.get(f"{PREFIX}/openapi.json")
    assert response.status_code == 200
    assert "paths" in response.json()


@pytest.mark.integration
def test_predict_returns_a_probability_and_a_band(api_client, valid_employee):
    response = api_client.post(f"{PREFIX}/predict/attrition", json=valid_employee)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["attrition_probability"] <= 1.0
    assert body["risk_band"] in {"HIGH", "MEDIUM", "LOW"}
    assert body["top_factors"], "a prediction without a reason is not actionable"
    assert "never as an automatic decision" in body["caveat"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Age", 250),  # out of range
        ("Age", 5),  # under the minimum
        ("OverTime", "Sometimes"),  # not in the allowed set
        ("JobSatisfaction", 9),  # outside the 1-4 scale
        ("MonthlyIncome", -100),  # non-positive
        ("YearsInCurrentRole", 99),  # exceeds YearsAtCompany
    ],
)
def test_invalid_input_is_rejected_before_it_reaches_the_model(
    api_client, valid_employee, field, value
):
    response = api_client.post(f"{PREFIX}/predict/attrition", json={**valid_employee, field: value})
    assert response.status_code == 422, f"{field}={value} should have been rejected"


@pytest.mark.integration
def test_missing_required_field_is_rejected(api_client, valid_employee):
    payload = {k: v for k, v in valid_employee.items() if k != "OverTime"}
    assert api_client.post(f"{PREFIX}/predict/attrition", json=payload).status_code == 422


@pytest.mark.integration
@pytest.mark.parametrize(
    "path",
    [
        "/dashboard/summary",
        "/dashboard/attrition-by-department",
        "/dashboard/engagement-by-department",
        "/dashboard/skill-gaps",
        "/dashboard/recommendations",
        "/dashboard/departments",
        "/dashboard/model-quality",
        "/predict/model",
        "/predict/log",
        "/skills/crosswalk",
        "/skills/role-requirements",
        "/intelligence/levers",
    ],
)
def test_read_endpoints_return_200(api_client, path):
    assert api_client.get(f"{PREFIX}{path}").status_code == 200


@pytest.mark.integration
def test_unknown_employee_returns_404(api_client):
    assert api_client.get(f"{PREFIX}/employees/A-99999999").status_code == 404


@pytest.mark.integration
def test_numeric_id_collision_returns_both_people_not_one(api_client):
    """Finding F1, as a live contract test.

    Employee id 1001 exists in both populations, for two different people. The
    API must surface both rather than silently picking one.
    """
    response = api_client.get(f"{PREFIX}/employees/resolve/1001")
    assert response.status_code == 200
    matches = response.json()["matches"]
    assert len(matches) == 2
    assert {m["population"] for m in matches} == {"A", "B"}
    # Genuinely different people, not the same record twice.
    assert matches[0]["role"] != matches[1]["role"]


@pytest.mark.integration
def test_population_b_never_receives_a_fabricated_risk_score(api_client):
    response = api_client.get(f"{PREFIX}/employees/B-1001")
    assert response.status_code == 200
    body = response.json()
    assert body["attrition_probability"] is None
    assert body["risk_band"] == "UNAVAILABLE"
    assert "does not transfer" in body["risk_unavailable_reason"]


@pytest.mark.integration
def test_counterfactual_is_refused_for_the_population_without_a_valid_model(api_client):
    assert api_client.get(f"{PREFIX}/intelligence/counterfactual/B-1001").status_code == 422
    assert api_client.get(f"{PREFIX}/intelligence/counterfactual/A-622").status_code == 200


@pytest.mark.integration
def test_action_plan_stays_within_budget(api_client):
    response = api_client.get(
        f"{PREFIX}/intelligence/action-plan",
        params={"budget": 80_000, "min_risk": 0.4, "max_employees": 40},
    )
    assert response.status_code == 200
    plan = response.json()
    assert plan["spend"] <= plan["budget"]
    assert plan["return_on_investment"] >= 0


@pytest.mark.integration
def test_action_plan_rejects_a_non_positive_budget(api_client):
    assert (
        api_client.get(f"{PREFIX}/intelligence/action-plan", params={"budget": 0}).status_code
        == 422
    )


@pytest.mark.integration
def test_skill_gap_responses_declare_that_skills_are_derived(api_client):
    """Finding F3 must travel all the way to the consumer, not stop at the docs."""
    for path in ("/dashboard/skill-gaps", "/dashboard/recommendations"):
        body = api_client.get(f"{PREFIX}{path}").json()
        assert body["skills_are_derived"] is True


@pytest.mark.integration
def test_prediction_is_written_to_the_drift_log(api_client, valid_employee):
    before = api_client.get(f"{PREFIX}/predict/log", params={"limit": 1000}).json()["count"]
    api_client.post(f"{PREFIX}/predict/attrition", json=valid_employee)
    after = api_client.get(f"{PREFIX}/predict/log", params={"limit": 1000}).json()["count"]
    assert after > before


@pytest.mark.integration
def test_request_id_is_echoed_for_traceability(api_client):
    response = api_client.get("/health", headers={"X-Request-ID": "test-req-123"})
    assert response.headers["X-Request-ID"] == "test-req-123"
