"""Retention ROI Copilot and fairness tests."""

from __future__ import annotations

import pytest

from hrai.utils.config import get


@pytest.fixture(scope="module")
def engine():
    from hrai.intelligence.counterfactual import CounterfactualEngine

    return CounterfactualEngine()


@pytest.fixture(scope="module")
def sample_row(attrition_processed):
    return attrition_processed.drop(columns=["attrition_flag"]).head(1)


@pytest.mark.unit
def test_protected_attributes_are_never_levers():
    """The system must not be able to propose an intervention on who someone is."""
    from hrai.intelligence.counterfactual import load_levers

    protected = set(get("retention_roi.protected_attributes", []))
    assert protected, "protected attributes must be configured"
    assert protected.isdisjoint({lever.feature for lever in load_levers()})


@pytest.mark.unit
def test_a_lever_on_a_protected_attribute_is_refused_at_construction(monkeypatch):
    """Defence in depth: even a mis-edited config cannot produce such a lever."""
    import hrai.intelligence.counterfactual as module

    bad = [{"name": "age_lever", "label": "Change age", "feature": "Age", "delta": 1}]
    monkeypatch.setattr(
        module,
        "get",
        lambda key, default=None: (
            bad
            if key == "retention_roi.levers"
            else (
                ["Age", "Gender", "MaritalStatus"]
                if key == "retention_roi.protected_attributes"
                else default
            )
        ),
    )
    assert module.load_levers() == []


@pytest.mark.unit
def test_counterfactual_returns_a_priced_ranked_plan(engine, sample_row):
    plan = engine.plan_for(sample_row)
    assert 0.0 <= plan.baseline_risk <= 1.0
    assert plan.single_lever, "at least one lever should apply"
    for intervention in plan.single_lever:
        assert 0.0 <= intervention.new_risk <= 1.0
        assert intervention.cost > 0
    # Ranked by return on investment, best first.
    rois = [i.roi for i in plan.single_lever]
    assert rois == sorted(rois, reverse=True)


@pytest.mark.unit
def test_every_plan_carries_the_causal_caveat(engine, sample_row):
    plan = engine.plan_for(sample_row).to_dict()
    assert "not causal inference" in plan["caveat"]


@pytest.mark.unit
def test_a_lever_actually_changes_the_input_it_claims_to(engine, sample_row):
    """Guards against a lever that reports a delta without moving anything."""
    row = sample_row.iloc[0]
    for lever in engine.levers:
        if not lever.applies_to(row):
            continue
        assert lever.apply(row)[lever.feature] != row[lever.feature]


@pytest.mark.unit
def test_calibration_is_smooth_enough_for_counterfactuals(engine, attrition_processed):
    """Isotonic calibration saturates and would make every lever read as zero effect."""
    features = attrition_processed.drop(columns=["attrition_flag"])
    probability = engine.model.predict_proba(features)[:, 1]
    saturated = ((probability <= 1e-6) | (probability >= 1 - 1e-6)).mean()
    assert saturated < 0.02, "saturated probabilities cannot respond to a perturbation"


@pytest.mark.unit
def test_action_plan_respects_its_budget(engine):
    from hrai.intelligence.counterfactual import build_action_plan

    budget = 60_000.0
    plan = build_action_plan(budget, engine=engine, min_risk=0.4, max_employees=40)
    assert plan["spend"] <= budget
    assert plan["employees_covered"] <= 40
    # One intervention per person by default.
    keys = [i["person_key"] for i in plan["interventions"]]
    assert len(keys) == len(set(keys))


@pytest.mark.unit
def test_action_plan_never_proposes_a_risk_increasing_intervention(engine):
    from hrai.intelligence.counterfactual import build_action_plan

    plan = build_action_plan(100_000.0, engine=engine, min_risk=0.4, max_employees=40)
    assert all(i["risk_reduction"] > 0 for i in plan["interventions"])


@pytest.mark.unit
def test_fairness_audit_reports_every_protected_attribute():
    from hrai.ml.fairness import audit

    payload = audit()
    assert "Gender" in payload["attributes"]
    assert "AgeBand" in payload["attributes"]
    for result in payload["attributes"].values():
        assert result["equal_opportunity_difference"] is not None
        assert len(result["groups"]) >= 2
    assert payload["interpretation"]
