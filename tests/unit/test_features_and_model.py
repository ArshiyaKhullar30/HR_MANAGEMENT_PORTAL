"""Feature, pipeline and prediction tests (Step 22)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hrai.features.engineering import FeatureEngineer, excluded_columns, feature_columns
from hrai.ml.evaluate import compute_metrics, risk_band, select_threshold


@pytest.mark.unit
def test_target_and_identity_columns_are_never_features(attrition_processed):
    """Finding F7 — the leakage register, enforced rather than remembered."""
    engineered = FeatureEngineer().fit(attrition_processed).transform(attrition_processed)
    features = set(feature_columns(engineered))
    for leak in ("Attrition", "attrition_flag", "employee_id", "EmployeeNumber"):
        assert leak not in features


@pytest.mark.unit
def test_leakage_register_includes_the_engagement_label_columns():
    blocked = excluded_columns()
    for column in ("ExitDate", "TerminationType", "EmployeeStatus", "is_voluntary_exit"):
        assert column in blocked


@pytest.mark.unit
def test_engineered_features_are_added_with_finite_values(attrition_processed):
    engineered = FeatureEngineer().fit(attrition_processed).transform(attrition_processed)
    for name in (
        "IncomePerYearAtCompany",
        "PromotionGap",
        "SatisfactionIndex",
        "ExperienceRatio",
        "TenureInRoleRatio",
        "ManagerStability",
    ):
        assert name in engineered.columns
        assert np.isfinite(engineered[name]).all(), f"{name} produced inf/NaN"


@pytest.mark.unit
def test_zero_tenure_does_not_divide_by_zero():
    """A +1 denominator, so a day-one employee yields 0 rather than infinity."""
    row = pd.DataFrame(
        [
            {
                "MonthlyIncome": 5000,
                "YearsAtCompany": 0,
                "YearsSinceLastPromotion": 0,
                "JobSatisfaction": 3,
                "EnvironmentSatisfaction": 3,
                "RelationshipSatisfaction": 3,
                "TotalWorkingYears": 0,
                "YearsInCurrentRole": 0,
                "YearsWithCurrManager": 0,
            }
        ]
    )
    out = FeatureEngineer().fit(row).transform(row)
    assert np.isfinite(out["IncomePerYearAtCompany"]).all()
    assert out["ExperienceRatio"].iloc[0] == 0.0


@pytest.mark.unit
def test_prediction_returns_a_real_probability(model_bundle, valid_employee):
    from app.ml.predictor import predict_one

    result = predict_one(valid_employee, explain=False, bundle=model_bundle, log_it=False)
    probability = result["attrition_probability"]
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0
    assert result["model_version"] == model_bundle.version


@pytest.mark.unit
@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.95, "HIGH"),
        (0.60, "HIGH"),
        (0.59, "MEDIUM"),
        (0.30, "MEDIUM"),
        (0.29, "LOW"),
        (0.0, "LOW"),
    ],
)
def test_risk_band_boundaries(probability, expected):
    assert risk_band(probability) == expected


@pytest.mark.unit
def test_metrics_reject_accuracy_and_report_the_useful_ones():
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.concatenate([np.full(90, 0.05), np.full(10, 0.9)])
    metrics = compute_metrics(y_true, y_prob, 0.5)
    assert metrics.recall == 1.0
    assert metrics.roc_auc == 1.0
    assert not hasattr(metrics, "accuracy")


@pytest.mark.unit
def test_cost_optimal_threshold_agrees_with_the_analytic_optimum():
    """On calibrated probabilities the empirical search must find the Bayes point.

    The identity `threshold = cost_fp / (cost_fp + cost_fn)` only holds when the
    scores are genuine probabilities, so the fixture draws labels *from* the
    scores rather than adding noise to the labels.
    """
    rng = np.random.default_rng(42)
    y_prob = rng.beta(1.2, 6.0, 20_000)  # realistic risk distribution
    y_true = rng.binomial(1, y_prob)  # calibrated by construction
    result = select_threshold(y_true, y_prob, np.full(len(y_prob), 50_000.0))
    assert not result["threshold_at_grid_boundary"]
    assert abs(result["threshold"] - result["bayes_optimal_threshold"]) < 0.05


@pytest.mark.unit
def test_capacity_threshold_flags_exactly_the_reviewable_share():
    from hrai.ml.evaluate import capacity_threshold

    rng = np.random.default_rng(7)
    y_prob = rng.beta(1.5, 5.0, 1000)
    result = capacity_threshold(y_prob, capacity_pct=0.10)
    assert result["employees_flagged"] == 100
    assert (y_prob >= result["threshold"]).sum() == pytest.approx(100, abs=2)


@pytest.mark.unit
def test_model_is_calibrated_on_the_population_it_was_trained_on(model_bundle, attrition_processed):
    """Predicted risk should track the observed rate, or risk bands are meaningless."""
    features = attrition_processed.drop(columns=["attrition_flag"])
    probability = model_bundle.pipeline.predict_proba(features)[:, 1]
    observed = attrition_processed["attrition_flag"].mean()
    assert abs(probability.mean() - observed) < 0.05
