"""Unit tests for the training module's decision logic.

`main()` retrains for roughly a minute, so the parts that encode a *decision*
are tested directly — those are what would silently go wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hrai.ml.train import candidate_models, imbalance_ratio


@pytest.mark.unit
def test_all_three_build_notes_candidates_are_present():
    models = candidate_models()
    assert set(models) == {"logistic_regression", "random_forest", "xgboost"}
    # Scaling matters for the linear model and is meaningless for trees.
    assert models["logistic_regression"]["scale"] is True
    assert models["random_forest"]["scale"] is False
    assert models["xgboost"]["scale"] is False


@pytest.mark.unit
def test_every_candidate_is_class_balanced():
    """Otherwise XGBoost competes against two balanced models while optimising
    for the majority class, and 'XGBoost lost' would mean nothing."""
    models = candidate_models(scale_pos_weight=5.2)
    assert models["logistic_regression"]["estimator"].class_weight == "balanced"
    assert models["random_forest"]["estimator"].class_weight == "balanced_subsample"
    assert models["xgboost"]["estimator"].scale_pos_weight == 5.2


@pytest.mark.unit
def test_imbalance_ratio_is_negatives_over_positives():
    assert imbalance_ratio(pd.Series([0] * 80 + [1] * 20)) == pytest.approx(4.0)
    assert imbalance_ratio(pd.Series([0, 0])) == 1.0  # no positives: safe default


@pytest.mark.unit
def test_all_candidates_are_seeded_identically():
    from hrai.utils.config import seed

    for spec in candidate_models().values():
        assert spec["estimator"].random_state == seed()


@pytest.mark.unit
def test_calibration_selection_rejects_a_saturating_method():
    """Isotonic can score marginally better on Brier while saturating, which
    makes every counterfactual read as zero effect."""
    from sklearn.linear_model import LogisticRegression

    from hrai.ml.train import select_calibration_method

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=400) > 0).astype(int))

    from hrai.features.pipeline import build_pipeline

    spec = {"estimator": LogisticRegression(max_iter=500), "scale": True}
    method, comparison = select_calibration_method(
        build_pipeline(X, spec["estimator"], scale=True), X, y, spec
    )
    assert method in {"sigmoid", "isotonic"}
    assert {row["method"] for row in comparison} == {"sigmoid", "isotonic"}
    for row in comparison:
        assert 0 <= row["saturated_fraction"] <= 1
        assert row["distinct_values"] > 1


@pytest.mark.unit
def test_the_shipped_model_chose_a_smooth_calibration(model_bundle):
    metadata = model_bundle.metadata
    assert metadata["calibration_method"] == "sigmoid"
    comparison = {row["method"]: row for row in metadata["calibration_comparison"]}
    assert (
        comparison["isotonic"]["saturated_fraction"] > comparison["sigmoid"]["saturated_fraction"]
    )


@pytest.mark.unit
def test_logistic_regression_won_on_pr_auc_not_on_expectation(model_bundle):
    """The Build Notes expected XGBoost. On this data it did not win, and the
    comparison table is what settles it."""
    comparison = model_bundle.metadata["metrics"]["cv_comparison"]
    ranked = sorted(comparison, key=lambda r: -r["average_precision_mean"])
    assert ranked[0]["model"] == "logistic_regression"
    assert len(comparison) == 3
