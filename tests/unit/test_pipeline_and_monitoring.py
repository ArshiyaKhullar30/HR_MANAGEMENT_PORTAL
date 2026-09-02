"""Registry, I/O, analytics, transfer and monitoring tests."""

from __future__ import annotations

import pandas as pd
import pytest

# ---- I/O and the scope rule --------------------------------------------


@pytest.mark.unit
def test_only_permitted_datasets_can_be_loaded():
    from hrai.utils.io import load_raw

    with pytest.raises(KeyError):
        load_raw("employee_performance_pro")  # an archive/ file — finding F8


@pytest.mark.unit
def test_frame_checksum_detects_any_change():
    from hrai.utils.io import frame_checksum

    frame = pd.DataFrame({"a": [1, 2, 3]})
    assert frame_checksum(frame) == frame_checksum(frame.copy())
    assert frame_checksum(frame) != frame_checksum(pd.DataFrame({"a": [1, 2, 4]}))
    assert frame_checksum(frame) != frame_checksum(frame.iloc[::-1])


@pytest.mark.unit
def test_processed_artifacts_are_recorded_with_checksums():
    from hrai.utils.io import read_manifest

    manifest = read_manifest()
    assert "employee_attrition_processed" in manifest
    for entry in manifest.values():
        assert len(entry["content_sha256"]) == 64
        assert entry["rows"] > 0


# ---- model registry ------------------------------------------------------


@pytest.mark.unit
def test_registry_round_trips_a_model_with_audit_metadata():
    from hrai.ml.registry import latest_version, list_versions, load_model

    version = latest_version()
    assert version and version.startswith("v")

    pipeline, metadata = load_model(version)
    assert hasattr(pipeline, "predict_proba")
    # What an audit actually needs to trace a prediction back to its origin.
    for key in (
        "algorithm",
        "training_date",
        "random_seed",
        "operating_threshold",
        "feature_columns",
        "data_checksums",
        "libraries",
        "artifact_sha256",
    ):
        assert key in metadata, f"metadata missing {key}"
    assert list_versions()


@pytest.mark.unit
def test_loading_an_unknown_version_fails_loudly():
    from hrai.ml.registry import load_model

    with pytest.raises(FileNotFoundError):
        load_model("v999")


# ---- engagement analytics (Step 10) -------------------------------------


@pytest.mark.unit
def test_engagement_summary_uses_the_correct_likert_scale():
    from hrai.intelligence.engagement import engagement_summary

    summary = engagement_summary()
    assert summary["scale"] == "1-5 Likert"
    assert 1 <= summary["average_engagement"] <= 5
    assert summary["employees"] == 3000


@pytest.mark.unit
@pytest.mark.parametrize("dimension", ["DepartmentType", "Division", "tenure_band"])
def test_engagement_breakdowns(dimension):
    from hrai.intelligence.engagement import engagement_by

    frame = engagement_by(dimension)
    assert not frame.empty
    assert frame["avg_engagement"].between(1, 5).all()
    # Aggregation runs on employee grain, so the parts sum to the whole.
    assert frame["employees"].sum() == 3000


@pytest.mark.unit
def test_unknown_engagement_dimension_raises():
    from hrai.intelligence.engagement import engagement_by

    with pytest.raises(KeyError):
        engagement_by("NotAColumn")


@pytest.mark.unit
def test_lowest_engagement_is_sorted_worst_first():
    from hrai.intelligence.engagement import lowest_engagement

    frame = lowest_engagement(n=20)
    assert len(frame) == 20
    assert frame["engagement_score"].is_monotonic_increasing


# ---- cross-population transfer ------------------------------------------


@pytest.mark.unit
def test_transfer_contract_excludes_features_that_cannot_transfer():
    from hrai.ml.transfer import contract_features

    features = set(contract_features())
    # Disjoint vocabularies across populations.
    assert "Department" not in features
    assert "JobRole" not in features
    # Population A takes only {3, 4}, so there is no variation to transfer.
    assert "PerformanceRating" not in features
    assert {"Age", "YearsAtCompany", "JobSatisfaction"} <= features


@pytest.mark.unit
def test_population_b_is_mapped_and_clipped_to_the_training_support():
    from hrai.ml.transfer import population_a_contract, population_b_contract

    features_a, _ = population_a_contract()
    features_b, y_b, info = population_b_contract()

    assert list(features_a.columns) == list(features_b.columns)
    assert len(features_b) == 3000
    assert y_b.isin([0, 1]).all()
    # Extrapolating beyond the fitted support is where silent nonsense begins.
    for column in features_a.select_dtypes("number").columns:
        assert features_b[column].min() >= features_a[column].min()
        assert features_b[column].max() <= features_a[column].max()
    assert info["clipping"], "Population B genuinely exceeds A's age range"


@pytest.mark.unit
def test_likert_rescale_maps_five_point_onto_four_point():
    from hrai.ml.transfer import population_b_contract
    from hrai.utils.io import load_processed

    features_b, _, _ = population_b_contract()
    raw = load_processed("engagement_processed")
    assert features_b["JobSatisfaction"].between(1, 4).all()
    # A 5 on the source scale must land on the top of the target scale.
    top = raw["satisfaction_score"] == 5
    assert features_b.loc[top.to_numpy(), "JobSatisfaction"].max() == pytest.approx(4.0)


@pytest.mark.unit
def test_psi_is_zero_for_identical_distributions_and_large_for_shifted_ones():
    from hrai.ml.transfer import population_shift

    frame = pd.DataFrame({"x": range(1000)})
    assert population_shift(frame, frame.copy())[0]["psi"] == pytest.approx(0.0, abs=1e-6)
    shifted = pd.DataFrame({"x": range(5000, 6000)})
    assert population_shift(frame, shifted)[0]["shift"] == "severe"


# ---- monitoring (Steps 25-27) -------------------------------------------


@pytest.mark.unit
def test_psi_flags_a_shifted_feature():
    from hrai.monitoring.drift import population_stability_index

    reference = pd.Series(range(1000))
    assert population_stability_index(reference, reference.copy()) == pytest.approx(0, abs=1e-6)
    assert population_stability_index(reference, pd.Series(range(5000, 6000))) > 0.25


@pytest.mark.unit
def test_feature_drift_reports_psi_and_a_ks_test():
    from hrai.monitoring.drift import feature_drift

    reference = pd.DataFrame({"Age": range(20, 60)})
    current = pd.DataFrame({"Age": range(50, 90)})
    rows = feature_drift(reference, current, features=["Age"])
    assert rows[0]["psi_drifted"] is True
    assert rows[0]["ks_drifted"] is True


@pytest.mark.unit
def test_retraining_trigger_is_relative_to_the_models_own_baseline():
    """An absolute F1 floor would fire on every run and be trained away as noise."""
    from hrai.monitoring.drift import retraining_decision

    healthy = {
        "current": {"f1": 0.40},
        "f1_below_retrain_threshold": False,
        "baseline_f1": 0.40,
        "f1_relative_limit": 0.34,
    }
    assert retraining_decision([], healthy)["retrain_recommended"] is False

    drifted = [{"feature": "Age", "psi": 0.9}]
    decision = retraining_decision(drifted, healthy)
    assert decision["retrain_recommended"] is True
    assert "Age" in decision["reasons"][0]


@pytest.mark.unit
def test_performance_monitor_labels_its_basis_honestly():
    from hrai.monitoring.drift import performance_monitor

    result = performance_monitor()
    assert "in-sample" in result["basis"]
    assert result["current"]["f1"] > 0
    assert result["baseline_f1"] > 0
