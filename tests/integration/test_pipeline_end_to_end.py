"""End-to-end pipeline tests.

Each numbered step is runnable as `python -m hrai.<module>`, and each is
deterministic, so the suite can invoke the real entry points rather than mocking
them. Regenerating the artifacts is the point: if a step stops being
reproducible, this is where it shows up.
"""

from __future__ import annotations

import json

import pytest

from hrai.utils.config import project_root


def _report(name: str) -> dict:
    path = project_root() / "docs" / name
    assert path.exists(), f"{name} was not generated"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_step_01_profiling_reproduces_the_audit_findings():
    from hrai.profiling.run import main

    assert main() == 0
    findings = _report("findings.json")

    # F1 — the two employee populations are not the same people.
    f1 = findings["F1_employee_population_identity"]
    assert f1["shared_key_count"] == 753
    assert f1["keys_refer_to_same_entities"] is False

    # F4 — no role title reaches O*NET by exact match.
    f4 = findings["F4_role_to_onet_matching"]
    assert f4["attrition_exact_matches"] == 0
    assert f4["engagement_exact_matches"] == 0

    # F5 — the engagement file is event-grain.
    assert findings["F5_engagement_grain"]["distinct_employees"] == 3000

    # F6 — the real data-quality defects.
    f6 = findings["F6_data_quality_defects"]
    assert f6["engagement_exit_date_but_not_terminated"] == 1198
    assert f6["engagement_likert_observed_range"]["Engagement Score"] == [1, 5]

    # Referential integrity of the skill ontology.
    integrity = findings["onet_referential_integrity"]
    assert integrity["essential_skills_subset_of_occupations"] is True
    assert integrity["software_skills_subset_of_occupations"] is True
    assert len(integrity["foundational_skills"]) == 10


@pytest.mark.integration
def test_step_02_validation_reports_the_defects_it_should():
    from hrai.validation.run import main

    assert main("report") == 0
    report = _report("validation_report.json")
    datasets = report["datasets"]

    # The attrition file is genuinely clean; the engagement file is not.
    assert datasets["employee_attrition"]["passed"] is True
    assert datasets["hr_performance_engagement"]["passed"] is False
    assert datasets["hr_performance_engagement"]["failure_count"] == 1198


@pytest.mark.integration
def test_step_03_cleaning_makes_strict_validation_pass():
    from hrai.cleaning.run import main

    assert main(strict=True) == 0
    report = _report("cleaning_report.json")
    assert all(d["passed"] for d in report["datasets"].values())
    assert len(report["artifacts"]) == 6


@pytest.mark.integration
def test_step_04_relationships_document_is_generated():
    from hrai.profiling.relationships import main

    assert main() == 0
    text = (project_root() / "docs" / "data_relationships.md").read_text(encoding="utf-8")
    assert "NO EMPLOYEE-LEVEL JOIN" in text
    assert "ROLE -> O*NET SOC CROSSWALK" in text


@pytest.mark.integration
def test_step_16_intelligence_table_is_rebuilt_consistently():
    from hrai.intelligence.employee_table import build_employee_intelligence, main

    assert main() == 0
    summary = _report("intelligence_summary.json")
    assert summary["employees"] == 4470
    assert summary["population_a"] == 1470
    assert summary["population_b"] == 3000
    # Population B must never receive a fabricated risk score.
    assert summary["risk_unavailable"] == 3000

    table, org_gaps = build_employee_intelligence()
    scored = table[table["population"] == "A"]
    unscored = table[table["population"] == "B"]
    assert scored["attrition_probability"].notna().all()
    assert unscored["attrition_probability"].isna().all()
    assert table["skills_are_derived"].all()
    assert table["person_key"].is_unique
    assert not org_gaps.empty


@pytest.mark.integration
def test_cross_population_transfer_is_measured_not_assumed():
    from hrai.ml.transfer import main, run_transfer_validation

    assert main() == 0
    result = run_transfer_validation()

    # The honest finding: the model does not survive the population change.
    assert result["population_b"]["external_metrics"]["roc_auc"] < 0.60
    assert result["generalises"] is False
    assert "does not transfer" in result["interpretation"]
    # And the diagnosis for why.
    severe = [s["feature"] for s in result["distribution_shift"] if s["shift"] == "severe"]
    assert "YearsAtCompany" in severe


@pytest.mark.integration
def test_monitoring_sweep_runs_and_decides_on_retraining():
    from hrai.monitoring.drift import main, run_monitoring

    assert main() == 0
    payload = run_monitoring()
    assert payload["feature_drift"]
    assert payload["performance"]["current"]["f1"] > 0
    decision = payload["retraining_decision"]
    assert isinstance(decision["retrain_recommended"], bool)
    assert "THEN retrain" in decision["rule"]


@pytest.mark.integration
def test_fairness_audit_entry_point():
    from hrai.ml.fairness import main

    assert main() == 0
    payload = _report("fairness_audit.json")
    assert payload["protected_attributes_excluded_from_levers"]
    assert "limitation" in payload


@pytest.mark.integration
def test_shap_global_report_entry_point():
    from hrai.ml.explain import write_global_report

    payload = write_global_report()
    assert payload["top_features"]
    top = payload["top_features"][0]
    assert {"feature", "label", "importance", "direction"} <= set(top)


@pytest.mark.integration
def test_mlflow_tracking_records_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'test.db'}")
    from hrai.ml.tracking import log_training_run, track_run

    report = _report("model_training_report.json")
    with track_run("pytest-run") as mlflow:
        log_training_run(
            mlflow,
            winner=report["winner"],
            comparison=report["comparison"],
            metrics=report["metrics"],
            threshold_result=report["threshold_selection"],
            version=report["version"],
            feature_count=57,
        )

    import mlflow as mlflow_module

    mlflow_module.set_tracking_uri(f"sqlite:///{tmp_path / 'test.db'}")
    runs = mlflow_module.search_runs(experiment_names=["attrition-prediction"])
    assert len(runs) == 1
    assert "metrics.test_roc_auc" in runs.columns


@pytest.mark.integration
def test_crosswalk_entry_point_writes_its_outputs():
    from hrai.skills.crosswalk import main

    assert main() == 0
    report = _report("role_crosswalk.json")
    assert report["roles"] == 40
    assert (project_root() / "conf" / "role_crosswalk_auto.yaml").exists()
