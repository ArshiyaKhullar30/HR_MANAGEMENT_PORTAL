"""Data drift and model performance monitoring (Steps 25-27).

The Build Notes stage this correctly: plain pandas statistics first, Evidently
once the basic version proves useful. Both are here, in that order — the pandas
implementation is the one that runs, and Evidently is generated alongside when
it is installed.

Three questions, answered separately because they fail separately:

1. **Has the input distribution moved?** PSI and a Kolmogorov-Smirnov test per
   watched feature, production against training. Needs no ground truth, so it is
   the earliest available warning.
2. **Has the prediction distribution moved?** Cheaper still, and it catches
   pipeline breakage that feature-level checks miss.
3. **Has performance actually dropped?** Only answerable once real outcomes
   arrive. Drift is a leading indicator, not proof of decay — a model can be
   fine on shifted data, and the retraining rule below treats it that way.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from hrai.utils.config import get, project_root
from hrai.utils.io import load_processed
from hrai.utils.logger import get_logger

log = get_logger(__name__)

PSI_BINS = 10


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = PSI_BINS
) -> float:
    """PSI between two distributions. >0.25 is the conventional 'significant' line."""
    reference, current = reference.dropna(), current.dropna()
    if reference.empty or current.empty:
        return float("nan")

    if pd.api.types.is_numeric_dtype(reference):
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
        cur_counts = np.histogram(current, bins=edges)[0].astype(float)
    else:
        categories = reference.astype(str).value_counts().index
        ref_counts = (
            reference.astype(str).value_counts().reindex(categories).fillna(0).to_numpy(float)
        )
        cur_counts = (
            current.astype(str).value_counts().reindex(categories).fillna(0).to_numpy(float)
        )

    # Clip so an empty bin cannot produce an infinite PSI.
    ref_share = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_share = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(((ref_share - cur_share) * np.log(ref_share / cur_share)).sum())


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None
) -> list[dict[str, Any]]:
    features = features or get("monitoring.drift_features", [])
    psi_threshold = float(get("monitoring.psi_threshold", 0.20))
    ks_alpha = float(get("monitoring.ks_pvalue_threshold", 0.05))

    rows = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        psi = population_stability_index(reference[feature], current[feature])
        ks_p = None
        if pd.api.types.is_numeric_dtype(reference[feature]):
            ks_p = float(
                stats.ks_2samp(reference[feature].dropna(), current[feature].dropna()).pvalue
            )
        rows.append(
            {
                "feature": feature,
                "psi": round(psi, 4),
                "psi_drifted": bool(psi > psi_threshold),
                "ks_pvalue": round(ks_p, 6) if ks_p is not None else None,
                "ks_drifted": bool(ks_p < ks_alpha) if ks_p is not None else None,
                "reference_mean": (
                    round(float(reference[feature].mean()), 3)
                    if pd.api.types.is_numeric_dtype(reference[feature])
                    else None
                ),
                "current_mean": (
                    round(float(current[feature].mean()), 3)
                    if pd.api.types.is_numeric_dtype(current[feature])
                    else None
                ),
                "severity": ("severe" if psi > 0.25 else "moderate" if psi > 0.10 else "stable"),
            }
        )
    return sorted(rows, key=lambda r: -(r["psi"] if r["psi"] == r["psi"] else 0))


def prediction_drift() -> dict[str, Any]:
    """Compare logged production predictions against the training distribution."""
    from app.services.prediction_log import read_predictions

    predictions = read_predictions()
    if predictions.empty:
        return {
            "available": False,
            "reason": "No predictions logged yet. Serve some requests first.",
        }

    reference_report = project_root() / "docs" / "model_training_report.json"
    training_positive_rate = 0.1612
    if reference_report.exists():
        payload = json.loads(reference_report.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {}).get("test_calibrated", {})
        if metrics.get("n"):
            training_positive_rate = metrics["support_positive"] / metrics["n"]

    flagged_rate = float(predictions["flagged"].mean())
    return {
        "available": True,
        "predictions_logged": int(len(predictions)),
        "mean_probability": round(float(predictions["probability"].mean()), 4),
        "flagged_rate": round(flagged_rate, 4),
        "training_positive_rate": round(training_positive_rate, 4),
        "risk_band_distribution": predictions["risk_band"].value_counts().to_dict(),
        "model_versions_seen": sorted(predictions["model_version"].unique().tolist()),
        "note": (
            "A flagged rate far above the training positive rate is expected here: "
            "the operating threshold is deliberately low because missing a leaver "
            "costs ~24x more than an unnecessary conversation."
        ),
    }


def performance_monitor(outcomes: pd.DataFrame | None = None) -> dict[str, Any]:
    """Step 26 — recompute metrics once real outcomes are known.

    In production, `outcomes` is a table of employee_id -> did_they_actually_leave,
    joined to the prediction log. Here it defaults to the labelled training
    population, which demonstrates the mechanism honestly: these are in-sample
    numbers and are labelled as such, not passed off as live performance.
    """
    from hrai.ml.evaluate import compute_metrics
    from hrai.ml.registry import load_model

    model, metadata = load_model()
    threshold = float(metadata.get("operating_threshold", 0.5))

    if outcomes is None:
        frame = load_processed("employee_attrition_processed")
        y_true = frame["attrition_flag"].astype(int).to_numpy()
        probabilities = model.predict_proba(frame.drop(columns=["attrition_flag"]))[:, 1]
        basis = "labelled training population (in-sample demonstration, not live performance)"
    else:
        y_true = outcomes["actual"].astype(int).to_numpy()
        probabilities = outcomes["probability"].to_numpy()
        basis = "joined production outcomes"

    current = compute_metrics(y_true, probabilities, threshold)
    baseline = metadata.get("metrics", {}).get("test_calibrated", {})
    baseline_f1 = float(baseline.get("f1", current.f1)) if baseline else current.f1

    # Judged against the model's own validated baseline. An absolute floor would
    # be arbitrary: this model's F1 ceiling is set by a deliberately low
    # operating threshold, not by how well it separates the classes.
    relative_drop = float(get("monitoring.retrain_triggers.f1_relative_drop", 0.15))
    absolute_floor = float(get("monitoring.retrain_triggers.f1_absolute_floor", 0.20))
    relative_limit = baseline_f1 * (1.0 - relative_drop)

    return {
        "basis": basis,
        "model_version": metadata.get("version"),
        "current": current.to_dict(),
        "baseline_at_training": baseline,
        "baseline_f1": round(baseline_f1, 4),
        "f1_delta": round(current.f1 - baseline_f1, 4),
        "f1_relative_limit": round(relative_limit, 4),
        "f1_absolute_floor": absolute_floor,
        "f1_below_retrain_threshold": bool(
            current.f1 < relative_limit or current.f1 < absolute_floor
        ),
    }


def retraining_decision(
    drift_rows: list[dict[str, Any]], performance: dict[str, Any]
) -> dict[str, Any]:
    """Step 27 — the retraining rule, evaluated rather than left to judgement.

        IF drift > threshold OR F1 below threshold OR 6 months of new data
        THEN retrain

    Written down in advance precisely so it is not a judgement call under
    pressure later.
    """
    triggers = get("monitoring.retrain_triggers", {}) or {}
    psi_limit = float(triggers.get("drift_psi_above", 0.20))

    drifted = [r["feature"] for r in drift_rows if r.get("psi", 0) > psi_limit]
    f1_low = bool(performance.get("f1_below_retrain_threshold"))

    reasons = []
    if drifted:
        reasons.append(f"input drift above PSI {psi_limit} on: {', '.join(drifted)}")
    if f1_low:
        reasons.append(
            f"F1 {performance['current']['f1']} fell below its retraining limit "
            f"({performance.get('f1_relative_limit')}, i.e. "
            f"{100 * float(triggers.get('f1_relative_drop', 0.15)):.0f}% under the "
            f"validated baseline of {performance.get('baseline_f1')})"
        )

    return {
        "retrain_recommended": bool(reasons),
        "reasons": reasons or ["No trigger met."],
        "rule": (
            "IF drift > threshold OR F1 below threshold OR "
            f"{triggers.get('months_of_new_data', 6)} months of new data THEN retrain"
        ),
        "next_step": (
            "new data -> validation -> training -> evaluation -> MLflow tracking -> "
            "human approval -> deploy new version"
            if reasons
            else "Continue monitoring; no action required."
        ),
    }


def run_monitoring() -> dict[str, Any]:
    """Full monitoring sweep. Population B is the natural drift comparison."""
    reference = load_processed("employee_attrition_processed")

    # Population B is a genuinely different workforce, which makes it the ideal
    # stand-in for "production data that has moved" — no simulation needed.
    engagement = load_processed("engagement_processed")
    current = pd.DataFrame(
        {
            "Age": engagement["age"].astype(float),
            "YearsAtCompany": engagement["tenure_years"].astype(float),
            "JobSatisfaction": engagement["satisfaction_score"].astype(float),
        }
    )

    drift_rows = feature_drift(
        reference, current, features=["Age", "YearsAtCompany", "JobSatisfaction"]
    )
    performance = performance_monitor()
    predictions = prediction_drift()
    decision = retraining_decision(drift_rows, performance)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "employee_attrition_processed (training population)",
        "current": "engagement_processed (Population B, standing in for shifted production data)",
        "feature_drift": drift_rows,
        "prediction_drift": predictions,
        "performance": performance,
        "retraining_decision": decision,
        "evidently_report": _evidently(reference, current),
    }
    log.info(
        "monitoring sweep complete",
        extra={
            "drifted_features": [r["feature"] for r in drift_rows if r["psi_drifted"]],
            "retrain_recommended": decision["retrain_recommended"],
        },
    )
    return payload


def _evidently(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Generate an Evidently HTML report when the library is available."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        columns = [c for c in current.columns if c in reference.columns]
        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=reference[columns], current_data=current[columns])
        path = project_root() / "docs" / "drift_report.html"
        result.save_html(str(path))
        return {"generated": True, "path": str(path.relative_to(project_root()))}
    except Exception as exc:  # noqa: BLE001 - reporting must never break monitoring
        log.warning("evidently report unavailable", extra={"error": str(exc)[:160]})
        return {"generated": False, "reason": str(exc)[:160]}


def main() -> int:
    from hrai.utils.logger import setup_logging

    setup_logging()
    payload = run_monitoring()
    (project_root() / "docs" / "monitoring_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
