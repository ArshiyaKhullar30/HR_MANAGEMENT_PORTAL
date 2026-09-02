"""MLflow experiment tracking (Step 09, second half).

The Build Notes are explicit about the order: hand-written `metadata.json`
first, MLflow "once manual versioning feels like it's holding me back". Both
now run — the JSON stays the source of truth the API reads at load time (no
tracking server required to serve a prediction), while MLflow accumulates the
run history that makes *comparing* versions tractable.

Tracking is best-effort by design: a training run must never fail because a
tracking backend is unavailable.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from hrai.utils.config import get, project_root
from hrai.utils.logger import get_logger

log = get_logger(__name__)

EXPERIMENT = "attrition-prediction"


def tracking_uri() -> str:
    """Local SQLite backend by default.

    MLflow 3.x put the filesystem store into maintenance mode and refuses it
    unless `MLFLOW_ALLOW_FILE_STORE=true`. SQLite is the supported local
    backend: still a single file, still no server to run, but on the path that
    receives updates.
    """
    import os

    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{project_root() / 'mlflow.db'}")


@contextmanager
def track_run(run_name: str):
    """Yield an MLflow run, or a no-op shim if MLflow cannot start."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=run_name) as run:
            log.info(
                "mlflow run started",
                extra={
                    "run_name": run_name,
                    "mlflow_run_id": run.info.run_id,
                    "tracking_uri": tracking_uri(),
                },
            )
            yield mlflow
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        log.warning("mlflow unavailable; continuing without tracking", extra={"error": str(exc)})

        class _NoOp:
            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        yield _NoOp()


def log_training_run(
    mlflow: Any,
    *,
    winner: str,
    comparison: list[dict[str, Any]],
    metrics: dict[str, Any],
    threshold_result: dict[str, Any],
    version: str,
    feature_count: int,
) -> None:
    """Record parameters, metrics and the comparison table for one training run."""
    mlflow.log_params(
        {
            "winner": winner,
            "model_version": version,
            "random_seed": get("random_seed", 42),
            "cv_folds": get("model.cv_folds"),
            "cv_repeats": get("model.cv_repeats"),
            "calibration": get("model.calibration.method"),
            "operating_threshold": threshold_result.get("threshold"),
            "bayes_threshold": threshold_result.get("bayes_optimal_threshold"),
            "cost_ratio_fn_to_fp": threshold_result.get("cost_ratio_fn_to_fp"),
            "encoded_features": feature_count,
            "trained_on": "Population A (employee_attrition)",
        }
    )

    test = metrics.get("test_calibrated", {})
    mlflow.log_metrics(
        {
            f"test_{k}": float(v)
            for k, v in test.items()
            if isinstance(v, (int, float)) and k != "threshold"
        }
    )
    mlflow.log_metrics(
        {
            "cost_expected": float(threshold_result.get("expected_cost", 0.0)),
            "cost_saving_vs_half": float(threshold_result.get("cost_saving_vs_0.5", 0.0)),
            "brier_improvement_from_calibration": float(
                metrics.get("brier_improvement_from_calibration", 0.0)
            ),
        }
    )

    # Every candidate's CV result, so model selection is auditable later.
    for row in comparison:
        name = row["model"]
        mlflow.log_metrics(
            {
                f"cv_{name}_{metric}": float(row[metric])
                for metric in row
                if metric.endswith("_mean") and isinstance(row[metric], (int, float))
            }
        )

    for artifact in (
        "docs/model_training_report.json",
        "docs/model_card.md",
        "docs/shap_global_importance.json",
    ):
        path = project_root() / artifact
        if path.exists():
            try:
                mlflow.log_artifact(str(path))
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "could not log artifact", extra={"artifact": artifact, "error": str(exc)}
                )

    log.info("mlflow run logged", extra={"winner": winner, "version": version})
