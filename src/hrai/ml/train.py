"""Model training and comparison (Steps 06-09).

    make train            # or: python -m hrai.ml.train

Sequence:

1. Stratified train/test split. The test set is touched exactly once, at the end.
2. Repeated stratified K-fold CV on train for all three candidates. With 237
   positive cases a single split is too high-variance to compare on, so every
   number is a mean +/- std across folds.
3. Winner chosen on PR-AUC (the honest metric under imbalance), with recall as
   the tie-breaker because missing a genuinely at-risk employee is the expensive
   error.
4. Winner calibrated, because Step 16 turns probabilities into risk bands.
5. Operating threshold chosen on out-of-fold predictions by expected cost —
   never on the test set.
6. Final evaluation on the held-out test set, then versioned to `models/vN/`.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from xgboost import XGBClassifier

from hrai.features.engineering import FeatureEngineer, feature_columns
from hrai.features.pipeline import build_pipeline, fitted_feature_names
from hrai.ml.evaluate import (
    capacity_threshold,
    compute_metrics,
    reliability_table,
    select_threshold,
)
from hrai.ml.registry import save_model
from hrai.ml.tracking import log_training_run, track_run
from hrai.utils.config import get, project_root, raw_path, seed
from hrai.utils.io import file_checksum, load_processed
from hrai.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

TARGET = "attrition_flag"


def candidate_models(scale_pos_weight: float = 1.0) -> dict[str, dict[str, Any]]:
    """The three candidates from the Build Notes, plus how each must be fed.

    `scale` matters only for the linear model; trees are invariant to it, and an
    unnecessary scaler in a saved artifact misleads whoever reads it later.

    `scale_pos_weight` is XGBoost's equivalent of `class_weight="balanced"`.
    Without it XGBoost competes against two class-balanced models while itself
    optimising for the majority class, which is not a fair comparison.
    """
    random_state = seed()
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state
            ),
            "scale": True,
            "role": "explainable baseline",
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            "scale": False,
            "role": "non-linear relationships the linear model cannot see",
        },
        "xgboost": {
            "estimator": XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            ),
            "scale": False,
            "role": "usually strongest on tabular data",
        },
    }


def _cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=int(get("model.cv_folds", 5)),
        n_repeats=int(get("model.cv_repeats", 3)),
        random_state=seed(),
    )


def imbalance_ratio(y: pd.Series) -> float:
    """negatives / positives — XGBoost's `scale_pos_weight`."""
    positives = int(y.sum())
    return float((len(y) - positives) / positives) if positives else 1.0


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Step 07 — identical pipeline and folds for all three candidates."""
    scoring = ["precision", "recall", "f1", "roc_auc", "average_precision", "neg_brier_score"]
    rows = []
    for name, spec in candidate_models(imbalance_ratio(y_train)).items():
        pipeline = build_pipeline(X_train, spec["estimator"], scale=spec["scale"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipeline, X_train, y_train, cv=_cv(), scoring=scoring, n_jobs=1, error_score="raise"
            )
        row = {"model": name, "role": spec["role"]}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            key = "brier" if metric == "neg_brier_score" else metric
            mean = -values.mean() if metric == "neg_brier_score" else values.mean()
            row[f"{key}_mean"] = round(float(mean), 4)
            row[f"{key}_std"] = round(float(values.std()), 4)
        rows.append(row)
        log.info(
            "candidate evaluated",
            extra={
                "model": name,
                "pr_auc": row["average_precision_mean"],
                "recall": row["recall_mean"],
                "roc_auc": row["roc_auc_mean"],
            },
        )
    table = pd.DataFrame(rows)
    # PR-AUC is the honest headline under imbalance; recall breaks ties because a
    # missed at-risk employee is the expensive error.
    return table.sort_values(["average_precision_mean", "recall_mean"], ascending=False)


def select_calibration_method(
    base_pipeline, X_train: pd.DataFrame, y_train: pd.Series, spec: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """Pick isotonic or sigmoid on out-of-fold Brier score and smoothness.

    `saturated_fraction` is the share of out-of-fold predictions pinned within
    1e-6 of 0 or 1. A saturated probability cannot respond to a counterfactual
    nudge, so a method that saturates heavily is rejected even when its Brier
    score is marginally better.
    """
    from sklearn.metrics import brier_score_loss

    folds = int(get("model.cv_folds", 5))
    results = []
    for candidate in ("sigmoid", "isotonic"):
        model = CalibratedClassifierCV(
            estimator=build_pipeline(X_train, spec["estimator"], scale=spec["scale"]),
            method=candidate,
            cv=folds,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof = cross_val_predict(
                model, X_train, y_train, cv=folds, method="predict_proba", n_jobs=1
            )[:, 1]
        saturated = float(((oof <= 1e-6) | (oof >= 1 - 1e-6)).mean())
        results.append(
            {
                "method": candidate,
                "oof_brier": round(float(brier_score_loss(y_train, oof)), 5),
                "saturated_fraction": round(saturated, 4),
                "distinct_values": int(len(np.unique(np.round(oof, 6)))),
            }
        )

    # Reject a method that saturates more than 2% of predictions; among the rest,
    # the best Brier score wins.
    usable = [r for r in results if r["saturated_fraction"] <= 0.02] or results
    winner = min(usable, key=lambda r: r["oof_brier"])
    log.info(
        "calibration method selected", extra={"method": winner["method"], "candidates": results}
    )
    return winner["method"], results


def main() -> int:
    setup_logging()
    np.random.seed(seed())
    root = project_root()

    df = load_processed("employee_attrition_processed")
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(get("model.test_size", 0.2)),
        stratify=y,
        random_state=seed(),
    )
    log.info(
        "data split",
        extra={
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_positive_rate": round(float(y_train.mean()), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
        },
    )

    comparison = compare_models(X_train, y_train)
    winner_name = str(comparison.iloc[0]["model"])
    spec = candidate_models(imbalance_ratio(y_train))[winner_name]
    log.info(
        "winner selected",
        extra={"model": winner_name, "pr_auc": comparison.iloc[0]["average_precision_mean"]},
    )

    base_pipeline = build_pipeline(X_train, spec["estimator"], scale=spec["scale"])

    # --- calibration: probabilities must be right, not just well-ordered ----
    # The method is chosen on evidence rather than declared. Two things matter:
    #   * out-of-fold Brier score, and
    #   * smoothness. Isotonic regression is a step function, so it saturates
    #     (a wide band of raw scores all map to exactly 1.0) and a small
    #     perturbation often moves the calibrated probability by exactly zero.
    #     That is fatal for the counterfactual engine, which measures risk as a
    #     *difference* after nudging one feature.
    # With 237 positive cases isotonic is also the more overfit-prone choice.
    method, calibration_comparison = select_calibration_method(
        base_pipeline, X_train, y_train, spec
    )
    calibrated = CalibratedClassifierCV(
        estimator=base_pipeline, method=method, cv=int(get("model.cv_folds", 5))
    )

    # --- threshold on out-of-fold train predictions, never on test ----------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof_prob = cross_val_predict(
            calibrated,
            X_train,
            y_train,
            cv=int(get("model.cv_folds", 5)),
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
    threshold_result = select_threshold(y_train.to_numpy(), oof_prob, X_train.get("MonthlyIncome"))
    threshold = threshold_result["threshold"]
    # A second, operationally-bounded operating point for the dashboard's
    # watchlist view — see `capacity_threshold` for why both exist.
    capacity_result = capacity_threshold(oof_prob)
    metrics_at_capacity_note = compute_metrics(
        y_train.to_numpy(), oof_prob, capacity_result["threshold"]
    )

    # --- fit final model, evaluate once on the held-out test set ------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated.fit(X_train, y_train)
    test_prob = calibrated.predict_proba(X_test)[:, 1]

    uncalibrated = build_pipeline(X_train, spec["estimator"], scale=spec["scale"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uncalibrated.fit(X_train, y_train)
    uncal_prob = uncalibrated.predict_proba(X_test)[:, 1]

    metrics_calibrated = compute_metrics(y_test.to_numpy(), test_prob, threshold)
    metrics_at_half = compute_metrics(y_test.to_numpy(), test_prob, 0.5)
    metrics_uncalibrated = compute_metrics(y_test.to_numpy(), uncal_prob, threshold)

    log.info(
        "held-out test performance",
        extra={
            "recall": metrics_calibrated.recall,
            "precision": metrics_calibrated.precision,
            "f1": metrics_calibrated.f1,
            "roc_auc": metrics_calibrated.roc_auc,
            "pr_auc": metrics_calibrated.pr_auc,
            "brier": metrics_calibrated.brier,
            "threshold": threshold,
        },
    )

    # --- persist -----------------------------------------------------------
    engineered = FeatureEngineer().fit(X_train).transform(X_train)
    columns = feature_columns(engineered)
    encoded = fitted_feature_names(uncalibrated)

    checksums = {name: file_checksum(raw_path(name))[:16] for name in ("employee_attrition",)}

    metrics_payload = {
        "test_calibrated": metrics_calibrated.to_dict(),
        "test_calibrated_at_0.5": metrics_at_half.to_dict(),
        "test_uncalibrated": metrics_uncalibrated.to_dict(),
        "brier_improvement_from_calibration": round(
            metrics_uncalibrated.brier - metrics_calibrated.brier, 4
        ),
        "cv_comparison": comparison.to_dict(orient="records"),
    }

    # The calibrated model is what predicts; the uncalibrated base pipeline is
    # what SHAP explains. Calibration is a monotonic transform of the score, so
    # it changes the number but not which features drove it — and explaining
    # through the calibration wrapper would need a slow model-agnostic explainer.
    import joblib as _joblib

    version, version_dir = save_model(
        calibrated,
        algorithm=f"{winner_name} + {method} calibration",
        metrics=metrics_payload,
        threshold=threshold,
        feature_columns=columns,
        encoded_feature_names=encoded,
        data_checksums=checksums,
        extra={
            "calibration_method": method,
            "calibration_comparison": calibration_comparison,
            "threshold_selection": {k: v for k, v in threshold_result.items() if k != "curve"},
            "capacity_operating_point": {
                **capacity_result,
                "oof_precision": metrics_at_capacity_note.precision,
                "oof_recall": metrics_at_capacity_note.recall,
            },
            "class_balance": "class_weight / scale_pos_weight",
            "cv": f"{get('model.cv_folds')}-fold x {get('model.cv_repeats')} repeats",
            "notes": (
                "Trained on Population A only. Population B is scored via the "
                "Common Feature Contract transfer model; see hrai.ml.transfer."
            ),
        },
    )

    _joblib.dump(uncalibrated, version_dir / "attrition_base.joblib")
    background = X_train.sample(n=min(200, len(X_train)), random_state=seed()).reset_index(
        drop=True
    )
    _joblib.dump(background, version_dir / "shap_background.joblib")
    log.info(
        "explainer assets saved",
        extra={"base_model": "attrition_base.joblib", "background_rows": len(background)},
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "version": version,
        "winner": winner_name,
        "comparison": comparison.to_dict(orient="records"),
        "calibration_comparison": calibration_comparison,
        "threshold_selection": threshold_result,
        "capacity_operating_point": {
            **capacity_result,
            "oof_precision": metrics_at_capacity_note.precision,
            "oof_recall": metrics_at_capacity_note.recall,
        },
        "metrics": metrics_payload,
        "reliability_calibrated": reliability_table(y_test.to_numpy(), test_prob),
        "reliability_uncalibrated": reliability_table(y_test.to_numpy(), uncal_prob),
    }
    (root / "docs" / "model_training_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    # Also drop the winner at the path the Build Notes name.
    import shutil

    shutil.copy2(
        version_dir / "attrition_pipeline.joblib",
        root / get("paths.models", "models") / "attrition_pipeline.joblib",
    )

    with track_run(f"{winner_name}-{version}") as mlflow:
        log_training_run(
            mlflow,
            winner=winner_name,
            comparison=comparison.to_dict(orient="records"),
            metrics=metrics_payload,
            threshold_result=threshold_result,
            version=version,
            feature_count=len(encoded),
        )

    log.info("steps 06-09 complete", extra={"version": version, "winner": winner_name})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
