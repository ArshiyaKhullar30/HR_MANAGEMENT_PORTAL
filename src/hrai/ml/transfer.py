"""Cross-population transfer and external validation.

Finding F1 said the two employee datasets are different companies, so they
cannot be joined. That is a constraint — but it is also an opportunity almost no
project at this scale gets: **Population B carries its own attrition label.**

So instead of assuming the model generalises, we measure it. Train on
Population A, restrict to features whose meaning genuinely transfers, score
Population B, and score that against B's own `Voluntarily Terminated` outcomes.
The answer to "does this model survive contact with a different workforce?"
becomes a number rather than a hope.

Being honest about the contract is most of the work:

* `Department` and `JobRole` are excluded — the vocabularies are disjoint
  (only "Sales" is shared; 0 of 9 vs 31 job titles overlap). One-hot encoding
  with `handle_unknown="ignore"` would quietly zero them and look like it
  worked.
* `PerformanceRating` is excluded — Population A only ever takes {3, 4}, so
  there is no variation to learn.
* Likert scales are rescaled: B is 1-5, A is 1-4.
* Population B's age and tenure run outside A's training support (tenure: A
  spans 0-40 years, B spans 0.4-5.4). Rows are clipped to A's observed range and
  the clipping is counted and reported, because extrapolating a fitted model
  beyond its support is exactly where silent nonsense comes from.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from hrai.features.pipeline import build_pipeline
from hrai.ml.evaluate import compute_metrics
from hrai.utils.config import get, project_root, seed
from hrai.utils.io import load_processed
from hrai.utils.logger import get_logger

log = get_logger(__name__)


def contract_features() -> list[str]:
    return list(get("features.common_feature_contract", []))


def population_a_contract() -> tuple[pd.DataFrame, pd.Series]:
    df = load_processed("employee_attrition_processed")
    return df[contract_features()].copy(), df["attrition_flag"].astype(int)


def population_b_contract() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Map Population B onto Population A's feature space, honestly."""
    df = load_processed("engagement_processed")
    lo_from, hi_from = get("features.transfer_likert_rescale.from_range", [1, 5])
    lo_to, hi_to = get("features.transfer_likert_rescale.to_range", [1, 4])

    def rescale(series: pd.Series) -> pd.Series:
        scaled = (series.astype(float) - lo_from) / (hi_from - lo_from)
        return (lo_to + scaled * (hi_to - lo_to)).round(3)

    out = pd.DataFrame(
        {
            "Age": df["age"].astype(float),
            "Gender": df["GenderCode"].astype(str).str.strip(),
            "YearsAtCompany": df["tenure_years"].astype(float),
            "JobSatisfaction": rescale(df["satisfaction_score"]),
            "WorkLifeBalance": rescale(df["work_life_balance_score"]),
        }
    )[contract_features()]

    # Clip to Population A's observed support and count what we clipped.
    a_features, _ = population_a_contract()
    clipping: dict[str, Any] = {}
    for column in out.select_dtypes(include=[np.number]).columns:
        lo, hi = float(a_features[column].min()), float(a_features[column].max())
        outside = int(((out[column] < lo) | (out[column] > hi)).sum())
        if outside:
            clipping[column] = {
                "clipped_rows": outside,
                "training_support": [lo, hi],
                "pct": round(100 * outside / len(out), 2),
            }
        out[column] = out[column].clip(lo, hi)

    y = df["is_voluntary_exit"].astype(int)
    return out, y, {"clipping": clipping, "rows": int(len(out))}


def population_shift(a: pd.DataFrame, b: pd.DataFrame) -> list[dict[str, Any]]:
    """Population Stability Index per feature — how far B is from A.

    PSI > 0.25 is the conventional "significant shift" line. Reporting it beside
    the transfer result is what turns a disappointing number into a diagnosis.
    """
    rows = []
    for column in a.columns:
        if not pd.api.types.is_numeric_dtype(a[column]):
            a_dist = a[column].value_counts(normalize=True)
            b_dist = b[column].value_counts(normalize=True).reindex(a_dist.index).fillna(1e-6)
            psi = float(((a_dist - b_dist) * np.log((a_dist + 1e-6) / (b_dist + 1e-6))).sum())
        else:
            edges = np.unique(np.quantile(a[column].dropna(), np.linspace(0, 1, 11)))
            if len(edges) < 3:
                continue
            a_counts = np.histogram(a[column].dropna(), bins=edges)[0].astype(float)
            b_counts = np.histogram(b[column].dropna(), bins=edges)[0].astype(float)
            a_share = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
            b_share = np.clip(b_counts / max(b_counts.sum(), 1), 1e-6, None)
            psi = float(((a_share - b_share) * np.log(a_share / b_share)).sum())
        rows.append(
            {
                "feature": column,
                "psi": round(psi, 4),
                "shift": ("severe" if psi > 0.25 else "moderate" if psi > 0.10 else "stable"),
                "population_a_mean": (
                    round(float(a[column].mean()), 2)
                    if pd.api.types.is_numeric_dtype(a[column])
                    else None
                ),
                "population_b_mean": (
                    round(float(b[column].mean()), 2)
                    if pd.api.types.is_numeric_dtype(b[column])
                    else None
                ),
            }
        )
    return sorted(rows, key=lambda r: -r["psi"])


def run_transfer_validation() -> dict[str, Any]:
    """Train restricted on A, validate on A, then externally validate on B."""
    features = contract_features()
    X_a, y_a = population_a_contract()
    X_b, y_b, mapping_info = population_b_contract()

    X_train, X_test, y_train, y_test = train_test_split(
        X_a,
        y_a,
        test_size=float(get("model.test_size", 0.2)),
        stratify=y_a,
        random_state=seed(),
    )

    estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed())
    pipeline = build_pipeline(X_train, estimator, scale=True, engineer=False)
    model = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    prob_a = model.predict_proba(X_test)[:, 1]
    prob_b = model.predict_proba(X_b)[:, 1]

    # Compare at each population's own base rate, so the two numbers are about
    # generalisation rather than about a threshold tuned on one of them.
    threshold_a = float(np.quantile(prob_a, 1 - y_a.mean()))
    threshold_b = float(np.quantile(prob_b, 1 - y_b.mean()))

    metrics_a = compute_metrics(y_test.to_numpy(), prob_a, threshold_a)
    metrics_b = compute_metrics(y_b.to_numpy(), prob_b, threshold_b)

    shift = population_shift(X_a, X_b)
    auc_drop = round(metrics_a.roc_auc - metrics_b.roc_auc, 4)

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "contract_features": features,
        "excluded_from_contract": {
            "Department, JobRole": "vocabularies are disjoint across populations",
            "PerformanceRating": "Population A takes only {3, 4} — no variation to transfer",
        },
        "population_a": {
            "rows": int(len(X_a)),
            "base_rate": round(float(y_a.mean()), 4),
            "held_out_metrics": metrics_a.to_dict(),
        },
        "population_b": {
            "rows": int(len(X_b)),
            "base_rate": round(float(y_b.mean()), 4),
            "external_metrics": metrics_b.to_dict(),
            **mapping_info,
        },
        "roc_auc_drop_on_transfer": auc_drop,
        "generalises": bool(metrics_b.roc_auc >= 0.60),
        "distribution_shift": shift,
        "interpretation": _interpret(metrics_a.roc_auc, metrics_b.roc_auc, shift),
    }

    log.info(
        "cross-population transfer validated",
        extra={
            "auc_population_a": metrics_a.roc_auc,
            "auc_population_b": metrics_b.roc_auc,
            "auc_drop": auc_drop,
            "generalises": result["generalises"],
            "severe_shift_features": [s["feature"] for s in shift if s["shift"] == "severe"],
        },
    )
    return result


def _interpret(auc_a: float, auc_b: float, shift: list[dict[str, Any]]) -> str:
    severe = [s["feature"] for s in shift if s["shift"] == "severe"]
    drop = auc_a - auc_b
    if auc_b < 0.55:
        verdict = (
            "The model does not transfer: on Population B it performs at close to "
            "chance. Attrition risk for Population B should be reported as "
            "unavailable rather than estimated."
        )
    elif drop > 0.15:
        verdict = (
            "The model transfers weakly. Population B scores are directionally "
            "useful for ranking but should not be read as calibrated probabilities."
        )
    else:
        verdict = (
            "The model transfers with modest degradation, which is the expected "
            "result for a restricted feature contract across two workforces."
        )
    if severe:
        verdict += (
            f" Severe distribution shift on {', '.join(severe)} explains much of "
            "the gap and is the first thing to address."
        )
    return verdict


def main() -> int:
    from hrai.utils.logger import setup_logging

    setup_logging()
    result = run_transfer_validation()
    (project_root() / "docs" / "transfer_validation.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
