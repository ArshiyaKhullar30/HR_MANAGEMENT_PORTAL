"""Fairness audit.

A model that influences decisions about people needs a bias check, and this one
is cheap to do properly. We slice performance by protected attribute and report
three things per group:

* **Equal opportunity** — do at-risk people get flagged at the same rate
  regardless of group? (difference in true-positive rate)
* **Predictive equality** — are people who stay wrongly flagged at the same
  rate? (difference in false-positive rate)
* **Calibration** — does a 40% score mean the same thing for every group?

The system also refuses, structurally, to propose an intervention on a protected
attribute: `retention_roi.protected_attributes` is excluded from the lever set
in `hrai.intelligence.counterfactual`, and a test asserts the two never
intersect.

A note on what this cannot do: these are *observational* fairness metrics on a
dataset whose own labels may encode historical bias. Parity here means the model
does not add measurable disparity — not that the underlying process is fair.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from hrai.ml.registry import load_model
from hrai.utils.config import get, project_root
from hrai.utils.io import load_processed
from hrai.utils.logger import get_logger

log = get_logger(__name__)

# Below this a group is too small for its rates to be meaningful.
MIN_GROUP_SIZE = 30
# Conventional "four-fifths"-style tolerance on rate differences.
PARITY_TOLERANCE = 0.10


def _group_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    positives, negatives = y_true == 1, y_true == 0
    return {
        "n": int(len(y_true)),
        "base_rate": round(float(y_true.mean()), 4),
        "flag_rate": round(float(y_pred.mean()), 4),
        "true_positive_rate": (
            round(float(y_pred[positives].mean()), 4) if positives.any() else None
        ),
        "false_positive_rate": (
            round(float(y_pred[negatives].mean()), 4) if negatives.any() else None
        ),
        "mean_predicted_risk": round(float(y_prob.mean()), 4),
        "calibration_gap": round(float(y_prob.mean() - y_true.mean()), 4),
    }


def audit(attributes: list[str] | None = None) -> dict[str, Any]:
    """Slice model performance by each protected attribute."""
    model, metadata = load_model()
    threshold = float(metadata.get("operating_threshold", 0.5))

    df = load_processed("employee_attrition_processed")
    y = df["attrition_flag"].astype(int).to_numpy()
    X = df.drop(columns=["attrition_flag"])
    prob = model.predict_proba(X)[:, 1]

    protected = attributes or [
        a for a in get("retention_roi.protected_attributes", []) if a in df.columns
    ]
    # Age is continuous; band it so groups are comparable.
    banded = df.copy()
    if "Age" in banded.columns:
        banded["AgeBand"] = pd.cut(
            banded["Age"],
            bins=[17, 29, 39, 49, 100],
            labels=["18-29", "30-39", "40-49", "50+"],
        ).astype(str)
        protected = [a for a in protected if a != "Age"] + ["AgeBand"]

    results: dict[str, Any] = {}
    for attribute in protected:
        if attribute not in banded.columns:
            continue
        groups: dict[str, Any] = {}
        for value, index in banded.groupby(attribute).groups.items():
            positions = banded.index.get_indexer(index)
            if len(positions) < MIN_GROUP_SIZE:
                continue
            groups[str(value)] = _group_metrics(y[positions], prob[positions], threshold)
        if len(groups) < 2:
            continue

        tprs = [
            g["true_positive_rate"] for g in groups.values() if g["true_positive_rate"] is not None
        ]
        fprs = [
            g["false_positive_rate"]
            for g in groups.values()
            if g["false_positive_rate"] is not None
        ]
        gaps = [abs(g["calibration_gap"]) for g in groups.values()]

        eo_diff = round(max(tprs) - min(tprs), 4) if tprs else None
        pe_diff = round(max(fprs) - min(fprs), 4) if fprs else None

        results[attribute] = {
            "groups": groups,
            "equal_opportunity_difference": eo_diff,
            "predictive_equality_difference": pe_diff,
            "max_calibration_gap": round(max(gaps), 4) if gaps else None,
            "within_tolerance": bool(
                (eo_diff is None or eo_diff <= PARITY_TOLERANCE)
                and (pe_diff is None or pe_diff <= PARITY_TOLERANCE)
            ),
        }

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": metadata.get("version"),
        "operating_threshold": threshold,
        "tolerance": PARITY_TOLERANCE,
        "min_group_size": MIN_GROUP_SIZE,
        "attributes": results,
        "all_within_tolerance": (
            all(r["within_tolerance"] for r in results.values()) if results else None
        ),
        "protected_attributes_excluded_from_levers": get("retention_roi.protected_attributes", []),
        "interpretation": _interpret(results),
        "limitation": (
            "Observational fairness metrics computed on labels that may themselves "
            "encode historical bias. Parity here means the model adds no measurable "
            "disparity, not that the underlying process is fair."
        ),
    }
    log.info(
        "fairness audit complete",
        extra={
            "attributes": list(results),
            "all_within_tolerance": payload["all_within_tolerance"],
        },
    )
    return payload


def _interpret(results: dict[str, Any]) -> dict[str, str]:
    """Distinguish miscalibration from genuine base-rate difference.

    These are not the same failure. A model can be perfectly calibrated inside
    every group and still flag one group far more often, simply because that
    group really does leave more often. Calibration and equalised odds cannot
    both hold when base rates differ — that is a theorem, not a bug — so the
    honest report says which of the two is being sacrificed and why.
    """
    out: dict[str, str] = {}
    for attribute, result in results.items():
        gaps = [abs(g["calibration_gap"]) for g in result["groups"].values()]
        base_rates = [g["base_rate"] for g in result["groups"].values()]
        max_gap = max(gaps) if gaps else 0.0
        base_spread = (max(base_rates) - min(base_rates)) if base_rates else 0.0

        if result["within_tolerance"]:
            out[attribute] = "Within tolerance on both equal opportunity and predictive equality."
        elif max_gap <= 0.05 and base_spread > 0.05:
            out[attribute] = (
                f"Flagged, but the cause is a genuine base-rate difference of "
                f"{base_spread:.1%} across groups, not miscalibration — every group's "
                f"calibration gap is under {max_gap:.3f}. The model is telling the truth "
                "about differing risk; the disparity is in the workforce, not added by "
                "the model. Calibration and equalised odds cannot both hold when base "
                "rates differ. Route flagged groups to human review and do not use the "
                "score as an automatic decision."
            )
        else:
            out[attribute] = (
                f"Flagged with a calibration gap of {max_gap:.3f}, which indicates the "
                "model itself is miscalibrated for at least one group rather than merely "
                "reflecting different base rates. This warrants retraining with group-aware "
                "calibration before the score is used operationally."
            )
    return out


def main() -> int:
    from hrai.utils.logger import setup_logging

    setup_logging()
    payload = audit()
    (project_root() / "docs" / "fairness_audit.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
