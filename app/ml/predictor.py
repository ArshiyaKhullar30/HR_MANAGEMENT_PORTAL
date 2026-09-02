"""Prediction service (Steps 18-21).

Wraps the loaded pipeline with the three things a caller actually needs beyond
a number: the risk band, the reason (SHAP), and a record in the prediction log.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml.model_loader import ModelBundle, get_bundle
from app.services.prediction_log import log_prediction
from hrai.ml.evaluate import risk_band
from hrai.utils.logger import get_logger

log = get_logger(__name__)

CAVEAT = (
    "Predicted attrition risk is a statistical estimate from historical patterns, "
    "not a statement about this individual's intentions. Use it to prioritise a "
    "human conversation, never as an automatic decision."
)


def predict_one(
    features: dict[str, Any],
    *,
    explain: bool = True,
    bundle: ModelBundle | None = None,
    log_it: bool = True,
) -> dict[str, Any]:
    """Score a single employee, with the top contributing factors."""
    bundle = bundle or get_bundle()

    # Validate here rather than only at the API boundary, so a caller using the
    # service directly gets the same defaults and the same clear error instead of
    # a cryptic ColumnTransformer KeyError about eighteen missing columns.
    from app.validation.employee_schema import EmployeeFeatures

    validated = EmployeeFeatures(**features).model_dump()
    employee_id = validated.get("employee_id")

    frame = pd.DataFrame([validated])
    probability = float(bundle.pipeline.predict_proba(frame)[:, 1][0])
    band = risk_band(probability)

    top_factors: list[dict[str, Any]] = []
    if explain:
        try:
            from hrai.ml.explain import get_explainer

            explainer = get_explainer(bundle.version)
            top_factors = [c.to_dict() for c in explainer.explain_row(frame, top_n=5)]
        except Exception as exc:  # noqa: BLE001 - a prediction must not fail on explainability
            log.warning(
                "explanation unavailable for this request",
                extra={"error": str(exc), "employee_id": employee_id},
            )

    if log_it:
        log_prediction(
            employee_id=employee_id,
            features=validated,
            probability=probability,
            risk_band=band,
            threshold=bundle.threshold,
            model_version=bundle.version,
        )

    return {
        "employee_id": employee_id,
        "attrition_probability": round(probability, 4),
        "risk_band": band,
        "threshold": bundle.threshold,
        "flagged": probability >= bundle.threshold,
        "model_version": bundle.version,
        "top_factors": top_factors,
        "caveat": CAVEAT,
    }
