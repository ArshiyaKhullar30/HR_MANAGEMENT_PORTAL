"""Attrition prediction endpoints (Step 18)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.ml.model_loader import get_bundle
from app.ml.predictor import predict_one
from app.services.prediction_log import read_predictions
from app.validation.employee_schema import AttritionPrediction, EmployeeFeatures
from hrai.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["attrition"])


@router.post(
    "/attrition",
    response_model=AttritionPrediction,
    summary="Score a single employee's attrition risk",
)
def predict_attrition(employee: EmployeeFeatures, explain: bool = True) -> AttritionPrediction:
    """Run the attrition model on one employee.

    Returns a calibrated probability, its risk band, and the top factors driving
    it. Input is validated by Pydantic first, so malformed data gets a 422 and
    never reaches the model.
    """
    log.info(
        "prediction request received",
        extra={"employee_id": employee.employee_id, "explain": explain},
    )
    try:
        result = predict_one(employee.model_dump(), explain=explain)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    log.info(
        "prediction completed",
        extra={
            "employee_id": employee.employee_id,
            "probability": result["attrition_probability"],
            "risk_band": result["risk_band"],
            "model_version": result["model_version"],
        },
    )
    return AttritionPrediction(**result)


@router.get("/model", summary="Metadata for the currently loaded model")
def model_info() -> dict:
    bundle = get_bundle()
    metadata = bundle.metadata
    return {
        "version": bundle.version,
        "algorithm": bundle.algorithm,
        "operating_threshold": bundle.threshold,
        "training_date": metadata.get("training_date"),
        "risk_bands": metadata.get("risk_bands"),
        "metrics": metadata.get("metrics", {}).get("test_calibrated"),
        "calibration_method": metadata.get("calibration_method"),
        "threshold_selection": metadata.get("threshold_selection"),
        "git_sha": metadata.get("git_sha"),
        "trained_on": metadata.get("trained_on"),
        "notes": metadata.get("notes"),
    }


@router.get("/log", summary="Recent predictions (drift-monitoring feed)")
def prediction_log(limit: int = 100) -> dict:
    frame = read_predictions(limit=limit)
    return {
        "count": int(len(frame)),
        "predictions": frame.to_dict(orient="records"),
        "note": (
            "Append-only, partitioned by date under data/predictions/. "
            "Consumed by drift monitoring."
        ),
    }
