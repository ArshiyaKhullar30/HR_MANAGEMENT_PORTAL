"""Dashboard data endpoints (Step 18)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.services.intelligence_service import get_intelligence_service, load_report
from hrai.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _service():
    service = get_intelligence_service()
    if not service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Employee intelligence table not built. "
                "Run `make pipeline && make train && make intelligence`."
            ),
        )
    return service


@router.get("/summary", summary="Headline KPIs")
def summary(department: str | None = Query(default=None)) -> dict:
    return _service().summary(department)


@router.get("/attrition-by-department", summary="Attrition risk per department")
def attrition_by_department() -> dict:
    return {
        "departments": _service().attrition_by_department(),
        "note": (
            "Population B is excluded: the model does not transfer to it "
            "(external ROC-AUC 0.50). See docs/transfer_validation.json."
        ),
    }


@router.get("/engagement-by-department", summary="Engagement per department")
def engagement_by_department() -> dict:
    return {"departments": _service().engagement_by_department(), "scale": "1-5 Likert"}


@router.get("/skill-gaps", summary="Organisation-wide skill gaps")
def skill_gaps(
    limit: int = Query(default=25, ge=1, le=200),
    severity: str | None = Query(default=None, pattern="^(HIGH|MEDIUM|LOW)$"),
) -> dict:
    return {
        "gaps": _service().skill_gaps(limit=limit, severity=severity),
        "skills_are_derived": True,
        "note": (
            "Employee skill profiles are derived from role requirements, tenure, "
            "performance and training history — no source dataset records "
            "individual skills (finding F3)."
        ),
    }


@router.get("/recommendations", summary="Per-employee upskilling recommendations")
def recommendations(
    limit: int = Query(default=100, ge=1, le=1000), department: str | None = Query(default=None)
) -> dict:
    return {
        "recommendations": _service().recommendations(limit=limit, department=department),
        "skills_are_derived": True,
    }


@router.get("/departments", summary="Department filter values")
def departments() -> dict:
    return {"departments": _service().departments()}


@router.get("/model-quality", summary="Model metrics, fairness and transfer validation")
def model_quality() -> dict:
    return {
        "training": load_report("model_training_report.json").get("metrics", {}),
        "global_drivers": load_report("shap_global_importance.json").get("top_features", [])[:10],
        "fairness": load_report("fairness_audit.json"),
        "transfer_validation": load_report("transfer_validation.json"),
    }
