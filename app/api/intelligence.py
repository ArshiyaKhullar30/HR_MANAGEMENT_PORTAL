"""Retention ROI Copilot endpoints — the prescriptive layer.

Predicting who will leave is the easy half. These endpoints answer the question
an HR director actually has: given a budget, who do we spend it on, on what, and
what do we get back?
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.services.intelligence_service import get_intelligence_service
from hrai.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/intelligence", tags=["retention copilot"])

_ENGINE = None


def _engine():
    """The counterfactual engine, built once per process."""
    global _ENGINE
    if _ENGINE is None:
        from hrai.intelligence.counterfactual import CounterfactualEngine

        _ENGINE = CounterfactualEngine()
        log.info("counterfactual engine initialised")
    return _ENGINE


@router.get(
    "/counterfactual/{person_key}",
    summary="What would actually reduce this person's risk, and at what cost",
)
def counterfactual(person_key: str) -> dict:

    from hrai.utils.io import load_processed

    if not person_key.upper().startswith("A-"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Counterfactual planning requires a model-scored employee. "
                "Only Population A ('A-<id>') has attrition scores — the model "
                "does not transfer to Population B (external ROC-AUC 0.50)."
            ),
        )

    employee_id = int(person_key.split("-", 1)[1])
    frame = load_processed("employee_attrition_processed")
    match = frame[frame["employee_id"] == employee_id]
    if match.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No Population A employee with id {employee_id}",
        )

    row = match.drop(columns=["attrition_flag"], errors="ignore").head(1)

    service = get_intelligence_service()
    record = service.employee(person_key) or {}
    course = record.get("recommendation")

    plan = _engine().plan_for(row, recommended_course=course)
    payload = plan.to_dict()
    payload["skill_gaps"] = record.get("skill_gaps")
    payload["recommended_course"] = course
    log.info(
        "counterfactual plan generated",
        extra={
            "person_key": person_key,
            "baseline_risk": payload["baseline_risk"],
            "interventions": len(payload["interventions"]),
        },
    )
    return payload


@router.get("/action-plan", summary="Budget-constrained retention plan for the workforce")
def action_plan(
    budget: float = Query(..., gt=0, description="Total retention budget."),
    min_risk: float = Query(default=0.30, ge=0.0, le=1.0),
    max_employees: int = Query(default=300, ge=1, le=1000),
) -> dict:
    """Choose the interventions that maximise expected value within the budget.

    A 0/1 knapsack solved greedily by return on investment. Employees below
    `min_risk` are not considered — spending retention budget on someone who was
    never going to leave is the most expensive kind of false positive.
    """
    from hrai.intelligence.counterfactual import build_action_plan

    plan = build_action_plan(
        budget, engine=_engine(), min_risk=min_risk, max_employees=max_employees
    )
    log.info(
        "action plan requested",
        extra={
            "budget": budget,
            "covered": plan["employees_covered"],
            "roi": plan["return_on_investment"],
        },
    )
    return plan


@router.get("/levers", summary="Which interventions the system may propose")
def levers() -> dict:
    from hrai.intelligence.counterfactual import CAVEAT, load_levers
    from hrai.utils.config import get

    return {
        "levers": [
            {
                "name": lever.name,
                "label": lever.label,
                "feature": lever.feature,
                "rationale": lever.rationale,
                "cost_multiple_monthly": lever.cost_multiple_monthly,
            }
            for lever in load_levers()
        ],
        "protected_attributes_excluded": get("retention_roi.protected_attributes", []),
        "why_excluded": (
            "The system will never propose an intervention on someone's "
            "age, gender or marital status. This is enforced in code and "
            "asserted by a test, not left to convention."
        ),
        "caveat": CAVEAT,
    }
