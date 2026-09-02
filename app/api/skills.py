"""Skills, employees and the Retention ROI Copilot endpoints (Step 18)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.services.intelligence_service import get_intelligence_service
from hrai.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["employees"])


@router.get("/employees/{person_key}", summary="Full intelligence record for one person")
def employee(person_key: str) -> dict:
    """Look up by `person_key` (e.g. `A-101`), never by bare numeric id.

    Population A spans IDs 1-2068 and Population B spans 1001-4000, so 753 IDs
    exist in both — for completely different people (finding F1). A numeric
    lookup could return the wrong person, so the composite key is required.
    """
    service = get_intelligence_service()
    record = service.employee(person_key)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No employee with person_key {person_key!r}. Use the form "
                "'A-<id>' or 'B-<id>'; try /employees/resolve/<id> to find it."
            ),
        )
    return record


@router.get("/employees/resolve/{employee_id}", summary="Find every person carrying a numeric id")
def resolve(
    employee_id: int, population: str | None = Query(default=None, pattern="^[ABab]$")
) -> dict:
    service = get_intelligence_service()
    matches = service.resolve_employee(employee_id, population)
    return {
        "employee_id": employee_id,
        "matches": matches,
        "count": len(matches),
        "note": (
            "More than one match means this numeric id exists in both populations "
            "for different people. Use person_key to disambiguate."
        ),
    }


@router.get("/skills/role-requirements", summary="What each role requires, both tiers")
def role_requirements(role: str | None = Query(default=None)) -> dict:
    from hrai.skills.ontology import role_requirement_summary

    summary = role_requirement_summary()
    if role:
        summary = summary[summary["role"].str.lower() == role.lower()]
        if summary.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown role: {role!r}"
            )
    return {
        "roles": summary.to_dict(orient="records"),
        "tiers": {
            "foundational": "O*NET Basic Skills, graded 1-7",
            "technical": "O*NET Technology Skills, binary tools",
        },
    }


@router.get("/skills/crosswalk", summary="Role -> O*NET SOC crosswalk")
def crosswalk() -> dict:
    from hrai.skills.crosswalk import resolved_crosswalk

    frame = resolved_crosswalk()
    columns = [
        c
        for c in [
            "role",
            "source",
            "soc_code",
            "occupation_title",
            "confidence",
            "reviewed",
            "needs_review",
        ]
        if c in frame.columns
    ]
    return {
        "roles": len(frame),
        "human_reviewed": int(frame["reviewed"].sum()) if "reviewed" in frame else 0,
        "mappings": frame[columns].to_dict(orient="records"),
        "note": (
            "0 of 40 role titles match an O*NET occupation exactly (finding F4), "
            "so this crosswalk is what makes the skills layer possible."
        ),
    }
