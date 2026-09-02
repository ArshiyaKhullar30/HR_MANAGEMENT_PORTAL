"""Engagement request models (Step 19).

Population B is on a 1-5 Likert scale, not 0-100 (finding F6). Encoding that in
the API schema means a caller sending a 0-100 score is rejected at the boundary
rather than silently producing a nonsense aggregate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hrai.utils.config import get

_LO, _HI = get("validation.engagement.likert_range", [1, 5])


class EngagementRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    employee_id: int = Field(..., ge=0)
    engagement_score: int = Field(
        ..., ge=_LO, le=_HI, description=f"Likert {_LO}-{_HI}, NOT 0-100."
    )
    satisfaction_score: int = Field(..., ge=_LO, le=_HI)
    work_life_balance_score: int = Field(..., ge=_LO, le=_HI)


class EngagementSummary(BaseModel):
    employees: int
    average_engagement: float
    average_engagement_pct: float
    average_satisfaction: float
    average_work_life_balance: float
    scale: str
    voluntary_exit_rate: float
    low_engagement_employees: int
