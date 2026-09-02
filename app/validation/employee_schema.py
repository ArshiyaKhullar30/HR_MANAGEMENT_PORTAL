"""Request and response models for the API (Step 19).

Pydantic v2 validates every request before it reaches business logic or the
model. Bad data gets a 422 and never produces a garbage prediction.

The bounds here are the *same* bounds as the Pandera schemas in
`hrai.validation.schemas` — both read the ranges from `conf/config.yaml`, so a
validation rule has exactly one definition in the codebase.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hrai.utils.config import get

_AGE_MIN, _AGE_MAX = get("validation.attrition.age_range", [18, 100])


class EmployeeFeatures(BaseModel):
    """One employee, as the attrition model expects them.

    Only the fields the model actually uses are required. Extra fields are
    allowed and ignored, so a caller can post a fuller HR record without having
    to strip it down first.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    Age: int = Field(..., ge=_AGE_MIN, le=_AGE_MAX, description="Employee age in years.")
    Department: str = Field(..., min_length=1)
    JobRole: str = Field(..., min_length=1)
    MonthlyIncome: float = Field(..., gt=0)
    OverTime: Literal["Yes", "No"]
    JobSatisfaction: int = Field(..., ge=1, le=4)
    EnvironmentSatisfaction: int = Field(default=3, ge=1, le=4)
    RelationshipSatisfaction: int = Field(default=3, ge=1, le=4)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    JobInvolvement: int = Field(default=3, ge=1, le=4)
    JobLevel: int = Field(default=1, ge=1, le=5)
    PerformanceRating: int = Field(default=3, ge=1, le=4)
    StockOptionLevel: int = Field(default=0, ge=0, le=3)
    YearsAtCompany: int = Field(..., ge=0, le=60)
    TotalWorkingYears: int = Field(default=0, ge=0, le=60)
    YearsInCurrentRole: int = Field(default=0, ge=0, le=60)
    YearsSinceLastPromotion: int = Field(default=0, ge=0, le=60)
    YearsWithCurrManager: int = Field(default=0, ge=0, le=60)
    TrainingTimesLastYear: int = Field(default=0, ge=0, le=10)
    NumCompaniesWorked: int = Field(default=1, ge=0, le=20)
    DistanceFromHome: int = Field(default=5, ge=1, le=100)
    PercentSalaryHike: int = Field(default=12, ge=0, le=100)
    Education: int = Field(default=3, ge=1, le=5)
    EducationField: str = Field(default="Life Sciences", min_length=1)
    Gender: Literal["Male", "Female"] = "Male"
    MaritalStatus: str = Field(default="Single", min_length=1)
    BusinessTravel: str = Field(default="Travel_Rarely", min_length=1)
    HourlyRate: int = Field(default=65, ge=1)
    DailyRate: int = Field(default=800, ge=1)
    MonthlyRate: int = Field(default=14000, ge=1)
    employee_id: int | None = Field(default=None, ge=0)

    @field_validator("YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager")
    @classmethod
    def _not_longer_than_a_career(cls, value: int, info) -> int:
        # Caught here rather than in the model, where it would silently produce a
        # nonsense engineered ratio instead of an error the caller can fix.
        tenure = info.data.get("YearsAtCompany")
        if tenure is not None and value > tenure:
            raise ValueError(f"{info.field_name} ({value}) cannot exceed YearsAtCompany ({tenure})")
        return value


class Contribution(BaseModel):
    feature: str
    label: str
    value: Any = None
    shap_value: float
    direction: str


class AttritionPrediction(BaseModel):
    employee_id: int | None = None
    attrition_probability: float = Field(..., ge=0, le=1)
    risk_band: Literal["HIGH", "MEDIUM", "LOW"]
    threshold: float
    flagged: bool
    model_version: str
    top_factors: list[Contribution] = Field(default_factory=list)
    caveat: str


class InterventionOut(BaseModel):
    lever: str
    label: str
    feature: str
    from_value: Any = None
    to_value: Any = None
    baseline_risk: float
    new_risk: float
    risk_reduction: float
    cost: float
    expected_value_saved: float
    roi: float
    recommended_course: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    data_loaded: bool
