"""Shared read model for the dashboard endpoints.

The Employee Intelligence Table is read once and cached in memory. It is a few
thousand rows, so re-reading it per request would be pure waste; a
`refresh()` hook exists for when the pipeline regenerates it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from hrai.utils.config import project_root
from hrai.utils.io import load_processed, processed_exists
from hrai.utils.logger import get_logger

log = get_logger(__name__)


class IntelligenceService:
    def __init__(self) -> None:
        self._table: pd.DataFrame | None = None
        self._org_gaps: pd.DataFrame | None = None

    # -- loading -----------------------------------------------------------
    @property
    def table(self) -> pd.DataFrame:
        if self._table is None:
            self._table = load_processed("employee_intelligence")
            log.info("intelligence table loaded", extra={"rows": len(self._table)})
        return self._table

    @property
    def org_gaps(self) -> pd.DataFrame:
        if self._org_gaps is None:
            self._org_gaps = load_processed("organisation_skill_gaps")
        return self._org_gaps

    def refresh(self) -> None:
        self._table = None
        self._org_gaps = None

    @staticmethod
    def is_available() -> bool:
        return processed_exists("employee_intelligence")

    # -- dashboard reads ---------------------------------------------------
    def summary(self, department: str | None = None) -> dict[str, Any]:
        df = self._filtered(department)
        scored = df[df["attrition_probability"].notna()]
        engaged = df[df["engagement_score"].notna()]
        return {
            "total_employees": int(len(df)),
            "population_a": int((df["population"] == "A").sum()),
            "population_b": int((df["population"] == "B").sum()),
            "employees_with_risk_score": int(len(scored)),
            "high_risk_employees": int((df["risk_band"] == "HIGH").sum()),
            "medium_risk_employees": int((df["risk_band"] == "MEDIUM").sum()),
            "risk_unavailable": int((df["risk_band"] == "UNAVAILABLE").sum()),
            "average_attrition_probability": (
                round(float(scored["attrition_probability"].mean()), 4) if len(scored) else None
            ),
            "average_engagement": (
                round(float(engaged["engagement_score"].mean()), 2) if len(engaged) else None
            ),
            "average_engagement_pct": (
                round(100 * float(engaged["engagement_score"].mean()) / 5, 1)
                if len(engaged)
                else None
            ),
            "employees_with_skill_gaps": int(df["gap_count"].fillna(0).gt(0).sum()),
            "skills_are_derived": True,
            "model_version": (
                df["model_version"].dropna().iloc[0] if df["model_version"].notna().any() else None
            ),
        }

    def attrition_by_department(self) -> list[dict[str, Any]]:
        scored = self.table[self.table["attrition_probability"].notna()]
        if scored.empty:
            return []
        grouped = (
            scored.groupby("department", as_index=False)
            .agg(
                employees=("person_key", "nunique"),
                average_risk=("attrition_probability", "mean"),
                high_risk=("risk_band", lambda s: int((s == "HIGH").sum())),
                actual_attrition_rate=("actual_attrition", "mean"),
            )
            .round(4)
            .sort_values("average_risk", ascending=False)
        )
        return grouped.to_dict(orient="records")

    def engagement_by_department(self) -> list[dict[str, Any]]:
        engaged = self.table[self.table["engagement_score"].notna()]
        if engaged.empty:
            return []
        grouped = (
            engaged.groupby("department", as_index=False)
            .agg(
                employees=("person_key", "nunique"),
                average_engagement=("engagement_score", "mean"),
                low_engagement=("engagement_score", lambda s: int((s <= 2).sum())),
            )
            .round(3)
            .sort_values("average_engagement")
        )
        return grouped.to_dict(orient="records")

    def skill_gaps(self, limit: int = 25, severity: str | None = None) -> list[dict[str, Any]]:
        gaps = self.org_gaps
        if severity:
            gaps = gaps[gaps["severity_band"] == severity.upper()]
        return gaps.head(limit).to_dict(orient="records")

    def recommendations(
        self, limit: int = 100, department: str | None = None
    ) -> list[dict[str, Any]]:
        df = self._filtered(department)
        df = df[df["recommendation"].notna()]
        columns = [
            "person_key",
            "employee_id",
            "population",
            "department",
            "role",
            "primary_gap",
            "skill_gaps",
            "recommendation",
            "recommendation_confidence",
            "recommendation_cost",
            "risk_band",
            "gap_severity_total",
        ]
        available = [c for c in columns if c in df.columns]
        return (
            df.sort_values("gap_severity_total", ascending=False)
            .head(limit)[available]
            .to_dict(orient="records")
        )

    def employee(self, person_key: str) -> dict[str, Any] | None:
        """One employee's full intelligence record.

        Keyed by `person_key` ("A-101"), never by the bare numeric id: IDs
        collide across the two populations, so a numeric lookup could return the
        wrong person entirely (finding F1).
        """
        match = self.table[self.table["person_key"] == person_key]
        if match.empty:
            return None
        record = match.iloc[0].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in record.items()}

    def resolve_employee(
        self, employee_id: int, population: str | None = None
    ) -> list[dict[str, Any]]:
        """All people carrying this numeric id — usually one per population."""
        match = self.table[self.table["employee_id"] == int(employee_id)]
        if population:
            match = match[match["population"] == population.upper()]
        return [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in match.to_dict(orient="records")
        ]

    def departments(self) -> list[str]:
        return sorted(self.table["department"].dropna().unique().tolist())

    def _filtered(self, department: str | None) -> pd.DataFrame:
        if not department or department.lower() == "all":
            return self.table
        return self.table[self.table["department"] == department]


@lru_cache(maxsize=1)
def get_intelligence_service() -> IntelligenceService:
    return IntelligenceService()


def load_report(name: str) -> dict[str, Any]:
    """Read one of the generated JSON reports under docs/."""
    path = project_root() / "docs" / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
