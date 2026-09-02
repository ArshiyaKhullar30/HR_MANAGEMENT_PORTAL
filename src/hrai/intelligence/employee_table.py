"""Employee Intelligence Table (Step 16) — the business output of the project.

Everything from Days 2 and 3 lands here: attrition risk, engagement, role, skill
gaps and the recommendation, one row per employee. The dashboard is a view onto
this table.

Two honesty rules are enforced structurally, not by convention:

**Population B carries no attrition probability.** The transfer validation
measured the restricted model at ROC-AUC 0.50 on Population B — chance. Rather
than print a plausible-looking number, those rows carry `attrition_probability =
null` and a `risk_unavailable_reason` that says exactly why. A number that is
indistinguishable from a coin toss is worse than an honest blank, because
someone will act on it.

**Skill data is labelled derived.** `skills_are_derived` is True on every row
and travels through the API and into the dashboard banner.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from hrai.ml.evaluate import risk_band_series
from hrai.ml.registry import load_model
from hrai.skills.employee_skills import build_employee_skills
from hrai.skills.gap import compute_skill_gaps, organisation_skill_gaps, per_employee_gaps
from hrai.skills.recommend import recommendation_lookup
from hrai.utils.config import project_root
from hrai.utils.io import load_processed, save_processed
from hrai.utils.logger import get_logger

log = get_logger(__name__)

TRANSFER_REPORT = "docs/transfer_validation.json"


def _transfer_verdict() -> dict[str, Any]:
    path = project_root() / TRANSFER_REPORT
    if not path.exists():
        return {"generalises": False, "reason": "Cross-population transfer has not been validated."}
    payload = json.loads(path.read_text(encoding="utf-8"))
    auc = payload.get("population_b", {}).get("external_metrics", {}).get("roc_auc")
    return {
        "generalises": bool(payload.get("generalises")),
        "external_roc_auc": auc,
        "reason": (
            f"The attrition model does not transfer to this population "
            f"(externally validated ROC-AUC {auc} against its own outcomes — "
            "indistinguishable from chance). Reporting a risk score here would be "
            "misleading, so none is shown."
        ),
    }


def build_employee_intelligence() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(employee_intelligence, organisation_skill_gaps)``."""
    skills = build_employee_skills()
    gaps = compute_skill_gaps(skills)
    per_person = per_employee_gaps(gaps)
    org_gaps = organisation_skill_gaps(gaps)

    all_gap_skills = sorted({s for row in per_person["top_gaps"].dropna() for s in row})
    recommendations = recommendation_lookup(all_gap_skills)

    # ---- Population A: model-scored ---------------------------------------
    attrition = load_processed("employee_attrition_processed")
    model, metadata = load_model()
    features = attrition.drop(columns=["attrition_flag"], errors="ignore")
    probabilities = model.predict_proba(features)[:, 1]

    population_a = pd.DataFrame(
        {
            "person_key": "A-" + attrition["employee_id"].astype(int).astype(str),
            "employee_id": attrition["employee_id"].astype(int),
            "population": "A",
            "source_dataset": "employee_attrition",
            "department": attrition["Department"].astype(str),
            "role": attrition["JobRole"].astype(str).str.strip(),
            "age": attrition["Age"].astype(int),
            "tenure_years": attrition["YearsAtCompany"].astype(float),
            "monthly_income": attrition["MonthlyIncome"].astype(float),
            "attrition_probability": pd.Series(probabilities.round(4), dtype="Float64"),
            "actual_attrition": attrition["attrition_flag"].astype(int),
            "engagement_score": pd.Series([pd.NA] * len(attrition), dtype="Int64"),
            "satisfaction_score": attrition["JobSatisfaction"].astype("Int64"),
            "work_life_balance_score": attrition["WorkLifeBalance"].astype("Int64"),
            "risk_unavailable_reason": pd.Series([pd.NA] * len(attrition), dtype="string"),
        }
    )
    population_a["risk_band"] = risk_band_series(population_a["attrition_probability"])

    # ---- Population B: engagement-scored, risk withheld --------------------
    engagement = load_processed("engagement_processed")
    verdict = _transfer_verdict()
    population_b = pd.DataFrame(
        {
            "person_key": "B-" + engagement["employee_id"].astype(int).astype(str),
            "employee_id": engagement["employee_id"].astype(int),
            "population": "B",
            "source_dataset": "hr_performance_engagement",
            "department": engagement["DepartmentType"].astype(str),
            "role": engagement["Title"].astype(str).str.strip(),
            "age": engagement["age"].astype("Int64"),
            "tenure_years": engagement["tenure_years"].astype(float),
            # Explicit float dtype: an all-NA object column makes pd.concat warn and
            # would leave the merged column untyped.
            "monthly_income": pd.Series([pd.NA] * len(engagement), dtype="Float64"),
            "attrition_probability": pd.Series([pd.NA] * len(engagement), dtype="Float64"),
            "actual_attrition": engagement["is_voluntary_exit"].astype(int),
            "engagement_score": engagement["engagement_score"].astype("Int64"),
            "satisfaction_score": engagement["satisfaction_score"].astype("Int64"),
            "work_life_balance_score": engagement["work_life_balance_score"].astype("Int64"),
            "risk_band": "UNAVAILABLE",
            "risk_unavailable_reason": verdict["reason"],
        }
    )

    out = pd.concat([population_a, population_b], ignore_index=True)

    # ---- attach skills, gaps and the recommendation -----------------------
    out = out.merge(
        per_person[
            [
                "person_key",
                "gap_count",
                "gap_severity_total",
                "primary_gap",
                "top_gaps",
                "technical_gap_count",
                "foundational_gap_count",
            ]
        ],
        on="person_key",
        how="left",
        validate="one_to_one",
    )
    out["skill_gaps"] = out["top_gaps"].apply(lambda g: ", ".join(g) if isinstance(g, list) else "")
    out["recommendation"] = out["primary_gap"].map(
        lambda s: recommendations.get(s, {}).get("course") if pd.notna(s) else None
    )
    out["recommendation_confidence"] = out["primary_gap"].map(
        lambda s: recommendations.get(s, {}).get("match_confidence") if pd.notna(s) else None
    )
    out["recommendation_cost"] = out["primary_gap"].map(
        lambda s: recommendations.get(s, {}).get("median_cost") if pd.notna(s) else None
    )
    out["skills_are_derived"] = True
    out["model_version"] = metadata.get("version")
    out = out.drop(columns=["top_gaps"])

    log.info(
        "employee intelligence table built",
        extra={
            "rows": len(out),
            "population_a": int((out["population"] == "A").sum()),
            "population_b": int((out["population"] == "B").sum()),
            "with_risk_score": int(out["attrition_probability"].notna().sum()),
            "high_risk": int((out["risk_band"] == "HIGH").sum()),
            "model_version": metadata.get("version"),
        },
    )
    return out, org_gaps


def main() -> int:
    from hrai.utils.logger import setup_logging

    setup_logging()
    table, org_gaps = build_employee_intelligence()
    save_processed(table, "employee_intelligence")
    save_processed(org_gaps, "organisation_skill_gaps")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "employees": int(len(table)),
        "population_a": int((table["population"] == "A").sum()),
        "population_b": int((table["population"] == "B").sum()),
        "high_risk": int((table["risk_band"] == "HIGH").sum()),
        "medium_risk": int((table["risk_band"] == "MEDIUM").sum()),
        "risk_unavailable": int((table["risk_band"] == "UNAVAILABLE").sum()),
        "average_engagement": round(float(table["engagement_score"].dropna().mean()), 2),
        "skills_are_derived": True,
        "critical_skill_gaps": int((org_gaps["severity_band"] == "HIGH").sum()),
    }
    (project_root() / "docs" / "intelligence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("step 16 complete", extra=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
