"""Skill gap engine (Steps 13-14).

The core operation is the plain set subtraction the Build Notes specify —
*required minus held* — applied twice, because finding F2 established there are
two kinds of skill here:

* **Technical (Tier 2)** — genuine set subtraction. Missing tools are weighted
  by market signal, so a Hot Technology gap outranks a legacy-tool gap.
* **Foundational (Tier 1)** — a graded comparison. Nobody has *zero* Critical
  Thinking, so the gap is `required_level - proficiency_level`, weighted by how
  important the skill is to that role.

Severity is `magnitude x weight` in both tiers, which puts them on one
comparable scale so a single ranked list can mix them.
"""

from __future__ import annotations

import pandas as pd

from hrai.utils.config import get
from hrai.utils.logger import get_logger

log = get_logger(__name__)

# Below this the shortfall is inside the noise of a derived proficiency estimate
# and would produce a gap for practically everyone.
FOUNDATIONAL_GAP_TOLERANCE = 0.5


def compute_skill_gaps(employee_skills: pd.DataFrame) -> pd.DataFrame:
    """One row per (person, skill) with a gap magnitude and severity score."""
    df = employee_skills.copy()

    # Coerce explicitly: when one tier is absent (a role with only technical
    # requirements, say) the column arrives as object dtype and arithmetic on it
    # raises rather than producing an empty result.
    for column in ("required_level", "proficiency_level", "importance", "holds_skill"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    foundational = df[df["tier"] == "foundational"].copy()
    foundational["gap_magnitude"] = (
        (foundational["required_level"] - foundational["proficiency_level"])
        .clip(lower=0)
        .round(3)
        .fillna(0.0)
    )
    foundational["has_gap"] = (foundational["gap_magnitude"] > FOUNDATIONAL_GAP_TOLERANCE).astype(
        int
    )
    # Importance is 1-5; normalise so both tiers score on a comparable scale.
    foundational["gap_weight"] = (foundational["importance"] / 5.0).round(3).fillna(0.0)

    technical = df[df["tier"] == "technical"].copy()
    # Genuine set subtraction: required minus held.
    technical["has_gap"] = (1 - technical["holds_skill"].fillna(0)).astype(int)
    technical["gap_magnitude"] = technical["has_gap"].astype(float)
    technical["gap_weight"] = (technical["importance"] / 2.0).clip(0, 1).round(3).fillna(0.0)

    parts = [frame for frame in (foundational, technical) if not frame.empty]
    out = (
        pd.concat(parts, ignore_index=True)
        if parts
        else df.assign(gap_magnitude=0.0, has_gap=0, gap_weight=0.0)
    )
    out["severity"] = (out["gap_magnitude"] * out["gap_weight"]).round(4)
    out = out.sort_values(["person_key", "severity"], ascending=[True, False])

    log.info(
        "skill gaps computed",
        extra={
            "rows": len(out),
            "people": int(out["person_key"].nunique()),
            "gaps_found": int(out["has_gap"].sum()),
            "foundational_gaps": int(foundational["has_gap"].sum()),
            "technical_gaps": int(technical["has_gap"].sum()),
        },
    )
    return out.reset_index(drop=True)


def per_employee_gaps(gaps: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """One row per person: their top-N gaps and a total severity score."""
    flagged = gaps[gaps["has_gap"] == 1].copy()

    top = (
        flagged.sort_values(["person_key", "severity"], ascending=[True, False])
        .groupby("person_key", as_index=False)
        .head(top_n)
    )
    summary = top.groupby(["person_key", "population", "employee_id", "role"], as_index=False).agg(
        top_gaps=("skill_name", lambda s: list(s)), top_gap_tiers=("tier", lambda s: list(s))
    )
    totals = flagged.groupby("person_key", as_index=False).agg(
        gap_count=("skill_name", "size"),
        gap_severity_total=("severity", "sum"),
        technical_gap_count=("tier", lambda s: int((s == "technical").sum())),
        foundational_gap_count=("tier", lambda s: int((s == "foundational").sum())),
    )
    out = summary.merge(totals, on="person_key", how="right")
    out["gap_severity_total"] = out["gap_severity_total"].round(3)
    out["primary_gap"] = out["top_gaps"].apply(
        lambda g: g[0] if isinstance(g, list) and g else None
    )
    out["is_derived"] = True
    return out.sort_values("gap_severity_total", ascending=False).reset_index(drop=True)


def organisation_skill_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    """Step 14 — the same logic rolled up across the whole organisation.

    The Build Notes set severity by absolute headcount (100+ HIGH, 50+ MEDIUM).
    Those thresholds are converted to a share of the workforce so they keep
    meaning as headcount changes — the same rule, made scalable.
    """
    total_people = int(gaps["person_key"].nunique())
    high_pct = float(get("skills.org_gap_severity.high_pct", 0.25))
    medium_pct = float(get("skills.org_gap_severity.medium_pct", 0.10))

    flagged = gaps[gaps["has_gap"] == 1]
    rollup = flagged.groupby(["skill_name", "tier"], as_index=False).agg(
        employees_missing=("person_key", "nunique"),
        mean_severity=("severity", "mean"),
        total_severity=("severity", "sum"),
    )
    rollup["pct_of_workforce"] = (rollup["employees_missing"] / total_people).round(4)
    rollup["severity_band"] = pd.cut(
        rollup["pct_of_workforce"],
        bins=[-0.01, medium_pct, high_pct, 1.01],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    rollup["mean_severity"] = rollup["mean_severity"].round(4)
    rollup["total_severity"] = rollup["total_severity"].round(2)
    rollup = rollup.sort_values(["employees_missing", "total_severity"], ascending=False)

    log.info(
        "organisation-wide skill gaps rolled up",
        extra={
            "skills_with_gaps": len(rollup),
            "workforce": total_people,
            "high_severity": int((rollup["severity_band"] == "HIGH").sum()),
            "medium_severity": int((rollup["severity_band"] == "MEDIUM").sum()),
        },
    )
    return rollup.reset_index(drop=True)
