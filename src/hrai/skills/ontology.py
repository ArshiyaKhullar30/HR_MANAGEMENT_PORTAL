"""The two-tier skill ontology (Steps 11-13).

Finding F2 established that this project has two different kinds of skill, and
they cannot be compared the same way:

* **Tier 1 — foundational.** O*NET Basic Skills: 10 cognitive competencies
  (Critical Thinking, Reading Comprehension, Mathematics, ...) scored per
  occupation on Importance (1-5) and Level (1-7). These are *graded* — everyone
  has some Critical Thinking; the question is how much the role needs.

* **Tier 2 — technical.** O*NET Technology Skills: 8,751 concrete tools across
  134 categories, flagged Hot Technology and In Demand. These are *binary* —
  you either work with Kubernetes or you do not.

So the gap engine does subtraction for Tier 2 and a graded comparison for
Tier 1. Treating them alike would either lose the grading or invent one.
"""

from __future__ import annotations

import pandas as pd

from hrai.skills.crosswalk import resolved_crosswalk
from hrai.utils.config import get
from hrai.utils.io import load_processed
from hrai.utils.logger import get_logger

log = get_logger(__name__)

# A role's tool list runs to 100+ entries; the long tail is incidental software
# nobody would train for. Ranking by market signal keeps the list actionable.
MAX_TOOLS_PER_ROLE = 25


def role_requirements() -> tuple[pd.DataFrame, pd.DataFrame]:
    """What each of the 40 roles requires, in both tiers.

    Returns ``(foundational, technical)``, both keyed by ``role``.
    """
    crosswalk = resolved_crosswalk()[
        ["role", "soc_code", "occupation_title", "confidence", "reviewed", "needs_review"]
    ]

    foundational = crosswalk.merge(
        load_processed("essential_skills_processed"),
        on="soc_code",
        how="inner",
        validate="many_to_many",
    ).rename(columns={"skill": "skill_name"})
    foundational["required_level"] = foundational["level"].round(2)
    foundational["importance"] = foundational["importance"].round(2)
    foundational["tier"] = "foundational"

    technical = crosswalk.merge(
        load_processed("software_skills_processed"),
        on="soc_code",
        how="inner",
        validate="many_to_many",
    )
    hot_weight = float(get("skills.technical.weight_hot_technology", 1.5))
    demand_weight = float(get("skills.technical.weight_in_demand", 2.0))
    technical["market_weight"] = (
        1.0
        + technical["hot_technology"] * (hot_weight - 1.0)
        + technical["in_demand"] * (demand_weight - 1.0)
    ).round(3)
    technical = technical.rename(columns={"tool": "skill_name"})
    technical["tier"] = "technical"

    # Deterministic ordering: market weight, then name. Never rely on file order.
    technical = (
        technical.sort_values(
            ["role", "market_weight", "skill_name"], ascending=[True, False, True]
        )
        .groupby("role", as_index=False)
        .head(MAX_TOOLS_PER_ROLE)
        .reset_index(drop=True)
    )
    # Linear decay from 1.0 (the role's most in-demand tool) to 0.2 (the 25th).
    # A 1/rank decay would drop to 0.25 by the fourth tool, implying employees
    # know almost nothing outside their top three — which is not how roles work.
    rank = technical.groupby("role").cumcount()
    depth = technical.groupby("role")["skill_name"].transform("size").clip(lower=1)
    technical["role_priority"] = (1.0 - 0.8 * (rank / depth)).clip(0.2, 1.0).round(4)

    log.info(
        "role requirements built",
        extra={
            "roles": int(crosswalk["role"].nunique()),
            "foundational_rows": len(foundational),
            "technical_rows": len(technical),
            "tools_per_role_cap": MAX_TOOLS_PER_ROLE,
        },
    )
    return foundational, technical


def role_requirement_summary() -> pd.DataFrame:
    """One row per role: how many requirements of each tier, and market heat."""
    foundational, technical = role_requirements()
    summary = technical.groupby(["role", "soc_code", "occupation_title"], as_index=False).agg(
        technical_skills=("skill_name", "nunique"),
        hot_technologies=("hot_technology", "sum"),
        in_demand_tools=("in_demand", "sum"),
    )
    found = foundational.groupby("role", as_index=False).agg(
        foundational_skills=("skill_name", "nunique"),
        mean_required_level=("required_level", "mean"),
    )
    out = summary.merge(found, on="role", how="outer")
    out["mean_required_level"] = out["mean_required_level"].round(2)
    return out.sort_values("role").reset_index(drop=True)
