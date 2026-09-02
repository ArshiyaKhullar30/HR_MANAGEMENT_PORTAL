"""Derived employee skill profiles (Step 12) — finding F3.

**None of the five source datasets records what skills an individual holds.**
The Build Notes anticipated this and pre-authorised the fallback: *"If it
doesn't → build a controlled table for the MVP so the rest of the pipeline has
something real to work with."*

This is that table, built to three rules so it is defensible rather than
decorative:

1. **Derived, never random.** Proficiency is a function of signals that are
   genuinely in the data — the role's O*NET requirement, tenure, performance
   rating, education level, and (for Population B) actual training history
   including whether the training was passed or failed.
2. **Deterministic.** Individual variation comes from a hash of
   ``(seed, employee_id, skill)``, not from a random number generator. The same
   inputs always produce byte-identical output, regardless of row order,
   parallelism or how many times the pipeline has run.
3. **Labelled everywhere.** Every row carries ``is_derived = True``. It flows
   through the API responses and is displayed as a banner on the dashboard.
   Derived proficiency is never presented as observed fact.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from hrai.skills.ontology import role_requirements
from hrai.utils.config import seed
from hrai.utils.io import load_processed
from hrai.utils.logger import get_logger

log = get_logger(__name__)

# Threshold on the capability score above which an employee is treated as
# holding a technical tool.
TECHNICAL_ACQUISITION_THRESHOLD = 0.55


def _deterministic_unit(employee_id: int, key: str) -> float:
    """A stable pseudo-random value in [0, 1) for one (employee, skill) pair.

    A hash rather than an RNG: no global state, no dependence on iteration
    order, reproducible across processes and machines.
    """
    digest = hashlib.sha256(f"{seed()}|{employee_id}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _capability_factors(population: str) -> pd.DataFrame:
    """Per-employee capability signals, normalised to [0, 1]."""
    if population == "A":
        df = load_processed("employee_attrition_processed")
        out = pd.DataFrame({"employee_id": df["employee_id"].astype(int)})
        out["tenure"] = (df["YearsAtCompany"] / 20.0).clip(0, 1)
        out["experience"] = (df["TotalWorkingYears"] / 30.0).clip(0, 1)
        out["performance"] = ((df["PerformanceRating"] - 1) / 3.0).clip(0, 1)
        out["education"] = ((df["Education"] - 1) / 4.0).clip(0, 1)
        out["training"] = (df["TrainingTimesLastYear"] / 6.0).clip(0, 1)
        out["seniority"] = ((df["JobLevel"] - 1) / 4.0).clip(0, 1)
        out["role"] = df["JobRole"].astype(str).str.strip()
    else:
        df = load_processed("engagement_processed")
        out = pd.DataFrame({"employee_id": df["employee_id"].astype(int)})
        out["tenure"] = (df["tenure_years"].fillna(0) / 20.0).clip(0, 1)
        out["experience"] = (df["age"].fillna(35).sub(22).div(35.0)).clip(0, 1)
        out["performance"] = ((df["current_employee_rating"].astype(float) - 1) / 4.0).clip(0, 1)
        # Training outcome is a real signal: passing raises capability, failing
        # does not. An employee who failed every course has not gained the skill.
        out["education"] = df["training_pass_rate"].fillna(0.5).clip(0, 1)
        out["training"] = (df["training_events"].fillna(0) / 3.0).clip(0, 1)
        out["seniority"] = out["tenure"]
        out["role"] = df["Title"].astype(str).str.strip()

    # Weights sum to 1. Tenure and performance dominate because they are the
    # signals most defensibly related to demonstrated capability.
    out["capability"] = (
        0.30 * out["tenure"]
        + 0.25 * out["performance"]
        + 0.15 * out["experience"]
        + 0.15 * out["education"]
        + 0.10 * out["training"]
        + 0.05 * out["seniority"]
    ).round(4)
    out["population"] = population
    return out


def build_employee_skills() -> pd.DataFrame:
    """One row per (employee, skill) for both populations and both tiers."""
    foundational_req, technical_req = role_requirements()
    frames = []

    for population in ("A", "B"):
        factors = _capability_factors(population)

        # ---- Tier 1: graded proficiency against the role's required level ----
        found = factors.merge(
            foundational_req[["role", "skill_name", "required_level", "importance", "tier"]],
            on="role",
            how="inner",
        )
        variation = np.array(
            [
                _deterministic_unit(int(e), f"F|{s}")
                for e, s in zip(found["employee_id"], found["skill_name"], strict=True)
            ]
        )
        # Centred so an average-capability employee lands *at* the role's
        # required level, not below it. An earlier calibration centred at ~0.81
        # and produced a "gap" for 63% of the workforce on every foundational
        # skill — technically consistent, but a signal that flags everyone
        # tells an HR team nothing.
        multiplier = (0.78 + 0.55 * found["capability"] + 0.34 * (variation - 0.5)).clip(0.35, 1.45)
        found["proficiency_level"] = (found["required_level"] * multiplier).clip(0, 7).round(2)
        found["holds_skill"] = 1
        frames.append(
            found[
                [
                    "employee_id",
                    "population",
                    "role",
                    "skill_name",
                    "tier",
                    "required_level",
                    "importance",
                    "proficiency_level",
                    "holds_skill",
                ]
            ]
        )

        # ---- Tier 2: binary acquisition of each tool ------------------------
        tech = factors.merge(
            technical_req[["role", "skill_name", "market_weight", "role_priority", "tier"]],
            on="role",
            how="inner",
        )
        variation = np.array(
            [
                _deterministic_unit(int(e), f"T|{s}")
                for e, s in zip(tech["employee_id"], tech["skill_name"], strict=True)
            ]
        )
        # Tools central to the role are more likely to be held than peripheral
        # ones, so role_priority carries the most weight.
        score = 0.40 * tech["role_priority"] + 0.35 * tech["capability"] + 0.25 * variation
        tech["holds_skill"] = (score >= TECHNICAL_ACQUISITION_THRESHOLD).astype(int)
        tech["proficiency_level"] = np.where(tech["holds_skill"] == 1, score.round(3), 0.0)
        tech["required_level"] = np.nan
        tech["importance"] = tech["market_weight"]
        frames.append(
            tech[
                [
                    "employee_id",
                    "population",
                    "role",
                    "skill_name",
                    "tier",
                    "required_level",
                    "importance",
                    "proficiency_level",
                    "holds_skill",
                ]
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    out["is_derived"] = True  # never presented as observed fact

    # Employee IDs collide across populations (A spans 1-2068, B spans 1001-4000),
    # so `employee_id` alone is NOT a person. Any groupby on it would silently
    # merge two different people — finding F1 in a new guise. `person_key` is the
    # only safe identity downstream.
    out["person_key"] = out["population"] + "-" + out["employee_id"].astype(int).astype(str)

    log.info(
        "employee skills derived",
        extra={
            "rows": len(out),
            "employees": int(out["employee_id"].nunique()),
            "foundational_rows": int((out["tier"] == "foundational").sum()),
            "technical_rows": int((out["tier"] == "technical").sum()),
            "technical_hold_rate": round(
                float(out.loc[out["tier"] == "technical", "holds_skill"].mean()), 4
            ),
            "is_derived": True,
        },
    )
    return out
