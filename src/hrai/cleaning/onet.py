"""Cleaning for the three O*NET reference files (Step 03).

These become the role master and the two-tier skill ontology (finding F2):

* `occupation_master`            — 1,016 occupations, the role reference table.
* `essential_skills_processed`   — Tier 1, foundational: 10 cognitive skills
  with Importance (1-5) and Level (1-7) per occupation, pivoted from the long
  IM/LV form into one row per (occupation, skill).
* `software_skills_processed`    — Tier 2, technical: 8,753 tools across 134
  categories, with the Hot Technology and In Demand market signals kept.
"""

from __future__ import annotations

import pandas as pd

from hrai.cleaning.text import canonical_skill_series, clean_category
from hrai.utils.logger import get_logger

log = get_logger(__name__)

SOC = "O*NET-SOC Code"


def clean_occupations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out = out.rename(
        columns={
            SOC: "soc_code",
            "Title": "occupation_title",
            "Description": "occupation_description",
        }
    )
    for col in ("soc_code", "occupation_title", "occupation_description"):
        out[col] = clean_category(out[col])
    out = out.drop_duplicates(subset=["soc_code"]).reset_index(drop=True)
    log.info("occupation master cleaned", extra={"occupations": len(out)})
    return out


def clean_essential_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long IM/LV form into one row per (occupation, skill)."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out = out.rename(
        columns={
            SOC: "soc_code",
            "Element Name": "skill",
            "Scale ID": "scale",
            "Data Value": "value",
        }
    )
    for col in ("soc_code", "skill", "scale"):
        out[col] = clean_category(out[col])

    # O*NET flags low-confidence estimates; excluding them keeps the ontology honest.
    suppressed = out["Recommend Suppress"].astype("string").str.strip().eq("Y")
    out = out[~suppressed]

    wide = (
        out.pivot_table(
            index=["soc_code", "skill"], columns="scale", values="value", aggfunc="mean"
        )
        .reset_index()
        .rename(columns={"IM": "importance", "LV": "level"})
    )
    wide.columns.name = None
    wide["importance"] = wide["importance"].round(3)
    wide["level"] = wide["level"].round(3)
    wide["tier"] = "foundational"
    wide = wide.dropna(subset=["importance", "level"])

    log.info(
        "essential skills cleaned",
        extra={
            "rows": len(wide),
            "occupations": int(wide["soc_code"].nunique()),
            "distinct_skills": int(wide["skill"].nunique()),
            "suppressed_rows_removed": int(suppressed.sum()),
        },
    )
    return wide.reset_index(drop=True)


def clean_software_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise tool names and keep the market-demand signals."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out = out.rename(
        columns={
            SOC: "soc_code",
            "Element Name": "tool_category",
            "Workplace Example": "tool_raw",
            "Hot Technology": "hot_technology",
            "In Demand": "in_demand",
        }
    )
    for col in ("soc_code", "tool_category", "tool_raw"):
        out[col] = clean_category(out[col])

    out["tool"] = canonical_skill_series(out["tool_raw"])
    out["hot_technology"] = out["hot_technology"].astype("string").str.strip().eq("Y").astype(int)
    out["in_demand"] = out["in_demand"].astype("string").str.strip().eq("Y").astype(int)
    out["tier"] = "technical"

    out = out[out["tool"].str.len() > 0]
    # One row per (occupation, tool); keep the strongest market signal seen.
    out = out.groupby(["soc_code", "tool"], as_index=False).agg(
        tool_category=("tool_category", "first"),
        hot_technology=("hot_technology", "max"),
        in_demand=("in_demand", "max"),
        tier=("tier", "first"),
    )

    log.info(
        "software skills cleaned",
        extra={
            "rows": len(out),
            "occupations": int(out["soc_code"].nunique()),
            "distinct_tools": int(out["tool"].nunique()),
            "hot_tools": int(out.loc[out["hot_technology"] == 1, "tool"].nunique()),
        },
    )
    return out.reset_index(drop=True)
