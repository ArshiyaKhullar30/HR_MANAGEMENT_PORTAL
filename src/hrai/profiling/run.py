"""Step 01 entry point — profile every source dataset and write the outputs.

    make profile          # or: python -m hrai.profiling.run

Produces:
  docs/data_dictionary.md      human-readable, one section per dataset
  conf/schemas/<name>.json     machine-readable profile, consumed by Step 02
  docs/findings.json           the reproduced evidence for F1, F4, F5, F6
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from hrai.cleaning.text import is_constant
from hrai.profiling.profiler import key_overlap_evidence, profile_dataset
from hrai.utils.config import get, project_root, raw_path
from hrai.utils.io import file_checksum, load_raw
from hrai.utils.logger import get_logger, run_id, setup_logging

log = get_logger(__name__)

DATASETS = [
    "employee_attrition",
    "hr_performance_engagement",
    "occupation_data",
    "essential_skills",
    "software_skills",
]


def _age_from_dob(dob: pd.Series, as_of: int = 2023) -> pd.Series:
    """Best-effort age from the engagement file's mixed-format DOB column."""
    parsed = pd.to_datetime(dob, format="%d-%m-%Y", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(dob, errors="coerce", format="mixed", dayfirst=True))
    years = as_of - parsed.dt.year
    # Two-digit years parse into the future; roll them back a century.
    return years.where(years > 0, years + 100)


def evidence_pack(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Reproduce, from the pipeline, the findings the plan is built on."""
    att, eng = frames["employee_attrition"], frames["hr_performance_engagement"]
    occ, ess, sw = (
        frames["occupation_data"],
        frames["essential_skills"],
        frames["software_skills"],
    )

    eng_aug = eng.copy()
    eng_aug["_age_from_dob"] = _age_from_dob(eng_aug["DOB"])

    # -- F1: are the two employee populations the same people? --------------
    f1 = key_overlap_evidence(
        att,
        eng_aug,
        "EmployeeNumber",
        "Employee ID",
        {"Gender": "GenderCode", "Age": "_age_from_dob"},
        tolerance={"Age": 1.0},
    )

    # -- F4: do role vocabularies reach O*NET at all? ------------------------
    occ_titles = set(occ["Title"].astype(str).str.strip())
    att_roles = set(att["JobRole"].astype(str).str.strip())
    eng_titles = set(eng["Title"].astype(str).str.strip())
    f4 = {
        "onet_occupations": len(occ_titles),
        "attrition_roles": len(att_roles),
        "attrition_exact_matches": len(att_roles & occ_titles),
        "engagement_titles": len(eng_titles),
        "engagement_exact_matches": len(eng_titles & occ_titles),
        "total_roles_needing_crosswalk": len(att_roles | eng_titles),
        "verdict": "A semantic role -> SOC crosswalk is mandatory; exact matching yields nothing.",
    }

    # -- F5: engagement grain -----------------------------------------------
    f5 = {
        "rows": int(len(eng)),
        "distinct_employees": int(eng["Employee ID"].nunique()),
        "employees_with_multiple_rows": int((eng["Employee ID"].value_counts() > 1).sum()),
        "grain": "employee x survey/training event",
        "verdict": "Not one row per employee; resolve grain before any aggregation.",
    }

    # -- F6: data-quality defects -------------------------------------------
    terminated = set(get("validation.engagement.terminated_statuses", []))
    has_exit_date = eng["ExitDate"].notna() & eng["ExitDate"].astype(str).str.strip().ne("")
    exit_but_active = int((has_exit_date & ~eng["EmployeeStatus"].isin(terminated)).sum())
    likert = get("validation.engagement.likert_columns", [])
    f6 = {
        "engagement_exit_date_but_not_terminated": exit_but_active,
        "engagement_duplicate_id_rows": int(len(eng) - eng["Employee ID"].nunique()),
        "attrition_constant_columns": [c for c in att.columns if is_constant(att[c])],
        "attrition_missing_cells": int(att.isna().sum().sum()),
        "engagement_likert_observed_range": {
            c: [int(eng[c].min()), int(eng[c].max())] for c in likert if c in eng.columns
        },
        "verdict": "Likert columns are 1-5, not 0-100; the 0-100 rule would catch nothing.",
    }

    # -- O*NET referential integrity ----------------------------------------
    occ_codes = set(occ["O*NET-SOC Code"])
    integrity = {
        "essential_skills_subset_of_occupations": bool(
            set(ess["O*NET-SOC Code"]).issubset(occ_codes)
        ),
        "software_skills_subset_of_occupations": bool(
            set(sw["O*NET-SOC Code"]).issubset(occ_codes)
        ),
        "occupations_without_essential_skills": len(occ_codes - set(ess["O*NET-SOC Code"])),
        "occupations_without_software_skills": len(occ_codes - set(sw["O*NET-SOC Code"])),
        "foundational_skills": sorted(ess["Element Name"].unique().tolist()),
        "technical_tool_count": int(sw["Workplace Example"].nunique()),
        "technical_category_count": int(sw["Element Name"].nunique()),
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id(),
        "F1_employee_population_identity": f1,
        "F4_role_to_onet_matching": f4,
        "F5_engagement_grain": f5,
        "F6_data_quality_defects": f6,
        "onet_referential_integrity": integrity,
    }


def _md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_data_dictionary(profiles: dict[str, Any], evidence: dict[str, Any]) -> str:
    root = project_root()
    lines: list[str] = [
        "# Data Dictionary",
        "",
        "> Generated by `make profile` (Step 01). Do not edit by hand — "
        "re-run the profiler instead.",
        "",
        f"- Generated: `{evidence['generated_at']}`",
        f"- Run ID: `{evidence['run_id']}`",
        "",
        "## Source datasets at a glance",
        "",
    ]

    overview = [
        [
            f"`{name}`",
            f"{p['rows']:,}",
            str(p["columns"]),
            ", ".join(f"`{k}`" for k in p["candidate_keys"]) or "_none_",
            f"{p['total_missing_cells']:,}",
            str(p["duplicate_rows"]),
        ]
        for name, p in profiles.items()
    ]
    lines += [
        _md_table(
            overview, ["Dataset", "Rows", "Cols", "Candidate key", "Missing cells", "Dup rows"]
        ),
        "",
    ]

    for name, p in profiles.items():
        lines += [f"## `{name}`", ""]
        lines += [
            f"- **Shape:** {p['rows']:,} rows x {p['columns']} columns",
            f"- **Candidate keys:** {', '.join(f'`{k}`' for k in p['candidate_keys']) or '_none_'}",
            f"- **ID-like columns:** {', '.join(f'`{c}`' for c in p['id_like_columns']) or '_none_'}",
            f"- **Constant columns:** {', '.join(f'`{c}`' for c in p['constant_columns']) or '_none_'}",
            f"- **Duplicate rows:** {p['duplicate_rows']:,}",
            f"- **SHA-256:** `{p['source_sha256'][:16]}...`",
            "",
        ]
        rows = []
        for col in p["columns_detail"]:
            note = []
            if col["is_constant"]:
                note.append("constant")
            if col["is_candidate_key"]:
                note.append("**key**")
            if col["has_leading_trailing_space"]:
                note.append("whitespace")
            if col["missing_pct"] > 0:
                note.append(f"{col['missing_pct']}% missing")
            rng = ""
            if col["numeric"]:
                rng = f"{col['numeric'].get('min')} – {col['numeric'].get('max')}"
            elif col["categories"] and len(col["categories"]) <= 8:
                rng = ", ".join(list(col["categories"])[:8])
            rows.append(
                [
                    f"`{col['name']}`",
                    col["dtype"],
                    f"{col['unique']:,}",
                    (rng[:70] + "...") if len(rng) > 70 else (rng or "—"),
                    ", ".join(note) or "—",
                ]
            )
        lines += [_md_table(rows, ["Column", "Dtype", "Distinct", "Range / values", "Notes"]), ""]

    lines += [
        "## Reproduced findings",
        "",
        "These are recomputed by the profiler on every run, so the plan's "
        "conclusions never drift from the data.",
        "",
    ]
    f1 = evidence["F1_employee_population_identity"]
    lines += [
        "### F1 — employee population identity",
        "",
        f"- Shared key values: **{f1['shared_key_count']}**",
        f"- Mean attribute agreement: **{f1.get('mean_agreement_pct')}%**",
        f"- Same entities: **{f1.get('keys_refer_to_same_entities')}**",
        "",
        f"> {f1['verdict']}",
        "",
    ]
    for label, detail in f1["attribute_agreement"].items():
        lines.append(
            f"  - `{label}` — {detail['agree']}/{detail['compared']} = {detail['agree_pct']}%"
        )
    lines += [
        "",
        "### F4 — role to O*NET matching",
        "",
        "```json",
        json.dumps(evidence["F4_role_to_onet_matching"], indent=2),
        "```",
        "",
        "### F5 — engagement grain",
        "",
        "```json",
        json.dumps(evidence["F5_engagement_grain"], indent=2),
        "```",
        "",
        "### F6 — data-quality defects",
        "",
        "```json",
        json.dumps(evidence["F6_data_quality_defects"], indent=2),
        "```",
        "",
        "### O*NET referential integrity",
        "",
        "```json",
        json.dumps(evidence["onet_referential_integrity"], indent=2),
        "```",
        "",
    ]

    path = root / "docs" / "data_dictionary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path.relative_to(root))


def main() -> int:
    setup_logging()
    root = project_root()
    schema_dir = root / "conf" / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    profiles: dict[str, Any] = {}

    for name in DATASETS:
        df = load_raw(name)
        frames[name] = df
        profile = profile_dataset(df, name).to_dict()
        profile["source_sha256"] = file_checksum(raw_path(name))
        profiles[name] = profile
        (schema_dir / f"{name}.json").write_text(
            json.dumps(profile, indent=2, default=str), encoding="utf-8"
        )

    evidence = evidence_pack(frames)
    (root / "docs" / "findings.json").write_text(
        json.dumps(evidence, indent=2, default=str), encoding="utf-8"
    )
    dict_path = write_data_dictionary(profiles, evidence)

    log.info(
        "step 01 complete",
        extra={
            "datasets_profiled": len(profiles),
            "data_dictionary": dict_path,
            "f1_same_entities": evidence["F1_employee_population_identity"].get(
                "keys_refer_to_same_entities"
            ),
        },
    )

    f1 = evidence["F1_employee_population_identity"]
    if f1.get("keys_refer_to_same_entities"):
        log.error(
            "F1 evidence changed: the employee populations now appear to match. "
            "Re-open docs/adr/001-two-population-architecture.md before proceeding."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
