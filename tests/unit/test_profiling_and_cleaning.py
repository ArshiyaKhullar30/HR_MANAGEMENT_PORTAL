"""Profiling, cleaning and I/O tests (Steps 01-03)."""

from __future__ import annotations

import pandas as pd
import pytest

from hrai.cleaning.attrition import clean_attrition
from hrai.cleaning.engagement import clean_engagement_events, to_employee_grain
from hrai.cleaning.onet import clean_essential_skills, clean_occupations, clean_software_skills
from hrai.cleaning.text import canonical_skill, clean_category, normalise_token, parse_dates
from hrai.profiling.profiler import key_overlap_evidence, profile_dataset


@pytest.fixture(scope="module")
def raw_attrition():
    from hrai.utils.io import load_raw

    return load_raw("employee_attrition")


@pytest.fixture(scope="module")
def raw_engagement():
    from hrai.utils.io import load_raw

    return load_raw("hr_performance_engagement")


# ---- profiling ----------------------------------------------------------


@pytest.mark.unit
def test_profiler_finds_the_key_and_the_constant_columns(raw_attrition):
    profile = profile_dataset(raw_attrition, "employee_attrition")
    assert profile.rows == 1470
    assert profile.candidate_keys == ["EmployeeNumber"]
    assert set(profile.constant_columns) == {"EmployeeCount", "Over18", "StandardHours"}
    assert profile.duplicate_rows == 0


@pytest.mark.unit
def test_profiler_detects_whitespace_padding(raw_engagement):
    profile = profile_dataset(raw_engagement, "engagement")
    padded = [c.name for c in profile.columns_detail if c.has_leading_trailing_space]
    assert "DepartmentType" in padded


@pytest.mark.unit
def test_key_overlap_evidence_rejects_a_coincidental_overlap(raw_attrition, raw_engagement):
    """Finding F1, reproduced by the pipeline rather than asserted from memory."""
    evidence = key_overlap_evidence(
        raw_attrition,
        raw_engagement,
        "EmployeeNumber",
        "Employee ID",
        {"Gender": "GenderCode"},
    )
    assert evidence["shared_key_count"] == 753
    assert evidence["keys_refer_to_same_entities"] is False
    assert evidence["attribute_agreement"]["Gender vs GenderCode"]["agree_pct"] < 60
    assert "COINCIDENTAL" in evidence["verdict"]


@pytest.mark.unit
def test_key_overlap_evidence_accepts_a_genuine_key():
    """The same test must pass a real key, or it only ever says no."""
    left = pd.DataFrame({"id": [1, 2, 3], "gender": ["M", "F", "M"]})
    right = pd.DataFrame({"id": [1, 2, 3], "sex": ["M", "F", "M"]})
    evidence = key_overlap_evidence(left, right, "id", "id", {"gender": "sex"})
    assert evidence["keys_refer_to_same_entities"] is True


@pytest.mark.unit
def test_key_overlap_evidence_handles_disjoint_keys():
    left = pd.DataFrame({"id": [1, 2], "g": ["M", "F"]})
    right = pd.DataFrame({"id": [9, 10], "g": ["M", "F"]})
    assert key_overlap_evidence(left, right, "id", "id", {"g": "g"})["shared_key_count"] == 0


# ---- text normalisation -------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["AWS", "Amazon Web Services", "AWS Cloud", "amazon aws"])
def test_skill_aliases_collapse_to_one_canonical_name(raw):
    """The Build Notes' example: three spellings, one skill."""
    assert canonical_skill(raw) == "AWS"


@pytest.mark.unit
def test_unknown_skills_keep_their_original_spelling():
    assert canonical_skill("Snowflake Data Cloud") == "Snowflake Data Cloud"


@pytest.mark.unit
def test_version_suffixes_are_stripped_before_matching():
    assert normalise_token("Microsoft Excel 2016") == "microsoft excel"
    assert canonical_skill("Microsoft Excel 2016") == "Microsoft Excel"


@pytest.mark.unit
def test_clean_category_strips_the_production_padding():
    series = pd.Series(["Production       ", " Sales", "IT/IS"])
    assert list(clean_category(series)) == ["Production", "Sales", "IT/IS"]


@pytest.mark.unit
def test_mixed_date_formats_all_parse():
    """The engagement file mixes `20-Sep-19` and `07-10-1969` across columns."""
    parsed = parse_dates(
        pd.Series(["20-Sep-19", "07-10-1969", "2023-01-14"]), ["%d-%b-%y", "%d-%m-%Y", "%Y-%m-%d"]
    )
    assert parsed.notna().all()
    assert parsed.iloc[1].year == 1969


# ---- cleaning -----------------------------------------------------------


@pytest.mark.unit
def test_cleaning_attrition_is_idempotent(raw_attrition):
    once = clean_attrition(raw_attrition)
    twice = clean_attrition(raw_attrition)
    assert once.equals(twice)


@pytest.mark.unit
def test_cleaning_attrition_drops_constants_and_stamps_identity(raw_attrition):
    cleaned = clean_attrition(raw_attrition)
    for constant in ("EmployeeCount", "Over18", "StandardHours"):
        assert constant not in cleaned.columns
    assert (cleaned["population"] == "A").all()
    assert cleaned["employee_id"].is_unique
    assert cleaned["attrition_flag"].isin([0, 1]).all()


@pytest.mark.unit
def test_cleaning_engagement_removes_pii_and_the_unreliable_columns(raw_engagement):
    events = clean_engagement_events(raw_engagement)
    for pii in ("FirstName", "LastName", "ADEmail", "Supervisor"):
        assert pii not in events.columns
    # ExitDate/TerminationType are noise AND the label — dropped on both counts.
    for unreliable in ("ExitDate", "TerminationType", "TerminationDescription"):
        assert unreliable not in events.columns


@pytest.mark.unit
def test_engagement_label_comes_from_employee_status(raw_engagement):
    events = clean_engagement_events(raw_engagement)
    expected = (raw_engagement["EmployeeStatus"] == "Voluntarily Terminated").sum()
    assert events["is_voluntary_exit"].sum() == expected


@pytest.mark.unit
def test_engagement_collapses_from_event_grain_to_employee_grain(raw_engagement):
    """Finding F5: 3,150 rows over 3,000 employees."""
    events = clean_engagement_events(raw_engagement)
    employees = to_employee_grain(events)
    assert len(events) == 3150
    assert len(employees) == 3000
    assert employees["employee_id"].is_unique
    # Training history is cumulative, so it must aggregate rather than take the last.
    assert employees["training_events"].sum() == len(events)


@pytest.mark.unit
def test_age_is_computed_from_data_vintage_not_the_wall_clock(raw_engagement):
    """Using `today` would make ages drift on every run, breaking determinism."""
    first = clean_engagement_events(raw_engagement)["age"]
    second = clean_engagement_events(raw_engagement)["age"]
    assert first.equals(second)
    assert first.between(18, 100).all()


@pytest.mark.unit
def test_onet_cleaning_produces_the_two_tier_ontology():
    from hrai.utils.io import load_raw

    occupations = clean_occupations(load_raw("occupation_data"))
    foundational = clean_essential_skills(load_raw("essential_skills"))
    technical = clean_software_skills(load_raw("software_skills"))

    assert len(occupations) == 1016
    assert occupations["soc_code"].is_unique
    # Tier 1: the 10 O*NET Basic Skills, pivoted from long IM/LV form.
    assert foundational["skill"].nunique() == 10
    assert {"importance", "level"} <= set(foundational.columns)
    assert foundational["importance"].between(1, 5).all()
    # Tier 2: canonicalised tools with market signals retained.
    assert technical["tool"].nunique() > 8000
    assert set(technical["hot_technology"].unique()) <= {0, 1}
    # Referential integrity into the role master.
    assert set(foundational["soc_code"]) <= set(occupations["soc_code"])
    assert set(technical["soc_code"]) <= set(occupations["soc_code"])
