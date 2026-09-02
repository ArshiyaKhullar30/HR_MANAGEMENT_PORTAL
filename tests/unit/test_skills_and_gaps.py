"""Skill ontology, gap engine and recommendation tests (Step 22)."""

from __future__ import annotations

import pandas as pd
import pytest

from hrai.skills.gap import compute_skill_gaps, organisation_skill_gaps


@pytest.fixture(scope="module")
def skills():
    from hrai.skills.employee_skills import build_employee_skills

    return build_employee_skills()


@pytest.mark.unit
def test_skill_gap_is_plain_set_subtraction():
    """The Build Notes' worked example, exactly as written."""
    required = {"Python", "SQL", "MLOps", "Docker", "AWS"}
    held = {"Python", "SQL", "AWS"}
    assert required - held == {"MLOps", "Docker"}


@pytest.mark.unit
def test_technical_gap_matches_the_expected_subtraction():
    """A hand-built frame with a known answer — a golden-file style check."""
    frame = pd.DataFrame(
        [
            {
                "person_key": "A-1",
                "population": "A",
                "employee_id": 1,
                "role": "Dev",
                "skill_name": "Python",
                "tier": "technical",
                "required_level": None,
                "importance": 1.5,
                "proficiency_level": 0.8,
                "holds_skill": 1,
            },
            {
                "person_key": "A-1",
                "population": "A",
                "employee_id": 1,
                "role": "Dev",
                "skill_name": "Docker",
                "tier": "technical",
                "required_level": None,
                "importance": 2.0,
                "proficiency_level": 0.0,
                "holds_skill": 0,
            },
            {
                "person_key": "A-1",
                "population": "A",
                "employee_id": 1,
                "role": "Dev",
                "skill_name": "SQL",
                "tier": "technical",
                "required_level": None,
                "importance": 1.0,
                "proficiency_level": 0.0,
                "holds_skill": 0,
            },
        ]
    )
    gaps = compute_skill_gaps(frame)
    missing = set(gaps.loc[gaps["has_gap"] == 1, "skill_name"])
    assert missing == {"Docker", "SQL"}
    # Docker carries the higher market weight, so it must outrank SQL.
    ranked = gaps[gaps["has_gap"] == 1].sort_values("severity", ascending=False)
    assert ranked.iloc[0]["skill_name"] == "Docker"


@pytest.mark.unit
def test_foundational_gap_is_graded_not_binary():
    """Nobody has zero Critical Thinking — Tier 1 compares levels, not membership."""
    frame = pd.DataFrame(
        [
            {
                "person_key": "A-1",
                "population": "A",
                "employee_id": 1,
                "role": "Dev",
                "skill_name": "Critical Thinking",
                "tier": "foundational",
                "required_level": 4.0,
                "importance": 5.0,
                "proficiency_level": 3.9,
                "holds_skill": 1,
            },
            {
                "person_key": "A-2",
                "population": "A",
                "employee_id": 2,
                "role": "Dev",
                "skill_name": "Critical Thinking",
                "tier": "foundational",
                "required_level": 4.0,
                "importance": 5.0,
                "proficiency_level": 1.5,
                "holds_skill": 1,
            },
        ]
    )
    gaps = compute_skill_gaps(frame).set_index("person_key")
    assert gaps.loc["A-1", "has_gap"] == 0  # 0.1 short — inside tolerance
    assert gaps.loc["A-2", "has_gap"] == 1  # 2.5 short — a real gap
    assert gaps.loc["A-2", "gap_magnitude"] == pytest.approx(2.5)


@pytest.mark.unit
def test_derived_skills_are_deterministic(skills):
    """Same seed, same inputs, byte-identical output — the determinism contract."""
    from hrai.skills.employee_skills import build_employee_skills

    assert skills.equals(build_employee_skills())


@pytest.mark.unit
def test_every_derived_skill_row_is_labelled_as_derived(skills):
    """Finding F3: derived proficiency must never be presented as observed fact."""
    assert skills["is_derived"].all()


@pytest.mark.unit
def test_person_key_never_collides_across_populations(skills):
    """Employee ids overlap between populations; person_key must not."""
    per_key = skills.groupby("person_key")["population"].nunique()
    assert (per_key == 1).all()


@pytest.mark.unit
def test_organisation_rollup_uses_percentage_bands(skills):
    gaps = compute_skill_gaps(skills)
    org = organisation_skill_gaps(gaps)
    assert set(org["severity_band"]) <= {"HIGH", "MEDIUM", "LOW"}
    assert (org["pct_of_workforce"] <= 1.0).all()
    # Severity must be monotonic in prevalence.
    high = org[org["severity_band"] == "HIGH"]["pct_of_workforce"]
    low = org[org["severity_band"] == "LOW"]["pct_of_workforce"]
    if len(high) and len(low):
        assert high.min() > low.max()


@pytest.mark.unit
def test_crosswalk_covers_every_role_with_real_skill_data():
    """Finding F4: without this the whole skills layer has no input."""
    from hrai.skills.crosswalk import resolved_crosswalk
    from hrai.utils.io import load_processed

    crosswalk = resolved_crosswalk()
    foundational = set(load_processed("essential_skills_processed")["soc_code"])
    technical = set(load_processed("software_skills_processed")["soc_code"])

    assert len(crosswalk) == 40
    assert crosswalk["reviewed"].all(), "every mapping must be human-reviewed"
    assert not crosswalk["needs_review"].any()
    assert set(crosswalk["soc_code"]) <= foundational
    assert set(crosswalk["soc_code"]) <= technical


@pytest.mark.unit
def test_recommendation_prefers_an_exact_rule_over_an_embedding():
    from hrai.skills.recommend import recommend_semantic, recommend_v1

    result = recommend_semantic(["SQL"], top_k=1).iloc[0]
    assert result["method"] == "rule"
    assert result["course"] == recommend_v1("SQL")
    assert result["match_confidence"] == 1.0


@pytest.mark.unit
def test_a_weak_semantic_match_falls_back_rather_than_bluffing():
    from hrai.skills.recommend import SEMANTIC_CONFIDENCE_FLOOR, recommend_semantic

    result = recommend_semantic(["Zzyzx Quantum Basketweaving"], top_k=1).iloc[0]
    assert result["match_confidence"] < SEMANTIC_CONFIDENCE_FLOOR
    assert result["method"] == "fallback_low_confidence"


@pytest.mark.unit
def test_course_costs_come_from_the_real_training_records():
    from hrai.skills.recommend import build_course_catalogue

    catalogue = build_course_catalogue()
    observed = catalogue[catalogue["source"] == "observed"]
    assert len(observed) == 5, "the five real Training Program Name values"
    assert (observed["median_cost"] > 0).all()
    assert (observed["times_delivered"] > 0).all()
