"""Config contract tests — the scope rule and determinism contract."""

import pytest

from hrai.utils.config import dataset_meta, get, project_root, raw_path, seed

PERMITTED = {
    "employee_attrition",
    "hr_performance_engagement",
    "occupation_data",
    "essential_skills",
    "software_skills",
}


@pytest.mark.unit
def test_only_the_five_permitted_datasets_are_declared():
    assert set(get("datasets")) == PERMITTED


@pytest.mark.unit
def test_raw_path_rejects_datasets_outside_the_scope_rule():
    with pytest.raises(KeyError):
        raw_path("employee_performance_pro")  # an archive/ file — out of scope
    with pytest.raises(KeyError):
        raw_path("Employee_Performance_Dataset")


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(PERMITTED))
def test_every_permitted_dataset_exists_on_disk(name):
    assert raw_path(name).exists(), f"missing source dataset: {name}"


@pytest.mark.unit
def test_seed_is_single_and_stable():
    assert seed() == 42
    assert get("random_seed") == seed()


@pytest.mark.unit
def test_engagement_grain_is_documented_as_event_level():
    """Finding F5: 3,150 rows over 3,000 employees — not one row per employee."""
    meta = dataset_meta("hr_performance_engagement")
    assert meta["expected_rows"] == 3150
    assert meta["distinct_employees"] == 3000
    assert "event" in meta["grain"]


@pytest.mark.unit
def test_leakage_register_blocks_the_label_columns():
    """Finding F7: these columns ARE the label."""
    blocked = set(get("leakage.hr_performance_engagement"))
    assert {"ExitDate", "TerminationType", "EmployeeStatus"} <= blocked


@pytest.mark.unit
def test_engagement_likert_range_is_one_to_five_not_zero_to_hundred():
    """Finding F6: the Build Notes' 0-100 rule would catch nothing here."""
    assert get("validation.engagement.likert_range") == [1, 5]


@pytest.mark.unit
def test_protected_attributes_are_never_retention_levers():
    """Risk R9 / WOW D: no intervention may be proposed on a protected attribute."""
    protected = set(get("retention_roi.protected_attributes"))
    levers = {lever["feature"] for lever in get("retention_roi.levers")}
    assert protected.isdisjoint(levers)


@pytest.mark.unit
def test_project_root_resolves_to_the_repo():
    assert (project_root() / "conf" / "config.yaml").exists()
