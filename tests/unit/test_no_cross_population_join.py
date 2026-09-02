"""The architectural invariant from finding F1, as an executable test.

If this ever fails, someone has joined the two employee populations and every
downstream number is suspect.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_no_cross_population_join.py"


@pytest.mark.unit
def test_no_cross_population_employee_join_in_codebase():
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, (
        "A cross-population employee join was found. The two employee datasets "
        "are different companies (finding F1).\n" + result.stderr
    )


@pytest.mark.unit
def test_invariant_is_declared_in_config():
    from hrai.utils.config import get

    assert get("invariants.forbid_cross_population_employee_join") is True
    assert set(get("invariants.populations").values()) == {
        "employee_attrition",
        "hr_performance_engagement",
    }
