#!/usr/bin/env python3
"""Enforce the project's central architectural invariant (finding F1 / risk R1).

``employee_attrition.csv`` and ``hr_performance_engagement.csv`` are different
companies. The 753 overlapping IDs are numeric coincidence — on those rows
gender agrees 48.6% (a coin flip) and age agrees 6.0%. Joining them on employee
identity fabricates 753 people who do not exist and silently corrupts the
attrition, engagement, skill-gap and recommendation layers at once.

What this flags: a ``merge``/``join`` **keyed on an employee id column**
(``employee_id``, ``EmployeeNumber``, ``Employee ID``) inside a file that touches
both populations.

What it deliberately does not flag, because none of these can fabricate a person:

* ``person_key`` joins — the key encodes the population (``"A-101"``), so it
  cannot match across them.
* ``pd.concat`` — vertical stacking keeps rows separate; it never merges two
  people into one row.
* ``str.join`` — ``", ".join(items)`` is string formatting, not a table join.

A guard that cries wolf gets disabled, so precision matters as much as coverage.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "docs", "conf"}
EXCLUDE_FILES = {
    "check_no_cross_population_join.py",
    "test_no_cross_population_join.py",
}

# A merge/join call, capturing its argument list. Excludes `str.join`, which is
# always preceded by a quote or a separator variable rather than a frame.
MERGE_CALL = re.compile(
    r"""(?<!['"])\b(?:pd\.merge|\w+\.merge|\w+\.join)\s*\(([^)]{0,400})""",
    re.DOTALL,
)

# The keys that identify a person only *within* one population.
UNSAFE_KEYS = re.compile(r"""['"](?:employee_id|EmployeeNumber|Employee\s+ID)['"]""")
# The key that is safe because it carries the population prefix.
SAFE_KEY = re.compile(r"""['"]person_key['"]""")

POP_A = re.compile(r"EmployeeNumber|employee_attrition", re.I)
POP_B = re.compile(r"Employee\s*ID|hr_performance_engagement|engagement_processed", re.I)


def iter_sources() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.suffix not in {".py", ".ipynb"} or path.name in EXCLUDE_FILES:
            continue
        if EXCLUDE_DIRS & set(path.relative_to(ROOT).parts):
            continue
        out.append(path)
    return out


def code_of(path: Path) -> str:
    """Source text — for notebooks, the code cells only (outputs are stripped)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".ipynb":
        return text
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return text
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def line_of(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def main() -> int:
    violations: list[tuple[Path, int, str]] = []

    for path in iter_sources():
        source = code_of(path)
        # Only files that touch both populations can commit this error at all.
        if not (POP_A.search(source) and POP_B.search(source)):
            continue

        for match in MERGE_CALL.finditer(source):
            arguments = match.group(1)
            if SAFE_KEY.search(arguments):
                continue  # person_key encodes the population — cannot cross
            if not UNSAFE_KEYS.search(arguments):
                continue  # not keyed on a person id at all
            lineno = line_of(source, match.start())
            snippet = source.splitlines()[lineno - 1].strip()
            violations.append((path.relative_to(ROOT), lineno, snippet))

    if not violations:
        return 0

    print("\n" + "=" * 78, file=sys.stderr)
    print(
        "BLOCKED — cross-population employee join detected (finding F1 / risk R1)", file=sys.stderr
    )
    print("=" * 78, file=sys.stderr)
    for path, lineno, snippet in violations:
        print(f"  {path}:{lineno}\n      {snippet}", file=sys.stderr)
    print(
        "\n  employee_attrition and hr_performance_engagement are DIFFERENT companies.\n"
        "  Their 753 shared IDs are coincidence: gender agrees 48.6%, age agrees 6.0%.\n"
        "  Join on 'person_key' (which encodes the population) or bridge the two\n"
        "  populations through the role -> O*NET skill ontology instead.\n"
        "  See docs/data_relationships.md and docs/adr/001-two-population-architecture.md\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
