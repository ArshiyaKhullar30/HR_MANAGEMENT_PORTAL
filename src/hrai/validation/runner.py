"""Schema validation runner (Step 02).

Two modes, per the plan:

* ``strict``  — raise on any failure. Used in the pipeline after cleaning.
* ``report``  — collect every failure and return a structured report. Used on
  the raw data, where violations are expected and are the point.

Pandera's ``lazy=True`` collects all failures in one pass rather than stopping
at the first, so one run tells you everything that is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

try:
    import pandera.pandas as pa
    from pandera.errors import SchemaErrors
except ImportError:  # pragma: no cover
    import pandera as pa
    from pandera.errors import SchemaErrors

from hrai.utils.logger import get_logger

log = get_logger(__name__)


class ValidationFailed(Exception):
    """Raised in strict mode when a dataset violates its contract."""


@dataclass
class ValidationReport:
    dataset: str
    schema: str
    rows: int
    passed: bool
    failure_count: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else f"FAIL ({self.failure_count} violations)"
        return f"{self.dataset:34} {status}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "schema": self.schema,
            "rows": self.rows,
            "passed": self.passed,
            "failure_count": self.failure_count,
            "failures": self.failures,
        }


def validate(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    dataset: str,
    mode: str = "report",
) -> ValidationReport:
    """Validate ``df`` against ``schema``.

    In ``report`` mode every violation is collected and returned. In ``strict``
    mode the first failing dataset raises :class:`ValidationFailed`.
    """
    report = ValidationReport(
        dataset=dataset, schema=schema.name or "unnamed", rows=len(df), passed=True
    )
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        report.passed = False
        failures = exc.failure_cases

        # A frame-level check reports the whole failing row, producing one entry
        # per column. Collapse those to distinct failing rows so the count is
        # "1,198 bad rows", not "1,198 x 39 columns".
        is_frame_level = failures.get("schema_context", pd.Series(dtype=str)).eq("DataFrameSchema")
        frame_failures = failures[is_frame_level]
        column_failures = failures[~is_frame_level]

        records: list[dict[str, Any]] = []

        if not column_failures.empty:
            grouped = (
                column_failures.groupby(["check", "column"], dropna=False)
                .agg(
                    count=("failure_case", "size"),
                    example=("failure_case", lambda s: str(s.iloc[0])[:80]),
                )
                .reset_index()
            )
            records += [
                {
                    "check": str(row["check"]),
                    "column": None if pd.isna(row["column"]) else str(row["column"]),
                    "scope": "column",
                    "count": int(row["count"]),
                    "example": row["example"],
                }
                for _, row in grouped.iterrows()
            ]

        # Missing columns are the single most actionable failure — surface WHICH
        # columns are absent rather than collapsing them into a count. Pandera
        # reports these with the schema name in `column` and the missing column
        # name in `failure_case`, so they need their own branch.
        missing_mask = frame_failures["check"].eq("column_in_dataframe")
        missing = frame_failures[missing_mask]
        frame_failures = frame_failures[~missing_mask]
        if not missing.empty:
            for column_name in sorted(missing["failure_case"].astype(str).unique()):
                records.append(
                    {
                        "check": "required column is missing",
                        "column": column_name,
                        "scope": "dataframe",
                        "count": 1,
                        "example": f"{column_name} not present in the dataframe",
                    }
                )

        if not frame_failures.empty:
            for check, group in frame_failures.groupby("check", dropna=False):
                rows_affected = (
                    int(group["index"].nunique()) if "index" in group else int(len(group))
                )
                records.append(
                    {
                        "check": str(check),
                        "column": None,
                        "scope": "dataframe",
                        "count": rows_affected,
                        "example": str(group["failure_case"].iloc[0])[:80],
                    }
                )

        report.failure_count = sum(r["count"] for r in records)
        report.failures = sorted(records, key=lambda r: -r["count"])
        log.warning(
            "schema validation failed",
            extra={
                "dataset": dataset,
                "violations": report.failure_count,
                "distinct_checks": len(report.failures),
                "mode": mode,
            },
        )
        if mode == "strict":
            raise ValidationFailed(
                f"{dataset} violated its contract: {report.failure_count} failures across "
                f"{len(report.failures)} checks. First: {report.failures[0] if report.failures else 'n/a'}"
            ) from exc
    else:
        log.info("schema validation passed", extra={"dataset": dataset, "rows": len(df)})
    return report


def validate_all(
    frames: dict[str, pd.DataFrame],
    schemas: dict[str, Any],
    mode: str = "report",
) -> dict[str, ValidationReport]:
    return {
        name: validate(df, schemas[name](), name, mode=mode)
        for name, df in frames.items()
        if name in schemas
    }
