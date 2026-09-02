"""Dataset profiling (Step 01).

The same checklist the Build Notes specify — shape, columns, dtypes, missing
values, duplicates, candidate join keys — run identically over all five
datasets so the results are comparable rather than ad hoc.

Also carries the evidence tests behind findings F1 (the two employee datasets
are different companies), F5 (engagement grain) and F6 (data-quality defects),
so those conclusions are reproduced by the pipeline rather than asserted from
a one-off notebook cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from hrai.utils.logger import get_logger

log = get_logger(__name__)

# Cardinality at or below this is treated as categorical for vocabulary capture.
_CATEGORICAL_MAX_UNIQUE = 40


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    is_constant: bool
    is_candidate_key: bool
    sample_values: list[Any] = field(default_factory=list)
    numeric: dict[str, float] | None = None
    categories: dict[str, int] | None = None
    has_leading_trailing_space: bool = False


@dataclass
class DatasetProfile:
    name: str
    rows: int
    columns: int
    duplicate_rows: int
    total_missing_cells: int
    constant_columns: list[str]
    candidate_keys: list[str]
    id_like_columns: list[str]
    columns_detail: list[ColumnProfile]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns_detail"] = [
            asdict(c) if not isinstance(c, dict) else c for c in self.columns_detail
        ]
        return payload


def profile_column(series: pd.Series, rows: int) -> ColumnProfile:
    non_null = int(series.notna().sum())
    missing = int(rows - non_null)
    unique = int(series.nunique(dropna=True))

    profile = ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        non_null=non_null,
        missing=missing,
        missing_pct=round(100 * missing / rows, 2) if rows else 0.0,
        unique=unique,
        unique_pct=round(100 * unique / rows, 2) if rows else 0.0,
        is_constant=unique <= 1,
        # A candidate key must be unique AND complete; uniqueness alone is not enough.
        is_candidate_key=bool(unique == rows and missing == 0 and rows > 0),
        sample_values=[_jsonable(v) for v in series.dropna().unique()[:5]],
    )

    if pd.api.types.is_numeric_dtype(series) and non_null:
        described = series.describe()
        profile.numeric = {
            k: _jsonable(described[k])
            for k in ("min", "25%", "50%", "75%", "max", "mean", "std")
            if k in described
        }

    if series.dtype == object or unique <= _CATEGORICAL_MAX_UNIQUE:
        counts = series.value_counts(dropna=True)
        if len(counts) <= _CATEGORICAL_MAX_UNIQUE:
            profile.categories = {str(k): int(v) for k, v in counts.items()}

    if series.dtype == object:
        stripped = series.dropna().astype(str)
        profile.has_leading_trailing_space = bool((stripped != stripped.str.strip()).any())

    return profile


def profile_dataset(df: pd.DataFrame, name: str) -> DatasetProfile:
    """Run the full Step 01 checklist over one dataset."""
    rows = len(df)
    columns = [profile_column(df[c], rows) for c in df.columns]

    profile = DatasetProfile(
        name=name,
        rows=rows,
        columns=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        total_missing_cells=int(df.isna().sum().sum()),
        constant_columns=[c.name for c in columns if c.is_constant],
        candidate_keys=[c.name for c in columns if c.is_candidate_key],
        # The Build Notes' heuristic: hunt for anything that looks like a join key.
        id_like_columns=[c for c in df.columns if "id" in str(c).lower()],
        columns_detail=columns,
    )
    log.info(
        "dataset profiled",
        extra={
            "dataset": name,
            "rows": profile.rows,
            "cols": profile.columns,
            "duplicates": profile.duplicate_rows,
            "constant_columns": len(profile.constant_columns),
            "candidate_keys": profile.candidate_keys,
        },
    )
    return profile


# --------------------------------------------------------------------------
# Finding F1 — is a shared ID actually the same person?
# --------------------------------------------------------------------------


def key_overlap_evidence(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_key: str,
    right_key: str,
    attribute_pairs: dict[str, str],
    tolerance: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Test whether two tables' shared key values refer to the same entities.

    A matching column name is not proof of a matching key. This compares
    independent attributes across the rows that *do* share a key value: if the
    keys are genuine, agreement approaches 100%; if the overlap is coincidence,
    agreement collapses to chance.
    """
    tolerance = tolerance or {}
    left_first = left.drop_duplicates(subset=[left_key]).set_index(left_key)
    right_first = right.drop_duplicates(subset=[right_key]).set_index(right_key)

    shared = left_first.index.intersection(right_first.index)
    result: dict[str, Any] = {
        "left_key": left_key,
        "right_key": right_key,
        "left_distinct_keys": int(left_first.index.nunique()),
        "right_distinct_keys": int(right_first.index.nunique()),
        "shared_key_count": int(len(shared)),
        "attribute_agreement": {},
    }
    if len(shared) == 0:
        result["verdict"] = (
            "DISJOINT — no shared key values; the tables cannot be joined on this key."
        )
        return result

    lhs, rhs = left_first.loc[shared], right_first.loc[shared]
    agreements: list[float] = []

    for left_col, right_col in attribute_pairs.items():
        if left_col not in lhs.columns or right_col not in rhs.columns:
            continue
        a, b = lhs[left_col], rhs[right_col]
        both = a.notna() & b.notna()
        if not both.any():
            continue
        if left_col in tolerance:
            match = (
                pd.to_numeric(a[both], errors="coerce") - pd.to_numeric(b[both], errors="coerce")
            ).abs() <= tolerance[left_col]
        else:
            match = (
                a[both].astype(str).str.strip().str.lower()
                == b[both].astype(str).str.strip().str.lower()
            )
        pct = round(100 * float(match.mean()), 1)
        agreements.append(pct)
        result["attribute_agreement"][f"{left_col} vs {right_col}"] = {
            "compared": int(both.sum()),
            "agree": int(match.sum()),
            "agree_pct": pct,
        }

    mean_agreement = float(np.mean(agreements)) if agreements else 0.0
    result["mean_agreement_pct"] = round(mean_agreement, 1)
    # Genuine keys agree near-perfectly. Anything below ~90% means the overlap
    # is coincidental and joining would fabricate records.
    result["keys_refer_to_same_entities"] = bool(mean_agreement >= 90.0)
    result["verdict"] = (
        "SAME ENTITIES — the key is genuine and the tables may be joined."
        if mean_agreement >= 90.0
        else (
            f"COINCIDENTAL OVERLAP — attributes agree only {mean_agreement:.1f}% on the "
            f"{len(shared)} shared key values. These are different populations; "
            "joining them would fabricate records."
        )
    )
    return result


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python so profiles serialise."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
