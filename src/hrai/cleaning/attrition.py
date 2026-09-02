"""Cleaning for `employee_attrition.csv` — Population A (Step 03).

This file arrives clean: no missing cells, no duplicate rows, every column
correctly typed. The work here is therefore structural rather than remedial —
drop the columns that carry no information, and stamp the identity columns the
rest of the pipeline joins on.
"""

from __future__ import annotations

import pandas as pd

from hrai.cleaning.text import is_constant
from hrai.utils.config import get
from hrai.utils.logger import get_logger

log = get_logger(__name__)


def clean_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """Idempotent cleaning of Population A."""
    out = df.copy()

    # Constant columns carry zero information and only widen the feature matrix.
    constants = [
        c for c in get("validation.attrition.constant_columns_to_drop", []) if c in out.columns
    ]
    also_constant = [c for c in out.columns if is_constant(out[c])]
    to_drop = sorted(set(constants) | set(also_constant))
    out = out.drop(columns=to_drop)

    # Categorical hygiene, applied even though this file is tidy — cleaning must
    # not depend on the current contents of the file to stay correct.
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype("string").str.strip()

    out = out.drop_duplicates()

    # Shared identity columns. `employee_id` is a within-population key only;
    # it is NEVER comparable to Population B's employee_id (finding F1).
    out["employee_id"] = out["EmployeeNumber"].astype(int)
    out["population"] = "A"
    out["attrition_flag"] = (out["Attrition"] == "Yes").astype(int)

    log.info(
        "attrition cleaned",
        extra={
            "rows": len(out),
            "dropped_constant_columns": to_drop,
            "positive_rate": round(float(out["attrition_flag"].mean()), 4),
        },
    )
    return out.reset_index(drop=True)
