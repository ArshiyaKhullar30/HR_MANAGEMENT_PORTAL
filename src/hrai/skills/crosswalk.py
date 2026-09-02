"""Role -> O*NET SOC crosswalk (Step 11) — the keystone of the skills layer.

Finding F4: **zero** of the 9 attrition job roles and zero of the 31 engagement
titles match an O*NET occupation title exactly. Without a crosswalk the three
O*NET files are inert and the entire skills half of the project has no input.

The approach the Build Notes already sanction for the recommendation engine
works here too: embed both vocabularies with a sentence-transformer and match on
cosine similarity, so "Sr. DBA" can reach "Database Administrators" even though
the strings barely overlap.

Two safeguards, because a plausible-looking wrong mapping would silently distort
every downstream skill gap:

* **Confidence is persisted, not discarded.** Every mapping carries its score,
  and low-confidence ones are flagged all the way through to the dashboard.
* **Human review is a real step.** 40 roles is small enough to check by hand.
  `conf/role_crosswalk_reviewed.yaml` overrides the automatic result, and the
  reviewed file always wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hrai.utils.config import get, project_root
from hrai.utils.io import load_processed
from hrai.utils.logger import get_logger

log = get_logger(__name__)

AUTO_FILE = "conf/role_crosswalk_auto.yaml"
REVIEWED_FILE = "conf/role_crosswalk_reviewed.yaml"


@dataclass
class RoleMapping:
    role: str
    source: str
    soc_code: str
    occupation_title: str
    confidence: float
    needs_review: bool
    reviewed: bool = False
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source": self.source,
            "soc_code": self.soc_code,
            "occupation_title": self.occupation_title,
            "confidence": round(float(self.confidence), 4),
            "needs_review": bool(self.needs_review),
            "reviewed": bool(self.reviewed),
            "alternatives": self.alternatives,
        }


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    name = get("crosswalk.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    log.info("loading embedding model", extra={"model": name})
    return SentenceTransformer(name)


def distinct_roles() -> pd.DataFrame:
    """Every role title used by either population, with its headcount."""
    attrition = load_processed("employee_attrition_processed")
    engagement = load_processed("engagement_processed")

    rows = [
        {"role": str(role).strip(), "source": "employee_attrition", "headcount": int(count)}
        for role, count in attrition["JobRole"].value_counts().items()
    ] + [
        {"role": str(role).strip(), "source": "hr_performance_engagement", "headcount": int(count)}
        for role, count in engagement["Title"].value_counts().items()
    ]
    df = pd.DataFrame(rows)
    # A title used by both populations is one role; keep the combined headcount.
    return (
        df.groupby("role", as_index=False)
        .agg(headcount=("headcount", "sum"), source=("source", lambda s: "+".join(sorted(set(s)))))
        .sort_values("headcount", ascending=False)
        .reset_index(drop=True)
    )


def _occupation_corpus(occupations: pd.DataFrame) -> list[str]:
    """Title carries the identity; a short slice of description adds context.

    The full description is several hundred words and dilutes the title signal,
    which is what actually distinguishes one occupation from its neighbours.
    """
    return [
        f"{row.occupation_title}. {str(row.occupation_description)[:300]}"
        for row in occupations.itertuples()
    ]


def build_crosswalk(top_k: int | None = None) -> list[RoleMapping]:
    """Embed both vocabularies and match each role to its nearest occupations."""
    top_k = top_k or int(get("crosswalk.top_k", 5))
    threshold = float(get("crosswalk.auto_accept_threshold", 0.75))

    roles = distinct_roles()
    occupations = load_processed("occupation_master")
    model = _model()

    role_vectors = model.encode(
        roles["role"].tolist(), normalize_embeddings=True, show_progress_bar=False
    )
    occupation_vectors = model.encode(
        _occupation_corpus(occupations), normalize_embeddings=True, show_progress_bar=False
    )
    similarity = np.asarray(role_vectors) @ np.asarray(occupation_vectors).T

    mappings: list[RoleMapping] = []
    for i, row in roles.iterrows():
        order = np.argsort(similarity[i])[::-1][:top_k]
        best = int(order[0])
        alternatives = [
            {
                "soc_code": str(occupations.iloc[int(j)]["soc_code"]),
                "occupation_title": str(occupations.iloc[int(j)]["occupation_title"]),
                "confidence": round(float(similarity[i][int(j)]), 4),
            }
            for j in order[1:]
        ]
        confidence = float(similarity[i][best])
        mappings.append(
            RoleMapping(
                role=str(row["role"]),
                source=str(row["source"]),
                soc_code=str(occupations.iloc[best]["soc_code"]),
                occupation_title=str(occupations.iloc[best]["occupation_title"]),
                confidence=confidence,
                needs_review=confidence < threshold,
                alternatives=alternatives,
            )
        )

    flagged = sum(m.needs_review for m in mappings)
    log.info(
        "crosswalk built",
        extra={
            "roles": len(mappings),
            "flagged_for_review": flagged,
            "auto_accept_threshold": threshold,
            "mean_confidence": round(float(np.mean([m.confidence for m in mappings])), 4),
        },
    )
    return mappings


def write_auto(mappings: list[RoleMapping]) -> Path:
    path = project_root() / AUTO_FILE
    payload = {
        "_generated_by": "python -m hrai.skills.crosswalk",
        "_note": (
            "Automatic output. Edit conf/role_crosswalk_reviewed.yaml to override; "
            "the reviewed file always wins."
        ),
        "mappings": [m.to_dict() for m in mappings],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def load_reviewed() -> dict[str, dict[str, Any]]:
    """Human-reviewed overrides, keyed by role title."""
    path = project_root() / REVIEWED_FILE
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(entry["role"]): entry for entry in payload.get("mappings", [])}


def resolved_crosswalk() -> pd.DataFrame:
    """The crosswalk the rest of the pipeline uses: auto, overridden by reviewed."""
    path = project_root() / AUTO_FILE
    if not path.exists():
        mappings = build_crosswalk()
        write_auto(mappings)
        records = [m.to_dict() for m in mappings]
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        records = payload.get("mappings", [])

    reviewed = load_reviewed()
    resolved = []
    for record in records:
        role = record["role"]
        if role in reviewed:
            override = reviewed[role]
            record = {**record, **{k: v for k, v in override.items() if k != "alternatives"}}
            record["reviewed"] = True
            record["needs_review"] = False
        resolved.append(record)

    df = pd.DataFrame(resolved)
    log.info(
        "crosswalk resolved",
        extra={
            "roles": len(df),
            "human_reviewed": int(df["reviewed"].sum()) if "reviewed" in df else 0,
            "still_flagged": int(df["needs_review"].sum()) if "needs_review" in df else 0,
        },
    )
    return df


def main() -> int:
    from hrai.utils.logger import setup_logging

    setup_logging()
    mappings = build_crosswalk()
    path = write_auto(mappings)

    report = {
        "roles": len(mappings),
        "flagged_for_review": sum(m.needs_review for m in mappings),
        "mappings": [m.to_dict() for m in mappings],
    }
    (project_root() / "docs" / "role_crosswalk.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log.info("step 11 crosswalk written", extra={"file": str(path.name)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
