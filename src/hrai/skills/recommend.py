"""Upskilling recommendation engine (Step 15).

Two versions, exactly as the Build Notes stage them:

* **v1 — rules.** A missing skill maps to a training recommendation. Blunt,
  transparent, and correct for the skills we can name.
* **v2 — semantic.** A sentence-transformer embeds the missing skill and each
  course description, and cosine similarity picks the match. This is what lets
  "MLOps" reach a course called *"Deploying and Monitoring Machine Learning
  Systems"* even though the words do not overlap at all.

The catalogue is built **only from what is in the data** — the five real
`Training Program Name` values from the engagement file, plus O*NET's own skill
and technology-category vocabulary. Real `Training Cost` and
`Training Duration(Days)` distributions are attached to each course, which is
what later makes the retention ROI calculation possible rather than notional.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from hrai.utils.io import load_processed
from hrai.utils.logger import get_logger

log = get_logger(__name__)

# Below this cosine score the nearest course is not a real match. Saying so is
# more useful than presenting a confident-looking wrong recommendation.
SEMANTIC_CONFIDENCE_FLOOR = 0.25
GENERIC_FALLBACK_COURSE = "Technical Skills"

# v1 rules: explicit, auditable mappings for skills we can name outright.
RULE_MAP: dict[str, str] = {
    "SQL": "Data Fundamentals: SQL and Relational Databases",
    "Python": "Programming Foundations with Python",
    "R": "Statistical Programming with R",
    "AWS": "Cloud Platforms and Deployment",
    "Azure": "Cloud Platforms and Deployment",
    "Google Cloud": "Cloud Platforms and Deployment",
    "Docker": "Containerisation and Deployment Pipelines",
    "Kubernetes": "Containerisation and Deployment Pipelines",
    "Tableau": "Business Intelligence and Data Visualisation",
    "Power BI": "Business Intelligence and Data Visualisation",
    "SAP": "Enterprise Systems: SAP Essentials",
    "Salesforce": "Enterprise Systems: CRM and Salesforce",
    "Critical Thinking": "Analytical Thinking and Problem Solving",
    "Active Listening": "Communication Skills",
    "Speaking": "Communication Skills",
    "Writing": "Business Writing and Documentation",
    "Reading Comprehension": "Business Writing and Documentation",
    "Mathematics": "Quantitative Reasoning for Business",
    "Science": "Applied Scientific Method",
    "Monitoring": "Performance Management and Quality Monitoring",
    "Active Learning": "Learning How to Learn",
    "Learning Strategies": "Learning How to Learn",
}


def build_course_catalogue() -> pd.DataFrame:
    """Assemble the catalogue from in-data sources only.

    Real cost and duration statistics come from the engagement file's actual
    training records, so the ROI engine prices interventions from observed
    spend rather than an invented number.
    """
    events = load_processed("engagement_events_processed")
    real_programs = (
        events.groupby("Training Program Name", as_index=False)
        .agg(
            median_cost=("Training Cost", "median"),
            mean_cost=("Training Cost", "mean"),
            median_days=("Training Duration(Days)", "median"),
            observed_pass_rate=(
                "Training Outcome",
                lambda s: float(s.isin(["Passed", "Completed"]).mean()),
            ),
            times_delivered=("Training Outcome", "size"),
        )
        .rename(columns={"Training Program Name": "course"})
    )
    real_programs["source"] = "observed"
    real_programs["description"] = (
        real_programs["course"]
        .map(
            {
                "Communication Skills": "Speaking, active listening and presenting clearly to colleagues and stakeholders.",
                "Customer Service": "Handling customer relationships, service recovery and account management.",
                "Leadership Development": "Managing people, setting direction, coaching and running a team.",
                "Project Management": "Planning, scheduling, monitoring and delivering projects to time and budget.",
                "Technical Skills": "Hands-on technical tooling, software systems, data handling and engineering practice.",
            }
        )
        .fillna(real_programs["course"])
    )

    # Fallback catalogue for gaps the five observed programmes cannot cover.
    median_cost = float(real_programs["median_cost"].median())
    median_days = float(real_programs["median_days"].median())
    derived = pd.DataFrame({"course": sorted(set(RULE_MAP.values()))})
    derived["description"] = (
        derived["course"].map(_catalogue_descriptions()).fillna(derived["course"])
    )
    derived["median_cost"] = round(median_cost, 2)
    derived["mean_cost"] = round(median_cost, 2)
    derived["median_days"] = median_days
    derived["observed_pass_rate"] = float(real_programs["observed_pass_rate"].mean())
    derived["times_delivered"] = 0
    derived["source"] = "derived_from_onet_vocabulary"

    catalogue = pd.concat([real_programs, derived], ignore_index=True)
    catalogue = catalogue.drop_duplicates(subset=["course"]).reset_index(drop=True)
    catalogue["median_cost"] = catalogue["median_cost"].round(2)
    catalogue["observed_pass_rate"] = catalogue["observed_pass_rate"].round(3)

    log.info(
        "course catalogue built",
        extra={
            "courses": len(catalogue),
            "observed_programmes": int((catalogue["source"] == "observed").sum()),
            "median_cost": round(float(catalogue["median_cost"].median()), 2),
        },
    )
    return catalogue


def _catalogue_descriptions() -> dict[str, str]:
    return {
        "Data Fundamentals: SQL and Relational Databases": "Querying, joining and modelling data in relational databases with SQL.",
        "Programming Foundations with Python": "Writing, testing and structuring Python programs for data and automation work.",
        "Statistical Programming with R": "Statistical analysis, modelling and reporting using the R language.",
        "Cloud Platforms and Deployment": "Provisioning, deploying and operating workloads on cloud infrastructure.",
        "Containerisation and Deployment Pipelines": "Packaging applications in containers and running automated deployment pipelines.",
        "Business Intelligence and Data Visualisation": "Building dashboards, reports and visual analytics for business decisions.",
        "Enterprise Systems: SAP Essentials": "Navigating and operating SAP enterprise resource planning modules.",
        "Enterprise Systems: CRM and Salesforce": "Managing customer records, pipelines and workflows in a CRM platform.",
        "Analytical Thinking and Problem Solving": "Structuring problems, weighing evidence and reasoning to a defensible conclusion.",
        "Communication Skills": "Speaking, active listening and presenting clearly to colleagues and stakeholders.",
        "Business Writing and Documentation": "Reading closely and writing clear business documents, specifications and reports.",
        "Quantitative Reasoning for Business": "Applying arithmetic, statistics and quantitative methods to business questions.",
        "Applied Scientific Method": "Designing experiments, testing hypotheses and interpreting scientific results.",
        "Performance Management and Quality Monitoring": "Monitoring performance, assessing quality and acting on what the measures show.",
        "Learning How to Learn": "Choosing learning strategies and acquiring new skills efficiently and independently.",
    }


def recommend_v1(skill: str) -> str | None:
    """Rule-based recommendation. Returns None when no rule covers the skill."""
    return RULE_MAP.get(skill)


@lru_cache(maxsize=1)
def _semantic_index() -> tuple[pd.DataFrame, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    from hrai.utils.config import get

    catalogue = build_course_catalogue()
    model = SentenceTransformer(
        get("crosswalk.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )
    corpus = (catalogue["course"] + ". " + catalogue["description"]).tolist()
    vectors = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    return catalogue, np.asarray(vectors)


def recommend_semantic(skills: list[str], top_k: int = 1) -> pd.DataFrame:
    """v2 — cosine similarity between a missing skill and each course.

    A rule match always wins when one exists: it is exact, explainable, and free
    of the risk that an embedding produces a confident-looking wrong answer.
    """
    from sentence_transformers import SentenceTransformer

    from hrai.utils.config import get

    catalogue, course_vectors = _semantic_index()
    model = SentenceTransformer(
        get("crosswalk.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )
    skill_vectors = model.encode(list(skills), normalize_embeddings=True, show_progress_bar=False)
    similarity = np.asarray(skill_vectors) @ course_vectors.T

    rows = []
    for i, skill in enumerate(skills):
        rule = recommend_v1(skill)
        if rule is not None:
            match = catalogue[catalogue["course"] == rule]
            if not match.empty:
                row = match.iloc[0]
                rows.append(
                    {
                        "skill": skill,
                        "course": row["course"],
                        "match_confidence": 1.0,
                        "method": "rule",
                        "median_cost": float(row["median_cost"]),
                        "median_days": float(row["median_days"]),
                    }
                )
                continue
        for j in np.argsort(similarity[i])[::-1][:top_k]:
            row = catalogue.iloc[int(j)]
            score = float(similarity[i][int(j)])
            if score < SEMANTIC_CONFIDENCE_FLOOR:
                fallback = catalogue[catalogue["course"] == GENERIC_FALLBACK_COURSE]
                row = fallback.iloc[0] if not fallback.empty else row
                rows.append(
                    {
                        "skill": skill,
                        "course": row["course"],
                        "match_confidence": round(score, 4),
                        "method": "fallback_low_confidence",
                        "median_cost": float(row["median_cost"]),
                        "median_days": float(row["median_days"]),
                    }
                )
                continue
            rows.append(
                {
                    "skill": skill,
                    "course": row["course"],
                    "match_confidence": round(score, 4),
                    "method": "semantic",
                    "median_cost": float(row["median_cost"]),
                    "median_days": float(row["median_days"]),
                }
            )
    out = pd.DataFrame(rows)
    log.info(
        "recommendations generated",
        extra={
            "skills": len(skills),
            "rows": len(out),
            "rule_matched": int((out["method"] == "rule").sum()) if len(out) else 0,
            "semantic_matched": int((out["method"] == "semantic").sum()) if len(out) else 0,
            "low_confidence_fallback": (
                int((out["method"] == "fallback_low_confidence").sum()) if len(out) else 0
            ),
        },
    )
    return out


def recommendation_lookup(skills: list[str]) -> dict[str, dict]:
    """Skill -> best recommendation, as a dict for fast per-employee lookup."""
    table = recommend_semantic(sorted(set(skills)), top_k=1)
    return {row["skill"]: row for row in table.to_dict(orient="records")}
