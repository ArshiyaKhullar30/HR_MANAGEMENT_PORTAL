"""Step 03 entry point — produce `data/processed/` from `data/raw/`.

    make clean-data       # or: python -m hrai.cleaning.run

Raw files are never modified. Outputs are written as Parquet with a CSV mirror
under the filenames the Build Notes specify, and every artifact is checksummed
into `data/processed/_manifest.json`.

After writing, the processed frames are re-validated in **strict** mode: the
defects that failed on the raw data (F6) must now pass, or the step fails.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from hrai.cleaning.attrition import clean_attrition
from hrai.cleaning.engagement import clean_engagement_events, to_employee_grain
from hrai.cleaning.onet import clean_essential_skills, clean_occupations, clean_software_skills
from hrai.utils.config import project_root
from hrai.utils.io import load_raw, save_processed
from hrai.utils.logger import get_logger, run_id, setup_logging
from hrai.validation.runner import validate
from hrai.validation.schemas import attrition_processed_schema, engagement_processed_schema

log = get_logger(__name__)


def main(strict: bool = True) -> int:
    setup_logging()
    manifest = {}

    # -- Population A ------------------------------------------------------
    attrition = clean_attrition(load_raw("employee_attrition"))
    manifest["employee_attrition_processed"] = save_processed(
        attrition, "employee_attrition_processed"
    )

    # -- Population B ------------------------------------------------------
    events = clean_engagement_events(load_raw("hr_performance_engagement"))
    manifest["engagement_events_processed"] = save_processed(events, "engagement_events_processed")
    engagement = to_employee_grain(events)
    manifest["engagement_processed"] = save_processed(engagement, "engagement_processed")

    # -- O*NET reference / skill ontology ----------------------------------
    occupations = clean_occupations(load_raw("occupation_data"))
    manifest["occupation_master"] = save_processed(occupations, "occupation_master")

    essential = clean_essential_skills(load_raw("essential_skills"))
    manifest["essential_skills_processed"] = save_processed(essential, "essential_skills_processed")

    software = clean_software_skills(load_raw("software_skills"))
    manifest["software_skills_processed"] = save_processed(software, "software_skills_processed")

    # -- prove the cleaning worked -----------------------------------------
    mode = "strict" if strict else "report"
    reports = {
        "employee_attrition_processed": validate(
            attrition, attrition_processed_schema(), "employee_attrition_processed", mode=mode
        ),
        "engagement_processed": validate(
            engagement, engagement_processed_schema(), "engagement_processed", mode=mode
        ),
    }

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id(),
        "mode": mode,
        "stage": "processed",
        "datasets": {k: v.to_dict() for k, v in reports.items()},
        "artifacts": manifest,
    }
    (project_root() / "docs" / "cleaning_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    log.info(
        "step 03 complete",
        extra={
            "artifacts": len(manifest),
            "all_processed_schemas_pass": all(r.passed for r in reports.values()),
        },
    )
    return 0 if all(r.passed for r in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
