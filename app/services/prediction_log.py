"""Prediction logging (Step 21).

Separate from application logs. Every prediction is appended to a
date-partitioned Parquet dataset under `data/predictions/`, recording the model
version, a hash of the input features, the probability and the risk band.

This is the feed that drift monitoring (Step 25) consumes: comparing today's
prediction distribution against the training distribution is the earliest
available warning that the model has drifted, and it needs no ground truth.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from hrai.utils.config import data_dir, get
from hrai.utils.logger import get_logger, run_id

log = get_logger(__name__)

# Appends come from request handlers, which may run concurrently.
_LOCK = threading.Lock()


def feature_hash(features: dict[str, Any]) -> str:
    """Stable hash of the input, so identical requests are recognisable."""
    payload = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def log_prediction(
    *,
    employee_id: int | None,
    features: dict[str, Any],
    probability: float,
    risk_band: str,
    threshold: float,
    model_version: str,
    source: str = "api",
) -> None:
    if not get("logging.prediction_log.enabled", True):
        return

    now = datetime.now(UTC)
    record = {
        "timestamp": now.isoformat(),
        "date": now.date().isoformat(),
        "run_id": run_id(),
        "employee_id": employee_id,
        "feature_hash": feature_hash(features),
        "model_version": model_version,
        "probability": round(float(probability), 6),
        "risk_band": risk_band,
        "threshold": float(threshold),
        "flagged": bool(probability >= threshold),
        "source": source,
    }

    directory = data_dir("predictions") / f"date={record['date']}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predictions.parquet"

    with _LOCK:
        frame = pd.DataFrame([record])
        if path.exists():
            frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        frame.to_parquet(path, index=False)

    log.info(
        "prediction logged",
        extra={
            "employee_id": employee_id,
            "model_version": model_version,
            "probability": record["probability"],
            "risk_band": risk_band,
        },
    )


def read_predictions(limit: int | None = None) -> pd.DataFrame:
    """Every logged prediction, newest last. Empty frame when none exist yet."""
    root = data_dir("predictions")
    files = sorted(root.glob("date=*/predictions.parquet"))
    if not files:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "employee_id",
                "model_version",
                "probability",
                "risk_band",
                "flagged",
            ]
        )
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    frame = frame.sort_values("timestamp")
    return frame.tail(limit) if limit else frame
