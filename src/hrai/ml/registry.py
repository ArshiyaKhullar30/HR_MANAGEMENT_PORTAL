"""Model registry and versioning (Step 09).

The Build Notes' scheme first — `models/vN/` with a `metadata.json` beside the
artifact — then MLflow, in that order. Simple versioning that works beats a
platform that is half-configured.

Metadata is expanded to what an audit actually needs: the git SHA, the input
data checksums, the exact feature list, metrics, the chosen threshold, the seed
and the library versions. Without those, "which model made this prediction?" has
no answer six months later.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from hrai.utils.config import get, project_root
from hrai.utils.logger import get_logger

log = get_logger(__name__)

ARTIFACT_NAME = "attrition_pipeline.joblib"
METADATA_NAME = "metadata.json"
_VERSION_DIR = re.compile(r"^v(\d+)$")


def models_dir() -> Path:
    path = project_root() / get("paths.models", "models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project_root(), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - git may be absent; not worth failing a train run
        return None


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("pandas", "numpy", "sklearn", "xgboost", "shap"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            versions[name] = "unavailable"
    return versions


def existing_versions() -> list[int]:
    return sorted(
        int(m.group(1))
        for path in models_dir().iterdir()
        if path.is_dir() and (m := _VERSION_DIR.match(path.name))
    )


def next_version() -> str:
    versions = existing_versions()
    return f"v{(versions[-1] + 1) if versions else 1}"


def save_model(
    pipeline: Any,
    *,
    algorithm: str,
    metrics: dict[str, Any],
    threshold: float,
    feature_columns: list[str],
    encoded_feature_names: list[str],
    data_checksums: dict[str, str],
    extra: dict[str, Any] | None = None,
    version: str | None = None,
) -> tuple[str, Path]:
    """Persist a pipeline plus the metadata needed to reproduce and audit it."""
    version = version or next_version()
    version_dir = models_dir() / version
    version_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = version_dir / ARTIFACT_NAME
    joblib.dump(pipeline, artifact_path)

    metadata: dict[str, Any] = {
        "model_name": "Attrition Prediction Model",
        "version": version,
        "algorithm": algorithm,
        "training_date": datetime.now(UTC).isoformat(),
        "random_seed": get("random_seed", 42),
        "operating_threshold": round(float(threshold), 4),
        "risk_bands": get("model.risk_bands"),
        "metrics": metrics,
        "feature_columns": feature_columns,
        "encoded_feature_count": len(encoded_feature_names),
        "target": get("model.target", "Attrition"),
        "trained_on": "employee_attrition (Population A)",
        "data_checksums": data_checksums,
        "git_sha": _git_sha(),
        "libraries": _library_versions(),
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": _sha256(artifact_path),
    }
    if extra:
        metadata.update(extra)

    (version_dir / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    # `latest` is a pointer file rather than a symlink so it survives Docker
    # builds and Windows checkouts.
    (models_dir() / "LATEST").write_text(version, encoding="utf-8")

    log.info(
        "model version saved",
        extra={
            "version": version,
            "algorithm": algorithm,
            "threshold": metadata["operating_threshold"],
            "path": str(version_dir),
        },
    )
    return version, version_dir


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def latest_version() -> str | None:
    pointer = models_dir() / "LATEST"
    if pointer.exists():
        version = pointer.read_text(encoding="utf-8").strip()
        if (models_dir() / version / ARTIFACT_NAME).exists():
            return version
    versions = existing_versions()
    return f"v{versions[-1]}" if versions else None


def load_model(version: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Load a pipeline and its metadata. Defaults to the latest version."""
    version = version or latest_version()
    if version is None:
        raise FileNotFoundError("No trained model found. Run `make train` first.")
    version_dir = models_dir() / version
    artifact = version_dir / ARTIFACT_NAME
    if not artifact.exists():
        raise FileNotFoundError(f"Model artifact missing: {artifact}")
    pipeline = joblib.load(artifact)
    metadata_path = version_dir / METADATA_NAME
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    log.info("model loaded", extra={"version": version, "algorithm": metadata.get("algorithm")})
    return pipeline, metadata


def list_versions() -> list[dict[str, Any]]:
    out = []
    for version in existing_versions():
        path = models_dir() / f"v{version}" / METADATA_NAME
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            out.append(
                {
                    k: meta.get(k)
                    for k in (
                        "version",
                        "algorithm",
                        "training_date",
                        "operating_threshold",
                        "metrics",
                    )
                }
            )
    return out
