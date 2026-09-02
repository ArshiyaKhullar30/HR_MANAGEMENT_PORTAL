"""Model loading for the API (Step 18).

The model is loaded **once**, at application startup, and held for the process
lifetime. Loading per request would add roughly 800 ms to every call for no
benefit — the single most common performance mistake in an ML service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrai.ml.registry import load_model, models_dir
from hrai.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ModelBundle:
    """Everything the API needs to serve a prediction, resolved at startup."""

    pipeline: Any
    metadata: dict[str, Any]
    version: str
    threshold: float

    @property
    def algorithm(self) -> str:
        return str(self.metadata.get("algorithm", "unknown"))


_BUNDLE: ModelBundle | None = None


def load_bundle(version: str | None = None) -> ModelBundle:
    pipeline, metadata = load_model(version)
    bundle = ModelBundle(
        pipeline=pipeline,
        metadata=metadata,
        version=str(metadata.get("version", version or "unknown")),
        threshold=float(metadata.get("operating_threshold", 0.5)),
    )
    log.info(
        "model bundle loaded",
        extra={
            "version": bundle.version,
            "algorithm": bundle.algorithm,
            "threshold": bundle.threshold,
        },
    )
    return bundle


def set_bundle(bundle: ModelBundle | None) -> None:
    global _BUNDLE
    _BUNDLE = bundle


def get_bundle() -> ModelBundle:
    """The process-wide model bundle. Raises if startup has not run."""
    if _BUNDLE is None:
        raise RuntimeError(
            "Model is not loaded. The API loads it during startup; if you are calling "
            "this outside the app lifespan, call set_bundle(load_bundle()) first."
        )
    return _BUNDLE


def model_is_available() -> bool:
    if _BUNDLE is not None:
        return True
    return (models_dir() / "LATEST").exists()
