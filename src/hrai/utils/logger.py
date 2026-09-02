"""Structured logging with run IDs (Phase 0.7 / Steps 20-21).

Set up once at process start, then ``get_logger(__name__)`` anywhere. Every
record carries the same ``run_id`` so a prediction in ``data/predictions/`` can
be traced back to the exact pipeline run that produced it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import UTC, datetime

_RUN_ID = os.getenv("HRAI_RUN_ID") or uuid.uuid4().hex[:12]

# Attributes LogRecord always defines; anything else was passed via `extra=`.
_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


def run_id() -> str:
    """The current process's run ID, stable for the life of the process."""
    return _RUN_ID


class JsonFormatter(logging.Formatter):
    """One JSON object per line — greppable, and parseable by log tooling."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": _RUN_ID,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """The readable format from the Build Notes, for notebooks and local runs."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure the root logger. Idempotent — safe to call from a notebook cell."""
    from hrai.utils.config import get

    level = (level or os.getenv("HRAI_LOG_LEVEL") or get("logging.level", "INFO")).upper()
    fmt = fmt or get("logging.format", "json")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else HumanFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


class SafeLogger(logging.Logger):
    """A Logger that cannot be crashed by a colliding ``extra=`` key.

    ``logging`` raises ``KeyError: Attempt to overwrite 'name' in LogRecord`` if
    a structured field shadows a reserved LogRecord attribute — and ``name``,
    ``module``, ``filename`` and ``args`` are all natural field names in a data
    pipeline. Rather than making every call site remember the reserved list, we
    prefix collisions here.
    """

    def makeRecord(  # noqa: PLR0913 - signature fixed by the stdlib
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        if extra:
            extra = {(f"field_{k}" if k in _STANDARD else k): v for k, v in extra.items()}
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)


logging.setLoggerClass(SafeLogger)


def get_logger(name: str) -> logging.Logger:
    """A logger for ``name``. Configures the root logger on first use.

    Returns a plain ``Logger``, not a ``LoggerAdapter``: the default adapter's
    ``process()`` overwrites ``kwargs["extra"]`` with its own dict, which would
    silently discard every structured field passed at the call site.
    """
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)
