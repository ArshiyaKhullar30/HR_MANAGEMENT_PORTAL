# syntax=docker/dockerfile:1
# =============================================================================
# Enterprise HR AI — multi-stage image (Step 24)
#
# One image, two entrypoints. The backend and frontend share every dependency
# and all of src/, so building them separately would double the build time and
# the disk footprint to save nothing. docker-compose runs the same image twice
# with different commands.
# =============================================================================

# ---- builder: compile wheels once, so the runtime layer stays small ---------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is the OpenMP runtime XGBoost links against. Without it,
# `import xgboost` fails at runtime with a dlopen error that looks nothing
# like a missing system package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt


# ---- runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app" \
    HRAI_CONFIG=conf/config.yaml \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Never run as root. A container that only serves predictions has no business
# with write access to its own filesystem beyond the data directories.
RUN useradd --create-home --shell /bin/bash hrai
WORKDIR /app

COPY --chown=hrai:hrai conf/ ./conf/
COPY --chown=hrai:hrai src/ ./src/
COPY --chown=hrai:hrai app/ ./app/
COPY --chown=hrai:hrai frontend/ ./frontend/
COPY --chown=hrai:hrai scripts/ ./scripts/
COPY --chown=hrai:hrai pyproject.toml requirements.txt ./

RUN mkdir -p data/raw data/interim data/processed data/predictions models .cache \
    && chown -R hrai:hrai /app

USER hrai
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
