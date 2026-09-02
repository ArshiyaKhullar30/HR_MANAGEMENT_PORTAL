"""FastAPI application (Steps 18-20).

Production shape rather than a script with decorators on it:

* **App factory**, so tests build an isolated instance instead of importing
  global state.
* **Lifespan startup** loads the model once. Loading per request would add
  roughly 800 ms to every call.
* **Routers per domain**, matching the module layout in the Build Notes.
* **Request-correlated structured logging**, so a prediction in the log can be
  traced back to the request that produced it.
* **Startup degrades rather than crashes**: if the model has not been trained
  yet, the app still serves `/health` and reports what is missing. A service
  that refuses to boot tells you nothing about why.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import attrition, dashboard, intelligence, skills
from app.ml.model_loader import load_bundle, model_is_available, set_bundle
from app.services.intelligence_service import get_intelligence_service
from app.validation.employee_schema import HealthResponse
from hrai.utils.config import get
from hrai.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

API_PREFIX = get("api.prefix", "/api/v1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("application starting", extra={"api_prefix": API_PREFIX})

    if model_is_available():
        try:
            set_bundle(load_bundle())
        except Exception as exc:  # noqa: BLE001
            log.error(
                "model failed to load; prediction endpoints will return 503",
                extra={"error": str(exc)},
            )
    else:
        log.warning("no trained model found — run `make train`")

    try:
        service = get_intelligence_service()
        if service.is_available():
            _ = service.table  # warm the cache at startup, not on first request
            log.info("dataset loaded", extra={"rows": len(service.table)})
        else:
            log.warning("employee intelligence table not built — run `make intelligence`")
    except Exception as exc:  # noqa: BLE001
        log.error("intelligence table failed to load", extra={"error": str(exc)})

    log.info("application ready")
    yield
    log.info("application shutting down")
    set_bundle(None)


def create_app() -> FastAPI:
    app = FastAPI(
        title=get("api.title", "Enterprise HR AI"),
        version=str(get("project.version", "0.1.0")),
        description=(
            "Workforce Intelligence & Upskilling Platform — attrition risk, engagement, "
            "skill gaps, upskilling recommendations, and the Retention ROI Copilot.\n\n"
            "**Two populations.** `employee_attrition` (A) and "
            "`hr_performance_engagement` (B) are different companies and are never "
            "joined on employee id. Use `person_key` (`A-101`), not a bare id."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local Streamlit frontend; tighten before deployment
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            log.exception(
                "unhandled error",
                extra={"request_id": request_id, "path": request.url.path, "error": str(exc)},
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Liveness plus what is actually loaded — a bare 'ok' helps nobody."""
        from app.ml.model_loader import _BUNDLE

        return HealthResponse(
            status="ok",
            model_loaded=_BUNDLE is not None,
            model_version=_BUNDLE.version if _BUNDLE else None,
            data_loaded=get_intelligence_service().is_available(),
        )

    @app.get("/ready", tags=["system"])
    def ready() -> JSONResponse:
        """Readiness: 503 until the service can actually answer real requests."""
        from app.ml.model_loader import _BUNDLE

        model_ok = _BUNDLE is not None
        data_ok = get_intelligence_service().is_available()
        ready_now = model_ok and data_ok
        return JSONResponse(
            status_code=200 if ready_now else 503,
            content={
                "ready": ready_now,
                "model_loaded": model_ok,
                "intelligence_table": data_ok,
                "hint": (
                    None if ready_now else "Run: make pipeline && make train && make intelligence"
                ),
            },
        )

    for router in (attrition.router, dashboard.router, skills.router, intelligence.router):
        app.include_router(router, prefix=API_PREFIX)

    return app


app = create_app()
