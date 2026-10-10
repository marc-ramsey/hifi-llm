"""FastAPI app factory — creates the app, registers middleware and routes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from config import ProxyConfig, get

logger = logging.getLogger(__name__)


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Each call builds a fresh app from the provided config (or the current
    global config if *config* is None).  Idempotent — safe to call on SIGHUP.
    """

    # ── App scaffold ──────────────────────────────────────────────────────

    async def _lifespan(app):
        # Health probe is managed by ConfigReloadableApp, not per-instance.
        yield

    app = FastAPI(
        title="Simple LLM Proxy",
        description="OpenAI-compatible reverse proxy for multiple LLM backends",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # ── Exception handlers ────────────────────────────────────────────────

    from adapters.base import BackendError

    @app.exception_handler(BackendError)
    async def backend_error_handler(request: Request, exc: BackendError):
        logger.warning("Backend error on %s: %s", request.url.path, exc.message)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": exc.message,
                    "type": "backend_error",
                    "param": None,
                    "code": exc.status_code,
                },
            },
        )

    # ── Middleware (outermost → innermost) ────────────────────────────────

    from .middleware.body_size import BodySizeMiddleware
    app.add_middleware(BodySizeMiddleware)

    cors_cfg = config.cors if config else get().cors
    if cors_cfg.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_cfg.allow_origins,
            allow_methods=cors_cfg.allow_methods,
            allow_headers=cors_cfg.allow_headers,
        )

    from .middleware.request_id import RequestIDMiddleware
    app.add_middleware(RequestIDMiddleware)

    metrics = _get_metrics_store()
    from .middleware.access_log import AccessLogMiddleware
    app.add_middleware(AccessLogMiddleware, metrics=metrics)

    from .middleware.rate_limit import RateLimitMiddleware
    rate_cfg = config.rate_limit if config else get().rate_limit
    app.add_middleware(RateLimitMiddleware, config=rate_cfg, metrics=metrics)

    from .middleware.auth import AuthMiddleware
    app.add_middleware(AuthMiddleware)

    # ── Routes ────────────────────────────────────────────────────────────

    from .router import router
    app.include_router(router)

    @app.get("/health")
    async def health(models: bool = False):
        result = {"status": "ok", "config_loaded": True}
        if models:
            from proxy.health import _current_report
            report = _current_report()
            result["models"] = {
                bh.name: {
                    "status": bh.status.value,
                    "error": bh.error,
                    "latency_ms": bh.latency_ms,
                    "model_count": bh.model_count,
                }
                for bh in report.backends
            }
        return result

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        return PlainTextResponse(content=metrics.generate())

    # ── Static file serving ───────────────────────────────────────────────

    for sf in config.static_files:
        for dir_path_str in sf.directories:
            dir_path = Path(dir_path_str)
            if dir_path.is_dir():
                app.mount(sf.path, StaticFiles(directory=str(dir_path)))
                logger.info("Serving static files at %s from %s", sf.path, dir_path)
            else:
                logger.warning("Static directory not found, skipping: %s", dir_path)

    app.state.metrics = metrics

    return app


def _get_metrics_store():
    """Return the singleton MetricsStore (survives SIGHUP reloads)."""
    from proxy.metrics import get_metrics_store
    return get_metrics_store()
