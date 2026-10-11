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


def _middleware_stack(cfg: ProxyConfig, metrics) -> list[tuple[type, dict]]:
    """The middleware stack in **execution order** (outermost first).

        RequestID → AccessLog → BodySize → CORS → RateLimit → Auth → routes

    Why this order:
        RequestID   outermost, so *every* response — including 413/429/401 —
                    carries X-Request-ID and access logs can be correlated.
        AccessLog   outside RateLimit/Auth so rejected requests are logged too.
        BodySize    before anything reads the body: auth parses the JSON body
                    for the model name, so the cap must be enforced first.
        CORS        outside Auth, otherwise browser preflights (which never
                    carry an Authorization header) get a 401.
        RateLimit   before Auth so credential guessing is itself throttled.
        Auth        innermost; runs immediately before routing.

    ``create_app`` registers these in reverse: Starlette's ``add_middleware``
    prepends, so the last one added is the outermost.
    """
    from .middleware.access_log import AccessLogMiddleware
    from .middleware.auth import AuthMiddleware
    from .middleware.body_size import BodySizeMiddleware
    from .middleware.rate_limit import RateLimitMiddleware
    from .middleware.request_id import RequestIDMiddleware

    stack: list[tuple[type, dict]] = [
        (RequestIDMiddleware, {}),
        (AccessLogMiddleware, {"metrics": metrics}),
        (BodySizeMiddleware, {"max_body_bytes": int(cfg.max_body_size_mb * 1024 * 1024)}),
    ]

    if cfg.cors.enabled:
        stack.append((
            CORSMiddleware,
            {
                "allow_origins": cfg.cors.allow_origins,
                "allow_methods": cfg.cors.allow_methods,
                "allow_headers": cfg.cors.allow_headers,
            },
        ))

    stack.append((RateLimitMiddleware, {"config": cfg.rate_limit, "metrics": metrics}))
    stack.append((AuthMiddleware, {}))

    return stack


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Each call builds a fresh app from the provided config.  When *config* is
    None the current config is read from the global registry via ``get()`` —
    that is what :meth:`ConfigReloadableApp.reload` relies on to build an app
    from freshly reloaded config, so *every* config value (CORS, rate limits,
    auth, static files) is re-read on every build.  Idempotent.

    Raises:
        RuntimeError: if *config* is None and no config has been loaded yet.
    """
    cfg = config if config is not None else get()

    # ── App scaffold ──────────────────────────────────────────────────────

    async def _lifespan(app):
        # Health probe is managed by ConfigReloadableApp, not per-instance.
        yield

    app = FastAPI(
        title="Simple LLM Proxy",
        description="OpenAI-compatible reverse proxy for multiple LLM backends. "
                    "Routes /v1/* requests to configured backend servers, merging per-model default parameters.",
        version="0.1.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    # ── Exception handlers ────────────────────────────────────────────────

    from adapters.base import BackendError, make_error_response

    @app.exception_handler(BackendError)
    async def backend_error_handler(request: Request, exc: BackendError):
        logger.warning("Backend error on %s: %s", request.url.path, exc.message)
        return JSONResponse(status_code=502, content=make_error_response(exc.message, "backend_error", exc.status_code))

    # ── Middleware ────────────────────────────────────────────────────────
    # Registered innermost-first; see _middleware_stack() for the order and
    # the reasoning behind it.

    from proxy.metrics import get_metrics_store
    metrics = get_metrics_store()

    for cls, kwargs in reversed(_middleware_stack(cfg, metrics)):
        app.add_middleware(cls, **kwargs)

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

    for sf in cfg.static_files:
        for dir_path_str in sf.directories:
            dir_path = Path(dir_path_str)
            if dir_path.is_dir():
                app.mount(sf.path, StaticFiles(directory=str(dir_path)))
                logger.info("Serving static files at %s from %s", sf.path, dir_path)
            else:
                logger.warning("Static directory not found, skipping: %s", dir_path)

    app.state.metrics = metrics

    return app
