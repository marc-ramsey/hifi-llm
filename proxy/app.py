"""FastAPI app factory — creates the app, registers middleware and routes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import asyncio

from config import ProxyConfig, get, set


logger = logging.getLogger(__name__)


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    async def _lifespan(app):
        from proxy.health import start_health_probe, _deferred_models
        if _deferred_models is not None:
            start_health_probe(_deferred_models)
        yield

    app = FastAPI(
        title="Simple LLM Proxy",
        description="OpenAI-compatible reverse proxy for multiple LLM backends",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # Convert BackendError (from adapters) into OpenAI-style 502 responses
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

    # Request size limit — reject bodies larger than 10 MB to prevent DoS
    MAX_BODY_SIZE = 10 * 1024 * 1024

    class BodyTooLargeError(Exception):
        """Raised when the request body exceeds MAX_BODY_SIZE."""
        pass

    class BodySizeMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Fast path: check Content-Length header
            content_length = request.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_BODY_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"message": "Request body too large (max 10 MB)", "type": "payload_too_large", "param": None, "code": 413}},
                )

            # For chunked encoding or missing Content-Length, we need to
            # enforce the limit by intercepting the receive callable.
            original_receive = request._receive  # type: ignore[attr-defined]
            total_size = 0

            async def limited_receive() -> dict:
                nonlocal total_size
                message = await original_receive()
                if message["type"] == "http.request":
                    body = message.get("body", b"")
                    total_size += len(body)
                    if total_size > MAX_BODY_SIZE:
                        raise BodyTooLargeError()
                return message

            try:
                request._receive = limited_receive  # type: ignore[attr-defined]
            except AttributeError:
                pass  # Some ASGI servers may not allow this; fall through

            try:
                return await call_next(request)
            except BodyTooLargeError:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"message": "Request body too large (max 10 MB)", "type": "payload_too_large", "param": None, "code": 413}},
                )

    app.add_middleware(BodySizeMiddleware)

    # CORS — allow all origins for development; lock down in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth middleware — registered before routes
    from .middleware import AuthMiddleware
    auth_cfg = config.auth if config else get().auth
    app.add_middleware(AuthMiddleware, api_key=auth_cfg.api_key)

    # Register routes under /v1
    from .router import router
    app.include_router(router)

    # Health endpoint — at root, not under /v1
    from proxy.health import _current_report
    @app.get("/health")
    async def health(models: bool = False):
        result = {"status": "ok", "config_loaded": True}
        if models:
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

    # Start periodic health probe task
    from proxy.health import start_health_probe, stop_health_probe
    interval = config.health_check_interval if config else 2.0
    start_health_probe(config.models, interval=interval)

    # Load plugins after routes are registered (plugins can add their own routes)
    plugins_dir = config.plugins_dir if config else get().plugins_dir
    if plugins_dir:
        from plugins.manager import load_plugins
        load_plugins(Path(plugins_dir), app)

    # Store cleanup hook on the app for graceful shutdown
    app.state.stop_health_probe = stop_health_probe

    return app
