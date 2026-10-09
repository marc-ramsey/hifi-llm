"""FastAPI app factory — creates the app, registers middleware and routes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import ProxyConfig, get, set


logger = logging.getLogger(__name__)


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Simple LLM Proxy",
        description="OpenAI-compatible reverse proxy for multiple LLM backends",
        version="0.1.0",
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

    class BodySizeMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            content_length = request.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_BODY_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"message": "Request body too large (max 10 MB)", "type": "payload_too_large", "param": None, "code": 413}},
                )
            return await call_next(request)

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
    from config import get
    @app.get("/health")
    async def health():
        return {"status": "ok", "config_loaded": True}

    # Load plugins after routes are registered (plugins can add their own routes)
    plugins_dir = config.plugins_dir if config else get().plugins_dir
    if plugins_dir:
        from plugins.manager import load_plugins
        load_plugins(Path(plugins_dir), app)

    return app
