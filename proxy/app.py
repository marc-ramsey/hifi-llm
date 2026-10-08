"""FastAPI app factory — creates the app, registers middleware and routes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ProxyConfig, get


logger = logging.getLogger(__name__)


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Simple LLM Proxy",
        description="OpenAI-compatible reverse proxy for multiple LLM backends",
        version="0.1.0",
    )

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
