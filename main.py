"""Simple LLM Proxy — entry point with signal handling."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from pathlib import Path

import uvicorn

from config import ProxyConfig, get, set
from config.loader import load_config
from proxy.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm-proxy")

_config_path: str | None = None
_shutdown_event: asyncio.Event | None = None
_server: uvicorn.Server | None = None


def _load_config(path: str | None = None) -> ProxyConfig:
    """Load and validate config, set in registry."""
    config = load_config(path)
    set(config)
    return config


def _shutdown(signum: int, _frame):
    """Handle SIGTERM/SIGINT — initiate graceful shutdown."""
    logger.info("Received signal %d, initiating shutdown...", signum)
    if _shutdown_event:
        _shutdown_event.set()
    if _server:
        _server.should_exit = True


def _reload(signum: int, _frame):
    """Handle SIGHUP — reload config without dropping connections."""
    logger.info("Received SIGHUP, reloading config...")
    try:
        _load_config(_config_path)
        logger.info("Config reloaded successfully")
    except Exception:
        logger.exception("Config reload failed — keeping old config")


def run(config_path: str | None = None) -> None:
    """Bootstrap the proxy and start serving."""
    global _config_path, _shutdown_event, _server

    _config_path = config_path
    _shutdown_event = asyncio.Event()

    # Load config on first startup
    config = _load_config(config_path)
    logger.info("Loaded config: %d model(s), auth=%s, plugins_dir=%s",
                len(config.models),
                "enabled" if config.auth.api_key else "disabled",
                config.plugins_dir)

    # Register signal handlers
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGHUP, _reload)

    # Create app
    app = create_app(config)

    # Determine server config
    listen_cfg = config.listen
    host = listen_cfg.host
    port = listen_cfg.port

    logger.info("Starting server on %s:%d", host, port)
    logger.info("Config file: %s", config_path or "proxy-config.yaml")

    # Start server
    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        loop="uvloop",
    )
    _server = uvicorn.Server(server_config)

    # Run server and wait for shutdown signal
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_server.serve())
    finally:
        loop.close()

    logger.info("Shutdown complete")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run(config_path)
