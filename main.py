"""Simple LLM Proxy — entry point with signal handling."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import uvicorn

from config import ProxyConfig, get, set
from config.loader import load_config, resolve_config_path
from proxy.app import create_app
from proxy.health import collect_health, log_health_report, stop_health_probe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm-proxy")

_config_path: str | None = None
_server: uvicorn.Server | None = None


def _load_config(path: str | None = None) -> ProxyConfig:
    """Load and validate config, set in registry. Probe backends after loading."""
    config = load_config(path)
    set(config)

    # Probe all backends and log status
    report = asyncio.run(collect_health(config.models, timeout=2.0))
    log_health_report(report)

    if report.any_unhealthy:
        logger.warning(
            "%d/%d backends unhealthy — proxy running in degraded mode",
            report.unhealthy_count, report.total,
        )

    return config


def _reload(signum: int, _frame):
    """Handle SIGHUP — reload config and rebuild the app.

    The existing server keeps running; a fresh app is created with the new
    config (including updated auth middleware). In-flight requests complete
    normally. This is not a full restart — uvicorn stays up, but the ASGI
    app object is replaced so middleware picks up the new settings.
    """
    global _server
    logger.info("Received SIGHUP, reloading config...")
    try:
        config = _load_config(_config_path)
        logger.info("Config reloaded successfully")

        # Create a fresh app with the new config — this picks up updated
        # auth middleware, CORS settings, body-size limits, etc.
        from proxy.app import create_app
        new_app = create_app(config)

        # Replace the running app in-place so uvicorn starts using it
        # for subsequent requests. In-flight requests on the old app finish.
        if _server is not None:
            _server.app = new_app
    except Exception:
        logger.exception("Config reload failed — keeping old config")


def run(config_path: str | None = None) -> None:
    """Bootstrap the proxy and start serving."""
    global _config_path, _server

    _config_path = config_path

    # Load config on first startup
    config = _load_config(config_path)
    logger.info("Loaded config: %d model(s), auth=%s, plugins_dir=%s",
                len(config.models),
                "enabled" if config.auth.api_key else "disabled",
                config.plugins_dir)

    # Register SIGHUP for config reload.
    # SIGINT/SIGTERM are handled by uvicorn's internal capture_signals() which
    # sets should_exit=True and performs graceful shutdown automatically.
    signal.signal(signal.SIGHUP, _reload)

    # Create app
    app = create_app(config)

    # Determine server config
    listen_cfg = config.listen
    host = listen_cfg.host
    port = listen_cfg.port

    logger.info("Starting server on %s:%d", host, port)
    resolved = resolve_config_path(config_path)
    logger.info("Config file: %s", resolved)

    # Start server
    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        loop="uvloop",
    )
    _server = uvicorn.Server(server_config)

    # Run server — asyncio.run manages the event loop lifecycle cleanly.
    # Uvicorn's serve() handles SIGINT/SIGTERM internally (capture_signals).
    try:
        asyncio.run(_server.serve())
    finally:
        # Cancel the periodic health probe task on shutdown
        stop_health_probe()

    logger.info("Shutdown complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple LLM Proxy")
    parser.add_argument("--config", "-c", default=None, help="Path to config file")
    args = parser.parse_args()
    run(args.config)
