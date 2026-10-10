"""Simple LLM Proxy — entry point with signal handling."""

import asyncio
import logging
import signal
import sys

import uvicorn

from config import ProxyConfig, get, set
from config.loader import load_config, resolve_config_path
from proxy.health import restart_health_probe, stop_health_probe, _current_report_value
from adapters.base import close_http_client
from proxy.process_manager import ProcessManager
from proxy.reloadable import ConfigReloadableApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm-proxy")

_config_path: str | None = None
_server: uvicorn.Server | None = None
_reloadable_app: ConfigReloadableApp | None = None


def _load_config(path: str | None = None) -> ProxyConfig:
    """Load and validate config, set in registry."""
    config = load_config(path)
    set(config)
    return config


def _reload(signum: int, _frame) -> None:
    """Handle SIGHUP — reload config and swap the ASGI app.

    The ConfigReloadableApp wrapper ensures uvicorn always sees the same
    object while new requests use the freshly-built app from updated
    config (including auth middleware, CORS settings, body-size limits,
    static files, etc.).  In-flight requests on the old app complete normally.
    """
    global _reloadable_app
    logger.info("Received SIGHUP, reloading config...")
    try:
        config = _load_config(_config_path)
        logger.info("Config reloaded successfully")

        # Restart the health probe with the new model list so it probes
        # the updated set of backends instead of the stale one.
        restart_health_probe(config.models, interval=config.health_check_interval)

        # Swap to a fresh ASGI app — this picks up updated auth middleware,
        # CORS settings, body-size limits, static files, etc.  In-flight requests
        # on the old app finish naturally; new requests use the new one.
        if _reloadable_app is not None:
            _reloadable_app.reload()

    except Exception:
        logger.exception("Config reload failed — keeping old config")


def run(config_path: str | None = None) -> None:
    """Bootstrap the proxy and start serving."""
    global _config_path, _server, _reloadable_app

    _config_path = config_path

    # Load config on first startup
    config = _load_config(config_path)
    logger.info("Loaded config: %d model(s), auth=%s, static_files=%d",
                len(config.models),
                "enabled" if config.auth.api_key else "disabled",
                len(config.static_files))

    # Register SIGHUP for config reload.
    # SIGINT/SIGTERM are handled by uvicorn's internal capture_signals() which
    # sets should_exit=True and performs graceful shutdown automatically.
    signal.signal(signal.SIGHUP, _reload)

    # Create the reloadable app wrapper — uvicorn always sees this same object.
    _reloadable_app = ConfigReloadableApp(initial_config=config)

    # Determine server config
    listen_cfg = config.listen
    host = listen_cfg.host
    port = listen_cfg.port

    logger.info("Starting server on %s:%d", host, port)
    resolved = resolve_config_path(config_path)
    logger.info("Config file: %s", resolved)

    # Start server — the reloadable wrapper is passed as the ASGI app.
    server_config = uvicorn.Config(
        _reloadable_app,
        host=host,
        port=port,
        log_level="info",
        loop="uvloop",
    )
    _server = uvicorn.Server(server_config)

    # Run server — asyncio.run manages the event loop lifecycle cleanly.
    # Uvicorn's serve() handles SIGINT/SIGTERM internally (capture_signals).
    # We wrap the serve call so we can run async cleanup within the same loop.
    async def _serve_and_cleanup():
        try:
            await _server.serve()
        finally:
            # Health probe stops synchronously; process manager and HTTP
            # clients shut down in parallel, awaited so exceptions aren't
            # silently lost and cleanup completes before exit.
            stop_health_probe()  # synchronous cancellation, returns immediately
            await asyncio.gather(
                ProcessManager.shutdown(),
                close_http_client(),
                return_exceptions=True,
            )

    try:
        asyncio.run(_serve_and_cleanup())
    except KeyboardInterrupt:
        # asyncio.run() re-raises KeyboardInterrupt from CancelledError
        # during cleanup; the shutdown already completed in the finally
        # block above, so this is expected on ^C.
        pass

    logger.info("Shutdown complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple LLM Proxy")
    parser.add_argument("--config", "-c", default=None, help="Path to config file")
    args = parser.parse_args()
    run(args.config)
