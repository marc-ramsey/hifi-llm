"""Config registry — holds the current ProxyConfig, swappable on SIGHUP."""

from __future__ import annotations

from .schema import ProxyConfig

_current: ProxyConfig | None = None


def get() -> ProxyConfig:
    if _current is None:
        raise RuntimeError("Config not loaded — call load_first() before serving")
    return _current


def set(config: ProxyConfig) -> None:
    global _current
    _current = config


def load_first(path: str | None = None) -> ProxyConfig:
    """Load config from file on first startup."""
    from .loader import load_config
    config = load_config(path) if path else load_config()
    set(config)
    return config
