"""Config registry — holds the current ProxyConfig, swappable on SIGHUP."""

from .schema import ProxyConfig

_current: ProxyConfig | None = None


def get() -> ProxyConfig:
    if _current is None:
        raise RuntimeError("Config not loaded — call load() before serving")
    return _current


def set(config: ProxyConfig) -> None:
    global _current
    _current = config
