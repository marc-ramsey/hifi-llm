"""Adapter registry and factory — dispatches to the correct backend adapter."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseAdapter, BackendError, OpenAICompatibleAdapter

logger = logging.getLogger(__name__)

# Registry: maps backend name string → adapter class
_registry: dict[str, type[BaseAdapter]] = {}


def register_adapter(name: str) -> Any:
    """Decorator to register an adapter class under a given backend name."""

    def wrapper(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        if name in _registry:
            logger.warning("Adapter '%s' already registered, overwriting", name)
        _registry[name] = cls
        return cls

    return wrapper


def get_adapter(backend: str) -> BaseAdapter:
    """Return a new adapter instance for the given backend name.

    Falls back to OpenAICompatibleAdapter if the backend is not found.
    """
    cls = _registry.get(backend)
    if cls is None:
        logger.info("Unknown backend '%s', falling back to OpenAICompatibleAdapter", backend)
        return OpenAICompatibleAdapter()
    return cls()


# Auto-register the built-in adapter
register_adapter("openai_compatible")(OpenAICompatibleAdapter)

__all__ = ["BaseAdapter", "BackendError", "OpenAICompatibleAdapter", "get_adapter", "register_adapter"]
