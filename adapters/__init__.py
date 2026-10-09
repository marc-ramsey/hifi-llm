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


def get_adapter(backend: str, api_key: str | None = None) -> BaseAdapter:
    """Return a new adapter instance for the given backend name.

    Falls back to OpenAICompatibleAdapter if the backend is not found.

    Args:
        backend: Backend name (e.g. "llama_cpp", "openai_compatible").
        api_key: Optional per-model backend API key (overrides adapter-level).
    """
    cls = _registry.get(backend)
    if cls is None:
        logger.info("Unknown backend '%s', falling back to OpenAICompatibleAdapter", backend)
        return OpenAICompatibleAdapter(api_key=api_key)
    return cls(api_key=api_key)


# ── Auto-register built-in adapters ─────────────────────────────────────

from .llama_cpp import LlamaCppAdapter  # noqa: E402 — imported after registry is defined
register_adapter("llama_cpp")(LlamaCppAdapter)
register_adapter("openai_compatible")(OpenAICompatibleAdapter)

__all__ = ["BaseAdapter", "BackendError", "LlamaCppAdapter", "OpenAICompatibleAdapter", "get_adapter", "register_adapter"]
