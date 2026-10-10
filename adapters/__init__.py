"""Adapter registry and factory — dispatches to the correct backend adapter."""

from __future__ import annotations

import logging
from typing import Any

from config.schema import ModelConfig

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


def get_adapter(
    backend: str,
    api_key: str | None = None,
    model_config: ModelConfig | None = None,
) -> BaseAdapter:
    """Return a new adapter instance for the given backend name.

    Falls back to OpenAICompatibleAdapter if the backend is not found.

    Args:
        backend: Backend name (e.g. "llama_cpp", "openai_compatible",
                 "managed_llama").
        api_key: Optional per-model backend API key (overrides adapter-level).
        model_config: Full model config, passed to adapters that need it
                      (e.g. ``managed_llama`` for process lifecycle).
    """
    cls = _registry.get(backend)
    if cls is None:
        logger.info("Unknown backend '%s', falling back to OpenAICompatibleAdapter", backend)
        return OpenAICompatibleAdapter(api_key=api_key)
    # Adapters that need model config accept it; others only take api_key.
    if model_config is not None:
        import inspect
        sig = inspect.signature(cls.__init__)
        if "model_config" in sig.parameters:
            return cls(model_config=model_config)
    return cls(api_key=api_key)


# ── Auto-register built-in adapters ─────────────────────────────────────

from .llama_cpp import LlamaCppAdapter  # noqa: E402 — imported after registry is defined
from .managed_llama import ManagedLlamaAdapter  # noqa: E402
register_adapter("llama_cpp")(LlamaCppAdapter)
register_adapter("openai_compatible")(OpenAICompatibleAdapter)
register_adapter("managed_llama")(ManagedLlamaAdapter)

__all__ = [
    "BaseAdapter", "BackendError",
    "LlamaCppAdapter", "ManagedLlamaAdapter", "OpenAICompatibleAdapter",
    "get_adapter", "register_adapter",
]
