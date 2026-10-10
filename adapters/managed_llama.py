"""Managed llama.cpp adapter — launches and manages a local llama-server process.

When ``backend`` is ``managed_llama``, HiFi owns the lifecycle of the backend
server process.  On first request the server is launched lazily, health probes
detect its status, and the process is auto-restarted on death (with backoff).

All actual request forwarding is delegated to :class:`LlamaCppAdapter`.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from config.schema import ModelConfig

from .base import BaseAdapter
from .llama_cpp import LlamaCppAdapter
from proxy.process_manager import ProcessManager

logger = logging.getLogger(__name__)


def _to_cli_flag(key: str) -> str:
    """Convert a snake_case config key to a --kebab-case CLI flag."""
    return "--" + key.replace("_", "-")


class ManagedLlamaAdapter(BaseAdapter):
    """Adapter for a locally-managed llama.cpp server process.

    The server is launched lazily on first request and auto-restarted on death.
    ``server_config`` keys are mapped to llama-server CLI flags at launch time;
    ``default_params`` flows through unchanged to the HTTP API.
    """

    accepts_model_config = True

    def __init__(self, model_config: ModelConfig) -> None:
        self._model_config = model_config
        self._delegate = LlamaCppAdapter()

    # ── command building ────────────────────────────────────────────────

    def _build_command(self) -> list[str]:
        """Build the llama-server command line from model config."""
        cmd = [self._model_config.llama_binary]

        # server_config → CLI flags (snake_case → --kebab-case)
        for key, value in self._model_config.server_config.items():
            cmd.extend([_to_cli_flag(key), str(value)])

        # Any extra raw args the user wants to pass through
        cmd.extend(self._model_config.default_params.get("server_args", []))

        return cmd

    # ── forwarding ──────────────────────────────────────────────────────

    async def forward_stream(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = 120000,
        verify_ssl: bool = True,
    ) -> AsyncIterator[bytes]:
        """Ensure the server is running, then delegate to LlamaCppAdapter."""
        cmd = self._build_command()
        ProcessManager.ensure_running(self._model_config.name, cmd)
        async for chunk in self._delegate.forward_stream(
            url=url, endpoint=endpoint, payload=payload,
            api_key=api_key, timeout_ms=timeout_ms, verify_ssl=verify_ssl,
        ):
            yield chunk

    async def forward_json(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = 120000,
        verify_ssl: bool = True,
    ) -> dict[str, Any]:
        """Ensure the server is running, then delegate to LlamaCppAdapter."""
        cmd = self._build_command()
        ProcessManager.ensure_running(self._model_config.name, cmd)
        return await self._delegate.forward_json(
            url=url, endpoint=endpoint, payload=payload,
            api_key=api_key, timeout_ms=timeout_ms, verify_ssl=verify_ssl,
        )
