"""Llama.cpp-specific adapter with proper provider headers and llama.cpp parameter support."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from .base import BaseAdapter, BackendError

logger = logging.getLogger(__name__)


class LlamaCppAdapter(BaseAdapter):
    """
    Adapter for llama.cpp's llama-server HTTP API.

    Unlike the generic OpenAI-compatible adapter, this one:
    - Sets `X-Provider: llama.cpp` header on all requests
    - Passes through llama.cpp-specific parameters (top_k, min_p, typical_p, etc.)
      without stripping them
    - Normalises llama-server response fields to OpenAI-compatible shapes
      where needed (e.g. `usage` field may be missing in non-streaming responses)
    """

    PROVIDER_HEADER = "X-Provider"
    PROVIDER_VALUE = "llama.cpp"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def forward_stream(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = 120000,
    ) -> AsyncIterator[bytes]:
        """Stream response chunks transparently with llama.cpp headers."""
        key = api_key or self._api_key
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            self.PROVIDER_HEADER: self.PROVIDER_VALUE,
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"

        async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
            async with client.stream("POST", f"{url}{endpoint}", json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

    async def forward_json(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = 120000,
    ) -> dict[str, Any]:
        """Return the full JSON response, normalising llama-server output to OpenAI shape."""
        key = api_key or self._api_key
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            self.PROVIDER_HEADER: self.PROVIDER_VALUE,
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"

        async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
            resp = await client.post(f"{url}{endpoint}", json=payload, headers=headers)
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                    msg = body.get("error", {}).get("message", resp.text[:200])
                except Exception:
                    msg = resp.text[:500]
                raise BackendError(resp.status_code, msg)

            data = resp.json()

            # Normalise llama-server responses that may lack standard OpenAI fields
            data = self._normalise_response(data)
            return data

    def _normalise_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalise llama-server response to OpenAI-compatible shape.

        llama-server may omit `usage` in some responses or use different field names.
        """
        # Ensure usage is present — llama-server includes it but let's be safe
        if "usage" not in data and "tokens_predicted" in data:
            data["usage"] = {
                "prompt_tokens": data.get("tokens_evaluated", 0),
                "completion_tokens": data.get("tokens_predicted", 0),
                "total_tokens": data.get("tokens_evaluated", 0) + data.get("tokens_predicted", 0),
            }

        # Ensure choices has proper structure
        if "choices" in data:
            for choice in data["choices"]:
                # llama-server may use `text` instead of `message` for completions-style output
                if "text" in choice and "message" not in choice:
                    choice["message"] = {"role": "assistant", "content": choice.pop("text")}

        return data



