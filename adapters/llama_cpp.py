"""Llama.cpp-specific adapter with proper provider headers.

Unlike the generic OpenAI-compatible adapter, this one:
- Sets `X-Provider: llama.cpp` header on all requests
- Passes through llama.cpp-specific parameters without stripping them
- Normalises llama-server non-streaming responses to OpenAI shape where needed

Streaming is transparent pass-through — no interpretation of response content.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from .base import _HTTP_CLIENT, BaseAdapter, BackendError

logger = logging.getLogger(__name__)


class LlamaCppAdapter(BaseAdapter):
    """Adapter for llama.cpp's llama-server HTTP API."""

    PROVIDER_HEADER = "X-Provider"
    PROVIDER_VALUE = "llama.cpp"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    @staticmethod
    def _error_sse(status_code: int, message: str) -> bytes:
        """Format an OpenAI-compatible error as SSE bytes."""
        error_data = {
            "error": {
                "message": message,
                "type": "backend_error",
                "param": None,
                "code": status_code,
            },
        }
        return f"data: {json.dumps(error_data)}\n\ndata: [DONE]\n\n".encode()

    async def forward_stream(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = 120000,
        verify_ssl: bool = True,
    ) -> AsyncIterator[bytes]:
        """Stream response bytes transparently with llama.cpp headers. Never raises."""
        key = api_key or self._api_key
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            self.PROVIDER_HEADER: self.PROVIDER_VALUE,
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"

        client = _HTTP_CLIENT if verify_ssl else httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            verify=False,
        )

        try:
            use_close = not verify_ssl
            async with client.stream(
                "POST", f"{url}{endpoint}", json=payload, headers=headers,
                timeout=timeout_ms / 1000,
            ) as resp:
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    msg = error_body.decode("utf-8", errors="replace")[:500]
                    yield self._error_sse(resp.status_code, msg)
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.TimeoutException:
            yield self._error_sse(408, f"Backend timed out after {timeout_ms}ms")
        except httpx.ConnectError as e:
            yield self._error_sse(502, f"Backend unreachable: {e}")
        except httpx.HTTPError as e:
            yield self._error_sse(502, str(e))
        finally:
            if use_close:
                await client.aclose()

    async def forward_json(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = 120000,
        verify_ssl: bool = True,
    ) -> dict[str, Any]:
        """Return the full JSON response, normalising llama-server output."""
        key = api_key or self._api_key
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            self.PROVIDER_HEADER: self.PROVIDER_VALUE,
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"

        client = _HTTP_CLIENT if verify_ssl else httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            verify=False,
        )

        try:
            resp = await client.post(
                f"{url}{endpoint}", json=payload, headers=headers,
                timeout=timeout_ms / 1000,
            )
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                    msg = body.get("error", {}).get("message", resp.text[:200])
                except Exception:
                    msg = resp.text[:500]
                raise BackendError(resp.status_code, msg)

            data = resp.json()
            data = self._normalise_response(data)
            return data
        finally:
            if not verify_ssl:
                await client.aclose()

    def _normalise_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalise llama-server non-streaming response to OpenAI-compatible shape."""
        if "usage" not in data and "tokens_predicted" in data:
            data["usage"] = {
                "prompt_tokens": data.get("tokens_evaluated", 0),
                "completion_tokens": data.get("tokens_predicted", 0),
                "total_tokens": data.get("tokens_evaluated", 0) + data.get("tokens_predicted", 0),
            }

        if "choices" in data:
            for choice in data["choices"]:
                if "text" in choice and "message" not in choice:
                    choice["message"] = {"role": "assistant", "content": choice.pop("text")}

        return data
