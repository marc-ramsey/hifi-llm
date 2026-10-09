"""Adapter interface and generic OpenAI-compatible adapter."""

from __future__ import annotations

import abc
import json
import os
from typing import Any, AsyncIterator

import httpx

TIMEOUT = int(os.environ.get("LLM_PROXY_TIMEOUT_MS", "120000"))


class BaseAdapter(abc.ABC):
    """Each adapter forwards a request to a backend and returns OpenAI-compatible output.

    Contract:
        forward_stream  — yields raw SSE bytes. NEVER raises; errors are
                          yielded as SSE error chunks instead.
        forward_json    — returns the full JSON response body or raises
                          BackendError (for non-streaming, exceptions are fine).
    """

    @abc.abstractmethod
    async def forward_stream(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = TIMEOUT,
    ) -> AsyncIterator[bytes]:
        """Yield raw SSE bytes from the backend. Never raises."""
        ...

    @abc.abstractmethod
    async def forward_json(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = TIMEOUT,
    ) -> dict[str, Any]:
        """Return the full JSON response body."""
        ...


class BackendError(Exception):
    """Raised when a backend returns an error response (non-streaming)."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Backend error {status_code}: {message}")


class OpenAICompatibleAdapter(BaseAdapter):
    """Generic adapter for any backend that speaks the OpenAI API format.

    Transparent pass-through — no interpretation of response content.
    Streaming errors are yielded as SSE-formatted error chunks so the
    generator never raises (StreamingResponse's async generator interface
    is fragile with exceptions).
    """

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
        timeout_ms: int = TIMEOUT,
    ) -> AsyncIterator[bytes]:
        """Stream response bytes transparently. Never raises."""
        key = api_key or self._api_key
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        try:
            async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
                async with client.stream(
                    "POST", f"{url}{endpoint}", json=payload, headers=headers
                ) as resp:
                    # Check status BEFORE streaming — catches immediate errors
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

    async def forward_json(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = TIMEOUT,
    ) -> dict[str, Any]:
        """Return the full JSON response."""
        key = api_key or self._api_key
        headers: dict[str, str] = {"Content-Type": "application/json"}
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
            return resp.json()
