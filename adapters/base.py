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
        verify_ssl: bool = True,
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
        verify_ssl: bool = True,
    ) -> dict[str, Any]:
        """Return the full JSON response body."""
        ...

    @staticmethod
    def _error_sse(status_code: int, message: str) -> bytes:
        """Format an OpenAI-compatible error as SSE bytes."""
        error_data = make_error_response(message, "backend_error", status_code)
        return f"data: {json.dumps(error_data)}\n\ndata: [DONE]\n\n".encode()

    def _build_headers(self, api_key: str | None) -> dict[str, str]:
        """Build HTTP headers for the backend request.

        Override in subclasses to add custom headers (e.g. provider hints).
        """
        return {"Content-Type": "application/json"}

    def _normalise_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalise the backend response to OpenAI-compatible shape.

        Override in subclasses that talk to non-OpenAI backends.
        The base implementation returns *data* unchanged.
        """
        return data


class BackendError(Exception):
    """Raised when a backend returns an error response (non-streaming)."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Backend error {status_code}: {message}")


def make_error_response(message: str, error_type: str, code: int | None) -> dict:
    """Build a standard OpenAI-compatible error dict.

    Used by adapters, middleware, and exception handlers to return
    consistent error shapes across the proxy.
    """
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        },
    }


_HTTP_CLIENT = httpx.AsyncClient(
    timeout=120.0,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)

_SSL_UNVERIFIED_CLIENT = httpx.AsyncClient(
    timeout=120.0,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    verify=False,
)


async def close_http_client() -> None:
    """Close all shared HTTP clients and drain keep-alive connections.

    Awaits graceful shutdown — any in-flight requests complete before
    the clients are closed. Call from an async context (e.g. shutdown hook).
    """
    await _HTTP_CLIENT.aclose()
    await _SSL_UNVERIFIED_CLIENT.aclose()


class OpenAICompatibleAdapter(BaseAdapter):
    """Generic adapter for any backend that speaks the OpenAI API format.

    Transparent pass-through — no interpretation of response content.
    Streaming errors are yielded as SSE-formatted error chunks so the
    generator never raises (StreamingResponse's async generator interface
    is fragile with exceptions).

    Subclasses can customise behaviour by overriding:
        _build_headers(api_key)   — add extra request headers
        _normalise_response(data) — transform non-OpenAI responses
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    # ── internals ───────────────────────────────────────────────────────

    def _get_client(self, verify_ssl: bool) -> httpx.AsyncClient:
        """Return the appropriate HTTP client for the given SSL setting."""
        return _HTTP_CLIENT if verify_ssl else _SSL_UNVERIFIED_CLIENT

    def _build_request_headers(self, api_key: str | None) -> dict[str, str]:
        """Build request headers including Content-Type and any custom headers."""
        headers = self._build_headers(api_key)
        headers["Content-Type"] = "application/json"
        return headers

    def _handle_error_response(self, resp: httpx.Response) -> str:
        """Extract an error message from a non-2xx response."""
        try:
            body = resp.json()
            msg = body.get("error", {}).get("message", resp.text[:200])
        except Exception:
            msg = resp.text[:500]
        return msg

    # ── streaming ───────────────────────────────────────────────────────

    async def forward_stream(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = TIMEOUT,
        verify_ssl: bool = True,
    ) -> AsyncIterator[bytes]:
        """Stream response bytes transparently. Never raises."""
        key = api_key or self._api_key
        headers = self._build_request_headers(key)
        client = self._get_client(verify_ssl)

        try:
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

    # ── non-streaming ───────────────────────────────────────────────────

    async def forward_json(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = TIMEOUT,
        verify_ssl: bool = True,
    ) -> dict[str, Any]:
        """Return the full JSON response."""
        key = api_key or self._api_key
        headers = self._build_request_headers(key)
        client = self._get_client(verify_ssl)

        resp = await client.post(
            f"{url}{endpoint}", json=payload, headers=headers,
            timeout=timeout_ms / 1000,
        )
        if resp.status_code >= 400:
            msg = self._handle_error_response(resp)
            raise BackendError(resp.status_code, msg)

        data = resp.json()
        return self._normalise_response(data)
