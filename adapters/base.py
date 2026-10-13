"""Adapter interface and generic OpenAI-compatible adapter."""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import weakref
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# ── timeouts ─────────────────────────────────────────────────────────────
#
# Non-streaming and streaming need *different* policies, so they are named
# separately and never share a value by accident:
#
#   non-streaming  one overall budget (connect + wait + body + parse).
#   streaming      bounded connect / pool / write, and by default an
#                  UNBOUNDED read gap — a thinking model can sit silent for
#                  minutes before its first token, and a long tool-calling
#                  turn is legitimate work, not a stall.  A streaming request
#                  ends when the backend says [DONE] or the client goes away;
#                  killing it on a wall clock produced truncated tool calls.

#: Overall budget for non-streaming requests.
# NOTE: Evaluated at import time. Hot-reloading the config at runtime will
# not update these values unless the module is reloaded.
TIMEOUT = int(os.environ.get("LLM_PROXY_TIMEOUT_MS", "120000"))

#: Time allowed to establish a TCP connection to a backend.
CONNECT_TIMEOUT_MS = int(os.environ.get("LLM_PROXY_CONNECT_TIMEOUT_MS", "5000"))

#: Time allowed to wait for a free connection from the pool.
POOL_TIMEOUT_MS = int(os.environ.get("LLM_PROXY_POOL_TIMEOUT_MS", "30000"))

#: Max gap between backend bytes on a stream.  0 (default) = no limit.
STREAM_READ_TIMEOUT_MS = int(os.environ.get("LLM_PROXY_STREAM_READ_TIMEOUT_MS", "0"))


def json_timeout(timeout_ms: int) -> httpx.Timeout:
    """Timeout policy for a buffered (non-streaming) request."""
    return httpx.Timeout(timeout_ms / 1000)


def stream_timeout(timeout_ms: int) -> httpx.Timeout:
    """Timeout policy for an SSE request.

    ``read`` is deliberately unbounded by default — see the note above.
    ``timeout_ms`` only bounds the write (upload) phase here.
    """
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_MS / 1000,
        read=(STREAM_READ_TIMEOUT_MS / 1000) if STREAM_READ_TIMEOUT_MS else None,
        write=timeout_ms / 1000,
        pool=POOL_TIMEOUT_MS / 1000,
    )


class BackendError(Exception):
    """Raised when a backend cannot produce a usable response.

    ``status_code`` is the *backend's* status (or 502/503 when the failure is
    the proxy's own), and ``code`` is what the proxy answers with.  ``body``
    carries the backend's original error payload when it had one, so the
    proxy can relay it instead of authoring a message.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.body = body
        self.headers = headers or {}
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


def error_message_from_body(body: bytes, limit: int = 2000) -> str:
    """Pull a human-readable message out of a backend error body.

    Prefers the OpenAI ``error.message`` field; falls back to the raw text.
    """
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return text[:limit]
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"][:limit]
        if isinstance(err, str):
            return err[:limit]
        if isinstance(parsed.get("message"), str):
            return parsed["message"][:limit]
    return text[:limit]


# ── shared HTTP clients ──────────────────────────────────────────────────
#
# Built lazily, one pair per event loop.  An httpx.AsyncClient binds its
# connection pool to whichever loop first uses it; creating it at import time
# (as this module used to) means a second loop — a test, or the app rebuilt
# on SIGHUP — can end up reusing sockets owned by a dead loop.

_MAX_CONNECTIONS = int(os.environ.get("LLM_PROXY_MAX_CONNECTIONS", "100"))
_MAX_KEEPALIVE = int(os.environ.get("LLM_PROXY_MAX_KEEPALIVE_CONNECTIONS", "20"))

_CLIENTS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[bool, httpx.AsyncClient]]" = (
    weakref.WeakKeyDictionary()
)


def _get_http_client(verify_ssl: bool) -> httpx.AsyncClient:
    """Return the shared client for *verify_ssl*, creating it on this loop.

    Must be called from an async context. Calling this synchronously will
    raise ``RuntimeError: no running event loop``.
    """
    loop = asyncio.get_running_loop()
    per_loop = _CLIENTS.get(loop)
    if per_loop is None:
        per_loop = {}
        _CLIENTS[loop] = per_loop

    client = per_loop.get(verify_ssl)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=None,  # every call passes an explicit policy
            verify=verify_ssl,
            limits=httpx.Limits(
                max_connections=_MAX_CONNECTIONS,
                max_keepalive_connections=_MAX_KEEPALIVE,
            ),
        )
        per_loop[verify_ssl] = client
    return client


async def close_http_client() -> None:
    """Close all shared HTTP clients and drain keep-alive connections.

    Awaits graceful shutdown — any in-flight requests complete before
    the clients are closed. Call from an async context (e.g. shutdown hook).
    """
    for per_loop in list(_CLIENTS.values()):
        for client in list(per_loop.values()):
            await client.aclose()
        per_loop.clear()


class BaseAdapter(abc.ABC):
    """Each adapter forwards a request to a backend and returns OpenAI-compatible output.

    Contract:
        forward_stream  — yields raw SSE bytes, unchanged.  If the backend
                          fails *before* any byte of body is produced, raise
                          :class:`BackendError` so the router can answer with
                          a real status code.  Once bytes have been yielded the
                          generator must not raise (the status line is already
                          on the wire); it logs and returns instead, ending
                          the stream.  It never invents ``data:`` frames.
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
        """Yield raw SSE bytes from the backend.

        Raises BackendError if the failure happens before the first byte.
        """
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

    def _build_headers(self, api_key: str | None) -> dict[str, str]:
        """Build HTTP headers for the backend request.

        Adds ``Authorization: Bearer <api_key>`` when a key is configured
        (per-request key, else the model's ``api_key`` from config).

        Override in subclasses to add custom headers (e.g. provider hints) —
        call ``super()._build_headers(api_key)`` to keep the auth header.
        """
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _normalise_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalise the backend response to OpenAI-compatible shape.

        Override in subclasses that talk to non-OpenAI backends.
        The base implementation returns *data* unchanged.
        """
        return data


class OpenAICompatibleAdapter(BaseAdapter):
    """Generic adapter for any backend that speaks the OpenAI API format.

    Transparent pass-through — no interpretation of response content, in
    either direction: SSE bodies are forwarded byte-for-byte and a backend
    error becomes a :class:`BackendError`, never a synthetic ``data:`` frame.

    Subclasses can customise behaviour by overriding:
        _build_headers(api_key)   — add extra request headers
        _normalise_response(data) — transform non-OpenAI responses
        PROVIDER                  — identity string for /v1/models display
    """

    PROVIDER: str = "openai_compatible"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    # ── internals ───────────────────────────────────────────────────────

    def _get_client(self, verify_ssl: bool) -> httpx.AsyncClient:
        """Return the appropriate HTTP client for the given SSL setting."""
        return _get_http_client(verify_ssl)

    def _build_request_headers(self, api_key: str | None) -> dict[str, str]:
        """Resolve the backend key and build the final request headers.

        *api_key* (per-request) overrides the adapter-level key; Content-Type
        is always present even if a subclass override forgets it.
        """
        headers = self._build_headers(api_key or self._api_key)
        headers.setdefault("Content-Type", "application/json")
        return headers

    def _handle_error_response(self, resp: httpx.Response) -> str:
        """Extract an error message from a non-2xx response."""
        return error_message_from_body(resp.content or resp.read())

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
        """Stream response bytes transparently.

        Raises:
            BackendError: the backend errored, or could not be reached,
                *before* any body byte was produced — the caller still owns
                the status line and can answer honestly.
        """
        headers = self._build_request_headers(api_key)
        client = self._get_client(verify_ssl)
        started = False

        try:
            async with client.stream(
                "POST", f"{url}{endpoint}", json=payload, headers=headers,
                timeout=stream_timeout(timeout_ms),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise BackendError(
                        resp.status_code,
                        error_message_from_body(body),
                        body=body,
                        headers=dict(resp.headers),
                    )
                async for chunk in resp.aiter_bytes():
                    started = True
                    yield chunk
        except (BackendError, asyncio.CancelledError):
            raise
        except httpx.TimeoutException as exc:
            if started:
                logger.warning("Stream to %s%s stalled: %s", url, endpoint, exc)
                return
            raise BackendError(502, f"Backend timed out connecting to {url}{endpoint}") from exc
        except httpx.ConnectError as exc:
            if started:
                logger.warning("Stream from %s%s lost: %s", url, endpoint, exc)
                return
            raise BackendError(502, f"Backend unreachable: {exc}") from exc
        except httpx.HTTPError as exc:
            if started:
                logger.warning("Stream from %s%s failed: %s", url, endpoint, exc)
                return
            raise BackendError(502, str(exc)) from exc

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
        headers = self._build_request_headers(api_key)
        client = self._get_client(verify_ssl)

        resp = await client.post(
            f"{url}{endpoint}", json=payload, headers=headers,
            timeout=json_timeout(timeout_ms),
        )
        if resp.status_code >= 400:
            raise BackendError(
                resp.status_code,
                self._handle_error_response(resp),
                body=resp.content,
                headers=dict(resp.headers),
            )

        data = resp.json()
        return self._normalise_response(data)
