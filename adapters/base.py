"""Adapter interface and generic OpenAI-compatible adapter."""

from __future__ import annotations

import abc
import asyncio
import os
from typing import Any, AsyncIterator

import httpx

TIMEOUT = int(os.environ.get("LLM_PROXY_TIMEOUT_MS", "120000"))


class BaseAdapter(abc.ABC):
    """Each adapter forwards a request to a backend and returns OpenAI-compatible output."""

    @abc.abstractmethod
    async def forward_stream(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = TIMEOUT,
    ) -> AsyncIterator[bytes]:
        """Yield raw response chunks from the backend (transparent pass-through)."""
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
    """Raised when a backend returns an error response."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Backend error {status_code}: {message}")


class OpenAICompatibleAdapter(BaseAdapter):
    """
    Generic adapter for any backend that speaks the OpenAI API format
    (llama.cpp server, vLLM, Azure OpenAI, etc.).
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def forward_stream(
        self,
        url: str,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None = None,
        timeout_ms: int = TIMEOUT,
    ) -> AsyncIterator[bytes]:
        """Stream response chunks transparently."""
        key = api_key or self._api_key
        headers: dict[str, str] = {
            "Content-Type": "application/json",
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
        timeout_ms: int = TIMEOUT,
    ) -> dict[str, Any]:
        """Return the full JSON response."""
        key = api_key or self._api_key
        headers: dict[str, str] = {
            "Content-Type": "application/json",
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
            return resp.json()
