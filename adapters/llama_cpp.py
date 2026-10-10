"""Llama.cpp-specific adapter with proper provider headers.

Unlike the generic OpenAI-compatible adapter, this one:
- Sets ``X-Provider: llama.cpp`` header on all requests
- Passes through llama.cpp-specific parameters without stripping them
- Normalises llama-server non-streaming responses to OpenAI shape where needed

Streaming is transparent pass-through — no interpretation of response content.
"""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter, OpenAICompatibleAdapter


class LlamaCppAdapter(OpenAICompatibleAdapter):
    """Adapter for llama.cpp's llama-server HTTP API."""

    PROVIDER = "llama.cpp"
    PROVIDER_HEADER = "X-Provider"
    PROVIDER_VALUE = "llama.cpp"

    def _build_headers(self, api_key: str | None) -> dict[str, str]:
        headers = super()._build_headers(api_key)
        headers[self.PROVIDER_HEADER] = self.PROVIDER_VALUE
        return headers

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
