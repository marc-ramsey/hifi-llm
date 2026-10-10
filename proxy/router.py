"""Route handlers — /v1/models and /v1/chat/completions.

The router is a dumb pipe: resolve the model, merge defaults, forward to the
adapter.  Streaming responses are passed through byte-for-byte — no JSON
parsing, no delta inspection.  The adapter guarantees that its generator
never raises; errors are yielded as SSE error chunks instead.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import get
from adapters import get_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


def _resolve_model_config(model_name: str):
    """Look up a model entry by name. Raises HTTPException if not found."""
    for model in get().models:
        if model.name == model_name:
            return model
    valid = ", ".join(m.name for m in get().models)
    raise HTTPException(
        status_code=400,
        detail={
            "message": f"Invalid model: {model_name}",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        },
    )


def _merge_defaults(payload: dict, defaults: dict) -> dict:
    """Merge default_params into the request payload.
    Client values override defaults.
    """
    merged = {**defaults, **payload}
    return merged


async def _resolve_and_merge(request: Request):
    """Resolve model config and merge defaults. Returns (model_config, payload).

    Caches the parsed body on request.state so it's only parsed once.
    """
    if not hasattr(request.state, "_request_body"):
        body = await request.json()
        request.state._request_body = body
        request.state._model = body.get("model", "")
    else:
        body = request.state._request_body
    model_name = request.state._model
    model_config = _resolve_model_config(model_name)
    payload = _merge_defaults(body, model_config.default_params)
    return model_config, payload


async def _forward_stream(request: Request, endpoint: str) -> StreamingResponse:
    """SSE pass-through proxy.

    The adapter's forward_stream yields raw bytes (already SSE-formatted).
    We buffer-split on newlines and forward complete SSE lines unchanged.
    Error handling is the adapter's responsibility — it never raises.
    """
    model_config, payload = await _resolve_and_merge(request)

    adapter = get_adapter(model_config.backend, api_key=model_config.api_key)
    stream = adapter.forward_stream(
        url=model_config.url,
        endpoint=endpoint,
        payload=payload,
    )

    async def chunk_iterator() -> AsyncIterator[str]:
        """Forward SSE lines from the backend unchanged."""
        buffer = b""
        async for chunk_bytes in stream:
            buffer += chunk_bytes
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                stripped = line.decode("utf-8").strip()
                if stripped.startswith("data: "):
                    data = stripped[6:]
                    yield f"data: {data}\n\n"
                    if data == "[DONE]":
                        return
                elif stripped.startswith("data:"):
                    data = stripped[5:].lstrip()
                    yield f"data:{data}\n\n"
                    if data == "[DONE]":
                        return

    return StreamingResponse(
        chunk_iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _forward_json(request: Request, endpoint: str) -> JSONResponse:
    """Generic JSON proxy for any endpoint."""
    model_config, payload = await _resolve_and_merge(request)

    adapter = get_adapter(model_config.backend, api_key=model_config.api_key)
    response = await adapter.forward_json(
        url=model_config.url,
        endpoint=endpoint,
        payload=payload,
    )
    return JSONResponse(content=response)


# ── GET /v1/models ──────────────────────────────────────────────────────────

@router.get("/models")
async def list_models():
    """Return the list of configured models in OpenAI format with status."""
    from proxy.health import _current_report

    report = _current_report()
    health_map = {bh.name: bh for bh in report.backends}

    models = get().models
    data = []
    for m in models:
        entry: dict = {
            "id": m.name,
            "object": "model",
            "created": 0,
            "owned_by": m.backend,
        }
        if m.provider:
            entry["provider"] = m.provider
        bh = health_map.get(m.name)
        if bh:
            status_value = "loaded" if bh.status.value == "healthy" else "unloaded"
            entry["status"] = {"value": status_value}
        data.append(entry)

    return {"object": "list", "data": data}


# ── POST /v1/chat/completions ───────────────────────────────────────────────

@router.post("/chat/completions")
async def chat_completions(request: Request):
    """Proxy /v1/chat/completions. Stream or JSON, delegated to adapter."""
    _, payload = await _resolve_and_merge(request)
    if payload.get("stream", False):
        return await _forward_stream(request, "/v1/chat/completions")
    else:
        return await _forward_json(request, "/v1/chat/completions")


# ── POST /v1/embeddings ─────────────────────────────────────────────────────

@router.post("/embeddings")
async def embeddings(request: Request):
    """Proxy /v1/embeddings to the backend."""
    return await _forward_json(request, "/v1/embeddings")


# ── POST /v1/completions (legacy) ───────────────────────────────────────────

@router.post("/completions")
async def completions(request: Request):
    """Proxy /v1/completions. Stream or JSON, delegated to adapter."""
    _, payload = await _resolve_and_merge(request)
    if payload.get("stream", False):
        return await _forward_stream(request, "/v1/completions")
    else:
        return await _forward_json(request, "/v1/completions")
