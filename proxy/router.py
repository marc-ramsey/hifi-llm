"""Route handlers — /v1/models and /v1/chat/completions."""

from __future__ import annotations

import json
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
    # Build list of valid names for the error message
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


def _normalize_delta(delta: dict) -> dict:
    """No-op — pass deltas through unchanged.

    Open WebUI handles reasoning_content via its structured output (Path 2),
    showing thinking in a collapsible section and content in the main area.
    We used to fold reasoning → content but that mixed thinking into the
    response text. Leave deltas as the backend sends them.
    """
    return delta


async def _resolve_and_merge(request: Request):
    """Resolve model config and merge defaults. Returns (model_config, payload)."""
    body = await request.json()
    model_name = body.get("model", "")
    model_config = _resolve_model_config(model_name)
    payload = _merge_defaults(body, model_config.default_params)
    return model_config, payload


async def _forward_stream(request: Request, endpoint: str) -> StreamingResponse:
    """Generic SSE streaming proxy for any endpoint."""
    model_config, payload = await _resolve_and_merge(request)

    adapter = get_adapter(model_config.backend, api_key=model_config.api_key)
    stream = adapter.forward_stream(
        url=model_config.url,
        endpoint=endpoint,
        payload=payload,
    )

    async def chunk_iterator() -> AsyncIterator[str]:
        """Yield SSE-formatted chunks from the backend response."""
        buffer = b""
        async for chunk_bytes in stream:
            buffer += chunk_bytes
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        parsed = json.loads(data)
                        for choice in parsed.get("choices", []):
                            delta = choice.get("delta")
                            if delta and isinstance(delta, dict):
                                choice["delta"] = _normalize_delta(delta)
                        yield f"data: {json.dumps(parsed, ensure_ascii=False)}\n\n"
                    except json.JSONDecodeError:
                        yield f"data: {data}\n\n"

    return StreamingResponse(
        chunk_iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
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
    """
    Proxy /v1/chat/completions.
    If stream=true, returns SSE StreamingResponse.
    Otherwise returns JSONResponse with the full body.
    """
    body = await request.json()
    if body.get("stream", False):
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
    """
    Proxy /v1/completions (legacy endpoint).
    If stream=true, returns SSE StreamingResponse.
    Otherwise returns JSONResponse with the full body.
    """
    body = await request.json()
    if body.get("stream", False):
        return await _forward_stream(request, "/v1/completions")
    else:
        return await _forward_json(request, "/v1/completions")



