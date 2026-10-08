"""Route handlers — /v1/models and /v1/chat/completions."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import get
from adapters.base import BackendError, OpenAICompatibleAdapter

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


def _backend_error_handler(func):
    """Decorator that converts BackendError to OpenAI-style 502 responses."""
    async def wrapper(request: Request):
        try:
            return await func(request)
        except BackendError as e:
            logger.warning("Backend error on %s: %s", request.url.path, e.message)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": e.message,
                        "type": "backend_error",
                        "param": None,
                        "code": e.status_code,
                    },
                },
            )
    return wrapper


def _merge_defaults(payload: dict, defaults: dict) -> dict:
    """Merge default_params into the request payload.
    Client values override defaults.
    """
    merged = {**defaults, **payload}
    return merged


# ── GET /v1/models ──────────────────────────────────────────────────────────

@router.get("/models")
async def list_models():
    """Return the list of configured models in OpenAI format."""
    models = get().models
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": m.backend,
            }
            for m in models
        ],
    }


# ── POST /v1/chat/completions ───────────────────────────────────────────────

async def _stream_handler(request: Request) -> StreamingResponse:
    """Handle /v1/chat/completions with streaming (SSE) to the client."""
    body = await request.json()
    model_name = body.get("model", "")

    model_config = _resolve_model_config(model_name)
    payload = _merge_defaults(body, model_config.default_params)

    adapter = OpenAICompatibleAdapter()
    stream = adapter.forward_stream(
        url=model_config.url,
        endpoint="/v1/chat/completions",
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
                    yield f"data: {data}\n\n"
                elif line_str.startswith("data:"):
                    data = line_str[5:].lstrip()
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
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


@_backend_error_handler
async def _json_handler(request: Request) -> JSONResponse:
    """Handle /v1/chat/completions returning a full JSON response."""
    body = await request.json()
    model_name = body.get("model", "")

    model_config = _resolve_model_config(model_name)
    payload = _merge_defaults(body, model_config.default_params)

    adapter = OpenAICompatibleAdapter()
    response = await adapter.forward_json(
        url=model_config.url,
        endpoint="/v1/chat/completions",
        payload=payload,
    )

    return JSONResponse(content=response)


@router.post("/chat/completions")
async def chat_completions(request: Request):
    """
    Proxy /v1/chat/completions.
    If stream=true, returns SSE StreamingResponse.
    Otherwise returns JSONResponse with the full body.
    """
    body = await request.json()
    if body.get("stream", False):
        return await _stream_handler(request)
    else:
        return await _json_handler(request)


# ── POST /v1/embeddings ─────────────────────────────────────────────────────

@_backend_error_handler
@router.post("/embeddings")
async def embeddings(request: Request):
    """Proxy /v1/embeddings to the backend."""
    body = await request.json()
    model_name = body.get("model", "")

    model_config = _resolve_model_config(model_name)
    payload = _merge_defaults(body, model_config.default_params)

    adapter = OpenAICompatibleAdapter()
    response = await adapter.forward_json(
        url=model_config.url,
        endpoint="/v1/embeddings",
        payload=payload,
    )

    return JSONResponse(content=response)


# ── POST /v1/completions (legacy) ───────────────────────────────────────────

async def _completions_stream_handler(request: Request) -> StreamingResponse:
    """Handle /v1/completions with streaming."""
    body = await request.json()
    model_name = body.get("model", "")

    model_config = _resolve_model_config(model_name)
    payload = _merge_defaults(body, model_config.default_params)

    adapter = OpenAICompatibleAdapter()
    stream = adapter.forward_stream(
        url=model_config.url,
        endpoint="/v1/completions",
        payload=payload,
    )

    async def chunk_iterator() -> AsyncIterator[str]:
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
                    yield f"data: {data}\n\n"
                elif line_str.startswith("data:"):
                    data = line_str[5:].lstrip()
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    yield f"data: {data}\n\n"

    return StreamingResponse(
        chunk_iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@_backend_error_handler
async def _completions_json_handler(request: Request) -> JSONResponse:
    """Handle /v1/completions returning a full JSON response."""
    body = await request.json()
    model_name = body.get("model", "")

    model_config = _resolve_model_config(model_name)
    payload = _merge_defaults(body, model_config.default_params)

    adapter = OpenAICompatibleAdapter()
    response = await adapter.forward_json(
        url=model_config.url,
        endpoint="/v1/completions",
        payload=payload,
    )

    return JSONResponse(content=response)


@router.post("/completions")
async def completions(request: Request):
    """
    Proxy /v1/completions (legacy endpoint).
    If stream=true, returns SSE StreamingResponse.
    Otherwise returns JSONResponse with the full body.
    """
    body = await request.json()
    if body.get("stream", False):
        return await _completions_stream_handler(request)
    else:
        return await _completions_json_handler(request)



