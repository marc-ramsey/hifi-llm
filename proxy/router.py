"""Route handlers — /v1/models and /v1/chat/completions.

The router is a dumb pipe: resolve the model, merge defaults, forward to the
adapter.  Streaming responses are passed through byte-for-byte — no JSON
parsing, no delta inspection, no invented frames.

Streaming error contract: the first upstream chunk is awaited *before* the
response is opened, so a backend that fails immediately still produces a real
status code instead of ``200 OK`` followed by a made-up ``data: {"error": …}``
event.  Once bytes are on the wire the status line is spent; the stream then
simply ends, exactly as the backend left it.  (A tool-calling client cannot
tell an injected error frame from an empty answer — it just never runs the
tool and shows nothing.)
"""

from __future__ import annotations

import logging
from copy import deepcopy

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from adapters import get_adapter, get_adapter_class
from adapters.base import BaseAdapter, make_error_response
from config import get
from proxy.health import _current_report
from proxy.metrics import get_metrics_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


def _resolve_model_config(model_name: str):
    """Look up a model entry by name. Raises HTTPException if not found."""
    for model in get().models:
        if model.name == model_name:
            return model
    valid = ", ".join(m.name for m in get().models)
    # HTTPException.detail is used directly as the response body by FastAPI,
    # so pass the inner dict (not wrapped in an "error" key).
    raise HTTPException(
        status_code=400,
        detail={
            "message": f"Invalid model: {model_name}",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        },
    )


# Response header naming the parameters the proxy supplied on this request.
# Anything that changes a request must be visible on the response, or a
# deployment spends an afternoon diffing byte counts to find out why sampling
# looks nothing like the config says it should.
DEFAULTED_HEADER = "X-Hifi-Defaulted"


def _merge_defaults(base: dict, defaults: dict) -> tuple[dict, list[str]]:
    """Fill in parameters *base* omits; client values always win.

    Returns ``(payload, injected)`` where *injected* names what the proxy
    supplied.  A nested key added inside a client-supplied dict is reported as
    ``parent.child``, so a partial injection is still visible in the header.

    Precedence is the point: a key present in the client request is left
    completely alone, because these are *defaults*, not policy.  Two edges are
    worth naming, because both were gotchas while writing this:

    * ``null`` counts as absent.  A client sending ``"temperature": null`` says
      it has no preference, which is exactly what a default answers.
    * An empty dict from the client is *present*, and stays empty.  Filling
      ``"logit_bias": {}`` from config would turn "no bias" into a bias — the
      one case where the client's value carries meaning precisely by being
      empty.  So emptiness is respected for dicts, unlike ``null``.

    Values are deep-copied.  A nested default (``logit_bias``,
    ``response_format``) merged into a request and mutated downstream would
    otherwise be inherited by every later request for that model, turning one
    odd request into permanent config drift.
    """
    result = dict(base)
    injected: list[str] = []
    for key, value in defaults.items():
        existing = result.get(key)
        if existing is None:
            result[key] = deepcopy(value)
            injected.append(key)
        elif existing and isinstance(existing, dict) and isinstance(value, dict):
            merged, sub = _merge_defaults(existing, value)
            result[key] = merged
            injected.extend(f"{key}.{name}" for name in sub)
    return result, injected


async def _resolve_and_merge(request: Request):
    """Resolve the model and apply its defaults. Returns (config, payload, injected).

    Caches the parsed body on request.state so it's only parsed once.
    *injected* names the parameters the proxy supplied; it is reported in
    ``X-Hifi-Defaulted`` and counted per model and parameter.
    """
    if not hasattr(request.state, "_request_body"):
        body = await request.json()
        request.state._request_body = body
        request.state._model = body.get("model", "")
    else:
        body = request.state._request_body
    model_name = request.state._model
    model_config = _resolve_model_config(model_name)
    payload, injected = _merge_defaults(body, model_config.default_params)

    if injected:
        # Loud on three channels: the response header (set by the caller, which
        # owns the response object), the log, and a counter.
        logger.debug(
            "model %s: proxy supplied default_params for: %s",
            model_name, ", ".join(injected),
        )
        for key in injected:
            get_metrics_store().record_default_param(model_name, key)
    return model_config, payload, injected


def _defaulted_headers(injected: list[str]) -> dict[str, str]:
    """Header dict naming supplied defaults, or empty when nothing was changed."""
    return {DEFAULTED_HEADER: ",".join(injected)} if injected else {}


def _make_adapter(model_config) -> BaseAdapter:
    """Create an adapter instance from a model config."""
    return get_adapter(
        model_config.backend,
        api_key=model_config.api_key,
        model_config=model_config,
    )


async def _forward_stream(
    model_config, payload: dict, injected: list[str], endpoint: str
) -> StreamingResponse:
    """Raw SSE pass-through proxy.

    The adapter's forward_stream yields raw bytes (already SSE-formatted).
    They are forwarded unchanged — no parsing, no splitting, no appended
    frames.

    The first chunk is pulled here, while the status line is still ours: a
    BackendError raised before any byte propagates to FastAPI's handler and
    becomes a real 5xx, the same way it does for non-streaming requests.

    *payload* is the request body with the model's ``default_params`` applied
    (see :func:`_merge_defaults`); *injected* names what was added and is
    echoed in ``X-Hifi-Defaulted``.
    """
    adapter = _make_adapter(model_config)

    upstream = adapter.forward_stream(
        url=model_config.url,
        endpoint=endpoint,
        payload=payload,
        verify_ssl=model_config.verify_ssl,
    )

    try:
        first = await upstream.__anext__()
    except StopAsyncIteration:  # backend produced an empty body
        first = None

    return StreamingResponse(
        _chain(first, upstream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            **_defaulted_headers(injected),
        },
    )


async def _chain(first: bytes | None, rest):
    """Yield an already-read *first* chunk, then the remainder of *rest*."""
    if first is not None:
        yield first
    async for chunk in rest:
        yield chunk


async def _forward_json(
    model_config, payload: dict, injected: list[str], endpoint: str
) -> JSONResponse:
    """Generic JSON proxy for any endpoint."""
    adapter = _make_adapter(model_config)

    response = await adapter.forward_json(
        url=model_config.url,
        endpoint=endpoint,
        payload=payload,
        verify_ssl=model_config.verify_ssl,
    )
    return JSONResponse(content=response, headers=_defaulted_headers(injected))


# ── GET /v1/models ──────────────────────────────────────────────────────────

@router.get("/models", summary="List available models", description="Returns the list of model objects available in the API. Each model includes its id, owned_by backend type, and current health status.")
async def list_models():
    """Return the list of configured models in OpenAI format with status."""

    report = _current_report()
    health_map = {bh.name: bh for bh in report.backends}

    models = get().models
    data = []
    for m in models:
        adapter_cls = get_adapter_class(m.backend)
        entry: dict = {
            "id": m.name,
            "object": "model",
            "created": 0,
            "owned_by": m.backend,
            "provider": adapter_cls.PROVIDER,
        }
        bh = health_map.get(m.name)
        if bh:
            status_value = "loaded" if bh.status.value == "healthy" else "unloaded"
            entry["status"] = {"value": status_value}
        data.append(entry)

    return {"object": "list", "data": data}


# ── POST /v1/chat/completions ───────────────────────────────────────────────

@router.post("/chat/completions", summary="Create chat completion", description="Creates a model response for the given conversation. The request body is forwarded to the backend after merging per-model default parameters (temperature, top_p, etc.). Supports both streaming and non-streaming responses.", responses={400: {"description": "Invalid model or request"}, 502: {"description": "Backend error"}})
async def chat_completions(request: Request):
    """Proxy /v1/chat/completions. Stream or JSON, delegated to adapter."""
    model_config, payload, injected = await _resolve_and_merge(request)
    if payload.get("stream", False):
        return await _forward_stream(
            model_config, payload, injected, "/v1/chat/completions"
        )
    return await _forward_json(model_config, payload, injected, "/v1/chat/completions")


# ── POST /v1/embeddings ─────────────────────────────────────────────────────

@router.post("/embeddings", summary="Create embeddings", description="Creates an embedding vector representing the input text. The request is forwarded to the configured embedding model backend.", responses={400: {"description": "Invalid model or request"}, 502: {"description": "Backend error"}})
async def embeddings(request: Request):
    """Proxy /v1/embeddings to the backend."""
    model_config, payload, injected = await _resolve_and_merge(request)
    return await _forward_json(model_config, payload, injected, "/v1/embeddings")


# ── POST /v1/completions (legacy) ───────────────────────────────────────────

@router.post("/completions", summary="Create legacy completion (deprecated)", description="Creates a completion for the provided prompt and parameters. Deprecated — use /chat/completions instead.", responses={400: {"description": "Invalid model or request"}, 502: {"description": "Backend error"}})
async def completions(request: Request):
    """Proxy /v1/completions. Stream or JSON, delegated to adapter."""
    model_config, payload, injected = await _resolve_and_merge(request)
    if payload.get("stream", False):
        return await _forward_stream(model_config, payload, injected, "/v1/completions")
    return await _forward_json(model_config, payload, injected, "/v1/completions")
