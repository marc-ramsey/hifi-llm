"""API-key authentication middleware.

Reads the API key dynamically from the config registry on every request so
that SIGHUP config reloads take effect immediately.  On each reload a fresh
FastAPI app (with fresh middleware instances) is built via ``create_app()``,
which reads the updated global config.  The dynamic ``get()`` call ensures
auth changes are picked up without requiring a process restart.

Supports two modes:
  - Single string: all requests must use that Bearer token (global key).
  - Dict: keys are model names, values are per-model tokens.  The proxy
    reads the ``model`` field from the request body to pick the right key.
    Requests that carry no model name (``GET /v1/models``, ``/docs``, a body
    that is not JSON, …) are matched against the ``"*"`` entry when one is
    configured — use it to let frontends list the catalogue:

        auth:
          api_key:
            "*": "catalogue-key"     # any request without a model
            "fast-agent": "key-a"

    With no ``"*"`` entry, model-less requests are rejected.

``PUBLIC_PATHS`` is served without credentials — uptime monitors and process
managers must be able to reach ``/health`` without a token.

Token comparison is constant-time (``hmac.compare_digest``).
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import get

logger = logging.getLogger("llm-proxy.auth")

# Dict-mode key matching requests that carry no model name.
WILDCARD_KEY = "*"

# Paths served without authentication.  /metrics deliberately stays protected
# (it exposes traffic data); /health is infrastructure and is also exempt from
# rate limiting.
PUBLIC_PATHS = frozenset({"/health"})


def _token_matches(supplied: str, expected: str) -> bool:
    """Compare tokens without leaking a prefix match through timing."""
    return hmac.compare_digest(
        supplied.encode("utf-8", "surrogateescape"),
        expected.encode("utf-8", "surrogateescape"),
    )


def _unauthorized_response() -> JSONResponse:
    """Return a standard 401 JSON error response."""
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Invalid or missing API key", "type": "unauthorized", "param": None, "code": 401}},
        headers={"Retry-After": "60"},
    )


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = get().auth.api_key

        # No key configured — skip auth
        if not api_key:
            return await call_next(request)

        # Infrastructure endpoints are reachable without credentials.
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Resolve the model name.  It may already be parsed by a prior middleware;
        # otherwise parse it from the request body now (cheap JSON parse, cached on
        # request.state so downstream handlers don't re-parse).
        model_name = getattr(request.state, "_model", None)
        if model_name is None:
            try:
                body = await request.json()
                request.state._model = model_name = body.get("model", "")
                # Also stash the parsed body so _resolve_and_merge skips re-parsing.
                request.state._request_body = body
            except Exception:
                model_name = None

        if isinstance(api_key, dict):
            # Per-model mode — read model name from request body.
            expected_key = api_key.get(model_name) if model_name else None
            if expected_key is None and not model_name:
                # No model in this request — fall back to the wildcard entry.
                expected_key = api_key.get(WILDCARD_KEY)
            if expected_key is None:
                # Unknown model, or no model and no wildcard configured.
                return _unauthorized_response()
        else:
            # Global key mode.
            expected_key = api_key

        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {expected_key}"

        if not _token_matches(auth, expected):
            return _unauthorized_response()

        return await call_next(request)
