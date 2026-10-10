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
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import get

logger = logging.getLogger("llm-proxy.auth")


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
            if model_name and model_name in api_key:
                expected_key = api_key[model_name]
            else:
                # Unknown model — reject (no fallback key configured for it).
                return _unauthorized_response()
        else:
            # Global key mode.
            expected_key = api_key

        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {expected_key}"

        if auth != expected:
            return _unauthorized_response()

        return await call_next(request)
