"""API-key authentication middleware.

Reads the API key dynamically from the config registry on every request so
that SIGHUP config reloads take effect immediately — no app rebuild needed
for auth changes.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import get


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Read the API key dynamically from the config registry so that
        # SIGHUP reloads (which swap _current) take effect immediately —
        # no app rebuild needed for auth changes.
        api_key = get().auth.api_key

        # No key configured — skip auth
        if not api_key:
            return await call_next(request)

        # No public endpoints — all requests require auth when api_key is set

        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {api_key}"

        if auth != expected:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid or missing API key", "type": "unauthorized", "param": None, "code": 401}},
                headers={"Retry-After": "60"},
            )

        return await call_next(request)
