"""API-key authentication middleware.

Reads the API key dynamically from the config registry on every request so
that SIGHUP config reloads take effect immediately — no app rebuild needed
for auth changes.
"""

from __future__ import annotations

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from config import get


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        # Store the key from construction time as a fallback.
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # Read the current API key from the config registry so that
        # SIGHUP reloads (which swap _current) take effect immediately.
        api_key = get().auth.api_key if self._api_key else self._api_key

        # No key configured — skip auth
        if not api_key:
            return await call_next(request)

        # Public endpoints — no auth required
        if request.url.path in ("/health",):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {api_key}"

        if auth != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
