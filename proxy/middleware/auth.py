"""API-key authentication middleware."""

from __future__ import annotations

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None):
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # No key configured — skip auth
        if not self._api_key:
            return await call_next(request)

        # Public endpoints — no auth required
        if request.url.path in ("/health",):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {self._api_key}"

        if auth != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
