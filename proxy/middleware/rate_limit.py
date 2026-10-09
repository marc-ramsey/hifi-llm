"""Rate-limiting middleware — per-IP sliding window with per-endpoint overrides.

Returns HTTP 429 when a client exceeds the configured request rate.
Skips /health to avoid interfering with uptime monitors.

Config (in config.yaml)::

    rate_limit:
      enabled: true
      requests_per_minute: 60
      endpoints:
        "/v1/chat/completions": 30
        "/v1/embeddings": 20

"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger("llm-proxy.rate_limit")


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config, metrics=None) -> None:  # type: ignore[override]
        super().__init__(app)
        self._config = config
        self._metrics = metrics
        # ip -> list of request timestamps (seconds since epoch)
        self._counters: dict[str, list[float]] = defaultdict(list)

    def _get_limit(self, path: str) -> int | None:
        """Return the rate limit for this path, or None if unlimited."""
        endpoints = self._config.endpoints
        # Exact match first
        if path in endpoints:
            return endpoints[path]
        # Prefix match — longest prefix wins
        best: tuple[str, int] | None = None
        for ep, limit in endpoints.items():
            if path.startswith(ep) and (best is None or len(ep) > len(best[0])):
                best = (ep, limit)
        return best[1] if best else self._config.requests_per_minute

    def _dispatch(self, request: Request):
        """Check rate limit. Returns 429 response or None."""
        if not self._config.enabled:
            return None

        # Skip health endpoint — uptime monitors shouldn't count against limits
        if request.url.path == "/health":
            return None

        ip = request.client.host if request.client else "unknown"
        limit = self._get_limit(request.url.path)
        if limit is None:
            return None  # explicitly unlimited

        now = time.time()
        window = 60.0  # sliding window in seconds

        # Prune old entries outside the window
        timestamps = self._counters[ip]
        cutoff = now - window
        self._counters[ip] = [t for t in timestamps if t > cutoff]

        if len(self._counters[ip]) >= limit:
            logger.warning("Rate limit exceeded for %s on %s (%d/%d)", ip, request.url.path, len(self._counters[ip]), limit)
            if self._metrics is not None:
                self._metrics.record_rate_limit(ip)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_exceeded",
                        "param": None,
                        "code": 429,
                    },
                },
                headers={"Retry-After": "60"},
            )

        self._counters[ip].append(now)
        return None

    async def dispatch(self, request: Request, call_next):
        response = self._dispatch(request)
        if response is not None:
            return response
        return await call_next(request)
