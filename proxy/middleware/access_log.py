"""Request logging middleware — one structured JSON line per request.

Logs method, path, status_code, duration_ms, and model for every
request that passes through the proxy.  Skips /health to avoid log noise.
Also records metrics for the /metrics endpoint.

For non-streaming requests, duration_ms measures total time.
For streaming responses, duration_ms measures time to first byte only.

Example output::

    {"method":"POST","path":"/v1/chat/completions","status":200,"duration_ms":1247.3,"model":"fast-agent"}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from .request_id import get_request_id


logger = logging.getLogger("llm-proxy.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics: Any = None) -> None:  # type: ignore[override]
        super().__init__(app)
        self._metrics = metrics

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        start = time.monotonic()
        response = await call_next(request)
        model = getattr(request.state, "_model", "") or ""
        status = response.status_code if hasattr(response, "status_code") else 200
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)

        # For streaming responses, elapsed_ms is only time-to-first-byte,
        # which is still useful for monitoring proxy responsiveness.
        log_entry = {
            "request_id": get_request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": elapsed_ms,
            "model": model,
        }
        logger.info(json.dumps(log_entry, ensure_ascii=False))

        if self._metrics is not None:
            self._metrics.record_request(request.url.path, status, elapsed_ms)

        return response
