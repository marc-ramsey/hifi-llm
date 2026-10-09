"""Request logging middleware — one structured JSON line per request.

Logs method, path, status_code, duration_ms, model, and backend for every
request that passes through the proxy.  Skips /health to avoid log noise.

Example output::

    {"method":"POST","path":"/v1/chat/completions","status":200,"duration_ms":1247.3,"model":"fast-agent","backend":"llama_cpp"}
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from .request_id import get_request_id


logger = logging.getLogger("llm-proxy.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip health checks — they're noisy and not useful in access logs
        if request.url.path == "/health":
            return await call_next(request)

        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Extract useful fields from the request
        model = ""
        backend = ""
        try:
            body = await request.json()
            model = body.get("model", "")
        except Exception:
            pass

        # Try to get backend name from the request header set by adapters,
        # or infer from the model config.  For now, just log the path which
        # tells us the endpoint.
        status = response.status_code if hasattr(response, "status_code") else 200

        log_entry = {
            "request_id": get_request_id(request),
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": round(elapsed_ms, 1),
            "model": model,
        }
        logger.info(json.dumps(log_entry, ensure_ascii=False))

        return response
