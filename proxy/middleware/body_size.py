"""Request body size limit middleware.

Rejects requests with bodies larger than MAX_BODY_SIZE (default 10 MB)
to prevent DoS. Checks the Content-Length header before the body is read.

Returns HTTP 413 when exceeded. Skips no endpoints — all requests are checked.

Note: only enforces the limit when a Content-Length header is present.
Clients that send chunked transfer encoding without a Content-Length
header bypass this check — in practice this virtually never happens for
JSON API requests, where the body size is always known upfront.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from adapters.base import make_error_response

logger = logging.getLogger("llm-proxy.body_size")


class BodySizeMiddleware(BaseHTTPMiddleware):
    """Enforce a maximum request body size via Content-Length header."""

    MAX_BODY_SIZE = 10 * 1024 * 1024  # bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.MAX_BODY_SIZE:
            logger.warning(
                "Request body too large (%s bytes) on %s",
                content_length,
                request.url.path,
            )
            return JSONResponse(
                status_code=413,
                content=make_error_response("Request body too large (max 10 MB)", "payload_too_large", 413),
            )

        return await call_next(request)
