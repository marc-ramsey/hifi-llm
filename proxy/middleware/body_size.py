"""Request body size limit middleware.

Rejects requests with bodies larger than 10 MB (configurable via MAX_BODY_SIZE)
to prevent DoS. Handles both Content-Length and chunked transfer encoding.

Returns HTTP 413 when exceeded. Skips no endpoints — all requests are checked.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("llm-proxy.body_size")


class BodyTooLargeError(Exception):
    """Raised when the request body exceeds MAX_BODY_SIZE."""


class BodySizeMiddleware(BaseHTTPMiddleware):
    """Enforce a maximum request body size (default 10 MB)."""

    MAX_BODY_SIZE = 10 * 1024 * 1024  # bytes

    async def dispatch(self, request: Request, call_next):
        # Fast path: check Content-Length header
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": "Request body too large (max 10 MB)",
                        "type": "payload_too_large",
                        "param": None,
                        "code": 413,
                    },
                },
            )

        # Slow path: intercept _receive for chunked encoding or missing CL
        original_receive = request._receive  # type: ignore[attr-defined]
        total_size = 0

        async def limited_receive() -> dict:
            nonlocal total_size
            message = await original_receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                total_size += len(body)
                if total_size > self.MAX_BODY_SIZE:
                    raise BodyTooLargeError()
            return message

        try:
            request._receive = limited_receive  # type: ignore[attr-defined]
        except AttributeError:
            pass  # Some ASGI servers may not allow this; fall through

        try:
            return await call_next(request)
        except BodyTooLargeError:
            logger.warning("Request body exceeded %d bytes on %s", self.MAX_BODY_SIZE, request.url.path)
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": "Request body too large (max 10 MB)",
                        "type": "payload_too_large",
                        "param": None,
                        "code": 413,
                    },
                },
            )
