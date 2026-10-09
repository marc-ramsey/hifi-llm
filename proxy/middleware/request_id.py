"""Request ID middleware — generates a UUID per request and injects it into
responses and structured access logs for end-to-end tracing.

Every request gets a unique `X-Request-ID` header in the response.  The ID
is also included in every structured log line produced by this proxy, so you
can grep logs for a single request:

    grep abc123 proxy.log

If the client sends an `X-Request-ID` header, that value is reused instead
of generating a new one — useful when chaining proxies.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


logger = logging.getLogger("llm-proxy.request_id")


class RequestIDMiddleware(BaseHTTPMiddleware):
    HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Reuse client-supplied ID if present (e.g. from a parent proxy)
        request_id = request.headers.get(self.HEADER, str(uuid.uuid4()))

        # Attach to the request state so downstream handlers can access it
        request.state.request_id = request_id  # type: ignore[attr-defined]

        response = await call_next(request)
        response.headers[self.HEADER] = request_id

        return response


def get_request_id(request: Request) -> str:
    """Return the request ID, falling back to a generated UUID."""
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())
