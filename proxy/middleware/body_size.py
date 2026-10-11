"""Request body size limit middleware.

Rejects oversized request bodies with HTTP 413.  The limit defaults to 10 MB
and is configurable (``max_body_size_mb`` in ``proxy.yaml``).

This middleware must sit *outside* anything that reads the body — auth parses
the JSON body to find the model name — which is why it is registered ahead of
it in ``proxy.app._middleware_stack``.

The limit is enforced twice over, and never by buffering:

1. From ``Content-Length``, before the body is read at all (cheap fast path).
2. By counting ``http.request`` messages on the ASGI receive channel as the
   application reads them.  This also caps chunked uploads sent without
   ``Content-Length``, which a header-only check silently allows through.

Nothing is parsed and the payload is never copied or held: the cap is a
counter, so raw byte-for-byte pass-through is preserved.

Once the limit is crossed the 413 is sent immediately and the application is
handed ``http.disconnect`` so it unwinds instead of parsing a partial body; any
response it tries to produce afterwards is dropped, since the 413 is already on
the wire.  The unread remainder is left to uvicorn (which closes the
connection) — the proxy never buffers it, which is the point of the cap.
"""

from __future__ import annotations

import logging

from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from adapters.base import make_error_response

logger = logging.getLogger("llm-proxy.body_size")

DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


class BodySizeMiddleware:
    """Enforce a maximum request body size on the ASGI receive channel."""

    def __init__(self, app: ASGIApp, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 1. Fast path — reject on the declared length without touching the body.
        declared = _content_length(scope)
        if declared is not None and declared > self.max_body_bytes:
            self._log(scope, declared)
            await self._reject(scope, send)
            return

        # 2. Count the body as the application reads it.
        state = {"seen": 0, "rejected": False}

        async def counting_receive() -> Message:
            message = await receive()

            if state["rejected"] or message["type"] != "http.request":
                return message

            state["seen"] += len(message.get("body") or b"")
            if state["seen"] <= self.max_body_bytes:
                return message

            # Over the limit: answer 413, then tell the application the client
            # went away so it unwinds instead of parsing a partial body.
            state["rejected"] = True
            self._log(scope, state["seen"])
            await self._reject(scope, send)
            return {"type": "http.disconnect"}

        async def send_wrapper(message: Message) -> None:
            if state["rejected"]:
                # The 413 is already sent; whatever the unwinding application
                # produces next is dropped on the floor.
                if message["type"] == "http.response.start":
                    logger.debug("Discarding application response after 413 on %s", scope.get("path", ""))
                return
            await send(message)

        try:
            await self.app(scope, counting_receive, send_wrapper)
        except Exception:
            if not state["rejected"]:
                raise
            # An error while unwinding after the 413 is not actionable.
            logger.debug("Application aborted after 413 on %s", scope.get("path", ""), exc_info=True)

    # ── helpers ───────────────────────────────────────────────────────────

    def _log(self, scope: Scope, size: int) -> None:
        logger.warning(
            "Request body too large (%s bytes, max %s) on %s",
            size, self.max_body_bytes, scope.get("path", ""),
        )

    async def _reject(self, scope: Scope, send: Send) -> None:
        response: Response = JSONResponse(
            status_code=413,
            content=make_error_response(
                f"Request body too large (max {self.max_body_bytes} bytes)",
                "payload_too_large",
                413,
            ),
        )
        # The real send plus a stub receive: the body has been dealt with.
        await response(scope, receive=_body_finished, send=send)


async def _body_finished() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


def _content_length(scope: Scope) -> int | None:
    """Return the declared Content-Length, or None if absent/unparseable."""
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None
