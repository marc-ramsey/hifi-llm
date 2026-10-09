"""ASGI wrapper that delegates to the latest FastAPI app.

This allows SIGHUP config reloads to swap the entire ASGI stack without
touching uvicorn internals.  Uvicorn always sees the same object; requests
are dispatched to whichever FastAPI instance is current.

In-flight requests on the old app complete naturally because they are
already deep in a handler call stack — the swap only affects new requests
entering ``__call__``.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from proxy.app import create_app

logger = logging.getLogger(__name__)

# Type aliases matching the ASGI spec
ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]


class ConfigReloadableApp:
    """Delegates all ASGI calls to the latest app built from current config.

    Usage::

        reloadable = ConfigReloadableApp(initial_config=config)
        uvicorn.Server(uvicorn.Config(app=reloadable, ...))

        # On SIGHUP:
        reloadable.reload()  # swaps in a fresh app from updated config
    """

    def __init__(self, initial_config: Any = None) -> None:
        """Create a reloadable ASGI wrapper.

        Args:
            initial_config: Optional ProxyConfig passed to the first app
                build.  If None, ``create_app()`` reads from the global
                config registry (which must already be loaded).
        """
        self._app: Any = None
        self._initial_config = initial_config
        self._initialized = False

    async def __call__(self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend) -> None:
        if not self._initialized:
            # First request — build the initial app from current config.
            self._app = create_app(self._initial_config)
            self._initialized = True
        return await self._app(scope, receive, send)

    def reload(self) -> None:
        """Swap to a fresh app built from the latest config.

        In-flight requests on the old app finish naturally; new requests
        immediately use the new app.  This is an atomic pointer swap — no
        locks needed because CPython's GIL makes single-assignment swaps
        atomic, and the FastAPI app is immutable after construction.
        """
        old_app = self._app
        self._app = create_app()
        if old_app is not None:
            logger.info("Replaced ASGI app with fresh instance from updated config")
