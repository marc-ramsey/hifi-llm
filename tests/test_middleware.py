"""Middleware stack tests — ordering, public endpoints, body-size cap.

Covers three behaviour contracts:

1. **Order** — ``RequestID → AccessLog → BodySize → CORS → RateLimit → Auth``.
   Consequences tested here: oversized bodies are rejected before auth looks at
   them, browser preflights are answered before auth, and credential guessing
   is throttled.
2. **``/health`` needs no credentials** — uptime monitors and process managers
   cannot hold a token; ``/metrics`` stays protected.
3. **The body-size cap cannot be bypassed** — chunked uploads with no
   ``Content-Length`` are counted while they are read, not just declared.
"""

from __future__ import annotations

import asyncio

import pytest
import requests

import config as config_registry
from config.schema import CORSConfig, ProxyConfig
from proxy.middleware.body_size import BodySizeMiddleware


# ────────────────────────────────────────────────────────────────────────
# 1. Order
# ────────────────────────────────────────────────────────────────────────

class TestMiddlewareOrder:
    def test_declared_order_matches_effective_order(self, restore_config):
        """The declared stack is what the built app actually runs."""
        from proxy.app import _middleware_stack, create_app

        config_registry.set(ProxyConfig(cors=CORSConfig(enabled=True)))
        app = create_app()

        expected = [cls.__name__ for cls, _ in _middleware_stack(config_registry.get(), None)]
        # user_middleware is outermost-first, same as _middleware_stack().
        assert [m.cls.__name__ for m in app.user_middleware] == expected

    def test_expected_sequence(self, restore_config):
        from proxy.app import _middleware_stack

        config_registry.set(ProxyConfig(cors=CORSConfig(enabled=True)))

        names = [cls.__name__ for cls, _ in _middleware_stack(config_registry.get(), None)]

        assert names == [
            "RequestIDMiddleware",
            "AccessLogMiddleware",
            "BodySizeMiddleware",
            "CORSMiddleware",
            "RateLimitMiddleware",
            "AuthMiddleware",
        ]

    def test_cors_optional(self, restore_config):
        from proxy.app import _middleware_stack

        config_registry.set(ProxyConfig(cors=CORSConfig(enabled=False)))

        names = [cls.__name__ for cls, _ in _middleware_stack(config_registry.get(), None)]

        assert "CORSMiddleware" not in names
        assert names[-1] == "AuthMiddleware"

    def test_body_size_cap_comes_from_config(self, restore_config):
        from proxy.app import _middleware_stack

        config_registry.set(ProxyConfig(max_body_size_mb=2.5))

        _, kwargs = _middleware_stack(config_registry.get(), None)[2]

        assert kwargs["max_body_bytes"] == int(2.5 * 1024 * 1024)


# ────────────────────────────────────────────────────────────────────────
# 2. Body-size cap — unit level, including the chunked path
# ────────────────────────────────────────────────────────────────────────

async def _run_body_middleware(*, max_body_bytes: int, headers: dict, chunks: list[bytes], trace: list):
    """Run BodySizeMiddleware over a fake ASGI app that reads its body.

    ``trace`` records what the inner application saw, so tests can assert both
    the response and whether the body ever reached the handler.
    """
    from starlette.responses import JSONResponse

    scope = {"type": "http", "http_version": "1.1", "method": "POST",
             "path": "/v1/chat/completions", "raw_path": b"/x", "query_string": b"",
             "root_path": "", "scheme": "http", "server": ("test", 1), "client": ("1.2.3.4", 1),
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]}
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    # receive() replays *chunks*, then reports the body finished.
    remaining = list(chunks)

    async def receive():
        trace.append("msg")
        if remaining:
            body = remaining.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(remaining)}
        trace.append("receive-exhausted")
        return {"type": "http.request", "body": b"", "more_body": False}

    async def reading_app(scope, receive, send):
        """A JSON handler: read the whole body, then answer."""
        trace.append("app-called")
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                trace.append("app-disconnected")
                return
            if not message.get("more_body", False):
                break
        await JSONResponse({"ok": True})(scope, receive, send)

    mw = BodySizeMiddleware(reading_app, max_body_bytes=max_body_bytes)
    await mw(scope, receive, send)
    return sent


def _status(sent):
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1, f"expected exactly one response, got {len(starts)}"
    return starts[0]["status"]


class TestBodySizeCap:
    def test_chunked_body_over_cap_rejected_without_content_length(self):
        """No Content-Length anywhere — the byte counter is the only defence."""
        trace: list = []
        sent = asyncio.run(_run_body_middleware(
            max_body_bytes=5_000, headers={}, chunks=[b"x" * 2_000] * 5, trace=trace,
        ))

        assert _status(sent) == 413
        assert "app-disconnected" in trace, "handler was not unwound after the 413"
        assert b"payload_too_large" in b"".join(
            m.get("body", b"") for m in sent if m["type"] == "http.response.body"
        )

    def test_body_under_cap_reaches_the_app(self):
        trace: list = []
        sent = asyncio.run(_run_body_middleware(
            max_body_bytes=5_000, headers={}, chunks=[b"x" * 2_000], trace=trace,
        ))

        assert _status(sent) == 200
        assert [t for t in trace if t != "msg"] == ["app-called"]

    def test_declared_content_length_rejected_before_app_runs(self):
        trace: list = []
        sent = asyncio.run(_run_body_middleware(
            max_body_bytes=1_000, headers={"Content-Length": "20000"}, chunks=[b"x"], trace=trace,
        ))

        assert _status(sent) == 413
        assert trace == [], "application ran despite an oversized Content-Length"

    def test_endless_chunked_body_is_bounded(self):
        """A client streaming forever is cut off, not drained forever."""
        trace: list = []
        endless = [b"x" * 1_000] * 10_000  # ~10 MB, never finishes
        sent = asyncio.run(_run_body_middleware(
            max_body_bytes=5_000, headers={}, chunks=endless, trace=trace,
        ))

        assert _status(sent) == 413
        # Reading stops at the chunk that crosses the limit — about 6 reads,
        # not one per chunk of the 10 MB still queued.
        assert trace.count("msg") < 50, f"kept reading: {trace.count('msg')} reads"

    def test_unparseable_content_length_is_ignored(self):
        trace: list = []
        sent = asyncio.run(_run_body_middleware(
            max_body_bytes=5_000, headers={"Content-Length": "not-a-number"},
            chunks=[b"x" * 100], trace=trace,
        ))
        assert _status(sent) == 200


# ────────────────────────────────────────────────────────────────────────
# E2E — real proxy processes
# ────────────────────────────────────────────────────────────────────────

CONFIG_TEMPLATE = """
listen:
  host: "0.0.0.0"
  port: {port}

auth:
  api_key: {api_key}

rate_limit:
  enabled: {rl_enabled}
  requests_per_minute: 2

max_body_size_mb: {max_body_mb}

models:
  - name: "model-a"
    url: "{backend_url}"
  - name: "model-b"
    url: "{backend_url}"

static_files: []
"""


def _config(api_key="\"sekrit\"", rl_enabled="false", max_body_mb="0.05"):
    return CONFIG_TEMPLATE.format(  # noqa: UP032 - {port}/{backend_url} replaced by proxy_factory
        port="{port}", backend_url="{backend_url}", api_key=api_key,
        rl_enabled=rl_enabled, max_body_mb=max_body_mb,
    )


class TestEndToEnd:
    def test_health_is_public_other_endpoints_are_not(self, proxy_factory, backend_url):
        _, base = proxy_factory(_config(), backend_url=backend_url)

        assert requests.get(f"{base}/health", timeout=5).status_code == 200
        assert requests.get(f"{base}/v1/models", timeout=5).status_code == 401
        assert requests.get(f"{base}/metrics", timeout=5).status_code == 401

        ok = requests.get(f"{base}/v1/models", headers={"Authorization": "Bearer sekrit"}, timeout=5)
        assert ok.status_code == 200

    def test_every_rejection_carries_request_id(self, proxy_factory, backend_url):
        _, base = proxy_factory(_config(), backend_url=backend_url)

        responses = [
            requests.get(f"{base}/v1/models", timeout=5),                                  # 401
            requests.get(f"{base}/v1/models", headers={"Authorization": "Bearer nope"}, timeout=5),  # 401
            requests.post(f"{base}/v1/chat/completions", data=b"x" * 5000, timeout=5),     # 413
        ]
        for resp in responses:
            assert resp.headers.get("X-Request-ID"), f"no X-Request-ID on {resp.status_code}"

    def test_oversized_body_rejected_before_auth(self, proxy_factory, backend_url):
        """BodySize is outside Auth: no credentials are needed to get a 413."""
        _, base = proxy_factory(_config(), backend_url=backend_url)

        resp = requests.post(
            f"{base}/v1/chat/completions",
            data=b"x" * 200_000,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert resp.status_code == 413

    def test_chunked_oversized_body_rejected_with_valid_credentials(
        self, proxy_factory, backend_url
    ):
        """Chunked upload (no Content-Length) + valid key → still 413, not 200/401."""
        _, base = proxy_factory(_config(), backend_url=backend_url)

        def body():
            # Just over the 0.05 MB (52,432 byte) cap, so the client finishes
            # its upload inside the drain budget and reads the 413 cleanly.
            for _ in range(6):
                yield b"x" * 10_000

        resp = requests.post(
            f"{base}/v1/chat/completions",
            data=body(),
            headers={"Authorization": "Bearer sekrit", "Content-Type": "application/json"},
            timeout=10,
        )
        assert resp.status_code == 413

    def test_cors_preflight_is_answered_without_credentials(self, proxy_factory, backend_url):
        """CORSMiddleware is outside Auth — preflights carry no Authorization."""
        _, base = proxy_factory(_config(), backend_url=backend_url)

        resp = requests.options(
            f"{base}/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
            timeout=5,
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_bad_credentials_are_throttled(self, proxy_factory, backend_url):
        """RateLimit is outside Auth, so guessing attempts hit 429."""
        _, base = proxy_factory(
            _config(rl_enabled="true"), backend_url=backend_url
        )

        responses = [
            requests.post(
                f"{base}/v1/chat/completions",
                json={"model": "model-a", "messages": []},
                headers={"Authorization": "Bearer wrong"},
                timeout=5,
            )
            for _ in range(5)
        ]
        codes = [r.status_code for r in responses]
        assert 429 in codes, f"auth attempts were not throttled: {codes}"
        throttled = next(r for r in responses if r.status_code == 429)
        assert throttled.headers.get("X-Request-ID")


class TestWildcardDictAuth:
    def test_model_less_requests_use_the_wildcard_key(self, proxy_factory, backend_url):
        _, base = proxy_factory(
            _config(api_key='{"*": "cat-key", "model-a": "key-a"}'), backend_url=backend_url
        )
        authed = {"Authorization": "Bearer cat-key"}

        # No model in the request → matched by "*".
        assert requests.get(f"{base}/v1/models", timeout=5).status_code == 401
        assert requests.get(f"{base}/v1/models", headers=authed, timeout=5).status_code == 200
        # The wildcard key must not grant model access.
        resp = requests.post(
            f"{base}/v1/chat/completions",
            json={"model": "model-a", "messages": []},
            headers=authed,
            timeout=10,
        )
        assert resp.status_code == 401

    def test_per_model_keys_are_independent(self, proxy_factory, backend_url):
        _, base = proxy_factory(
            _config(api_key='{"*": "cat-key", "model-a": "key-a", "model-b": "key-b"}'),
            backend_url=backend_url,
        )

        def post(model: str, key: str) -> int:
            return requests.post(
                f"{base}/v1/chat/completions",
                json={"model": model, "messages": []},
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            ).status_code

        assert post("model-a", "key-a") == 200
        assert post("model-b", "key-b") == 200
        assert post("model-a", "key-b") == 401
        assert post("model-b", "key-a") == 401
