"""Regression tests for upstream (backend) authentication.

Bug being guarded against: the adapters computed
``key = api_key or self._api_key`` but ``_build_headers()`` returned only
``Content-Type`` — the key was never emitted, so a model's ``api_key`` was
silently dropped and auth-protected backends were unreachable.
``managed_llama`` dropped it twice: its constructor did not accept ``api_key``.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from adapters import get_adapter
from adapters.base import OpenAICompatibleAdapter
from adapters.llama_cpp import LlamaCppAdapter
from adapters.managed_llama import ManagedLlamaAdapter
from config.schema import ModelConfig

from .conftest import _start_proxy, find_free_port

BACKEND_KEY = "backend-secret"


# ────────────────────────────────────────────────────────────────────────
# Recording backend — captures the headers of the last request
# ────────────────────────────────────────────────────────────────────────

class _RecordingHandler(BaseHTTPRequestHandler):
    """Answers like a minimal OpenAI backend and records request headers."""

    last_headers: dict[str, str] = {}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        self.rfile.read(length)
        type(self).last_headers = {k.lower(): v for k, v in self.headers.items()}

        body = json.dumps({
            "object": "chat.completion",
            "model": "echo",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # type: ignore[override]
        pass


@pytest.fixture(scope="module")
def recording_backend():
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def captured_headers():
    def _get() -> dict[str, str]:
        return dict(_RecordingHandler.last_headers)
    return _get


@pytest.fixture(scope="module")
def loop():
    """One event loop for the module.

    The adapters share a module-level ``httpx.AsyncClient``; giving every test
    its own loop (``asyncio.run``) would bind that client to a dead loop.
    """
    lo = asyncio.new_event_loop()
    yield lo
    lo.close()


def _model_config(url: str, **kwargs) -> ModelConfig:
    return ModelConfig(name="auth-model", url=url, **kwargs)


# ────────────────────────────────────────────────────────────────────────
# Header construction
# ────────────────────────────────────────────────────────────────────────

class TestHeaderConstruction:
    def test_bearer_token_emitted_when_key_set(self):
        headers = OpenAICompatibleAdapter(api_key="k1")._build_request_headers(None)
        assert headers["Authorization"] == "Bearer k1"
        assert headers["Content-Type"] == "application/json"

    def test_per_request_key_overrides_adapter_key(self):
        headers = OpenAICompatibleAdapter(api_key="adapter")._build_request_headers("request")
        assert headers["Authorization"] == "Bearer request"

    def test_no_authorization_header_without_key(self):
        headers = OpenAICompatibleAdapter()._build_request_headers(None)
        assert "Authorization" not in headers

    def test_llama_cpp_keeps_auth_and_provider(self):
        headers = LlamaCppAdapter(api_key="k2")._build_request_headers(None)
        assert headers["Authorization"] == "Bearer k2"
        assert headers["X-Provider"] == "llama.cpp"


# ────────────────────────────────────────────────────────────────────────
# On the wire
# ────────────────────────────────────────────────────────────────────────

class TestKeyOnTheWire:
    def test_openai_compatible_sends_key(self, recording_backend, captured_headers, loop):
        mc = _model_config(recording_backend, api_key=BACKEND_KEY)
        adapter = get_adapter(mc.backend, api_key=mc.api_key, model_config=mc)

        loop.run_until_complete(adapter.forward_json(
            url=mc.url, endpoint="/v1/chat/completions", payload={"model": "auth-model"},
        ))

        assert captured_headers().get("authorization") == f"Bearer {BACKEND_KEY}"

    def test_llama_cpp_sends_key(self, recording_backend, captured_headers, loop):
        mc = _model_config(recording_backend, backend="llama_cpp", api_key=BACKEND_KEY)
        adapter = get_adapter("llama_cpp", api_key=mc.api_key, model_config=mc)

        loop.run_until_complete(adapter.forward_json(
            url=mc.url, endpoint="/v1/chat/completions", payload={"model": "auth-model"},
        ))

        assert captured_headers().get("authorization") == f"Bearer {BACKEND_KEY}"
        assert captured_headers().get("x-provider") == "llama.cpp"

    def test_managed_llama_keeps_key(self, recording_backend, captured_headers, loop):
        """The key must survive both the registry and the delegate hop."""
        mc = _model_config(
            recording_backend,
            backend="managed_llama",
            api_key=BACKEND_KEY,
            llama_binary="/bin/true",  # exits immediately; launch still "succeeds"
        )
        adapter = get_adapter("managed_llama", api_key=mc.api_key, model_config=mc)

        assert isinstance(adapter, ManagedLlamaAdapter)
        assert adapter._api_key == BACKEND_KEY
        assert adapter._delegate._api_key == BACKEND_KEY

        loop.run_until_complete(adapter.forward_json(
            url=mc.url, endpoint="/v1/chat/completions", payload={"model": "auth-model"},
        ))

        assert captured_headers().get("authorization") == f"Bearer {BACKEND_KEY}"

    def test_backend_without_key_sends_none(self, recording_backend, captured_headers, loop):
        mc = _model_config(recording_backend)
        adapter = get_adapter(mc.backend, api_key=mc.api_key, model_config=mc)

        loop.run_until_complete(adapter.forward_json(
            url=mc.url, endpoint="/v1/chat/completions", payload={"model": "auth-model"},
        ))

        assert "authorization" not in captured_headers()


# ────────────────────────────────────────────────────────────────────────
# E2E — real proxy process, config api_key → backend sees the token
# ────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_proxy_forwards_config_api_key_to_backend(
        self, proxy_config_dir, recording_backend, captured_headers
    ):
        port = find_free_port()
        config_path = proxy_config_dir / f"backend-auth-{port}.yaml"
        config_path.write_text(
            f"""
listen:
  host: "0.0.0.0"
  port: {port}

auth:
  api_key: null

models:
  - name: "auth-model"
    url: "{recording_backend}"
    api_key: "{BACKEND_KEY}"

static_files: []
"""
        )
        proc = _start_proxy(port, config_path)
        try:
            resp = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "auth-model", "messages": [{"role": "user", "content": "hi"}]},
                timeout=10,
            )
            assert resp.status_code == 200, resp.text
            assert captured_headers().get("authorization") == f"Bearer {BACKEND_KEY}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
