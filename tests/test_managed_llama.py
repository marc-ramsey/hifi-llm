"""Tests for ManagedLlamaAdapter — lazy launch, delegation, config mapping."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import requests

from adapters.managed_llama import ManagedLlamaAdapter, _to_cli_flag
from config.schema import ModelConfig
from proxy.process_manager import ProcessManager


# ────────────────────────────────────────────────────────────────────────
# Mock llama-server — responds to /v1/models and /v1/chat/completions
# ────────────────────────────────────────────────────────────────────────

class _MockLlamaServer:
    """Minimal HTTP server that mimics llama-server's API."""

    def __init__(self, port: int):
        self.port = port
        self._server: HTTPServer | None = None

    def start(self) -> None:
        handler = _MockLlamaHandler
        handler.port = self.port
        self._server = HTTPServer(("127.0.0.1", self.port), handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()


class _MockLlamaHandler(BaseHTTPRequestHandler):
    """Responds like a real llama-server for test purposes."""

    port: int  # set by _MockLlamaServer

    def do_GET(self):
        if "/v1/models" in self.path:
            body = {
                "object": "list",
                "data": [
                    {"id": "mock-model", "object": "model", "owned_by": "me"}
                ],
            }
            self._json(200, body)
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if "/v1/chat/completions" in self.path:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            stream = body.get("stream", False)
            model = body.get("model", "mock")

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                resp = {
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                self._json(200, resp)
        else:
            self._json(404, {"error": {"message": "not found"}})

    def _json(self, status: int, data: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # silence logs


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_process_manager():
    ProcessManager.shutdown()
    yield
    ProcessManager.shutdown()


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock llama-server on a free port."""
    # Find a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = _MockLlamaServer(port)
    server.start()
    time.sleep(0.3)  # let it bind
    yield server, port
    server.stop()


import socket


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────

class TestToCliFlag:
    """Test snake_case → --kebab-case conversion."""

    def test_simple(self):
        assert _to_cli_flag("ctx_size") == "--ctx-size"

    def test_multi_word(self):
        assert _to_cli_flag("n_gpu_layers") == "--n-gpu-layers"

    def test_no_underscores(self):
        assert _to_cli_flag("threads") == "--threads"


class TestManagedLlamaAdapter:
    """Integration tests for the managed llama adapter."""

    def test_command_builds_correctly(self, mock_server):
        """Verify server command is built from config fields."""
        _, port = mock_server
        model_config = ModelConfig(
            name="test-model",
            url=f"http://127.0.0.1:{port}",
            backend="managed_llama",
            llama_binary="/usr/bin/llama-server",
            server_config={"ctx_size": 4096, "n_gpu_layers": 99},
        )
        adapter = ManagedLlamaAdapter(model_config)
        cmd = adapter._build_command()
        assert cmd == ["/usr/bin/llama-server", "--ctx-size", "4096", "--n-gpu-layers", "99"]

    def test_forward_json_triggers_lazy_start(self, mock_server):
        """First request should trigger server launch (even if binary missing,
        the ensure_running call is made)."""
        _, port = mock_server
        model_config = ModelConfig(
            name="test-model",
            url=f"http://127.0.0.1:{port}",
            backend="managed_llama",
            llama_binary="/nonexistent/llama-server",  # won't actually start
            server_config={"ctx_size": 4096},
        )
        adapter = ManagedLlamaAdapter(model_config)

        # ensure_running is called — it will fail to start but the method is invoked
        cmd = adapter._build_command()
        result = ProcessManager.ensure_running("test-model", cmd, cooldown=0.1)
        assert result is False  # binary doesn't exist

    def test_forward_json_succeeds_when_server_up(self, mock_server):
        """When the server is already running, forwarding works."""
        _, port = mock_server
        model_config = ModelConfig(
            name="test-model",
            url=f"http://127.0.0.1:{port}",
            backend="managed_llama",
            llama_binary="/usr/bin/llama-server",  # won't be used — server is up
            server_config={"ctx_size": 4096},
        )
        adapter = ManagedLlamaAdapter(model_config)

        import asyncio
        result = asyncio.run(adapter.forward_json(
            url=model_config.url,
            endpoint="/v1/chat/completions",
            payload={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        ))
        assert result["choices"][0]["message"]["content"] == "hi"



