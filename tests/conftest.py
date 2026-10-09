"""Fixtures for proxy E2E tests.

Starts two real subprocesses in the test session:
  - A lightweight threaded HTTP server that mimics an OpenAI-compatible backend
  - The hifi proxy pointing at that backend

This removes all dependency on external services (Arkestra, port :8080).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────────
# Lightweight backend server (real TCP socket, not a mock)
# ────────────────────────────────────────────────────────────────────────

class _BackendHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible backend that returns deterministic responses."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if "/chat/completions" in self.path:
            self._handle_chat(body)
        elif "/embeddings" in self.path:
            self._handle_embeddings(body)
        elif "/completions" in self.path:
            # Legacy completions — not supported by this backend
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": {"message": "not found"}}).encode()
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if "/models" in self.path:
            resp = {
                "object": "list",
                "data": [
                    {"id": "gemma-4-26B-instruct", "object": "model", "created": 0, "owned_by": "test"},
                    {"id": "qwen3.6-35B-instruct", "object": "model", "created": 0, "owned_by": "test"},
                    {"id": "nomic-embed", "object": "model", "created": 0, "owned_by": "test"},
                ],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_response(404)
            self.end_headers()

    # ── chat/completions ────────────────────────────────────────────────

    def _handle_chat(self, body: dict) -> None:
        stream = body.get("stream", False)
        model = body.get("model", "test")

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            # First chunk — role
            self._sse({"object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})

            # Content chunks
            words = ["hello", "world", "from", "the", "test", "server"]
            for word in words:
                self._sse({"object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"content": word + " ", "role": "assistant"}}]})

            # Final chunk — finish_reason
            self._sse({"object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self._sse_done()
        else:
            resp = {
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello world from the test server"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 30, "total_tokens": 35},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())

    def _handle_embeddings(self, body: dict) -> None:
        input_val = body.get("input", "")
        if isinstance(input_val, str):
            input_val = [input_val]
        resp = {
            "object": "list",
            "data": [{"embedding": [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)]} for i in range(len(input_val))],
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    # ── SSE helpers ─────────────────────────────────────────────────────

    def _sse(self, obj: dict) -> None:
        self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
        self.wfile.flush()

    def _sse_done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # Silence request logs
    def log_message(self, *args):  # type: ignore[override]
        pass


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def backend():
    """Start a real threaded HTTP server as an OpenAI-compatible backend.

    Yields (host, port).  Teardown after the session.
    """
    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), _BackendHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    yield "http://127.0.0.1", port

    server.shutdown()


@pytest.fixture(scope="session")
def backend_url(backend):
    """Full base URL of the test backend: http://127.0.0.1:<port>."""
    host, port = backend
    return f"{host}:{port}"


@pytest.fixture(scope="session")
def proxy_config_dir():
    """Create a temp directory for test configs."""
    d = tempfile.mkdtemp(prefix="llm-proxy-test-")
    yield Path(d)


@pytest.fixture(scope="session")
def proxy(proxy_config_dir, backend_url):
    """Start the hifi proxy as a subprocess, pointing at the test backend.

    Yields (port, proc).  Teardown after the session.
    """
    port = _find_free_port()

    config_path = proxy_config_dir / "proxy-config.yaml"
    config_path.write_text(
        f"""
listen:
  host: 0.0.0.0
  port: {port}

auth:
  api_key: null

models:
  - name: "gemma-4-26B-instruct"
    url: "{backend_url}"
    default_params:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256

  - name: "qwen3.6-35B-instruct"
    url: "{backend_url}"
    default_params:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 256

plugins_dir: null
"""
    )

    env = os.environ.copy()
    env["LLM_PROXY_TIMEOUT_MS"] = "60000"

    # Use the venv Python if available so the subprocess has access to
    # installed dependencies regardless of whether the test runner is
    # activated into the venv.
    venv_python = ROOT / ".venv" / "bin" / "python3"
    python_bin = str(venv_python) if venv_python.exists() else sys.executable

    proc = subprocess.Popen(
        [python_bin, str(ROOT / "main.py"), "--config", str(config_path)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to be ready (up to 30s)
    ready = False
    for _ in range(60):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.5)

    if not ready:
        proc.kill()
        stdout, _ = proc.communicate(timeout=5)
        pytest.fail(f"Proxy failed to start.\nstdout:\n{stdout}")

    yield port, proc

    # Teardown
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture
def client(proxy):
    """Return a requests session pointing at the running proxy."""
    port, _ = proxy
    base = f"http://127.0.0.1:{port}"
    s = requests.Session()

    def _request(method, path, **kwargs):
        return s.request(method, base + path, **kwargs)

    s.get = lambda path, **kw: _request("GET", path, **kw)
    s.post = lambda path, **kw: _request("POST", path, **kw)
    return s


@pytest.fixture
def write_config(proxy_config_dir):
    """Write a YAML config file and return its path."""

    def _write(content: str, name: str = "proxy-config.yaml") -> Path:
        path = proxy_config_dir / name
        path.write_text(content)
        return path

    return _write
