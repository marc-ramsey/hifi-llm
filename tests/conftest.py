"""Fixtures for proxy E2E tests.

Starts two real subprocesses in the test session:
  - A lightweight threaded HTTP server that mimics an OpenAI-compatible backend
  - The hifi proxy pointing at that backend

This removes all dependency on external services (Arkestra, port :8080).
"""

from __future__ import annotations

import json
import socket
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

import config as config_registry

ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────────
# Shared test utilities
# ────────────────────────────────────────────────────────────────────────

def find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def parse_sse_chunks(raw_text: str):
    """Parse SSE data lines into a list of dicts and a done-flag.

    Returns ``(chunks, done)`` where *done* is True when a ``[DONE]``
    sentinel was encountered.
    """
    chunks: list[dict] = []
    for line in raw_text.replace("\r", "").split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:].strip()
            if data == "[DONE]":
                return chunks, True
            try:
                chunks.append(json.loads(data))
            except ValueError:
                pass
    return chunks, False


# ────────────────────────────────────────────────────────────────────────
# Lightweight backend server (real TCP socket, not a mock)
# ────────────────────────────────────────────────────────────────────────

class _BackendHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible backend that returns deterministic responses."""

    # Upstream oracle: every body the proxy forwarded, in order.
    #
    # The suite used to assert only on what the proxy returned, and a working
    # pipe always returns something — so `default_params` shipped as a silent
    # no-op behind two green tests that claimed to cover it.  Anything about
    # *what the proxy sends* needs to be observable here.
    forwarded: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        type(self).forwarded.append({"path": self.path, "body": body})

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


def _start_proxy(port: int, config_path: Path, cwd: Path = ROOT,
                 env_extra: dict[str, str] | None = None,
                 log_path: str | None = None) -> subprocess.Popen:
    """Start the hifi proxy as a subprocess and wait for it to be ready.

    *env_extra* overrides the environment (e.g. ``LLM_PROXY_TIMEOUT_MS``) for
    tests that need a specific timeout policy.

    Returns the ``Popen`` instance.  On failure the process is killed and
    ``pytest.fail`` is called with stdout so the error is visible in test output.
    """
    env = os.environ.copy()
    env.setdefault("LLM_PROXY_TIMEOUT_MS", "60000")
    env.update(env_extra or {})

    venv_python = cwd / ".venv" / "bin" / "python3"
    python_bin = str(venv_python) if venv_python.exists() else sys.executable

    # log_path keeps the proxy's stdout around for post-mortem debugging.
    log_file = open(log_path, "w") if log_path else subprocess.PIPE

    proc = subprocess.Popen(
        [python_bin, str(cwd / "main.py"), "--config", str(config_path)],
        cwd=str(cwd),
        env=env,
        stdout=log_file,
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
        if log_path:
            detail = Path(log_path).read_text()
        else:
            detail, _ = proc.communicate(timeout=5)
        pytest.fail(f"Proxy failed to start.\nstdout:\n{detail}")

    return proc


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def backend():
    """Start a real threaded HTTP server as an OpenAI-compatible backend.

    Yields (host, port).  Teardown after the session.
    """
    port = find_free_port()
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
    port = find_free_port()

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

  - name: "nomic-embed"
    url: "{backend_url}"
    default_params: {{}}

static_files: []
"""
    )

    proc = _start_proxy(port, config_path)

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
def forwarded():
    """The upstream oracle: what the proxy actually sent, not what it returned.

    Gives a callable returning ``[{"path": …, "body": …}, …]`` recorded since
    the fixture ran, with ``.clear()`` to restart recording.  Use it for any
    claim about forwarding: the proxy always answers *something*, so asserting
    on the response alone cannot detect a body that was mangled, defaulted, or
    dropped on the way out.
    """
    _BackendHandler.forwarded.clear()

    def _got() -> list[dict]:
        return list(_BackendHandler.forwarded)

    _got.clear = lambda: _BackendHandler.forwarded.clear()
    return _got


@pytest.fixture(scope="session")
def proxy_with_static(proxy_config_dir, backend_url):
    """Start the hifi proxy with static file serving enabled.

    Serves ./tests/fixtures/static_docs at /docs and
    ./tests/fixtures/assets at /assets.

    Yields (port, proc).  Teardown after the session.
    """
    port = find_free_port()

    config_path = proxy_config_dir / "proxy-config-static.yaml"
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

static_files:
  - path: "/docs"
    directories:
      - "{ROOT}/tests/fixtures/static_docs"
  - path: "/assets"
    directories:
      - "{ROOT}/tests/fixtures/assets"
"""
    )

    proc = _start_proxy(port, config_path)

    yield port, proc

    # Teardown
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture
def proxy_factory(proxy_config_dir):
    """Start throwaway proxies from config text, for config-specific tests.

    The template's ``{port}`` placeholder becomes the chosen free port and any
    other ``{name}`` placeholder is replaced from *vars*::

        proc, base = proxy_factory(CONFIG, backend_url=backend_url)

    Every proxy started through the factory is terminated at teardown.
    """
    procs: list[subprocess.Popen] = []

    def _start(config_text: str, name: str | None = None, env: dict[str, str] | None = None, **vars_):
        port = find_free_port()
        text = config_text
        for key, value in {"port": port, **vars_}.items():
            text = text.replace("{" + key + "}", str(value))
        path = proxy_config_dir / (name or f"throwaway-{port}.yaml")
        path.write_text(text)
        proc = _start_proxy(port, path, env_extra=env)
        procs.append(proc)
        return proc, f"http://127.0.0.1:{port}"

    yield _start

    for proc in procs:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture
def restore_config():
    """Restore the global config registry after an in-process test.

    Tests that call ``config.set()`` directly would otherwise leak their
    config into later tests in the same process.
    """
    saved = config_registry._current
    try:
        yield
    finally:
        config_registry._current = saved


@pytest.fixture
def write_config(proxy_config_dir):
    """Write a YAML config file and return its path."""

    def _write(content: str, name: str = "proxy-config.yaml") -> Path:
        path = proxy_config_dir / name
        path.write_text(content)
        return path

    return _write
