"""Fixtures for proxy e2e tests.

Starts the proxy as a subprocess pointing at Arkestra (on :8080).
Config is written to a temp dir so each test can use a custom config.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
ARKESTRA_URL = "http://127.0.0.1:8080"

# ────────────────────────────────────────────────────────────────────
# Default test config (YAML string)
# ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = f"""
listen:
  host: 0.0.0.0
  port: {{port}}

auth:
  api_key: null

models:
  - name: "gemma-4-26B-instruct"
    url: "{ARKESTRA_URL}"
    default_params:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256

  - name: "qwen3.6-35B-instruct"
    url: "{ARKESTRA_URL}"
    default_params:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 256

plugins_dir: null
"""


@pytest.fixture(scope="session")
def proxy_config_dir():
    """Create a temp directory for test configs. All proxy instances share this."""
    d = tempfile.mkdtemp(prefix="llm-proxy-test-")
    yield Path(d)


@pytest.fixture(scope="session")
def proxy(proxy_config_dir, tmp_path_factory):
    """Start the proxy as a subprocess, teardown after all tests.

    Uses a random port to avoid clashes.
    """
    port = 19081  # fixed for simplicity; bump if occupied

    config_path = proxy_config_dir / "proxy-config.yaml"
    config_path.write_text(DEFAULT_CONFIG.format(port=port))

    env = os.environ.copy()
    env["LLM_PROXY_TIMEOUT_MS"] = "60000"

    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "main.py"), "--config", str(config_path)],
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
    s.base_url = base

    def _request(method, path, **kwargs):
        url = base + path
        return s.request(method, url, **kwargs)

    s.get = lambda path, **kw: _request("GET", path, **kw)
    s.post = lambda path, **kw: _request("POST", path, **kw)

    return s


@pytest.fixture
def write_config(proxy_config_dir):
    r'''Write a YAML config file and return its path.

    Usage:
        cfg = write_config("""
        listen:
          host: 0.0.0.0
          port: 19081
        models:
          - name: "test-model"
            url: "..."
        """)
        # then send SIGHUP or restart proxy
    '''
    def _write(content: str, name: str = "proxy-config.yaml") -> Path:
        path = proxy_config_dir / name
        path.write_text(content)
        return path

    return _write
