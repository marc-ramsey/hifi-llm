"""Tests for ProcessManager — subprocess lifecycle management."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from proxy.process_manager import ProcessManager


# ────────────────────────────────────────────────────────────────────────
# Mock server — a real process that can be configured to handle or ignore
# SIGTERM, and serves a minimal HTTP endpoint.
# ────────────────────────────────────────────────────────────────────────

def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


class _MockServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass  # silence logs


def _mock_server_script(port: int, ignore_sigterm: bool = False) -> str:
    """Return source code for a mock server subprocess.

    The script starts an HTTP server on *port*.  If *ignore_sigterm* is
    True it installs a no-op SIGTERM handler so the process survives the
    initial signal and requires SIGKILL to terminate.
    """
    return (
        f"import http.server, threading, signal, sys\n"
        f"\n"
        f"port = {port}\n"
        f"ignore_sigterm = {ignore_sigterm!r}\n"
        f"\n"
        f"if ignore_sigterm:\n"
        f"    signal.signal(signal.SIGTERM, lambda *a: None)\n"
        f"\n"
        f"srv = http.server.HTTPServer(('127.0.0.1', port), http.server.BaseHTTPRequestHandler)\n"
        f"t = threading.Thread(target=srv.serve_forever, daemon=True)\n"
        f"t.start()\n"
        f"srv.serve_forever()\n"
    )


def _make_mock_server(cmd: list[str]) -> subprocess.Popen:
    """Start a mock server process via the Python interpreter."""
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_process_manager():
    asyncio.run(ProcessManager.shutdown())
    yield
    asyncio.run(ProcessManager.shutdown())


# ────────────────────────────────────────────────────────────────────────
# Unit tests
# ────────────────────────────────────────────────────────────────────────

class TestProcessManager:
    """Unit tests for the process manager singleton."""

    def test_is_running_returns_false_for_unknown(self):
        assert ProcessManager.is_running("nonexistent") is False

    def test_start_and_is_running(self):
        cmd = ["python3", "-c", "import time; time.sleep(60)"]
        result = ProcessManager.start("test-proc", cmd)
        assert result is True
        assert ProcessManager.is_running("test-proc") is True
        ProcessManager.stop("test-proc")

    def test_stop_removes_from_registry(self):
        cmd = ["python3", "-c", "import time; time.sleep(60)"]
        ProcessManager.start("test-proc2", cmd)
        assert ProcessManager.is_running("test-proc2") is True
        ProcessManager.stop("test-proc2")
        assert ProcessManager.is_running("test-proc2") is False

    def test_ensure_running_starts_if_not_running(self):
        cmd = ["python3", "-c", "import time; time.sleep(60)"]
        result = ProcessManager.ensure_running("test-proc3", cmd, cooldown=0.1)
        assert result is True
        assert ProcessManager.is_running("test-proc3") is True
        ProcessManager.stop("test-proc3")

    def test_ensure_running_skips_if_already_running(self):
        cmd = ["python3", "-c", "import time; time.sleep(60)"]
        ProcessManager.start("test-proc4", cmd)
        result = ProcessManager.ensure_running("test-proc4", cmd, cooldown=1.0)
        assert result is True  # returns True because it's running
        ProcessManager.stop("test-proc4")

    def test_ensure_running_cooldown(self):
        """After a failed start, subsequent calls within cooldown return False."""
        cmd = ["/nonexistent/binary/that/does/not/exist"]
        result1 = ProcessManager.ensure_running("test-proc5", cmd, cooldown=2.0)
        assert result1 is False  # failed to start
        result2 = ProcessManager.ensure_running("test-proc5", cmd, cooldown=2.0)
        assert result2 is False  # still in cooldown
        ProcessManager.stop("test-proc5")

    def test_shutdown_kills_all(self):
        """Multiple processes are all terminated on shutdown."""
        cmd = ["python3", "-c", "import time; time.sleep(60)"]
        ProcessManager.start("proc-a", cmd)
        ProcessManager.start("proc-b", cmd)
        assert ProcessManager.is_running("proc-a") is True
        assert ProcessManager.is_running("proc-b") is True
        asyncio.run(ProcessManager.shutdown())
        assert ProcessManager.is_running("proc-a") is False
        assert ProcessManager.is_running("proc-b") is False


class TestShutdownSIGTERM:
    """Tests for SIGTERM→SIGKILL escalation during shutdown."""

    def test_graceful_sigterm_exit(self):
        """A process that handles SIGTERM exits within the timeout."""
        port = _find_free_port()
        script = _mock_server_script(port, ignore_sigterm=False)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            cmd = [sys.executable, script_path]
            ProcessManager.start("graceful-proc", cmd)
            assert ProcessManager.is_running("graceful-proc") is True

            # Shutdown should send SIGTERM and the process exits cleanly.
            asyncio.run(ProcessManager.shutdown())
            assert ProcessManager.is_running("graceful-proc") is False
        finally:
            os.unlink(script_path)

    def test_sigterm_escalates_to_sigkill(self):
        """A process that ignores SIGTERM is killed with SIGKILL after timeout."""
        port = _find_free_port()
        script = _mock_server_script(port, ignore_sigterm=True)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            cmd = [sys.executable, script_path]
            ProcessManager.start("stubborn-proc", cmd)
            assert ProcessManager.is_running("stubborn-proc") is True

            # Shutdown sends SIGTERM (ignored), then SIGKILL after 20s.
            # We reduce the kill timeout for the test by patching the constant.
            import proxy.process_manager as pm
            original_timeout = pm._KILL_TIMEOUT_S
            pm._KILL_TIMEOUT_S = 1.0  # 1 second instead of 20

            try:
                asyncio.run(ProcessManager.shutdown())
            finally:
                pm._KILL_TIMEOUT_S = original_timeout

            assert ProcessManager.is_running("stubborn-proc") is False
        finally:
            os.unlink(script_path)

    def test_shutdown_parallel(self):
        """Multiple stubborn processes are killed in parallel, not serially."""
        port1 = _find_free_port()
        port2 = _find_free_port()
        script1 = _mock_server_script(port1, ignore_sigterm=True)
        script2 = _mock_server_script(port2, ignore_sigterm=True)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f1:
            f1.write(script1)
            path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f2:
            f2.write(script2)
            path2 = f2.name

        try:
            cmd1 = [sys.executable, path1]
            cmd2 = [sys.executable, path2]
            ProcessManager.start("stubborn-a", cmd1)
            ProcessManager.start("stubborn-b", cmd2)
            assert ProcessManager.is_running("stubborn-a") is True
            assert ProcessManager.is_running("stubborn-b") is True

            import proxy.process_manager as pm
            original_timeout = pm._KILL_TIMEOUT_S
            pm._KILL_TIMEOUT_S = 1.0

            start = time.monotonic()
            try:
                asyncio.run(ProcessManager.shutdown())
            finally:
                pm._KILL_TIMEOUT_S = original_timeout

            elapsed = time.monotonic() - start
            # If serial, would take ~2s (1s timeout × 2).
            # Parallel should be ~1s. Use 1.5s as margin.
            assert elapsed < 1.5, f"Shutdown took {elapsed:.1f}s — processes killed serially?"

            assert ProcessManager.is_running("stubborn-a") is False
            assert ProcessManager.is_running("stubborn-b") is False
        finally:
            os.unlink(path1)
            os.unlink(path2)
