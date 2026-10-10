"""Tests for ProcessManager — subprocess lifecycle management."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from .conftest import find_free_port
from proxy.process_manager import ProcessManager


# ────────────────────────────────────────────────────────────────────────
# Mock server — a real process that can be configured to handle or ignore
# SIGTERM, and serves a minimal HTTP endpoint.
# ────────────────────────────────────────────────────────────────────────

class _MockServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass  # silence logs


def _mock_server_script(
    port: int,
    ignore_sigterm: bool = False,
    crash_after: float | None = None,
) -> str:
    """Return source code for a mock server subprocess.

    The script starts an HTTP server on *port*.  If *ignore_sigterm* is
    True it installs a no-op SIGTERM handler so the process survives the
    initial signal and requires SIGKILL to terminate.  If *crash_after*
    is set, the process sleeps that many seconds then exits with code 1
    (simulating an OOM crash or segfault).
    """
    lines = [
        f"import http.server, threading, signal, sys, time",
        "",
        f"port = {port}",
        f"ignore_sigterm = {ignore_sigterm!r}",
        f"crash_after = {crash_after!r}",
        "",
    ]
    if ignore_sigterm:
        lines.append("signal.signal(signal.SIGTERM, lambda *a: None)")
    lines.extend([
        "",
        "srv = http.server.HTTPServer(('127.0.0.1', port), http.server.BaseHTTPRequestHandler)",
        "t = threading.Thread(target=srv.serve_forever, daemon=True)",
        "t.start()",
    ])
    if crash_after is not None:
        lines.append(f"time.sleep({crash_after})")
        lines.append("sys.exit(1)")  # simulate crash
    else:
        lines.append("srv.serve_forever()")
    return "\n".join(lines) + "\n"


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
        port = find_free_port()
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
        port = find_free_port()
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
        port1 = find_free_port()
        port2 = find_free_port()
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


class TestAutoRestart:
    """Tests for process auto-restart after server failure."""

    def test_ensure_restarts_dead_process(self):
        """When a managed process dies, ensure_running relaunches it."""
        port = find_free_port()
        # Server crashes after 0.5s (simulates OOM / segfault)
        script = _mock_server_script(port, crash_after=0.5)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            cmd = [sys.executable, script_path]

            # Start the process and let it die.
            ProcessManager.start("crash-proc", cmd)
            assert ProcessManager.is_running("crash-proc") is True

            # Wait for it to crash.
            time.sleep(1.0)
            assert ProcessManager.is_running("crash-proc") is False

            # ensure_running should detect it's dead and restart it.
            result = ProcessManager.ensure_running(
                "crash-proc", cmd, cooldown=0.1,
            )
            assert result is True
            assert ProcessManager.is_running("crash-proc") is True
        finally:
            os.unlink(script_path)
            # Clean up the restarted process.
            asyncio.run(ProcessManager.shutdown())

    def test_restart_limit_prevents_self_dos(self):
        """After too many crashes in a short window, stop restarting."""
        import proxy.process_manager as pm
        original_window = pm._RESTART_WINDOW_S
        original_max = pm._MAX_RESTARTS

        # Use tight values for the test.
        pm._RESTART_WINDOW_S = 5.0
        pm._MAX_RESTARTS = 3

        port = find_free_port()
        script = _mock_server_script(port, crash_after=0.2)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            cmd = [sys.executable, script_path]

            # Crash 3 times (hits the limit).
            for i in range(3):
                ProcessManager.start("limit-proc", cmd)
                time.sleep(0.5)  # let it crash
                assert ProcessManager.is_running("limit-proc") is False
                result = ProcessManager.ensure_running("limit-proc", cmd, cooldown=0.1)
                assert result is True, f"Restart {i+1} should succeed"

            # 4th attempt — should be blocked by restart limit.
            time.sleep(0.5)  # let the last one crash
            result = ProcessManager.ensure_running("limit-proc", cmd, cooldown=0.1)
            assert result is False, "Should not restart after hitting limit"

            # No new process was spawned — registry still has the dead entry.
            assert ProcessManager.is_running("limit-proc") is False
        finally:
            pm._RESTART_WINDOW_S = original_window
            pm._MAX_RESTARTS = original_max
            os.unlink(script_path)
            asyncio.run(ProcessManager.shutdown())
