"""Tests for ProcessManager — subprocess lifecycle management."""

from __future__ import annotations

import asyncio
import time

import pytest

from proxy.process_manager import ProcessManager


class TestProcessManager:
    """Unit tests for the process manager singleton."""

    def setup_method(self):
        """Reset state before each test."""
        asyncio.run(ProcessManager.shutdown())

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
        # Use a nonexistent binary — start will fail
        cmd = ["/nonexistent/binary/that/does/not/exist"]
        result1 = ProcessManager.ensure_running("test-proc5", cmd, cooldown=2.0)
        assert result1 is False  # failed to start
        # Immediately retry — should be in cooldown
        result2 = ProcessManager.ensure_running("test-proc5", cmd, cooldown=2.0)
        assert result2 is False  # still in cooldown
        ProcessManager.stop("test-proc5")

    def test_shutdown_kills_all(self):
        cmd = ["python3", "-c", "import time; time.sleep(60)"]
        ProcessManager.start("proc-a", cmd)
        ProcessManager.start("proc-b", cmd)
        assert ProcessManager.is_running("proc-a") is True
        assert ProcessManager.is_running("proc-b") is True
        asyncio.run(ProcessManager.shutdown())
        assert ProcessManager.is_running("proc-a") is False
        assert ProcessManager.is_running("proc-b") is False
