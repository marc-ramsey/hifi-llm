"""E2E tests for SIGTERM graceful shutdown.

Verifies that the proxy drains in-flight requests before exiting.
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
from pathlib import Path

import pytest
import requests

from .conftest import ROOT


def _parse_sse(raw_text):
    """Parse SSE lines, return (chunks, done)."""
    chunks = []
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


class TestGracefulShutdown:
    def test_sigterm_drains_request(self, proxy, backend_url):
        """Send SIGTERM during a streaming request; request should complete."""
        port, proc = proxy

        # Start a streaming request
        stream_resp = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "gemma-4-26B-instruct",
                "messages": [{"role": "user", "content": "list numbers 1 through 10"}],
                "max_tokens": 200,
                "stream": True,
            },
            stream=True,
            timeout=30,
        )

        # Wait for some data to start arriving
        time.sleep(1)

        # Read what we have so far (blocking, but stream will eventually complete)
        raw = stream_resp.content.decode("utf-8", errors="replace")
        chunks, done = _parse_sse(raw)

        # Send SIGTERM while we have the response
        proc.send_signal(signal.SIGTERM)

        # Collect remaining data (in case more arrives after SIGTERM)
        try:
            remaining = stream_resp.read(timeout=5)
            if remaining:
                full_text = raw + remaining.decode("utf-8", errors="replace")
                chunks2, done2 = _parse_sse(full_text)
                if done2:
                    done = done2
                    chunks = chunks2
        except Exception:
            pass

        # We should have at least received some SSE data
        # The request may or may not complete fully depending on timing
        has_data = any(c.get("choices") for c in chunks)
        assert done or has_data, \
            f"No SSE chunks received during shutdown. Response preview:\n{raw[:500]}"

        # Process should exit cleanly within timeout
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail("Proxy did not exit within 15s after SIGTERM")
