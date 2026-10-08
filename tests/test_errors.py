"""E2E tests for error handling.

Verifies that backend errors and invalid requests produce proper
OpenAI-compatible error responses.
"""

from __future__ import annotations

import requests


def _is_openai_error(data):
    """Check that an error response follows the OpenAI error shape."""
    assert "error" in data or "detail" in data


class TestInvalidRequests:
    """Client-side errors (4xx)."""

    def test_invalid_model_name(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        data = resp.json()
        assert data["detail"]["type"] == "invalid_request_error"
        assert "nonexistent-model" in data["detail"]["message"]
        assert data["detail"]["param"] is None

    def test_invalid_model_for_embeddings(self, client):
        resp = client.post("/v1/embeddings", json={
            "model": "nonexistent-model",
            "input": "test",
        })
        assert resp.status_code == 400
        data = resp.json()
        assert "nonexistent-model" in data["detail"]["message"]

    def test_invalid_model_for_completions(self, client):
        resp = client.post("/v1/completions", json={
            "model": "nonexistent-model",
            "prompt": "test",
        })
        assert resp.status_code == 400
        data = resp.json()
        assert "nonexistent-model" in data["detail"]["message"]

    def test_missing_messages(self, client):
        """Missing 'messages' field should pass through to backend."""
        resp = client.post("/v1/chat/completions", json={
            "model": "gemma-4-26B-instruct",
            # no messages
        })
        # Backend may return 400 or 500 — proxy should not crash
        assert resp.status_code in (400, 500, 502)
        if resp.status_code == 502:
            data = resp.json()
            assert data["error"]["type"] == "backend_error"


class TestBackendErrors:
    """Server-side errors (5xx from backend)."""

    def test_backend_404_becomes_502(self, client):
        """Proxy /v1/completions → Arkestra returns 404 → proxy returns 502."""
        resp = client.post("/v1/completions", json={
            "model": "gemma-4-26B-instruct",
            "prompt": "test",
        })
        assert resp.status_code == 502
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "backend_error"
        assert data["error"]["code"] == 404

    def test_502_error_has_message(self, client):
        """The error message should be human-readable."""
        resp = client.post("/v1/completions", json={
            "model": "gemma-4-26B-instruct",
            "prompt": "test",
        })
        assert resp.status_code == 502
        data = resp.json()
        msg = data["error"]["message"]
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_streaming_backend_error(self, client):
        """Streaming should not crash when backend errors mid-stream."""
        # Use an endpoint that returns 404 from the backend
        resp = client.post("/v1/completions", json={
            "model": "gemma-4-26B-instruct",
            "prompt": "test",
            "stream": True,
        })
        # For streaming, the error may arrive as an SSE data line
        # or the connection may be closed. Either is acceptable.
        assert resp.status_code in (200, 502, 503)
