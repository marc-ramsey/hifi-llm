"""E2E tests for /metrics endpoint, rate-limiting middleware, and request IDs."""

from __future__ import annotations

import uuid

import requests


class TestMetricsEndpoint:
    def test_metrics_returns_text_plain(self, client):
        """GET /metrics should return text/plain content type."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["Content-Type"]

    def test_metrics_contains_help_lines(self, client):
        """Metrics output should contain HELP and TYPE declarations."""
        resp = client.get("/metrics")
        text = resp.text
        assert "# HELP proxy_requests_total" in text
        assert "# TYPE proxy_requests_total counter" in text

    def test_metrics_after_requests(self, client):
        """After making requests, counters should reflect them."""
        # Make a few requests first
        client.get("/health")
        client.get("/v1/models")
        client.post("/v1/chat/completions", json={
            "model": "gemma-4-26B-instruct",
            "messages": [{"role": "user", "content": "hi"}],
        })

        resp = client.get("/metrics")
        text = resp.text

        # Should have request count entries for paths hit
        assert 'path="/v1/models"' in text or 'path="/v1/chat/completions"' in text
        # Should have a sum and count for duration histogram
        assert "proxy_request_duration_seconds_count" in text
        assert "proxy_request_duration_seconds_sum" in text

    def test_metrics_includes_rate_limit_section(self, client):
        """Metrics should always include the rate-limit section header."""
        resp = client.get("/metrics")
        text = resp.text
        assert "# HELP proxy_rate_limits_total" in text
        assert "# TYPE proxy_rate_limits_total counter"


class TestRequestID:
    def test_x_request_id_in_response(self, client):
        """Every response should include X-Request-ID header."""
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers
        uuid.UUID(resp.headers["X-Request-ID"])

    def test_x_request_id_different_per_request(self, client):
        """Each request should get a unique X-Request-ID."""
        ids = set()
        for _ in range(5):
            resp = client.get("/health")
            ids.add(resp.headers["X-Request-ID"])
        assert len(ids) == 5

    def test_client_supplied_id_preserved(self, proxy):
        """If client sends X-Request-ID, it should be echoed back."""
        port, _ = proxy
        custom_id = "my-custom-trace-id-12345"
        resp = requests.get(
            f"http://127.0.0.1:{port}/health",
            headers={"X-Request-ID": custom_id},
        )
        assert resp.status_code == 200
        assert resp.headers["X-Request-ID"] == custom_id
