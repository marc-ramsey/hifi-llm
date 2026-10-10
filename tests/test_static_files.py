"""E2E tests for static file serving."""

from __future__ import annotations

import requests


class TestStaticFiles:
    """Test static file serving via config.static_files[]."""

    def test_file_served(self, proxy_with_static):
        """A file in the static directory should be served at the configured path."""
        port, _ = proxy_with_static

        resp = requests.get(f"http://127.0.0.1:{port}/docs/hello.txt")
        assert resp.status_code == 200
        assert resp.text.strip() == "Hello from static files!"

    def test_file_not_found(self, proxy_with_static):
        """A missing file should return 404."""
        port, _ = proxy_with_static

        resp = requests.get(f"http://127.0.0.1:{port}/docs/nonexistent.txt")
        assert resp.status_code == 404

    def test_path_prefix_serves_file(self, proxy_with_static):
        """Files at the configured path prefix should be accessible."""
        port, _ = proxy_with_static

        # The fixture serves from ./tests/fixtures/static_docs at /docs
        resp = requests.get(f"http://127.0.0.1:{port}/docs/README.md")
        assert resp.status_code == 200
        assert "# Static Docs" in resp.text

    def test_different_prefix(self, proxy_with_static):
        """A second static mount at a different prefix should also work."""
        port, _ = proxy_with_static

        resp = requests.get(f"http://127.0.0.1:{port}/assets/logo.png")
        assert resp.status_code == 200
        # Binary-ish content — just check it's not empty
        assert len(resp.content) > 0

    def test_static_does_not_interfere_with_routes(self, proxy_with_static):
        """Static file serving should not interfere with API routes."""
        port, _ = proxy_with_static

        resp = requests.get(f"http://127.0.0.1:{port}/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"

    def test_health_unchanged(self, proxy_with_static):
        """Health endpoint should work normally with static files enabled."""
        port, _ = proxy_with_static

        resp = requests.get(f"http://127.0.0.1:{port}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestStaticFilesInvalidConfig:
    """Test that invalid static file configs are handled gracefully."""

    def test_nonexistent_directory_skipped(self, proxy_config_dir, backend_url):
        """A config referencing a non-existent directory is skipped with a warning; proxy still starts."""
        from tests.conftest import _find_free_port, _start_proxy, ROOT

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
    default_params: {{}}

static_files:
  - path: "/docs"
    directories:
      - /nonexistent/path/that/does/not/exist
"""
        )

        proc = _start_proxy(port, config_path)

        # Verify the proxy is functional
        resp = requests.get(f"http://127.0.0.1:{port}/v1/models")
        assert resp.status_code == 200

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
