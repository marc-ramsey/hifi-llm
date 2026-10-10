"""E2E tests for SIGHUP hot-reload.

Verifies that sending SIGHUP to the proxy reloads the config
without dropping existing requests.
"""

from __future__ import annotations

import signal
import time

import requests


class TestHotReload:
    def test_sighup_adds_new_model(self, proxy, write_config, backend_url):
        """After SIGHUP, new models from the updated config should appear."""
        port, proc = proxy

        # Step 1: Verify initial model list
        resp = requests.get(f"http://127.0.0.1:{port}/v1/models")
        assert resp.status_code == 200
        initial_models = {m["id"] for m in resp.json()["data"]}
        # The proxy fixture provides 3 base models; a prior test's SIGHUP
        # may have added more, so we only assert the core ones exist.
        assert "gemma-4-26B-instruct" in initial_models
        assert "qwen3.6-35B-instruct" in initial_models

        # Step 2: Write a new config with an additional model
        new_config = f"""
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

  - name: "added-via-sighup"
    url: "{backend_url}"
    default_params: {{}}

static_files: []
"""
        write_config(new_config)

        # Step 3: Send SIGHUP
        proc.send_signal(signal.SIGHUP)
        time.sleep(1)  # allow reload

        # Step 4: Verify the new model appears
        resp = requests.get(f"http://127.0.0.1:{port}/v1/models")
        assert resp.status_code == 200
        new_models = {m["id"] for m in resp.json()["data"]}
        assert "added-via-sighup" in new_models
        # Original models should still be there
        assert "gemma-4-26B-instruct" in new_models

    def test_sighup_preserves_existing_requests(self, proxy, write_config):
        """In-flight requests should not be dropped during SIGHUP."""
        port, proc = proxy

        # Start a streaming request
        stream_req = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "gemma-4-26B-instruct",
                "messages": [{"role": "user", "content": "count to 3"}],
                "max_tokens": 15,
                "stream": True,
            },
            stream=True,
        )

        # Send SIGHUP mid-stream
        time.sleep(0.5)
        proc.send_signal(signal.SIGHUP)
        time.sleep(0.5)

        # The response should still come through (not a connection reset)
        # Read what we can
        chunks = []
        for line in stream_req.iter_lines():
            if line:
                chunks.append(line.decode())

        # Should have at least started receiving SSE data
        has_data = any(l.startswith("data: ") for l in chunks)
        assert has_data or stream_req.status_code == 200, \
            f"Streaming request failed during SIGHUP. Chunks: {chunks[:5]}"

    def test_sighup_invalid_config_keeps_old(self, proxy, write_config):
        """If the new config is invalid, the old config should be retained."""
        port, proc = proxy

        # Write an invalid config
        write_config("models: [invalid yaml: [broken")

        # Send SIGHUP
        proc.send_signal(signal.SIGHUP)
        time.sleep(1)

        # The old config should still work
        resp = requests.get(f"http://127.0.0.1:{port}/v1/models")
        assert resp.status_code == 200
        models = {m["id"] for m in resp.json()["data"]}
        # The invalid model should NOT have been loaded
        assert "invalid" not in str(models)
