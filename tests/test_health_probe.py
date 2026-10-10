"""E2E tests for health probe behavior.

Verifies that periodic health probes run, update the /health/models endpoint,
and correctly restart after SIGHUP config reload (Fix #1).
"""

from __future__ import annotations

import json
import signal
import time

import requests


class TestHealthProbe:
    def test_health_endpoint_includes_models(self, proxy):
        """The /health endpoint should report backend health for configured models."""
        port, _ = proxy

        # Wait a couple probe cycles (interval is 2s by default)
        time.sleep(5)

        resp = requests.get(f"http://127.0.0.1:{port}/health?models=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        # The proxy fixture has 3 models, all should appear
        assert len(data["models"]) == 3
        for model_name in ["gemma-4-26B-instruct", "qwen3.6-35B-instruct", "nomic-embed"]:
            assert model_name in data["models"], f"Model {model_name} missing from health report"

    def test_health_probe_reports_status(self, proxy):
        """Health probes should report healthy status for working backends."""
        port, _ = proxy

        time.sleep(5)

        resp = requests.get(f"http://127.0.0.1:{port}/health?models=true")
        data = resp.json()
        for model_name, info in data["models"].items():
            assert info["status"] == "healthy", \
                f"Expected {model_name} to be healthy, got {info['status']}"

    def test_health_probe_restarts_after_sighup(self, proxy, write_config, backend_url):
        """After SIGHUP, the health probe task should restart with the new model list.

        This is the regression test for the bug where restart_health_probe()
        cancelled the old task but never created a new one, causing probes to
        silently stop after config reload.
        """
        port, proc = proxy

        # Step 1: Wait for initial probes to run and capture baseline
        time.sleep(5)

        resp = requests.get(f"http://127.0.0.1:{port}/health?models=true")
        assert resp.status_code == 200
        initial_models = set(resp.json()["models"].keys())
        assert len(initial_models) >= 3, f"Expected at least 3 models in health, got {initial_models}"

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
    default_params: {{}}

  - name: "qwen3.6-35B-instruct"
    url: "{backend_url}"
    default_params: {{}}

  - name: "nomic-embed"
    url: "{backend_url}"
    default_params: {{}}

  - name: "added-via-sighup"
    url: "{backend_url}"
    default_params: {{}}

static_files: []
"""
        write_config(new_config)

        # Step 3: Send SIGHUP to trigger reload
        proc.send_signal(signal.SIGHUP)
        time.sleep(1)

        # Step 4: Wait for the restarted probe cycle (up to 5s for interval + latency)
        time.sleep(5)

        # Step 5: Verify the new model appears in health report
        resp = requests.get(f"http://127.0.0.1:{port}/health?models=true")
        assert resp.status_code == 200
        reload_models = set(resp.json()["models"].keys())

        # The new model should be present — this would fail if probes stopped after SIGHUP
        assert "added-via-sighup" in reload_models, \
            f"New model 'added-via-sighup' not found in health report after SIGHUP. " \
            f"Models: {reload_models}. Health probe likely did not restart."

        # Original models should still be there too
        assert initial_models <= reload_models, \
            f"Original models missing after reload. Was: {initial_models}, Now: {reload_models}"
