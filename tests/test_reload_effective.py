"""Regression tests: SIGHUP reload must actually rebuild the ASGI app.

Bug being guarded against: ``create_app()`` dereferenced its ``config``
argument for ``static_files`` while every other read fell back to the global
registry.  ``ConfigReloadableApp.reload()`` calls ``create_app()`` with no
argument, so *every* SIGHUP raised ``AttributeError: 'NoneType' object has no
attribute 'static_files'`` and the app was never swapped.

The failure was easy to miss because anything reading ``config.get()`` per
request (``/v1/models``, auth) still appeared to reload, while everything
captured at app-build time (rate limits, CORS, static mounts) was silently
ignored — and ``test_hot_reload.py`` only asserted on the model list.
"""

from __future__ import annotations

import signal
import subprocess
import time

import requests

import config as config_registry
from config.schema import ProxyConfig, RateLimitConfig, StaticFileConfig
from proxy.app import create_app
from proxy.reloadable import ConfigReloadableApp

from .conftest import ROOT, _start_proxy, find_free_port

FIXTURE_DOCS = ROOT / "tests" / "fixtures" / "static_docs"


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def _middleware_kwargs(app, cls_name: str) -> dict:
    """Return the kwargs registered for the middleware named *cls_name*."""
    for mw in app.user_middleware:
        if mw.cls.__name__ == cls_name:
            return mw.kwargs or {}
    raise AssertionError(f"{cls_name} not registered on app")


def _mount_paths(app) -> list[str]:
    """URL prefixes of every Starlette Mount on *app*."""
    return [r.path.rstrip("/") for r in app.routes if type(r).__name__ == "Mount"]


def _rate_limit_enabled(app) -> bool:
    return _middleware_kwargs(app, "RateLimitMiddleware")["config"].enabled


# ────────────────────────────────────────────────────────────────────────
# Unit tests — in-process
# ────────────────────────────────────────────────────────────────────────

class TestCreateAppWithoutArgument:
    """create_app() must fall back to the global registry, not crash."""

    def test_create_app_no_args_uses_global_registry(self, restore_config):
        config_registry.set(ProxyConfig(
            static_files=[StaticFileConfig(path="/files", directories=[str(FIXTURE_DOCS)])],
        ))

        app = create_app()  # used to raise AttributeError

        assert "/files" in _mount_paths(app)

    def test_config_argument_takes_precedence(self, restore_config):
        config_registry.set(ProxyConfig())

        arg_config = ProxyConfig(
            static_files=[StaticFileConfig(path="/from-arg", directories=[str(FIXTURE_DOCS)])],
        )
        assert "/from-arg" in _mount_paths(create_app(arg_config))
        assert "/from-arg" not in _mount_paths(create_app())


class TestReloadSwapsApp:
    """reload() must build a fresh app from the *newly loaded* config."""

    def test_reload_reapplies_build_time_config(self, restore_config):
        off = ProxyConfig(rate_limit=RateLimitConfig(enabled=False))
        config_registry.set(off)

        wrapper = ConfigReloadableApp(initial_config=off)
        wrapper.reload()
        assert _rate_limit_enabled(wrapper._app) is False
        first_app = wrapper._app

        on = ProxyConfig(rate_limit=RateLimitConfig(enabled=True, requests_per_minute=5))
        config_registry.set(on)
        wrapper.reload()

        assert wrapper._app is not first_app, "reload() did not swap the ASGI app"
        assert _rate_limit_enabled(wrapper._app) is True
        assert wrapper._config is on

    def test_reload_picks_up_static_mounts(self, restore_config):
        config_registry.set(ProxyConfig())
        wrapper = ConfigReloadableApp(initial_config=ProxyConfig())
        wrapper.reload()
        assert "/files" not in _mount_paths(wrapper._app)

        config_registry.set(ProxyConfig(
            static_files=[StaticFileConfig(path="/files", directories=[str(FIXTURE_DOCS)])],
        ))
        wrapper.reload()
        assert "/files" in _mount_paths(wrapper._app)


# ────────────────────────────────────────────────────────────────────────
# E2E — real proxy process, SIGHUP, real signal handling
# ────────────────────────────────────────────────────────────────────────

_CONFIG_TEMPLATE = """
listen:
  host: "0.0.0.0"
  port: {port}

auth:
  api_key: null

rate_limit:
  enabled: {rl_enabled}
  requests_per_minute: 3

models:
  - name: "echo-model"
    url: "{backend_url}"

static_files: []
"""


class TestSighupAppliesCapturedConfig:
    def test_sighup_enables_rate_limit(self, proxy_config_dir, backend_url):
        """Rate limiting is captured at app-build time — SIGHUP must rebuild."""
        port = find_free_port()
        config_path = proxy_config_dir / f"reload-rl-{port}.yaml"
        config_path.write_text(
            _CONFIG_TEMPLATE.format(port=port, backend_url=backend_url, rl_enabled="false")
        )
        proc = _start_proxy(port, config_path)
        base = f"http://127.0.0.1:{port}"

        try:
            codes = [requests.get(f"{base}/v1/models", timeout=5).status_code for _ in range(6)]
            assert codes == [200] * 6, f"baseline should be unthrottled, got {codes}"

            config_path.write_text(
                _CONFIG_TEMPLATE.format(port=port, backend_url=backend_url, rl_enabled="true")
            )
            proc.send_signal(signal.SIGHUP)
            time.sleep(1.0)

            codes = [requests.get(f"{base}/v1/models", timeout=5).status_code for _ in range(6)]
            assert 429 in codes, f"rate_limit not applied after SIGHUP: {codes}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
