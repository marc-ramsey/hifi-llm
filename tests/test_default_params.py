"""`default_params` are applied, client-wins, and visible on the response.

History: this feature shipped, was documented in three places, and had two
passing tests named after it — while `_forward_json`/`_forward_stream` sent the
raw request body and threw the merged payload away.  Deleting `_merge_defaults`
outright left the suite green, which is the bug class rather than one bug: the
harness had no way to ask what the proxy forwarded, so every test could only
assert on what it got back, and a pipe that returns 200 looks healthy.

The oracle is the `forwarded` fixture (bodies recorded by the test backend).
Every test here asserts on it.  A response-only assertion in this file would be
worse than no test, because it would re-create the illusion.
"""

from __future__ import annotations

import requests

import pytest

from config.schema import ModelConfig
from proxy.router import DEFAULTED_HEADER, _merge_defaults

from .conftest import _start_proxy, find_free_port

CONFIG = """
listen: {{host: "127.0.0.1", port: {port}}}
auth: {{api_key: null}}
models:
  - name: "defs"
    url: "{backend_url}"
    default_params:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256
      logit_bias: {{"42": 1.0, "99": 2.0}}

  - name: "plain"
    url: "{backend_url}"
    default_params: {{}}

static_files: []
"""


@pytest.fixture(scope="module")
def dp_proxy(backend_url, tmp_path_factory):
    """One proxy for the module, over a config whose models do and do not
    carry defaults.  Module-scoped on purpose: `proxy_factory` is
    function-scoped, and ten proxies for ten assertions is ten slow failures."""
    port = find_free_port()
    cfg = tmp_path_factory.mktemp("default-params") / "proxy-config.yaml"
    cfg.write_text(CONFIG.format(port=port, backend_url=backend_url))
    proc = _start_proxy(port, cfg)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _post(base, path="/v1/chat/completions", **body):
    body.setdefault("model", "defs")
    if "completions" in path:
        body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return requests.post(base + path, json=body, timeout=30)


class TestApplication:
    """The configured defaults reach the backend, or nothing else matters."""

    def test_omitted_params_are_supplied(self, dp_proxy, forwarded):
        resp = _post(dp_proxy)
        assert resp.status_code == 200

        sent = forwarded()[-1]["body"]
        assert sent["temperature"] == 0.7
        assert sent["top_p"] == 0.95
        assert sent["max_tokens"] == 256
        assert sent["logit_bias"] == {"42": 1.0, "99": 2.0}

    def test_client_values_win(self, dp_proxy, forwarded):
        """A default is a fallback, not a policy override.

        The original `_merge_defaults` inverted this (`result[key] = value`),
        which was harmless only because its output was discarded.  Wiring that
        up unchanged would have silently overridden every client's sampling
        settings — strictly worse than the no-op it replaced.
        """
        resp = _post(dp_proxy, temperature=0.1, max_tokens=5, user="alice")
        assert resp.status_code == 200

        sent = forwarded()[-1]["body"]
        assert sent["temperature"] == 0.1
        assert sent["max_tokens"] == 5
        assert sent["user"] == "alice"

    def test_explicit_null_means_no_preference(self, dp_proxy, forwarded):
        """`null` is the client saying "nothing chosen", which a default answers."""
        resp = _post(dp_proxy, temperature=None)
        assert resp.status_code == 200

        sent = forwarded()[-1]["body"]
        assert sent["temperature"] == 0.7, (
            "a null temperature should fall through to the default, not be "
            "sent upstream where the backend picks its own"
        )

    def test_streaming_requests_get_defaults_too(self, dp_proxy, forwarded):
        """Both paths must apply defaults; streaming was the one that didn't."""
        resp = _post(dp_proxy, stream=True)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["Content-Type"]

        sent = forwarded()[-1]["body"]
        assert sent["stream"] is True
        assert sent["temperature"] == 0.7

    def test_embeddings_get_defaults_too(self, dp_proxy, forwarded):
        """FR-03: every request naming this model, not just chat completions."""
        resp = _post(dp_proxy, path="/v1/embeddings", input="hello")
        assert resp.status_code == 200

        sent = forwarded()[-1]["body"]
        assert sent["input"] == "hello"
        assert sent["temperature"] == 0.7

    def test_explicit_empty_dict_survives_the_round_trip(self, dp_proxy, forwarded):
        resp = _post(dp_proxy, logit_bias={})
        assert resp.status_code == 200

        sent = forwarded()[-1]["body"]
        assert sent["logit_bias"] == {}, "config bias leaked into an opt-out request"
        assert "logit_bias" not in resp.headers.get(DEFAULTED_HEADER, "")

    def test_nested_dicts_merge_key_by_key(self, dp_proxy, forwarded):
        """Client logit_bias entries survive; only missing entries are added."""
        resp = _post(dp_proxy, logit_bias={"99": 9.0, "7": 3.0})
        assert resp.status_code == 200

        sent = forwarded()[-1]["body"]
        assert sent["logit_bias"] == {"42": 1.0, "99": 9.0, "7": 3.0}, (
            "a nested default must add only the entries the client omitted"
        )
        # The entry added inside a client-supplied dict is reported by path,
        # so a partial injection is never invisible.
        assert "logit_bias.42" in resp.headers[DEFAULTED_HEADER]


class TestVisibility:
    """A changed request is announced, or it becomes a debugging mystery."""

    def test_header_names_exactly_the_supplied_params(self, dp_proxy):
        resp = _post(dp_proxy, temperature=0.1)
        assert resp.headers[DEFAULTED_HEADER] == "top_p,max_tokens,logit_bias"

    def test_nothing_supplied_means_no_header(self, dp_proxy):
        resp = _post(dp_proxy, temperature=0.1, top_p=0.1, max_tokens=1,
                     logit_bias={})
        assert DEFAULTED_HEADER not in resp.headers, (
            "the header must be absent, not empty — 'nothing was changed' has "
            "to be distinguishable from 'something was changed'"
        )

    def test_model_without_defaults_forwards_the_body_unchanged(self, dp_proxy, forwarded):
        sent_body = {"model": "plain",
                     "messages": [{"role": "user", "content": "hi"}],
                     "temperature": 0.33}
        resp = requests.post(dp_proxy + "/v1/chat/completions", json=sent_body, timeout=30)
        assert resp.status_code == 200
        assert DEFAULTED_HEADER not in resp.headers

        assert forwarded()[-1]["body"] == sent_body

    def test_metrics_count_each_param_per_model(self, dp_proxy):
        """`which knob is firing, how often, on which model` from /metrics alone."""
        metric = 'proxy_default_params_total{model="defs",param="temperature"}'

        def count() -> int:
            body = requests.get(dp_proxy + "/metrics", timeout=30).text
            lines = [line for line in body.splitlines() if line.startswith(metric)]
            assert lines, f"{metric} missing from /metrics:\n{body[:500]}"
            return int(lines[0].split()[-1])

        before = count()
        _post(dp_proxy)
        _post(dp_proxy)
        assert count() - before == 2, (
            "exactly one count per injected parameter per request — a total "
            "with no breakdown is the number you can't act on"
        )


class TestMergeSemantics:
    """Unit level: the precedence rule and isolation, independent of HTTP."""

    def test_config_defaults_are_not_mutated_by_requests(self):
        """A nested default must be deep-copied per request.

        Without the copy, one request mutating `logit_bias` would permanently
        change the model's behaviour for every later request — config drift
        caused by serving traffic, invisible in the config file.
        """
        defaults = {"logit_bias": {"42": 1.0}, "response_format": {"type": "json"}}

        payload, injected = _merge_defaults({}, defaults)
        payload["logit_bias"]["42"] = 999.0
        payload["extra"] = True

        assert defaults == {"logit_bias": {"42": 1.0},
                            "response_format": {"type": "json"}}, (
            "a request mutated the model's default_params in place"
        )
        assert injected == ["logit_bias", "response_format"]

    def test_empty_dict_from_the_client_is_an_explicit_choice(self):
        """`logit_bias: {}` means "no bias", and must not be filled in.

        The one place presence and emptiness disagree: `null` is no preference,
        `{}` is a decision.  Filling it would flip "no bias" into whatever the
        deployment configured, on a request that asked for none.
        """
        payload, injected = _merge_defaults(
            {"logit_bias": {}}, {"logit_bias": {"42": 1.0}, "temperature": 0.7}
        )
        assert payload["logit_bias"] == {}
        assert injected == ["temperature"]

    def test_nested_additions_are_reported_with_a_path(self):
        """A key injected inside a client dict is visible, not hidden by the parent."""
        _payload, injected = _merge_defaults(
            {"logit_bias": {"99": 9.0}}, {"logit_bias": {"42": 1.0, "99": 2.0}}
        )
        assert injected == ["logit_bias.42"]

    def test_empty_defaults_never_touch_the_body(self):
        body = {"model": "x", "messages": []}
        payload, injected = _merge_defaults(body, {})
        assert payload == body
        assert injected == []


class TestServerArgsSeparation:
    """Launch arguments are process config, and must never ride in a body."""

    def test_legacy_default_params_server_args_is_promoted(self):
        cfg = ModelConfig(
            name="managed",
            url="http://127.0.0.1:9999",
            backend="managed_llama",
            llama_binary="/usr/bin/true",
            default_params={"temperature": 0.5, "server_args": ["--top-k", "20"]},
        )
        assert cfg.server_args == ["--top-k", "20"]
        assert "server_args" not in cfg.default_params
        assert cfg.default_params == {"temperature": 0.5}

    def test_command_line_uses_the_field(self):
        from adapters.managed_llama import ManagedLlamaAdapter

        cfg = ModelConfig(
            name="managed",
            url="http://127.0.0.1:9999",
            backend="managed_llama",
            llama_binary="/usr/bin/llama-server",
            server_config={"ctx_size": 4096},
            default_params={"server_args": ["--top-k", "20"], "temperature": 0.5},
        )
        cmd = ManagedLlamaAdapter(model_config=cfg)._build_command()
        assert cmd == ["/usr/bin/llama-server", "--ctx-size", "4096", "--top-k", "20"]

    def test_promoted_args_cannot_reach_a_body(self):
        """After migration `default_params` holds only request parameters."""
        cfg = ModelConfig(
            name="managed",
            url="http://127.0.0.1:9999",
            backend="managed_llama",
            llama_binary="/usr/bin/true",
            default_params={"server_args": ["--top-k", "20"]},
        )
        payload, _injected = _merge_defaults({"model": "managed", "messages": []},
                                            cfg.default_params)
        assert "server_args" not in payload, (
            "a llama-server launch flag was forwarded inside a request body"
        )
