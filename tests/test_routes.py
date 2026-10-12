"""E2E tests for the proxy's main routes.

Tests against a real threaded HTTP backend server via the proxy.
"""

from __future__ import annotations

import json

import pytest

from .conftest import ROOT, parse_sse_chunks


def _is_openai_chat_completion(resp_json):
    """Validate that a response matches the OpenAI chat.completion schema."""
    assert resp_json["object"] == "chat.completion"
    assert len(resp_json["choices"]) == 1
    choice = resp_json["choices"][0]
    assert "index" in choice
    assert "message" in choice
    assert "role" in choice["message"]
    assert "finish_reason" in choice
    assert "usage" in resp_json
    usage = resp_json["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


def _is_openai_chat_completion_chunk(resp_json):
    """Validate that a SSE chunk matches the OpenAI chat.completion.chunk schema."""
    assert resp_json["object"] == "chat.completion.chunk"
    assert len(resp_json["choices"]) >= 1
    assert "delta" in resp_json["choices"][0]


class TestHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["config_loaded"] is True


class TestModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 2
        ids = [m["id"] for m in data["data"]]
        assert "gemma-4-26B-instruct" in ids
        assert "qwen3.6-35B-instruct" in ids

    def test_model_shape(self, client):
        resp = client.get("/v1/models")
        model = resp.json()["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model


class TestChatCompletions:
    """Non-streaming chat completions."""

    def test_basic_completion(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "gemma-4-26B-instruct",
            "messages": [{"role": "user", "content": "say hi"}],
            "max_tokens": 10,
        })
        assert resp.status_code == 200
        _is_openai_chat_completion(resp.json())

    def test_default_params_merged(self, client, forwarded):
        """Params the client omits come from the model's ``default_params``.

        Asserted on the body the backend received.  This test used to assert
        ``status == 200`` and "one choice" while claiming to verify the merge —
        and it passed for months with ``default_params`` discarded on the floor.
        """
        resp = client.post("/v1/chat/completions", json={
            "model": "gemma-4-26B-instruct",
            "messages": [{"role": "user", "content": "count 1 to 2"}],
            "max_tokens": 10,
        })
        assert resp.status_code == 200
        assert resp.json()["object"] == "chat.completion"

        sent = forwarded()[-1]["body"]
        assert sent["temperature"] == 0.7, "configured default never applied"
        assert sent["top_p"] == 0.95
        assert sent["max_tokens"] == 10, "default must not overwrite a client value"
        # And the proxy says what it supplied, on the response.
        assert resp.headers["X-Hifi-Defaulted"] == "temperature,top_p"

    def test_invalid_model(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "this-model-does-not-exist",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        data = resp.json()
        assert data["detail"]["type"] == "invalid_request_error"
        assert "this-model-does-not-exist" in data["detail"]["message"]

    def test_client_params_override_defaults(self, client, forwarded):
        """A client value is left alone, and is not reported as defaulted."""
        resp = client.post("/v1/chat/completions", json={
            "model": "gemma-4-26B-instruct",
            "messages": [{"role": "user", "content": "count to 3"}],
            "max_tokens": 10,
            "temperature": 0.1,
        })
        assert resp.status_code == 200

        sent = forwarded()[-1]["body"]
        assert sent["temperature"] == 0.1
        defaulted = set((resp.headers.get("X-Hifi-Defaulted") or "").split(",")) - {""}
        assert defaulted == {"top_p"}, (
            "temperature was supplied by the client, so it must not be reported "
            f"as a default; got {defaulted}"
        )


class TestChatCompletionsStreaming:
    """Streaming chat completions via SSE."""

    def test_streaming_basic(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "gemma-4-26B-instruct",
            "messages": [{"role": "user", "content": "count 1 to 3"}],
            "max_tokens": 15,
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["Content-Type"]

        chunks, done = parse_sse_chunks(resp.text)
        assert done is True
        assert len(chunks) >= 1

        # First chunk should have role in delta
        first = chunks[0]
        _is_openai_chat_completion_chunk(first)
        assert first["choices"][0]["delta"].get("role") == "assistant"

    def test_streaming_collected_text(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "gemma-4-26B-instruct",
            "messages": [{"role": "user", "content": "say hi"}],
            "max_tokens": 8,
            "stream": True,
        })
        assert resp.status_code == 200
        chunks, done = parse_sse_chunks(resp.text)
        assert done is True
        text = "".join(
            c["choices"][0]["delta"].get("content") or ""
            for c in chunks
            if c.get("choices")
        )
        assert len(text) > 0

    def test_streaming_response_headers(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "gemma-4-26B-instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "stream": True,
        })
        assert "no-cache" in resp.headers.get("Cache-Control", "")
        assert "keep-alive" in resp.headers.get("Connection", "")


class TestEmbeddings:
    def test_embeddings_basic(self, client):
        resp = client.post("/v1/embeddings", json={
            "model": "nomic-embed",
            "input": "the quick brown fox",
        })
        # Arkestra may take time to load nomic-embed; skip if backend is unavailable
        if resp.status_code in (400, 502):
            pytest.skip("nomic-embed not loaded in Arkestra yet")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        emb = data["data"][0]
        assert "embedding" in emb
        assert isinstance(emb["embedding"], list)
        assert len(emb["embedding"]) > 0
        # All values should be floats
        assert all(isinstance(v, (int, float)) for v in emb["embedding"])

    def test_embeddings_multiple_inputs(self, client):
        resp = client.post("/v1/embeddings", json={
            "model": "nomic-embed",
            "input": ["hello", "world"],
        })
        if resp.status_code in (400, 502):
            pytest.skip("nomic-embed not loaded in Arkestra yet")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2


class TestLegacyCompletions:
    def test_legacy_completions_not_supported(self, client):
        """Arkestra does not serve /v1/completions — expect 502."""
        resp = client.post("/v1/completions", json={
            "model": "gemma-4-26B-instruct",
            "prompt": "test prompt",
        })
        assert resp.status_code == 502
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "backend_error"
        assert data["error"]["code"] == 404
