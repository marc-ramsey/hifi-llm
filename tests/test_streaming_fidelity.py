"""Streaming fidelity tests — SSE framing, error honesty, timeout policy.

This file exists because the proxy has repeatedly regressed by *authoring*
stream content: rewriting ``delta`` objects (which drops ``tool_calls``),
injecting synthetic error frames into a 200 SSE stream, or cutting a stream on
a wall clock.  Each test pins one rule:

1. what the backend writes is *exactly* what the client reads — byte for byte,
   including events split mid-line and mid-codepoint;
2. a backend failure **before the first byte** is a real status code, never a
   ``200 OK`` stream carrying a made-up ``data: {"error": …}`` event (a
   tool-calling client reads that as an empty answer and silently skips the
   tool);
3. a stream that dies mid-flight ends where it died — nothing appended;
4. a long silence between chunks is legitimate work, not a timeout;
5. (known gap, plan #3b B6) a client that goes away should cancel the
   upstream request — asserted, currently expected to fail while the hot path
   runs on ``BaseHTTPMiddleware``.

The backend is a raw asyncio TCP server so tests control framing at the byte
level: multiple events per write, an event split across writes, a split inside
a multi-byte UTF-8 character, a ``: comment`` keep-alive, and abrupt closes.
"""

from __future__ import annotations

import asyncio
import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest
import requests

from tests.conftest import _start_proxy, find_free_port

pytestmark = pytest.mark.timeout(120)


# ────────────────────────────────────────────────────────────────────────
# Wire script: what the torture backend writes, in exact byte chunks
# ────────────────────────────────────────────────────────────────────────

def _ev(obj) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n"


_FRAME_ROLE = _ev({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})

# A tool-call lifecycle fragmenting the SSE payload the way a real tokenizer
# does it: mid-line, mid-key, mid-UTF-8-codepoint, two events in one write.
_FRAME_TC_OPEN = _ev({"choices": [{"index": 0, "delta": {"tool_calls": [
    {"index": 0, "id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": ""}}]}}]})

_FRAME_TC_ARGS = _ev({"choices": [{"index": 0, "delta": {"tool_calls": [
    {"index": 0, "function": {"arguments": "{\"query\":\"current price of BTC\"}"}}]}}]})

_FRAME_TC_DONE = _ev({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})

# one event split across three writes
_TC_SPLIT = _FRAME_TC_OPEN
_TC_CUT_A = len(_TC_SPLIT) // 2
_TC_CUT_B = _TC_SPLIT.index(b"arguments") + 3

# one event split *inside* a multi-byte character
_UNICODE_EVENT = _ev({"choices": [{"index": 0, "delta": {"content": "café 日本語 → ok"}}]})
_UNICODE_CUT = _UNICODE_EVENT.index("日".encode("utf-8")) + 1  # mid-codepoint

TWO_EVENTS = (_ev({"choices": [{"index": 0, "delta": {"content": "α"}}]})
              + _ev({"choices": [{"index": 0, "delta": {"content": "β"}}]}))

KEEPALIVE = b": keep-alive\n\n"

TOOLCALL_CHUNKS: list[bytes] = [
    _FRAME_ROLE,
    _TC_SPLIT[:_TC_CUT_A],
    _TC_SPLIT[_TC_CUT_A:_TC_CUT_B],
    _TC_SPLIT[_TC_CUT_B:],
    KEEPALIVE,
    _FRAME_TC_ARGS,
    _UNICODE_EVENT[:_UNICODE_CUT],
    _UNICODE_EVENT[_UNICODE_CUT:],
    TWO_EVENTS,
    _FRAME_TC_DONE,
    b"data: [DONE]\n\n",
]

TOOLCALL_BODY = b"".join(TOOLCALL_CHUNKS)

FRAMES_PREFIX = [
    _ev({"choices": [{"index": 0, "delta": {"content": "partial "}}]}),
    _ev({"choices": [{"index": 0, "delta": {"content": "answer"}}]}),
]

SSE_HEAD = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")


def _chunk(data: bytes) -> bytes:
    return b"%x\r\n%s\r\n" % (len(data), data)


# ────────────────────────────────────────────────────────────────────────
# Raw TCP torture backend
# ────────────────────────────────────────────────────────────────────────

class TortureBackend:
    """HTTP/1.1 backend whose behaviour is selected by the request's model name.

    Scenario per model: ``torture-<scenario>``.  Records observable facts in
    ``state`` (frames written, whether/when the peer hung up).
    """

    def __init__(self) -> None:
        self.state: dict = {}
        self.port: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> str:
        self.port = find_free_port()
        ready = threading.Event()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve(ready))
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        assert ready.wait(10), "torture backend did not start"
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    async def _serve(self, ready: threading.Event) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        ready.set()

    # ── connection handling ─────────────────────────────────────────────

    async def _handle(self, reader: asyncio.StreamReader, writer) -> None:
        try:
            req = await self._read_request(reader)
            if req is None:
                return
            path, body = req
            try:
                model = json.loads(body or b"{}").get("model", "")
            except ValueError:
                model = ""
            scenario = model[len("torture-"):].replace("-", "_") if model.startswith("torture-") else "toolcalls"
            handler = getattr(self, f"_scenario_{scenario}", None)
            if handler is None:  # never fall back silently — it hides typos
                body = json.dumps({"error": {"message": f"unknown scenario: {scenario}"}}).encode()
                await self._write(
                    writer,
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n"
                    b"Content-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body),
                )
                return
            await handler(reader, writer)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    @staticmethod
    async def _read_request(reader) -> tuple[str, bytes] | None:
        head = b""
        while b"\r\n\r\n" not in head:
            piece = await reader.read(65536)
            if not piece:
                return None
            head += piece
        head_bytes, _, rest = head.partition(b"\r\n\r\n")
        lines = head_bytes.split(b"\r\n")
        path = lines[0].split(b" ")[1].decode("latin-1")
        length = 0
        for line in lines[1:]:
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                length = int(value.strip())
        body = rest
        while len(body) < length:
            piece = await reader.read(length - len(body))
            if not piece:
                break
            body += piece
        return path, body

    # ── scenarios ───────────────────────────────────────────────────────

    async def _write(self, writer, data: bytes) -> None:
        writer.write(data)
        await writer.drain()

    async def _scenario_toolcalls(self, reader, writer) -> None:
        """Tool-call deltas with adversarial framing. One chunk per TCP write."""
        await self._write(writer, SSE_HEAD)
        for piece in TOOLCALL_CHUNKS:
            await self._write(writer, _chunk(piece))
            await asyncio.sleep(0.02)
        writer.write(b"0\r\n\r\n")
        await writer.drain()

    async def _scenario_error_before(self, reader, writer) -> None:
        """Fail immediately, before any body byte."""
        body = json.dumps({"error": {"message": "backend exploded: invalid input batch"}}).encode()
        await self._write(
            writer,
            b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body),
        )

    async def _scenario_error_after(self, reader, writer) -> None:
        """Send frames, die mid-flight (RST after a flush)."""
        await self._write(writer, SSE_HEAD)
        for piece in FRAMES_PREFIX:
            await self._write(writer, _chunk(piece))
        await asyncio.sleep(0.2)  # let the frames reach the proxy
        try:
            writer.transport.abort()  # ECONNRESET, no terminating chunk
        except Exception:
            pass

    async def _scenario_slow_mid(self, reader, writer) -> None:
        """A 2 s silence between chunks: legitimate thinking, not a stall."""
        await self._write(writer, SSE_HEAD)
        await self._write(writer, _chunk(FRAMES_PREFIX[0]))
        await asyncio.sleep(2.0)
        await self._write(writer, _chunk(FRAMES_PREFIX[1]))
        await self._write(writer, _chunk(b"data: [DONE]\n\n"))
        writer.write(b"0\r\n\r\n")
        await writer.drain()

    async def _scenario_hang(self, reader, writer) -> None:
        """Stream forever, recording when the peer goes away."""
        await self._write(writer, SSE_HEAD)
        await self._write(writer, _chunk(FRAMES_PREFIX[0]))
        self.state["hang"] = {"frames": 1, "disconnected": False, "t0": time.monotonic()}
        try:
            while True:
                try:
                    data = await asyncio.wait_for(reader.read(1), timeout=0.3)
                except asyncio.TimeoutError:
                    data = None
                if data == b"":  # peer closed
                    self.state["hang"]["disconnected"] = True
                    self.state["hang"]["disconnect_s"] = time.monotonic() - self.state["hang"]["t0"]
                    return
                if data is None:
                    await self._write(writer, _chunk(FRAMES_PREFIX[1]))
                    self.state["hang"]["frames"] += 1
        except (ConnectionResetError, BrokenPipeError, OSError):
            self.state["hang"]["disconnected"] = True


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────

_MODELS = ["toolcalls", "error-before", "error-after", "slow-mid", "hang"]


def _config_text(backend_url: str, port: int) -> str:
    models = "\n".join(
        f'  - name: "torture-{name}"\n    url: "{backend_url}"\n    default_params: {{}}'
        for name in _MODELS
    )
    return f"""
listen:
  host: 127.0.0.1
  port: {port}
auth:
  api_key: null
models:
{models}
static_files: []
"""


@pytest.fixture(scope="module")
def torture_backend():
    backend = TortureBackend()
    backend.start()
    yield backend
    backend.stop()


def _launch(backend_url: str, env_extra: dict[str, str] | None = None):
    """Start a throwaway proxy on a free port pointing at *backend_url*."""
    port = find_free_port()
    cfg_dir = Path(tempfile.mkdtemp(prefix="hifi-fidelity-"))
    cfg = cfg_dir / "config.yaml"
    cfg.write_text(_config_text(backend_url, port))
    proc = _start_proxy(port, cfg, env_extra=env_extra,
                        log_path=f"/tmp/hifi-fidelity-{port}.log")
    return proc, f"http://127.0.0.1:{port}"


def _terminate(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="module")
def proxy(torture_backend):
    """Proxy with the default timeout policy."""
    proc, base = _launch(torture_backend_url(torture_backend))
    yield base
    _terminate(proc)


def torture_backend_url(backend: TortureBackend) -> str:
    return f"http://127.0.0.1:{backend.port}"


# ────────────────────────────────────────────────────────────────────────
# 1. framing fidelity
# ────────────────────────────────────────────────────────────────────────

def test_tool_call_stream_is_byte_identical(proxy):
    """The client reads exactly the bytes the backend wrote."""
    resp = requests.post(
        f"{proxy}/v1/chat/completions",
        json={"model": "torture-toolcalls", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        stream=True,
        timeout=30,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = b"".join(resp.iter_content(chunk_size=None))
    assert body == TOOLCALL_BODY


def test_tool_call_survives_reassembly(proxy):
    """A consumer that parses the stream still gets a runnable tool call.

    Regression guard for the historical bug where the proxy parsed each chunk
    and rewrote ``delta``, silently dropping ``tool_calls``.
    """
    resp = requests.post(
        f"{proxy}/v1/chat/completions",
        json={"model": "torture-toolcalls", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        stream=True,
        timeout=30,
    )
    name = ""
    args = ""
    finish = None
    for line in (b"".join(resp.iter_content(chunk_size=None))).decode().splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        choice = json.loads(line[6:])["choices"][0]
        delta = choice.get("delta") or {}
        for call in delta.get("tool_calls", []):
            fn = call.get("function") or {}
            name += fn.get("name") or ""
            args += fn.get("arguments") or ""
        finish = choice.get("finish_reason") or finish

    assert finish == "tool_calls"
    assert name == "web_search"
    assert json.loads(args) == {"query": "current price of BTC"}


def test_comment_keepalive_is_not_swallowed(proxy):
    """: comment lines (proxy/pad keep-alives) must reach the client."""
    resp = requests.post(
        f"{proxy}/v1/chat/completions",
        json={"model": "torture-toolcalls", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        stream=True,
        timeout=30,
    )
    assert b": keep-alive" in b"".join(resp.iter_content(chunk_size=None))


# ────────────────────────────────────────────────────────────────────────
# 2. honest errors before the first byte
# ────────────────────────────────────────────────────────────────────────

def test_backend_error_before_first_byte_is_a_real_status(proxy):
    """A backend 500 arriving before any byte must not look like a 200 stream.

    This is the failure that makes tools "silently do nothing": a client that
    sees ``200 OK`` + ``text/event-stream`` waits for deltas, gets an error
    object it may ignore, and reports an empty turn.
    """
    resp = requests.post(
        f"{proxy}/v1/chat/completions",
        json={"model": "torture-error-before", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        timeout=30,
    )
    assert resp.status_code == 502, "proxy must not report success for a failed stream"
    assert not resp.headers["content-type"].startswith("text/event-stream")
    data = resp.json()
    assert data["error"]["type"] == "backend_error"
    assert data["error"]["code"] == 500, "the backend's own status belongs in the error code"
    assert "invalid input batch" in data["error"]["message"]


def test_backend_error_matches_non_streaming_shape(proxy):
    """Streaming and non-streaming failures answer identically."""
    payload = {"model": "torture-error-before", "messages": [{"role": "user", "content": "hi"}]}
    streamed = requests.post(f"{proxy}/v1/chat/completions", json={**payload, "stream": True}, timeout=30)
    buffered = requests.post(f"{proxy}/v1/chat/completions", json={**payload, "stream": False}, timeout=30)
    assert streamed.status_code == buffered.status_code == 502
    assert streamed.json() == buffered.json()


# ────────────────────────────────────────────────────────────────────────
# 3. honest stream death
# ────────────────────────────────────────────────────────────────────────

def test_midstream_death_ends_the_stream_with_no_invented_frame(proxy):
    """Once bytes are on the wire the proxy adds nothing — not even [DONE]."""
    resp = requests.post(
        f"{proxy}/v1/chat/completions",
        json={"model": "torture-error-after", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        stream=True,
        timeout=30,
    )
    assert resp.status_code == 200
    body = b"".join(resp.iter_content(chunk_size=None))
    assert body == b"".join(FRAMES_PREFIX)
    assert b"[DONE]" not in body
    assert b'"error"' not in body


# ────────────────────────────────────────────────────────────────────────
# 4. timeouts
# ────────────────────────────────────────────────────────────────────────

def test_long_silence_between_chunks_is_not_a_timeout(torture_backend):
    """A stream must not be cut by the non-streaming request budget.

    ``LLM_PROXY_TIMEOUT_MS=800`` used to kill a stream after 800 ms of
    silence, truncating tool calls emitted by thinking models.
    """
    proc, base = _launch(torture_backend_url(torture_backend), {"LLM_PROXY_TIMEOUT_MS": "800"})
    try:
        resp = requests.post(
            f"{base}/v1/chat/completions",
            json={"model": "torture-slow-mid", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            stream=True,
            timeout=30,
        )
        assert resp.status_code == 200
        body = b"".join(resp.iter_content(chunk_size=None))
        assert body == b"".join(FRAMES_PREFIX) + b"data: [DONE]\n\n"
    finally:
        _terminate(proc)


def test_unreachable_backend_is_not_reported_as_an_empty_stream(torture_backend):
    """Connection refused before the first byte → real 502, JSON, not SSE."""
    proc, base = _launch("http://127.0.0.1:%d" % find_free_port())
    try:
        resp = requests.post(
            f"{base}/v1/chat/completions",
            json={"model": "torture-toolcalls", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            timeout=30,
        )
        assert resp.status_code == 502
        assert not resp.headers["content-type"].startswith("text/event-stream")
        assert resp.json()["error"]["type"] == "backend_error"
    finally:
        _terminate(proc)


# ────────────────────────────────────────────────────────────────────────
# 5. client lifecycle
# ────────────────────────────────────────────────────────────────────────

def test_client_disconnect_reaches_the_backend(torture_backend):
    """OWUI aborting a stream must free the llama.cpp slot behind it.

    This passes today: Starlette's BaseHTTPMiddleware does propagate the
    cancellation, so an abandoned stream does not wedge a backend slot.
    Pinned as a regression guard — plan #3b B6 rewrites this layer for speed,
    and this is the behaviour that must survive the rewrite.
    """
    proc, base = _launch(torture_backend_url(torture_backend))
    try:
        host, port = base.split("//")[1].split(":")
        port = int(port)
        body = json.dumps({
            "model": "torture-hang",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }).encode()
        sock = socket.create_connection((host, port), timeout=10)
        sock.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: %s:%d\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
            % (host.encode(), port, len(body)) + body
        )
        sock.recv(65536)  # first bytes prove the stream opened
        sock.close()

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if torture_backend.state.get("hang", {}).get("disconnected"):
                return
            time.sleep(0.1)
        raise AssertionError(
            f"backend still streaming {torture_backend.state.get('hang')} after the client left"
        )
    finally:
        _terminate(proc)
