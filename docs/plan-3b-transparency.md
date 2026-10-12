# Plan #3b — make the proxy a transparent pipe

**Status:** proposal, not implemented. Behaviour-changing; needs sign-off.
**Depends on:** #3a (middleware order, public `/health`, constant-time token
compare, real body-size cap) — landed.

## Design intent

The proxy is a pipe: prompts go upstream, responses come back, **nothing is
parsed and nothing is regenerated**. It exists to route by model name, to
authenticate, and to observe — not to author content on the backend's behalf.

Today it breaks that contract in six places. Each one is either invisible
re-encoding or invented content, and each one has to go before the "raw pipe"
claim is true.

## What the pipe does today

| # | Site | Behaviour today | Why it violates the intent |
|---|------|-----------------|---------------------------|
| 1 | `adapters/base.py:187`, `:220` | `client.stream(..., json=payload)` / `client.post(..., json=payload)` | The request body is parsed to a dict and **re-serialised**. Key order, number formatting (`1` vs `1.0`), duplicate keys, unknown fields and whitespace are all rewritten. The bytes the client sent are not the bytes the backend receives. |
| 2 | `proxy/router.py:74` | `_merge_defaults(body, model_config.default_params)` computes a merged payload that `_forward_json`/`_forward_stream` never use — they send `request.state._request_body` | `default_params` is a **silent no-op** while the endpoint docs promise merging. Either the pipe merges, or it doesn't; a knob that does nothing is worse than no knob. |
| 3 | `adapters/base.py:161-167` → `BackendError` | Backend `4xx`/`5xx` is discarded and replaced by a synthesised `502`; message truncated to 200/500 chars | The client can't tell "backend said my request was bad" from "proxy could not reach the backend". Error content is authored by the proxy. |
| 4 | `adapters/base.py:227-228` + `proxy/router.py:126` | Non-streaming success: `resp.json()` → `_normalise_response` → `JSONResponse(content=…)` | Round-trip through Python objects; upstream status and headers (`Retry-After`, `X-RateLimit-*`, `openai-organization`, content-type) dropped. |
| 5 | `adapters/base.py:191-194` + `proxy/router.py:106` | Streaming is byte-for-byte, **except** upstream error mid-stream becomes a synthetic SSE frame and `text/event-stream` is forced | Mostly honest; the synthetic frame is invented content. |
| 6 | all five middleware | `BaseHTTPMiddleware` | Every SSE chunk is shuttled through an anyio task + in-memory stream: added latency, memory copy, murky client-disconnect semantics. Not content-generation, but it is the tax paid on the "raw" path. |

## Target architecture (option B — sniff for routing, forward original bytes)

Keep the OpenAI-identical URL layout and dict-mode auth. Read the body **once**,
for routing and auth only, and keep the original bytes alongside the parse:

```
raw: bytes  ────────────────────────────────────────────────► upstream (content=raw)
     └─► json.loads (auth/router: model name only)
```

Routing never mutates what goes upstream. Config-driven parameter defaults, if
they stay, become an explicit opt-in that re-serialises — and say so in the
response.

## Work items

Ordered so each step is shippable and testable on its own.

**B1 — Forward the original bytes.** *(small, breaking only for byte-pedants)*
`Request.state` keeps `raw_body: bytes`. `OpenAICompatibleAdapter` sends
`content=raw` instead of `json=payload`, and sets `Content-Type` via
`setdefault` rather than forcing `application/json`.
*Test:* byte round-trip — a body with unusual spacing, duplicate JSON keys,
`"temperature": 1` and unknown fields arrives upstream **identically**
(`hashlib.sha256` of the recorded request equals the sha256 of the sent body).

**B2 — Decide `default_params`.** *(done — (b), make it work and be loud)*
Applied on both the JSON and streaming paths, with client precedence, and every
injected parameter announced three ways: `X-Hifi-Defaulted: top_p,logit_bias.42`
(absent when nothing was added, nested additions named by path),
`proxy_default_params_total{model,param}` on `/metrics`, and a `DEBUG` line.

Three decisions the tests forced, none of them obvious from the original plan:

- **`null` is absence.** `"temperature": null` means the client chose nothing,
  which is what a default answers. An explicit value is never overwritten.
- **An empty dict is presence, not absence.** Filling a client's
  `"logit_bias": {}` from config turns "no bias" into a bias — a behavioural
  flip caused by emptiness rather than omission. Empty dicts stay empty.
- **Launch arguments moved out of `default_params`.** `server_args` was read out
  of it for `managed_llama`, so now that defaults really are applied, a CLI flag
  would have been forwarded inside every request body. It is a first-class field
  now, with a warning-and-migrate path for old configs.

Deep-copied per request: one request mutating a nested default would otherwise
be inherited by every later request for that model — config drift caused by
serving traffic, invisible in the config file.

*Test:* `tests/test_default_params.py` (18 tests) asserts on the body the
backend received, not on the response. Mutation-checked: dropping the merge →
12 failures; inverting precedence → 9; suppressing the header → 4; removing the
counter → 1. Before this change, all four mutations left the suite green.

**B3 — Stop regenerating errors.** *(medium, the user-visible one)*
`_handle_error_response` returns the upstream response as-is: status, headers
(sanctioned hop-by-hop removal only), body. `BackendError` remains for
transport failures (connect error, timeout), which *are* the proxy's own
problems and keep a 502/504.
*Breaking:* clients that today see `502` for a backend `400` will see the
backend's `400` and body. This is the intended fix, but it is a visible change.
*Test:* backend returns `400 {"error":{"message":"unknown field 'foo'"}}` →
client sees status 400 and that exact body; backend unreachable → still 502 with
an OpenAI-shaped error.

**B4 — Pass through success status + headers.** *(small)*
Build the response from upstream status/headers/body instead of
`JSONResponse(content=…)`; keep `_normalise_response` only for adapters that
actually translate a non-OpenAI dialect (`llama_cpp`), and mark them as
translating in the response.
*Test:* backend sets `Retry-After`/`X-RateLimit-Remaining`/custom content-type
→ client receives them; a non-JSON (`text/plain`) body is forwarded verbatim.

**B5 — Streaming: keep it raw, honest about errors.** *(done)*
Keep byte passthrough. For an upstream error while streaming, forward the
upstream status and body when the stream has not started; once bytes are on the
wire, close the stream instead of injecting a synthetic frame.
*Test:* upstream error before first byte → real status/body; error after first
byte → stream ends without an invented `data:` frame.

Shipped: `forward_stream` now raises `BackendError(status, message, body,
headers)` instead of yielding `_error_sse(...)`, and `router._forward_stream`
applies the same trick it already used for compression sniffing — await the
first upstream chunk *before* constructing `StreamingResponse`, so a failure
with nothing on the wire still reaches the `BackendError` handler and produces
a real `502` (`error.code` = the backend's status, message extracted from the
backend body). Mid-stream failure logs and ends the stream; `_error_sse` is
deleted, so there is no code path left that can author an SSE frame.

**B7 — Split stream timeouts from request timeouts.** *(done, not in the
original list — found while investigating tool calls that never fired)*
One `httpx.Timeout(120.0)` was applied to streams as well as buffered calls, so
any stream with a >120 s total life, or a >120 s gap between chunks, was killed
mid-flight by the proxy. A thinking model emitting a tool call after a long
reasoning phase loses the call: the client sees a truncated turn, not an error.
`stream_timeout()` now bounds only connect/pool/write and leaves read
unbounded (`LLM_PROXY_STREAM_READ_TIMEOUT_MS=0`); `json_timeout()` keeps the
overall budget for buffered calls. Clients stuck on a dead backend are covered
by health probing plus connect/pool timeouts, not by cutting live streams.
*Test:* `test_long_silence_between_chunks_is_not_a_timeout` runs a proxy with
`LLM_PROXY_TIMEOUT_MS=800` against a backend that pauses 2 s mid-stream.

**B6 — Move the hot path off `BaseHTTPMiddleware`.** *(largest, do last —
performance only)* Correctness of client cancellation is *not* at issue:
measured with `test_client_disconnect_reaches_the_backend` (client aborts
mid-stream, raw-socket tap confirms the upstream sees the disconnect), current
Starlette does propagate the cancellation through the middleware stack, so an
abandoned stream does not wedge a llama.cpp slot. B6 stays open for per-chunk
overhead only, and that test is the guard that the rewrite must not regress it.
Port `RequestID`, `AccessLog`, `RateLimit` and `Auth` to pure ASGI (or drop them
onto the router as dependencies). `Auth`'s body sniff becomes a shared
`raw_body` helper so the body is parsed exactly once per request.
*Test:* SSE latency A/B (measure per-chunk overhead before/after); disconnect
propagation test — client aborts mid-stream, upstream request is cancelled;
`#3a`'s middleware-order and 413 tests keep passing unchanged.

## Rollout

0. **Harness first** *(done)* — the missing capability was never assertions but
   an **upstream oracle**: `tests/conftest.py` now records every body the proxy
   forwards (`forwarded` fixture), because a suite that can only see responses
   cannot fail a pipe that returns `200`. That is how `default_params` shipped
   as a no-op behind two green tests named after it, and why the streaming tests
   could not see a dropped `tool_calls` delta. With it in place,
   `tests/test_streaming_fidelity.py` pins the
   contract with an adversarial raw-TCP backend: events split across writes and
   split inside a UTF-8 codepoint, two events in one segment, `:` comment
   keep-alives, tool-call deltas reassembled by a consumer, backend failure
   before/after first byte, mid-stream RST, long silences, client disconnect.
   `tools/difftap.py` answers the same questions about *production* traffic:
   tap the client arm and the proxy arm, then `compare` reports whether the
   request bytes, the response bytes, or only the timings differ.
   Measured on a tool-call stream (client→tap→model vs client→proxy→tap→model):
   response bytes identical (775 B, 8 SSE events); request bytes differ only
   because httpx re-serialises with compact separators (133 B → 121 B, same
   parsed payload). `json=` also escapes non-ASCII, so CJK/tool JSON is sent as
   `\uXXXX` — semantically equivalent, ~6× the bytes. That is B1's real cost
   today: nothing observed breaking, everything slightly rewritten upstream.

1. B1 + B3 + B4 in one PR (B2 done) behind the same release, since together they are
   the "transparent" contract change; B5 alongside.
2. B6 separate — pure performance and lifecycle, no contract change.
3. Add a `transparency` note to README ("what the proxy changes about your
   request: nothing, unless `default_params` is set") and keep
   `SPECIFICATION.md` FR-05/table honest.
4. Version bump + changelog line: "backend errors are forwarded, not rewritten
   into 502" — someone's retry logic depends on the old status.

## Non-goals

- No response schema validation or repair.
- No retries, request queueing or load balancing (would re-introduce authored
  responses under failure).
- No OpenAI→llama.cpp translation on the default path; adapters that translate
  are named as translators in config (`backend: llama_cpp`), not silently.

## Open questions for sign-off

1. ~~`default_params`: remove (B2a) or make-real-and-loud (B2b)?~~ **Resolved:
   B2b**, implemented; precedence is client-wins and injection is reported in
   `X-Hifi-Defaulted`.
2. Backend `4xx` reaching clients as-is — acceptable breaking change?
3. Should `X-Request-ID` be forwarded upstream (today it is not), so backend
   logs can be joined with proxy logs?
4. Is hop-by-hop header stripping (`connection`, `keep-alive`, `transfer-encoding`,
   `content-length`, `content-encoding`) enough, or maintain an allow-list?
