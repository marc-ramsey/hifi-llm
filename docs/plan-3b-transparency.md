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

**B2 — Decide `default_params`.** *(small, product decision)*
Pick one:
- **(a) remove** — purest pipe; deletes `proxy/router.py:45-56` and the config field; or
- **(b) make it work and be loud** — merge only when `default_params` is
  non-empty, re-serialise in that case, and document that the response body is
  then the backend's, not the client's bytes.

Recommendation: **(b)** — it is already in users' configs, and "no config →
byte-identical" preserves the pipe claim for everyone else.
*Test:* with `default_params` empty → bytes identical upstream; non-empty →
merged keys present, all other bytes untouched.

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

**B5 — Streaming: keep it raw, honest about errors.** *(small)*
Keep byte passthrough. For an upstream error while streaming, forward the
upstream status and body when the stream has not started; once bytes are on the
wire, close the stream instead of injecting a synthetic frame.
*Test:* upstream error before first byte → real status/body; error after first
byte → stream ends without an invented `data:` frame.

**B6 — Move the hot path off `BaseHTTPMiddleware`.** *(largest, do last)*
Port `RequestID`, `AccessLog`, `RateLimit` and `Auth` to pure ASGI (or drop them
onto the router as dependencies). `Auth`'s body sniff becomes a shared
`raw_body` helper so the body is parsed exactly once per request.
*Test:* SSE latency A/B (measure per-chunk overhead before/after); disconnect
propagation test — client aborts mid-stream, upstream request is cancelled;
`#3a`'s middleware-order and 413 tests keep passing unchanged.

## Rollout

1. B1 + B2 + B3 + B4 in one PR behind the same release, since together they are
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

1. `default_params`: remove (B2a) or make-real-and-loud (B2b)?
2. Backend `4xx` reaching clients as-is — acceptable breaking change?
3. Should `X-Request-ID` be forwarded upstream (today it is not), so backend
   logs can be joined with proxy logs?
4. Is hop-by-hop header stripping (`connection`, `keep-alive`, `transfer-encoding`,
   `content-length`, `content-encoding`) enough, or maintain an allow-list?
