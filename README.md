# Simple LLM Proxy

A lightweight, OpenAI-compatible reverse proxy that routes requests across multiple LLM backends behind a single API surface.

## Features

- **Single OpenAI-compatible API** — `/v1/*` endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/embeddings`)
- **Multi-backend routing** — map model names to different backends by URL
- **Transparent SSE streaming** — raw byte pass-through, in both directions; the proxy never parses or rewrites a stream
- **Single global API key** auth — optional Bearer token guard
- **Hot-reload config** — `SIGHUP` triggers reload without dropping connections
- **Static file serving** — serve files from filesystem directories at configured URL paths
- **Graceful shutdown** — `SIGTERM`/`SIGINT` drain active requests

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the proxy
python3 main.py
```

The proxy listens on `0.0.0.0:8080` by default.

## Configuration

Config resolution priority (first match wins):

1. `--config <path>` / `-c <path>` CLI flag
2. `HIFI_CONFIG` environment variable
3. `~/.config/hifi/config.yaml` (default)

```yaml
listen:
  host: 0.0.0.0      # bind address
  port: 8080          # bind port

cors:
  enabled: true       # set false to disable CORS entirely
  allow_origins: ["*"]  # lock down to specific origins in production
  allow_methods: ["*"]
  allow_headers: ["*"]

auth:
  api_key: null       # set a string (global) or dict of model->key for per-model auth

rate_limit:           # per-IP sliding window rate limiting (disabled by default)
  enabled: false
  requests_per_minute: 60
  endpoints:
    "/v1/chat/completions": 30
    "/v1/embeddings": 20

max_body_size_mb: 10  # request bodies larger than this get HTTP 413

models:
  - name: "fast-agent"
    url: "http://backend:8080"
    provider: llama.cpp          # display hint for frontends like OWUI
    backend: openai_compatible   # adapter type: "llama_cpp", "managed_llama", or any registered
    api_key: null                # optional per-model backend auth token (${VAR} expanded)
    default_params:              # used when the request omits them; client wins
      temperature: 0.7
      top_p: 0.95
    verify_ssl: true             # set false for self-signed backend certs

  - name: "slow-thinker"
    url: "http://backend:8081"
    default_params:
      temperature: 0.7
      top_p: 0.9

# ── Managed local model (HiFi launches the server process) ───────────
  - name: "local-llama"
    url: "http://127.0.0.1:8080"
    backend: managed_llama
    llama_binary: "/usr/local/bin/llama-server"
    server_config:
      ctx_size: 4096
      n_gpu_layers: 99
      threads: 8
    server_args: ["--parallel", "2"]   # raw args appended to the command line

# ── Static file serving ─────────────────────────────────────────────
static_files:
  - path: "/docs"
    directories:
      - ./docs

health_check_interval: 2.0      # seconds between backend health probes
```

Environment variable expansion: use `${VAR_NAME}` in config string values and they'll be resolved from the process environment.

## Endpoints

### `GET /health`

Health check. Returns `{"status": "ok", "config_loaded": true}`.

Never requires credentials, even when `auth.api_key` is set — uptime monitors
and process managers cannot hold a token. Everything else, `/metrics` included,
is still gated.

### `GET /v1/models`

Lists configured models in OpenAI format.

### `POST /v1/chat/completions`

Forward to the appropriate backend. Supports both streaming (`"stream": true`) and non-streaming responses.

### `POST /v1/completions`

Legacy completions endpoint (forward to backends that support it).

### `POST /v1/embeddings`

Forward to the appropriate backend.

## Per-model defaults

`default_params` supplies request parameters the client did not send — the
mechanism that makes two model entries pointing at the same backend behave
differently, which is the only way `work/x` and `think/x` can differ at all:

```yaml
models:
  - name: "work/Qwen3.6-35B"
    url: "http://localhost:8082"
    default_params: {temperature: 0.3, top_p: 0.8}

  - name: "think/Qwen3.6-35B"
    url: "http://localhost:8082"
    default_params: {temperature: 1.0, top_p: 0.95, thinking: {type: enabled}}
```

Precedence and the two edges worth knowing:

| Client sent | Result |
|---|---|
| nothing, or `"temperature": null` | default applies — `null` means "no preference" |
| `"temperature": 0.1` | `0.1`; the config value is ignored |
| `"logit_bias": {"7": 3.0}` | merged key by key, client entries win |
| `"logit_bias": {}` | stays empty — an empty dict is an opt-out, not an omission |

**Every injected parameter is announced**, because a default that applies
silently is an afternoon of diffing byte counts to find out why sampling looks
nothing like the config:

- response header `X-Hifi-Defaulted: top_p,max_tokens,logit_bias.42` — absent
  when nothing was added; nested additions are named by path
- `proxy_default_params_total{model="…",param="…"}` on `/metrics`
- one `DEBUG` log line per request

Launch arguments are not request parameters: they belong in `server_args`
(append-only, verbatim) or `server_config` (snake_case → `--kebab-case`).
`default_params.server_args` used to work and is migrated with a warning, since
now that defaults are genuinely applied it would otherwise put a CLI flag inside
every request body.

> This knob shipped broken: the merge was computed and discarded, so the config
> did nothing while two passing tests named after it asserted only that the
> proxy returned `200`. Deleting the merge left the suite green. `default_params`
> is now applied upstream of both the streaming and JSON paths, and the tests
> assert on the body the backend received (`tests/test_default_params.py`).

## Frontend auth

When `auth.api_key` is set, every request must include:

```
Authorization: Bearer <your-key>
```

Set `auth.api_key: null` (or omit) to disable auth.

It can also be a map of model name → key to gate each model with its own token:

```yaml
auth:
  api_key:
    fast-agent: "key-a"
    slow-thinker: "key-b"
```

Requests that carry no model — `GET /v1/models`, for instance — are checked
against a `"*"` entry. Add one to allow those requests, leave it out to reject
them:

```yaml
auth:
  api_key:
    "*": "key-readonly"
    fast-agent: "key-a"
```

## Limits

`max_body_size_mb` (default 10) caps request bodies; anything larger is
answered with `413` and an OpenAI-style error body. The cap is enforced while
the body is being read, so a chunked upload with no `Content-Length` cannot
slip past it, and nothing is buffered beyond the cap on the way.

Requests pass through the middleware stack in this order:

```
request-id → access log → body size → CORS → rate limit → auth → route
```

which is what makes the protections work as advertised: oversized bodies are
rejected before credentials are examined, browser preflights are answered
without an `Authorization` header, and guessed credentials are throttled
instead of being replayed against the backend forever.

## Streaming

The response stream is forwarded byte for byte. The proxy parses nothing: no
delta rewriting, no injected frames, no `[DONE]` of its own. (An earlier build
rewrote `delta` objects to normalize reasoning content; that silently dropped
`delta.tool_calls`, which is why tool calls "broke" whenever this code was
touched. It is gone, and `tests/test_streaming_fidelity.py` keeps it gone.)

**Errors.** A backend failure before the first byte is reported as a real HTTP
status — `502` with the backend's own status in `error.code` — never as a
`200 OK` `text/event-stream` carrying a made-up error event, because a
tool-calling client reads that as an empty answer and skips the tool call.
Once bytes are on the wire the stream simply ends; nothing is appended.

**Timeouts.** The non-streaming request budget does not apply to streams:

| Variable | Default | Applies to |
|---|---|---|
| `LLM_PROXY_TIMEOUT_MS` | 120000 | non-streaming requests, overall |
| `LLM_PROXY_CONNECT_TIMEOUT_MS` | 5000 | connecting to a backend |
| `LLM_PROXY_POOL_TIMEOUT_MS` | 30000 | waiting for a free connection |
| `LLM_PROXY_STREAM_READ_TIMEOUT_MS` | 0 (no limit) | gap between streamed chunks |
| `LLM_PROXY_MAX_CONNECTIONS` / `..._MAX_KEEPALIVE_CONNECTIONS` | 100 / 20 | upstream pool |

A stream is therefore never cut for thinking too long — a reasoning model can
be silent for minutes. Set `LLM_PROXY_STREAM_READ_TIMEOUT_MS` to a positive
value to opt back in to a stall ceiling, and note that connect/pool timeouts
still apply before the stream starts.

To compare this proxy against a direct connection, `tools/difftap.py` taps both
arms and reports whether differences are content or timing:

```bash
tools/difftap.py serve --listen 127.0.0.1:8099 --upstream 127.0.0.1:8082 \
    --log /tmp/tap-direct.jsonl --capture-bodies
tools/difftap.py compare /tmp/tap-direct.jsonl /tmp/tap-hifi.jsonl \
    --path chat/completions
```

## Backend auth

A model's `api_key` is the *upstream* credential — unrelated to `auth.api_key`
above — and is sent to that model's backend as `Authorization: Bearer <api_key>`
on every request (also for `managed_llama`, whose key goes to the proxied
`llama-server`). Omit it and no `Authorization` header is sent.

## Hot Reload

Send `SIGHUP` to reload the config file without restarting:

```bash
kill -HUP $(lsof -ti:8080)
```

The new file is validated first — if it does not parse, the error is logged and
the running config stays in place. The app is then rebuilt, so model list,
frontend and backend auth keys, health-probe targets, rate limiting, CORS and
static mounts all take effect. In-flight requests finish on the old config.

## Static Files

Serve files from filesystem directories at configured URL paths:

```yaml
static_files:
  - path: "/docs"
    directories:
      - ./docs
```

Multiple directories per path are supported — the first directory that contains the requested file wins.

## Running as a Service

```bash
# With a custom config
python3 main.py --config /path/to/config.yaml

# Or via environment variable
HIFI_CONFIG=/path/to/config.yaml python3 main.py

# Run in background
nohup python3 main.py --config config.yaml > proxy.log 2>&1 &
```

## Testing

E2E tests against live backends:

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```
