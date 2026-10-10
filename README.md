# Simple LLM Proxy

A lightweight, OpenAI-compatible reverse proxy that routes requests across multiple LLM backends behind a single API surface.

## Features

- **Single OpenAI-compatible API** — `/v1/*` endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/embeddings`)
- **Multi-backend routing** — map model names to different backends by URL
- **Transparent SSE streaming** — raw SSE pass-through for streaming responses, with reasoning-content normalization for Open WebUI compatibility
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

models:
  - name: "fast-agent"
    url: "http://backend:8080"
    provider: llama.cpp          # display hint for frontends like OWUI
    backend: openai_compatible   # adapter type: "llama_cpp", "managed_llama", or any registered
    api_key: null                # optional per-model backend auth token (${VAR} expanded)
    default_params:
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

### `GET /v1/models`

Lists configured models in OpenAI format.

### `POST /v1/chat/completions`

Forward to the appropriate backend. Supports both streaming (`"stream": true`) and non-streaming responses.

### `POST /v1/completions`

Legacy completions endpoint (forward to backends that support it).

### `POST /v1/embeddings`

Forward to the appropriate backend.

## Auth

When `auth.api_key` is set, every request must include:

```
Authorization: Bearer <your-key>
```

Set `auth.api_key: null` (or omit) to disable auth.

## Hot Reload

Send `SIGHUP` to reload the config file without restarting:

```bash
kill -HUP $(lsof -ti:8080)
```

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
