# SPECFICATION.md
**Simple LLM‑Proxy – A language‑agnostic, Ubuntu 26.04‑targeted design**
Version: 1.0.0
Date: 2026‑10‑08

---

## 1. Overview

The **Simple LLM‑Proxy** is a lightweight HTTP reverse‑proxy that presents a **single OpenAI‑compatible API surface** (`/v1/*`) while routing requests to any number of backend LLM servers (e.g., `llama.cpp`, `vLLM`, Azure OpenAI, custom OpenAI‑compatible services).  

Key goals

| Goal | Description |
|------|-------------|
| **Declarative configuration** | All backends, default parameters, and authentication are defined in a single YAML file. |
| **Multiple instances of the same checkpoint** | The same model can be launched multiple times with distinct URLs and sampling parameters. |
| **Plug‑in architecture** | Optional modules can register extra routes, intercept requests/responses, add caching, tool‑calling, or expose custom agents. |
| **Stateless & horizontally scalable** | No persistent state is required; the proxy can be replicated behind a load‑balancer. |
| **Ubuntu 26.04 ready** | All dependencies are available via `apt`, `pip`, or `npm` on Ubuntu 26.04. |
| **Security‑first** | API‑key authentication, optional TLS termination, rate‑limiting, and IP allow‑lists. |

---

## 2. High‑Level Architecture

```
+-------------------+      +---------------------------+      +-------------------+
|   Client / Agent  | ---> |   Simple LLM‑Proxy        | ---> |   Backend LLM     |
| (OpenAI‑compatible)   |   |   (Node/Express, FastAPI) |   |   (llama‑cpp, vLLM, |
|                     |   |   - Config loader          |   |   OpenAI‑compatible) |
+-------------------+      |   - Router/Dispatcher      |      +-------------------+
                           |   - Plug‑in manager        |
                           |   - Health/metrics endpoint|
                           +---------------------------+
```

* **Router/Dispatcher** selects the backend based on the `model` field (or a custom header) and merges per‑model default parameters with the client payload. 
* **Plug‑in manager** loads optional JavaScript/TypeScript (or Python) modules from a `plugins/` directory. 
* **Health/metrics** expose `/health` (JSON) and `/metrics` (Prometheus format). 

---

## 3. Functional Requirements

| ID | Requirement | Description |
|----|-------------|-------------|
| FR‑01 | **OpenAI‑compatible endpoints** | Must implement at least: `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`. |
| FR‑02 | **Config‑driven backend list** | All backends are defined in `proxy-config.yaml`. |
| FR‑03 | **Per‑model default parameters** | Each entry in `models[]` may share a `url` with other entries; the **only identifier** that distinguishes them is the `name`. The proxy must apply the entry‑specific `default_params` to every request that references that `name`. |
| FR‑04 | **Multiple instances of the same checkpoint** | Interpreted as *multiple logical versions* of the same physical checkpoint, **all using the same backend URL**. The config must allow duplicate URLs. |
| FR‑05 | **API‑key authentication (optional)** | If `auth.api_keys` is present, requests must include a matching `Authorization: Bearer <key>` header. |
| FR‑06 | **Plug‑in registration** | Any file in `plugins/` that exports a `register(app)` function is loaded at startup. |
| FR‑07 | **Request/Response hooks** | Plug‑ins can register `onRequest(req, ctx)` and `onResponse(resp, ctx)` callbacks to modify payloads. |
| FR‑08 | **Custom agents** | Plug‑ins may expose arbitrary routes (e.g., `/v1/agents/echo`) and be referenced from the config as a synthetic model. |
| FR‑09 | **Hot reload** | Receiving `SIGHUP` causes the proxy to reload `proxy-config.yaml` without dropping existing connections. |
| FR‑10 | **Observability** | Expose Prometheus metrics: request count, latency, error count, cache hit‑rate (if caching plug‑in used). |
| FR‑11 | **Graceful shutdown** | On `SIGTERM`/`SIGINT` stop accepting new connections, finish in‑flight requests, then exit. |
| FR‑12 | **TLS termination (optional)** | Proxy can be placed behind an external TLS terminator (e.g., Nginx, HAProxy) or run with self‑signed certs. |

---

## 4. Non‑Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR‑01 | **Performance** | ≤ 30 ms added latency per request (excluding backend latency) under 100 RPS. |
| NFR‑02 | **Reliability** | 99.9 % uptime; automatic restart via `systemd`. |
| NFR‑03 | **Scalability** | Stateless → horizontal scaling via load balancer. |
| NFR‑04 | **Security** | No code execution from untrusted payloads; input validation; optional rate limiting (e.g., 10 RPS per API key). |
| NFR‑05 | **Portability** | Works on Ubuntu 26.04 LTS with Node 20.x or Python 3.12+. |
| NFR‑06 | **Maintainability** | All source files are under version control; configuration is human‑readable YAML; plug‑ins are isolated modules. |

---

## 5. Configuration Schema

File: `proxy-config.yaml` (YAML)

```yaml
# --------------------------------------------------------------
# Global listener configuration
listen:
  host: 0.0.0.0          # IP address to bind
  port: 8080             # TCP port

# --------------------------------------------------------------
# Optional API‑key authentication
auth:
  api_keys:               # map of external key → internal token
    "sk-proxy-abc123": "token-1"
    "sk-proxy-def456": "token-2"

# --------------------------------------------------------------
# Backend model definitions
models:
  # ───────────────────────────────────────────────────────────────────────
  # Example: three “versions” of the same checkpoint (llama‑7B) that live
  # on a single llama_cpp server listening on http://127.0.0.1:8000
  # ───────────────────────────────────────────────────────────────────────
  - name: "llama-7b-default"
    backend: "llama_cpp"
    url: "http://127.0.0.1:8000"          # ← SAME URL as the next two entries
    default_params:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 1024

  - name: "llama-7b-creative"
    backend: "llama_cpp"
    url: "http://127.0.0.1:8000"          # ← SAME URL
    default_params:
      temperature: 1.2
      top_p: 0.95
      repetition_penalty: 1.1
      max_tokens: 1024

  - name: "llama-7b-precise"
    backend: "llama_cpp"
    url: "http://127.0.0.1:8000"          # ← SAME URL
    default_params:
      temperature: 0.2
      top_p: 0.8
      presence_penalty: 0.6
      max_tokens: 512

  # ───────────────────────────────────────────────────────────────────────
  # Other back‑ends can still have distinct URLs; the rule only affects
  # uniqueness of `name`, not of `url`.
  # ───────────────────────────────────────────────────────────────────────
  - name: "gpt-4o"
    backend: "openai"
    url: "https://my-azure-openai.openai.azure.com"
    api_key: "${AZURE_OPENAI_KEY}"   # environment variable substitution
    default_params:
      temperature: 0.5
      max_tokens: 2048

# --------------------------------------------------------------
# Plug‑in discovery
plugins_dir: "./plugins"   # relative to the proxy working directory
```

### 5.1 Validation Rules

| Field | Rule |
|-------|------|
| `listen.host` | Must be a valid IPv4/IPv6 address or `0.0.0.0`. |
| `listen.port` | Integer 1‑65535, not in use at startup. |
| `auth.api_keys` | Keys must be unique strings; values may be any opaque identifier. |
| `models[].name` | **Must be globally unique** (used as the OpenAI‑compatible `model` identifier). |
| `models[].backend` | Must correspond to a known adapter module (`llama_cpp`, `vllm`, `openai`, …). |
| `models[].url` | Valid URL with scheme `http` or `https`. **May appear multiple times** – duplicate URLs are allowed. |
| `models[].default_params` | Keys must be valid OpenAI sampling parameters (`temperature`, `top_p`, `max_tokens`, `repetition_penalty`, …). |
| `models[].api_key` | If present, may contain `${VAR}` placeholders that are expanded from the process environment. |
| `plugins_dir` | Path must exist and be readable. |

---

## 6. Adapter Interface (Language‑agnostic contract)

Each backend **adapter** must expose a single asynchronous function (or method) with the following signature:

```
forward({
    url: string,            // base URL of the backend (e.g. http://localhost:8000)
    apiKey?: string,        // optional auth token for the backend
    endpoint: string,       // e.g. "/v1/chat/completions"
    payload: object,        // JSON body after merging default_params
    timeoutMs?: number      // optional per‑request timeout
}) => Promise<object>       // response body already converted to OpenAI schema
```

* The adapter is responsible for translating any mismatched field names, adding backend‑specific auth headers, and converting the backend’s response into **exactly** the OpenAI JSON format (including `choices`, `usage`, `model`, etc.).
* Errors must be thrown as JavaScript `Error` (or Python `Exception`) with a message that will be turned into an OpenAI‑compatible error payload.

---

## 7. Plug‑in API

Plug‑ins are ordinary modules (JavaScript/TypeScript **or** Python) that export a single function:

```js
export function register(app) {
    // `app` is the underlying HTTP framework instance
    // (Express for Node, FastAPI for Python). Use its routing API.
}
```

### 7.1 Available Hooks

| Hook | Signature | When invoked |
|------|-----------|--------------|
| `onRequest(req, ctx)` | `async (req, ctx) => void` | Before model resolution; can modify `req.body` or add to `ctx`. |
| `onResponse(resp, ctx)` | `async (resp, ctx) => void` | After backend response, before sending to client; can edit `resp`. |
| `registerRoutes(app)` | `() => void` | Called during startup; allows arbitrary route registration. |
| `healthCheck()` | `async () => { ok: boolean, details?: any }` | Polled by `/health` endpoint; contributes to overall health status. |

* `ctx` is a per‑request mutable object (e.g., `{ startTime: Date, model: string, authToken?: string }`).

### 7.2 Example Plug‑in Skeleton (Node)

```js
// plugins/example-cache.js
import LRU from "lru-cache";

const cache = new LRU({ max: 500 });

export function register(app) {
  app.use(async (req, res, next) => {
    if (req.path !== "/v1/chat/completions") return next();

    const key = JSON.stringify({
      model: req.body.model,
      messages: req.body.messages,
      params: req.body,
    });

    const cached = cache.get(key);
    if (cached) return res.json(cached);

    const originalJson = res.json.bind(res);
    res.json = (body) => {
      cache.set(key, body);
      return originalJson(body);
    };
    next();
  });
}
```

---

## 8. Deployment on Ubuntu 26.04

### 8.1 System Packages

```bash
# Node.js implementation (recommended)
sudo apt update
sudo apt install -y nodejs npm

# Python implementation (optional)
sudo apt install -y python3 python3-pip python3-venv
```

### 8.2 Directory Layout

```
/opt/llm-proxy/
├─ src/                # source code (Node or Python)
│   ├─ index.js        # entry point
│   ├─ adapters/
│   │   ├─ llama_cpp.js
│   │   └─ openai.js
│   └─ plugins/        # optional plug‑ins
├─ config/
│   └─ proxy-config.yaml
├─ logs/
└─ venv/               # Python virtualenv (if using Python)
```

### 8.3 Service Unit (systemd)

```ini
# /etc/systemd/system/llm-proxy.service
[Unit]
Description=Simple LLM Proxy
After=network.target

[Service]
Type=simple
User=llmproxy
Group=llmproxy
WorkingDirectory=/opt/llm-proxy
ExecStart=/usr/bin/node src/index.js   # or: /opt/llm-proxy/venv/bin/python3 src/main.py
Restart=on-failure
EnvironmentFile=/opt/llm-proxy/.env   # optional env vars (e.g., AZURE_OPENAI_KEY)
StandardOutput=append:/var/log/llm-proxy/out.log
StandardError=append:/var/log/llm-proxy/err.log
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

Enable & start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable llm-proxy
sudo systemctl start llm-proxy
```

### 8.4 TLS (optional)

If TLS termination is desired inside the proxy, generate a self‑signed cert or use Let’s Encrypt and configure the underlying framework (Express `https.createServer`, FastAPI `uvicorn --ssl-keyfile …`). Otherwise place an Nginx/HAProxy in front and forward plain HTTP.

### 8.5 Monitoring

* **Prometheus** – scrape `/metrics` (exposed by the proxy). 
* **Grafana** – visualize request latency, error rate, cache hit‑rate. 
* **Systemd** – `systemctl status llm-proxy` for health, `journalctl -u llm-proxy -f` for logs.

---

## 9. Testing Strategy

| Test Type | Description | Tool |
|-----------|-------------|------|
| **Unit** | Each adapter’s `forward` function with mock HTTP server. | Jest (Node) / pytest (Python) |
| **Integration** | End‑to‑end request through the proxy to a real backend (e.g., local `llama.cpp`). | `curl` scripts, Postman collection |
| **Load** | 100 RPS sustained for 5 min, measuring added latency. | `hey`, `wrk`, or `k6` |
| **Security** | Verify API‑key enforcement, rate‑limit, and request size limits. | `curl` with missing/invalid keys, `ab` for DoS simulation |
| **Hot‑Reload** | Send `SIGHUP` while serving requests; ensure no 5xx errors. | `kill -HUP <pid>` during load test |
| **Graceful Shutdown** | Send `SIGTERM`; confirm in‑flight requests complete. | `kill -TERM <pid>` while a client is waiting |

All tests must pass on a clean Ubuntu 26.04 VM.

---

## 10. Maintenance & Extensibility

1. **Adding a new backend**  
   * Implement an adapter module conforming to the **Adapter Interface**. 
   * Add a new entry to `proxy-config.yaml` with `backend: "<module_name>"`. 
2. **Adding a new plug‑in**  
   * Create a file under `plugins/` that exports `register(app)`. 
   * Optionally expose custom routes or request/response hooks. 
3. **Versioning**  
   * Increment `SPECFICATION.md` minor version for each backward‑compatible change, major for breaking changes. 
   * Tag releases in Git (e.g., `v1.0.0`). 
4. **Documentation**  
   * Keep `README.md` up to date with installation steps. 
   * Auto‑generate API docs from code comments using `typedoc` (Node) or `pydoc` (Python). 

---

## 11. Glossary

| Term | Meaning |
|------|---------|
| **Checkpoint** | The *binary / model file* (e.g. `llama‑7B.gguf`). |
| **Version** | A *named configuration* that tells the backend **how** to run that checkpoint (temperature, top‑p, repetition‑penalty, etc.). |
| **Backend** | An LLM server exposing an OpenAI‑compatible HTTP API (e.g., `llama.cpp`, `vLLM`). |
| **Adapter** | Language‑specific module that forwards a request to a backend and normalises the response. |
| **Plug‑in** | Optional module that can register extra routes or intercept traffic. |
| **Model name** | The identifier supplied by the client in the `model` field; maps to a `models[].name` entry. |
| **Default parameters** | Sampling and generation settings defined per model in the config file. |
| **Hot reload** | Reloading configuration without stopping the process (triggered by `SIGHUP`). |
| **Graceful shutdown** | Completing in‑flight requests before exiting (triggered by `SIGTERM`). |

---

## 12. References

* OpenAI API specification – https://platform.openai.com/docs/api-reference 
* `llama.cpp` HTTP server – https://github.com/ggerganov/llama.cpp/tree/master/examples/server 
* `vLLM` – https://github.com/vllm-project/vllm 
* Prometheus client libraries – https://prometheus.io/docs/instrumenting/clientlibs/ 

---

**End of Specification**

---

## 13. Language Choice – Python vs JavaScript/TypeScript vs Compiled

| Aspect | **Python** | **JavaScript / TypeScript (Node.js)** | **Compiled (Go / Rust / C++)** |
|--------|------------|----------------------------------------|--------------------------------|
| **Development speed** | • Very rapid prototyping – batteries‑included std‑lib, REPL, dynamic typing.<br>• Rich scientific‑ML ecosystem (requests, pydantic, FastAPI). | • Fast iteration with `npm`/`yarn` scripts; hot‑reload via `nodemon`.<br>• TypeScript adds optional static typing while keeping JavaScript ergonomics. | • Slower to write the first version – need to define structs, error handling, build steps.<br>• IDE support is excellent but the mental overhead is higher. |
| **Type safety / correctness** | • Runtime‑only checks; can add `pydantic`/`dataclasses` for validation but still dynamic.<br>• Easy to miss mismatched JSON shapes until runtime. | • **TypeScript** provides compile‑time guarantees for request/response shapes, adapter signatures, and plug‑in contracts.<br>• Pure JavaScript has no safety. | • Full compile‑time type safety (Go’s static typing, Rust’s ownership model).<br>• Guarantees that adapters conform to the exact `forward` signature; impossible to pass a mismatched payload. |
| **Performance (latency & throughput)** | • Interpreted; per‑request overhead ~ 1‑2 ms for JSON parsing + async I/O.<br>• Adequate when the dominant cost is the LLM backend (tens‑to‑hundreds ms). | • V8 JIT is very fast for JSON handling; similar latency to Python in practice.<br>• Non‑blocking event loop makes it easy to handle many concurrent connections with low memory per connection. | • Native binaries have the lowest per‑request overhead (sub‑millisecond JSON + network I/O).<br>• Go’s goroutine scheduler or Rust’s async runtimes can handle massive concurrency with minimal RAM. |
| **Concurrency model** | • `asyncio` + `uvicorn` (FastAPI) – single‑threaded event loop, can spawn multiple workers with `gunicorn`/`uvicorn‑workers`.<br>• Simpler for CPU‑light workloads. | • Single‑threaded event loop (Node) + `cluster` module or PM2 for multi‑core scaling.<br>• Natural for I/O‑bound proxy. | • **Go**: goroutine per connection, built‑in scheduler; excellent for high‑connection counts.<br>• **Rust**: async/await with `tokio` or `async‑std`; zero‑cost abstractions but more boilerplate. |
| **Ecosystem for HTTP / OpenAI** | • `fastapi` + `httpx` – OpenAPI auto‑generation, pydantic validation, easy dependency injection.<br>• Mature OpenAI SDK (`openai` python) for downstream calls. | • `express` / `fastify` + `axios` or native `http` – lightweight, many middleware libraries.<br>• Official `openai-node` SDK; good for streaming responses. | • **Go**: `net/http`, `chi`, `gin`; `openai-go` SDK is emerging.<br>• **Rust**: `warp`, `actix‑web`, `hyper`; `openai-rust` community crates. |
| **Plug‑in model** | • Dynamically import Python modules (`importlib`).<br>• No compile step – plug‑ins can be dropped in at runtime. | • `require` / `import` of JS/TS files; hot‑reload possible with `nodemon`.<br>• TypeScript plug‑ins benefit from the same type definitions as core. | • Plug‑ins must be compiled into the binary or loaded as shared libraries (`dlopen`).<br>• More friction; versioning must be managed carefully. |
| **Packaging & deployment** | • `venv` + `pip` – simple, but need Python runtime on the target host.<br>• Dockerfile size ~ 120 MB (official python‑slim). | • Single‑file `node` binary + `npm ci` – similar Docker size (~ 120 MB node‑slim).<br>• Can bundle with `pkg` or `esbuild` for a single executable if desired. | • Statically linked binaries (especially Rust) → Docker image < 20 MB.<br>• No runtime interpreter needed – easier to run on minimal VMs/containers. |
| **Observability & tooling** | • `prometheus_client`, `opentelemetry` integrations are mature.<br>• Debugging with `pdb`, `ipython` REPL. | • `prom-client`, `opentelemetry-js`; built‑in `debug` logging.<br>• Node inspector (`chrome://inspect`). | • `prometheus/client_golang`, `opentelemetry-go` or `opentelemetry-rust`.<br>• Debuggers (`dlv` for Go, `gdb`/`lldb` for Rust). |
| **Team skill‑set & hiring** | • Python is ubiquitous in ML teams; most LLM engineers already know it. | • Many full‑stack engineers are comfortable with JavaScript/TS; good if the same repo also contains front‑end tooling. | • Fewer engineers are fluent in Go/Rust; hiring may be harder, but those who know them bring strong systems‑engineering expertise. |
| **Long‑term maintenance** | • Dynamic language can accumulate runtime bugs; type‑checking via `mypy` helps but is optional.<br>• Frequent library updates (e.g., FastAPI) but stable. | • TypeScript can enforce contracts, reducing regression risk.<br>• Node ecosystem evolves fast; occasional breaking changes in major versions. | • Binary compatibility is stable; once compiled you ship the exact version you tested.<br>• Language upgrades (e.g., Rust 2021 edition) are non‑breaking for most code. |
| **Community & examples** | • Many open‑source LLM proxies (e.g., `text-generation-webui` extensions) are Python‑based. | • Several Node‑based OpenAI reverse‑proxies exist (e.g., `openai-proxy`, `oai-proxy`). | • Fewer ready‑made examples, but projects like `ollama` (Go) and `tgi` (Rust) show the pattern. |

**TL;DR Summary & Recommendation**

| Goal | Best Fit |
|------|----------|
| **Rapid prototyping / tight integration with existing ML pipelines** | **Python** – you get the fastest start‑up, a familiar ML stack, and excellent OpenAI SDK support. |
| **Strong type safety **and** easy plug‑in development without a compile step** | **TypeScript (Node.js)** – adds static typing to the familiar JavaScript runtime, good for teams that already use web‑tech. |
| **Maximum throughput, minimal footprint, and zero‑runtime dependencies** | **Compiled language (Go or Rust)** – best for production environments that need to serve thousands of concurrent connections on modest VMs or edge devices. |

**Practical compromise**
1. **Core proxy in TypeScript** – you gain compile‑time guarantees for the routing/plug‑in contract while keeping the same runtime as many front‑end services.
2. **Adapters written in the language that best matches each backend** – e.g., a thin Python wrapper around `llama.cpp` if you already have a Python server, or a Go adapter for a high‑performance vLLM gateway.
3. **Plug‑ins** can be authored in TS (for most use‑cases) and optionally compiled to a native binary if a particular plug‑in has heavy CPU work (e.g., a Rust‑based reranker).

This mixed‑language approach lets you **start fast**, **maintain type safety**, and **opt‑out to compiled modules** only where performance or binary‑only deployment is a strict requirement.