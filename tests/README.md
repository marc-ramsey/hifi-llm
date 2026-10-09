# LLM-Proxy E2E Tests

All tests run the proxy as a real subprocess backed by a **real threaded HTTP server** (a lightweight OpenAI-compatible mock server) — no external services required.

## Running

```bash
# All tests
python3 -m pytest tests/ -v

# Specific suite
python3 -m pytest tests/test_routes.py -v
python3 -m pytest tests/test_errors.py -v
python3 -m pytest tests/test_hot_reload.py -v
python3 -m pytest tests/test_shutdown.py -v
```

## Test Structure

| File | Tests |
|------|-------|
| `test_routes.py` | `/health`, `/v1/models`, `/v1/chat/completions` (stream+json), `/v1/embeddings`, `/v1/completions` |
| `test_errors.py` | Invalid model names (400), backend 404→502, streaming error handling |
| `test_hot_reload.py` | SIGHUP adds model, in-flight requests survive reload, invalid config keeps old |
| `test_shutdown.py` | SIGTERM during streaming request completes, process exits cleanly |

## Requirements

- None — the test backend starts automatically as a threaded HTTP server fixture.
