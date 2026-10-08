# LLM-Proxy E2E Tests

All tests are **live-endpoint** tests — they run the proxy as a real subprocess and send real HTTP requests to Arkestra (on `:8080`).

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

- Arkestra running at `http://127.0.0.1:8080` with `gemma-4-26B-instruct` loaded
- Embedding tests are skipped if `nomic-embed` is not loaded in Arkestra
