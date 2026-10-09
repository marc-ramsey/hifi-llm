# OpenAI-Compatible vs llama-server Chat Endpoints: Parameter Comparison

## Overview

Both **OpenAI's `/v1/chat/completions`** and **llama-server's (`llama.cpp`) `/v1/chat/completions`** expose REST endpoints with the same URL path, but their parameter sets differ significantly. llama-server is designed to be *mostly* OpenAI-compatible while also exposing many llama.cpp-specific parameters that have no OpenAI equivalent.

---

## 1. Shared / Overlapping Parameters

These parameters exist in **both APIs**, often with the same names and similar semantics:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model identifier |
| `messages` | array | Array of message objects (`role`, `content`) |
| `stream` | bool | Enable server-sent events streaming |
| `temperature` | float (0–2) | Sampling randomness (0 = greedy) |
| `top_p` | float (0–1) | Nucleus sampling threshold |
| `top_k` | int | Limit to K most probable tokens (OpenAI: deprecated/implicit; llama.cpp: explicit) |
| `max_tokens` / `max_completion_tokens` | int | Maximum tokens to generate |
| `n` | int | Number of completions to generate |
| `stop` | string or array | Stop sequences |
| `presence_penalty` | float (-2 to 2) | Penalize tokens that appear in the text so far |
| `frequency_penalty` | float (-2 to 2) | Penalize tokens based on existing frequency |
| `logit_bias` | object/map | Map token IDs to bias values |
| `seed` | int | RNG seed for deterministic sampling |
| `response_format` | object | Constrain output (e.g., `{ "type": "json_object" }`) |
| `tools` | array | Tool/function definitions |
| `tool_choice` | string/object | Control tool calling behavior (`"auto"`, `"none"`, `"required"`, or specific tool) |
| `parallel_tool_calls` | bool | Allow multiple parallel tool calls |
| `stream_options` | object | `{ "include_usage": true }` to include token counts in stream |

---

## 2. OpenAI-Exclusive Parameters

These parameters are supported by **OpenAI** but have **no equivalent** in llama-server:

| Parameter | Type | Description |
|-----------|------|-------------|
| `service_tier` | string | Specify service tier (`"auto"`, `"flex"`, `"quiet"`) for latency optimization |
| `modalities` | array | Output modalities (e.g., `["text", "audio"]`) |
| `verbosity` | string | Response verbosity level |
| `reasoning_effort` | string | Effort level for reasoning models (`"low"`, `"medium"`, `"high"`) |
| `audio` | object | Audio output parameters (`voice`, `format`) — for Realtime API |
| `web_search_options` | object | Web search tool configuration |
| `store` | bool | Whether to store the response for distillation/evals |
| `moderation` | object | Content moderation configuration |
| `prediction` | object | Predicted output for faster responses (cache known content) |
| `logprobs` | bool | Return log probabilities of output tokens |
| `top_logprobs` | int (0–20) | Number of most likely tokens to return with logprobs |
| `user` | string | End-user identifier for abuse monitoring |
| `best_of` | int | Generate N completions server-side, return best (completions endpoint only) |
| `functions` / `function_call` | deprecated | Legacy function calling (replaced by `tools`/`tool_choice`) |

---

## 3. llama-server-Exclusive Parameters

These are **llama.cpp-specific** parameters with no OpenAI equivalent. They give fine-grained control over sampling, decoding, and generation behavior:

### Sampling Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `top_k` | int (0 = disabled) | Limit next token selection to K most probable tokens |
| `min_p` | float (0–1) | Minimum probability relative to most likely token |
| `typical_p` | float | Locally typical sampling with parameter p |
| `xtc_probability` | float (0–1) | Chance for token removal via XTC sampler |
| `xtc_threshold` | float (0–1) | Minimum probability threshold for XTC token removal |
| `dynatemp_range` | float | Dynamic temperature range: final temp ∈ [temp−range, temp+range] |
| `dynatemp_exponent` | float | Dynamic temperature exponent (controls entropy→temperature mapping) |
| `mirostat` | int (0/1/2) | Enable Mirostat sampling (0=disabled, 1=Mirostat, 2=Mirostat 2.0) |
| `mirostat_tau` | float | Mirostat target entropy parameter τ |
| `mirostat_eta` | float | Mirostat learning rate parameter η |
| `adaptive_target` | float | Adaptive sampling target entropy (negative = disabled) |
| `adaptive_decay` | float (0–0.99) | EMA decay for adaptive sampling |
| `repeat_last_n` | int | Last N tokens to consider for repetition penalty |
| `repeat_penalty` | float | Repetition penalty factor |
| `dry_multiplier` | float | DRY repetition penalty multiplier |
| `dry_base` | float (≥1.0) | DRY repetition penalty base value |
| `dry_allowed_length` | int | Tokens extending beyond this length get exponential penalty |
| `dry_penalty_last_n` | int | How many tokens to scan for repetitions (DRY) |
| `dry_sequence_breakers` | array | Sequence breakers for DRY sampling |
| `samplers` | array/string | Order of samplers applied (e.g., `["top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]`) |
| `backend_sampling` | bool | Use backend sampling instead of llama.cpp sampling |

### Generation Control

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_predict` | int | Max tokens to predict (alias: `max_completion_tokens`, `max_tokens`) |
| `n_keep` | int (-1 = all) | Number of prompt tokens to keep when context exceeds size |
| `n_discard` | int | Tokens after n_keep to discard on context shift (0 = half) |
| `n_cmpl` / `n` | int | Number of completions to generate |
| `n_cache_reuse` | int | Min chunk size to attempt KV cache reuse |
| `n_indent` | int | Minimum line indentation for code completion |
| `t_max_predict_ms` | int64 | Max time in ms for prediction phase |
| `ignore_eos` | bool | Ignore end-of-sequence token, continue generating |
| `echo` | bool | Echo input tokens in output |
| `n_probs` / `logprobs` | int | Output probabilities of top N tokens per generated token |
| `post_sampling_probs` | bool | Return probabilities after applying the sampling chain |

### Context & Caching

| Parameter | Type | Description |
|-----------|------|-------------|
| `cache_prompt` | bool | Reuse KV cache from a previous request (KV shifting) |
| `sse_ping_interval` | int | SSE ping interval in seconds (-1 = disabled) |

### Grammar & Constrained Generation

| Parameter | Type | Description |
|-----------|------|-------------|
| `json_schema` | object | JSON schema for constrained output (converted to grammar internally) |
| `grammar` | string | GBNF grammar string for constrained generation |
| `grammar_lazy` | bool | Apply grammar constraints lazily (only when triggered) |
| `grammar_triggers` | array | Strings/patterns that trigger grammar-constrained generation |

### Chat / Template Control

| Parameter | Type | Description |
|-----------|------|-------------|
| `chat_format` | int | Internal chat format selector |
| `reasoning_format` | string | Reasoning format for chain-of-thought models |
| `reasoning_in_content` | bool | Include reasoning in content field |
| `generation_prompt` | string | Prompt appended to chat template output |
| `parse_tool_calls` | bool | Parse tool calls from generated output |
| `chat_parser` | string | Chat parser configuration |
| `continue_final_message` | bool/obj | Continue the final message of the chat template |
| `chat_template_kwargs` | object | Extra params for Jinja2 templating (e.g., `{ "enable_thinking": false }`) |

### Reasoning Budget Control

| Parameter | Type | Description |
|-----------|------|-------------|
| `reasoning_control` | bool | Enable real-time reasoning control (can end early via API) |
| `reasoning_budget_tokens` | int (-1 = disabled) | Token budget for reasoning/thinking phase |
| `reasoning_budget_start_tag` | string | Token marking start of reasoning budget section |
| `reasoning_budget_end_tags` | array | Tokens marking end of reasoning budget section |
| `reasoning_budget_message` | string | Message prepended when forcing reasoning end |

### LoRA Adapters

| Parameter | Type | Description |
|-----------|------|-------------|
| `lora` | array | List of `{ "id": int, "scale": float }` for LoRA adapter application per-request |

### Advanced / Debug

| Parameter | Type | Description |
|-----------|------|-------------|
| `verbose` | bool | Include `__verbose` field with debug info in response |
| `timings_per_token` | bool | Include per-token timing in each streamed response |
| `return_tokens` | bool | Return raw generated token IDs in `tokens` field |
| `return_progress` | bool | Include prompt processing progress events in stream |
| `response_fields` | array | Select which fields to return (e.g., `["generation_settings/n_predict"]`) |
| `preserved_tokens` | array | Token strings that must not be split during tokenization |

---

## 4. Key Differences Summary

### Scope & Philosophy
- **OpenAI**: Curated parameter set focused on what's exposed by their hosted service. Simpler, fewer knobs.
- **llama-server**: Broad parameter set exposing the full llama.cpp sampling/decoding pipeline. Designed for local/self-hosted inference where you want fine-grained control.

### Sampling Richness
llama-server supports **~20+ distinct sampling strategies** (top_k, top_p, min_p, typical_p, XTC, DRY, Mirostat 1/2, adaptive, dynamic temperature, custom sampler ordering) that have no OpenAI equivalents. OpenAI only exposes `temperature`, `top_p`, `frequency_penalty`, and `presence_penalty`.

### Constrained Generation
- **OpenAI**: Structured outputs via `response_format` with `json_schema` type (native schema enforcement).
- **llama-server**: Supports both `json_schema` (converted to GBNF grammar internally) and raw `grammar` strings for full GBNF constraint control.

### Reasoning Models
- **OpenAI**: `reasoning_effort` parameter (`"low"`/`"medium"`/`"high"`) — abstracted, opaque.
- **llama-server**: Granular reasoning budget with token limits, start/end tags, real-time control via `/v1/chat/completions/control`, and configurable output format.

### Context Management
- **OpenAI**: No explicit context management parameters (handled server-side).
- **llama-server**: `n_keep`, `n_discard`, `cache_prompt`, `n_cache_reuse` — full KV cache control.

### Multimodal
- **OpenAI**: Native support via `modalities`, `audio`, vision in messages.
- **llama-server**: Multimodal via `--mmproj` projector; images/audio/video supported in message content but requires specific model setup.

### Tool Calling
Both support OpenAI-style `tools`/`tool_choice`/`parallel_tool_calls`. llama-server additionally supports Anthropic-style tool calling via `/v1/messages`.

### Additional Endpoints (llama-server only)
- `/v1/responses` — OpenAI Responses API compatibility
- `/v1/messages` — Anthropic Messages API compatibility
- `/v1/systemone` — TypeSafe decision model API
- `/rerank` / `/reranking` — Document reranking
- `/embeddings` — Embeddings (both OpenAI-compatible and non-compatible)
- `/infill` — Code infilling (FIM)
- `/tokens/count` — Token counting
- `/slots` — Slot/KV cache state inspection
- `/props` — Server properties
- `/metrics` — Prometheus metrics
- `/lora-adapters` — LoRA adapter management

---

## 5. Compatibility Notes

llama-server explicitly states: *"While no strong claims of compatibility with OpenAI API spec are being made, in our experience it suffices to support many apps."*

**Practical compatibility**: The `openai` Python SDK and most OpenAI-compatible clients work with llama-server for basic chat completions. However:

- Parameters like `logprobs`, `top_logprobs`, `best_of`, `user`, `service_tier`, `modalities`, `audio`, `store`, `moderation`, `web_search_options`, `prediction`, and `reasoning_effort` will be **silently ignored** by llama-server.
- llama-server-specific parameters like `mirostat`, `min_p`, `typical_p`, `xtc_*`, `dry_*`, `samplers`, `grammar`, `lora`, `n_keep`, etc. have no effect on OpenAI's API (and would cause errors if sent).
- llama-server uses the `openai` Python SDK with `base_url` pointing to its server — this is the recommended integration pattern.
