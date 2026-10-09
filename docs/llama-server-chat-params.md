# Complete JSON Body for llama-server `/v1/chat/completions`

Every field below can be placed in the request body sent to `POST /v1/chat/completions`. The fields are processed by two layers:

1. **Schema layer** (`server-schema.cpp`) — explicitly declared parameters
2. **Pass-through layer** (`server-common.cpp`) — anything not consumed gets forwarded to llama's sampling layer

---

## Standard OpenAI-Compatible Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model identifier |
| `messages` | array | Chat messages (role + content) |
| `stream` | bool | Enable SSE streaming |
| `max_tokens` / `max_completion_tokens` | int | Max tokens to generate (aliases for `n_predict`) |
| `n` | int | Number of completions to generate |
| `stop` | string or array | Stop sequences |
| `temperature` | float | Sampling temperature (0 = greedy) |
| `top_p` | float (0–1) | Nucleus sampling threshold |
| `top_k` | int | Limit to K most probable tokens (0 = disabled) |
| `presence_penalty` | float (-2 to 2) | Penalize tokens that appear in text so far |
| `frequency_penalty` | float (-2 to 2) | Penalize tokens by existing frequency |
| `logit_bias` | object or array | Map token IDs/strings to bias values (or `[token, bias]` pairs; use `false` to ban) |
| `seed` | int | RNG seed (-1 = random) |
| `response_format` | object | Constrain output format (`{ "type": "json_object" }`) |
| `tools` | array | Tool/function definitions |
| `tool_choice` | string or object | Control tool calling (`"auto"`, `"none"`, `"required"`, or `{ "type": "function", "name": "..." }`) |
| `parallel_tool_calls` | bool | Allow multiple parallel tool calls |
| `stream_options` | object | `{ "include_usage": true }` to include token counts in stream |

---

## Sampling Parameters (llama.cpp-native)

| Field | Type | Description |
|-------|------|-------------|
| `min_p` | float (0–1) | Minimum probability relative to most likely token |
| `typical_p` | float | Locally typical sampling (1.0 = disabled) |
| `xtc_probability` | float (0–1) | Chance for token removal via XTC sampler |
| `xtc_threshold` | float (0–1) | Minimum probability threshold for XTC removal (> 0.5 disables) |
| `top_n_sigma` | float | Keep tokens within n standard deviations of top logit (< 0 = disabled) |
| `dynatemp_range` | float | Dynamic temperature range: final temp ∈ [temp−range, temp+range] |
| `dynatemp_exponent` | float | Dynamic temperature exponent (entropy→temperature mapping) |
| `repeat_last_n` | int | Last N tokens for repetition penalty (0 = disabled) |
| `repeat_penalty` | float | Repetition penalty factor (1.0 = disabled) |
| `mirostat` | int (0/1/2) | Mirostat sampling (0=disabled, 1=Mirostat, 2=Mirostat 2.0) |
| `mirostat_tau` | float | Mirostat target entropy τ |
| `mirostat_eta` | float | Mirostat learning rate η |
| `adaptive_target` | float | Adaptive sampling target entropy (negative = disabled) |
| `adaptive_decay` | float (0–0.99) | EMA decay for adaptive sampling |
| `samplers` | array or string | Order of samplers applied (e.g., `["top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]`) |
| `backend_sampling` | bool | Use backend sampling instead of llama.cpp sampling |

---

## DRY Sampling (Don't Repeat Yourself)

| Field | Type | Description |
|-------|------|-------------|
| `dry_multiplier` | float | DRY repetition penalty multiplier (0 = disabled) |
| `dry_base` | float (≥1.0) | DRY repetition penalty base value |
| `dry_allowed_length` | int | Tokens beyond this get exponential penalty |
| `dry_penalty_last_n` | int | How many tokens to scan for repetitions |
| `dry_sequence_breakers` | array of strings | Sequence breakers for DRY (e.g., `["\n", ":", "\""]`) |

---

## Reasoning / Thinking Parameters

| Field | Type | Description |
|-------|------|-------------|
| `reasoning_effort` | string (`"low"` / `"medium"` / `"high"` / `"none"`) | Passed to Jinja2 template. `"none"` disables thinking entirely |
| `reasoning_format` | string (`"deepseek"` / `"deepseek-legacy"`) | How reasoning content is returned in response |
| `reasoning_budget_tokens` | int (-1 = disabled) | Token budget for the reasoning/thinking phase |
| `thinking_budget_tokens` | int | Alias for `reasoning_budget_tokens` |
| `reasoning_budget_start_tag` | string | Start tag of thinking section (usually auto-populated from template) |
| `reasoning_budget_end_tags` / `reasoning_budget_end_tag` | array or string | End tags that terminate thinking (usually auto-populated) |
| `reasoning_budget_message` | string | Message prepended when forcing reasoning end |
| `reasoning_control` | bool | Enable real-time control to end reasoning early via `/v1/chat/completions/control` |

---

## Constrained Generation

| Field | Type | Description |
|-------|------|-------------|
| `json_schema` | object | JSON schema for constrained output (converted to GBNF grammar) |
| `grammar` | string | Raw GBNF grammar string for constrained generation |
| `grammar_lazy` | bool | Apply grammar constraints lazily (only when triggered) |
| `grammar_triggers` | array of strings | Patterns that trigger grammar-constrained generation |

---

## Chat / Template Control

| Field | Type | Description |
|-------|------|-------------|
| `chat_template_kwargs` | object | Extra params for Jinja2 template (e.g., `{ "enable_thinking": true, "reasoning_effort": "high" }`) |
| `chat_format` | int | Internal chat format selector |
| `reasoning_format` | string | Reasoning output format (`"deepseek"` / `"deepseek-legacy"`) |
| `generation_prompt` | string | Text appended after the chat template output |
| `parse_tool_calls` | bool | Parse tool calls from generated output |
| `chat_parser` | string | Chat parser configuration string |
| `continue_final_message` | object or bool | Continue the final message of the chat template (used for "Continue" button) |
| `add_generation_prompt` | bool | Whether to add generation prompt at end |
| `echo` | bool | Echo input tokens in output |

---

## Context / KV Cache Control

| Field | Type | Description |
|-------|------|-------------|
| `cache_prompt` | bool | Reuse KV cache from a previous request (KV shifting) |
| `n_keep` | int (-1 = all) | Tokens from initial prompt to keep when context exceeds size |
| `n_discard` | int | Tokens after n_keep to discard on context shift (0 = half) |
| `n_cache_reuse` | int | Min chunk size to attempt KV cache reuse |

---

## Generation Control

| Field | Type | Description |
|-------|------|-------------|
| `n_predict` | int | Max tokens to predict (aliases: `max_tokens`, `max_completion_tokens`) |
| `n_indent` | int | Minimum line indentation for code completion |
| `t_max_predict_ms` | int64 | Max time in ms for prediction phase |
| `ignore_eos` | bool | Ignore end-of-sequence token, continue generating |
| `n_probs` / `logprobs` | int | Output probabilities of top N tokens per generated token |
| `post_sampling_probs` | bool | Return probabilities after applying the sampling chain |
| `preserved_tokens` | array of strings | Token strings that must not be split during tokenization |

---

## LoRA Adapters

| Field | Type | Description |
|-------|------|-------------|
| `lora` | array of `{ "id": int, "scale": float }` | Per-request LoRA adapter application. Unlisted adapters default to scale 0.0 |

---

## Speculative Decoding

| Field | Type | Description |
|-------|------|-------------|
| `speculative.n_max` | int | Max draft tokens during speculative decoding |
| `speculative.n_min` | int | Min draft tokens for speculative decoding |
| `speculative.p_min` | float (0–1) | Min probability for draft tokens |
| `speculative.type` | string | Speculative decoding method |
| `speculative.ngram_size_n` | int | N-gram size for ngram-based speculative decoding |
| `speculative.ngram_size_m` | int | M-gram size for speculative decoding |
| `speculative.ngram_min_hits` | int | Min hits at ngram lookup |

---

## Debug / Observability

| Field | Type | Description |
|-------|------|-------------|
| `verbose` | bool | Include `__verbose` field with debug info in response |
| `timings_per_token` | bool | Include per-token timing in each streamed response |
| `return_tokens` | bool | Return raw generated token IDs in `tokens` field |
| `return_progress` | bool | Include prompt processing progress events in stream mode |
| `sse_ping_interval` | int | SSE ping interval in seconds (-1 = disabled) |
| `response_fields` | array of strings | Select which response fields to return (e.g., `["generation_settings/n_predict"]`) |

---

## Catch-All: Any Unknown Field

The pass-through at the end of `server-common.cpp`:

```cpp
for (const auto & item : body.items()) {
    if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
        llama_params[item.key()] = item.value();
    }
}
```

This means **any unrecognized field** in the JSON body gets forwarded directly to llama's sampling layer. So if a future version of llama.cpp adds a new parameter, it will work via pass-through even before it's formally declared in the schema.

---

## Example: Full Reasoning Request

```json
{
  "model": "kimi-k2-instruct",
  "messages": [
    { "role": "user", "content": "Explain quantum entanglement" }
  ],
  "stream": true,
  "reasoning_effort": "high",
  "chat_template_kwargs": {
    "enable_thinking": true
  },
  "reasoning_budget_tokens": 2048,
  "reasoning_control": true,
  "temperature": 0.8,
  "top_p": 0.95,
  "min_p": 0.05,
  "max_tokens": 4096,
  "stop": ["</think>"],
  "response_format": { "type": "json_object" }
}
```
