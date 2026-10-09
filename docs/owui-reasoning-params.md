# How llama-server Handles `reasoning_effort` (and Reasoning Parameters) — Mid-2026 Update

## The Key Discovery

llama-server **does** now accept `reasoning_effort` at the chat completion API level. It was added as both a CLI arg and an API parameter. Here's exactly how it works:

---

## 1. CLI Level (`--reasoning-effort`)

From `common/arg.cpp`:
```cpp
{"--reasoning-effort"}, "LEVEL",
"reasoning effort level given to the chat template: 'default' to keep the template default,\n"
"'low', 'medium', or 'high' (if supported by the template)", ...
    params.default_template_kwargs["reasoning_effort"] = json(value).dump();
```

This sets a **server-wide default** that gets baked into every chat request's Jinja2 template context.

---

## 2. API Level — It Goes Through `chat_template_kwargs`

In `tools/server/server-common.cpp`, the chat completion handler processes reasoning params like this:

```cpp
// Step 1: Merge CLI defaults with per-request kwargs
auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
inputs.chat_template_kwargs = opt.chat_template_kwargs;  // CLI defaults first
for (const auto & item : chat_template_kwargs_object.items()) {
    inputs.chat_template_kwargs[item.key()] = item.value().dump();
}

// Step 2: Parse "enable_thinking" kwarg
auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
if (enable_thinking_kwarg == "true")    inputs.enable_thinking = true;
else if (enable_thinking_kwarg == "false")  inputs.enable_thinking = false;

// Step 3: Parse OAI "reasoning_effort" field directly from the request body!
if (body.contains("reasoning_effort")) {
    auto reasoning_effort = json_value(body, "reasoning_effort", std::string(""));
    if (reasoning_effort == "none") {
        inputs.enable_thinking = false;
        inputs.chat_template_kwargs.erase("reasoning_effort");
    } else if (!reasoning_effort.empty()) {
        inputs.chat_template_kwargs["reasoning_effort"] = json(reasoning_effort).dump();
    }
}
```

### So `reasoning_effort` can be sent in TWO ways:

**A) Direct field on the request body** (OpenAI-compatible):
```json
{
  "model": "...",
  "messages": [...],
  "reasoning_effort": "medium"
}
```

**B) Nested inside `chat_template_kwargs`**:
```json
{
  "model": "...",
  "messages": [...],
  "chat_template_kwargs": {
    "reasoning_effort": "low"
  }
}
```

Both paths end up in the same place: `inputs.chat_template_kwargs["reasoning_effort"]`.

---

## 3. How It Reaches the Jinja2 Template

The merged kwargs flow into `common_chat_templates_apply()`, which eventually calls the Jinja2 template engine. In `common/jinja/caps.cpp`:

```cpp
void caps_apply_reasoning_effort(jinja::context & ctx, const std::string & effort) {
    value var = mk_val<value_string>(effort);
    ctx.set_val("reasoning_effort",   var);
    ctx.set_val("reasoning_strength", var);  // alias for compatibility
}
```

The template can then use `{{ reasoning_effort }}` or `{{ reasoning_strength }}` to control thinking/reasoning behavior. This is how models like DeepSeek, Kimi K2, and others get their effort-level instructions into the prompt.

---

## 4. Other Reasoning Parameters Available at the API Level

Beyond `reasoning_effort`, llama-server now accepts these reasoning-specific fields directly in the chat completion body:

| Parameter | Type | Description |
|-----------|------|-------------|
| `reasoning_effort` | string (`"low"` / `"medium"` / `"high"` / `"none"`) | Passed to Jinja2 template. `"none"` disables thinking entirely. |
| `enable_thinking` | bool (via `chat_template_kwargs`) | Explicitly enable/disable reasoning/thinking mode |
| `reasoning_budget_tokens` | int | Token budget for the reasoning phase (-1 = disabled) |
| `thinking_budget_tokens` | int | Alias for `reasoning_budget_tokens` |
| `reasoning_budget_start_tag` | string (auto-populated from template) | Start tag of thinking section |
| `reasoning_budget_end_tags` | array (auto-populated from template) | End tags that terminate thinking |
| `reasoning_budget_message` | string | Message prepended when forcing reasoning end |
| `reasoning_control` | bool | Enable real-time control to end reasoning early via `/v1/chat/completions/control` |
| `reasoning_format` | string (`"deepseek"` / `"deepseek-legacy"`) | How reasoning content is returned in the response |

### How they're wired:

```cpp
// Reasoning budget: pass parameters through to sampling layer
{
    int reasoning_budget = json_value(body, "reasoning_budget_tokens",
                           json_value(body, "thinking_budget_tokens", -1));
    if (reasoning_budget == -1) {
        reasoning_budget = opt.reasoning_budget;  // fall back to CLI default
    }

    if (!chat_params.thinking_end_tags.empty()) {
        llama_params["reasoning_budget_tokens"] = reasoning_budget;
        llama_params["reasoning_budget_start_tag"] = chat_params.thinking_start_tag;
        llama_params["reasoning_budget_end_tags"] = chat_params.thinking_end_tags;
        llama_params["reasoning_budget_message"] = json_value(body, "reasoning_budget_message", opt.reasoning_budget_message);
        llama_params["reasoning_control"] = json_value(body, "reasoning_control", false);
    }
}
```

Note: `reasoning_budget_tokens` and related params are passed through to the **sampling layer** (the actual token generation), not just the template. This is different from `reasoning_effort` which only affects the Jinja2 template.

---

## 5. What OWUI Actually Sends (Revised)

Now let's re-examine OWUI's behavior with this knowledge:

### OWUI's `apply_model_params_to_body_openai()`:
```python
mappings = {
    'reasoning_effort': str,   # ← FORWARDED as-is to the request body
    ...
}
```

So OWUI **does** forward `reasoning_effort` in the payload. When the upstream is llama-server, it gets picked up by the code path above:

```cpp
if (body.contains("reasoning_effort")) {
    inputs.chat_template_kwargs["reasoning_effort"] = json(reasoning_effort).dump();
}
```

### But OWUI still doesn't expose other reasoning params:
- `reasoning_budget_tokens` — **not in OWUI's param mappings**
- `reasoning_format` — **not in OWUI's param mappings**
- `reasoning_control` — **not in OWUI's param mappings**
- `enable_thinking` — **not in OWUI's param mappings**
- `reasoning_budget_message` — **not in OWUI's param mappings**

### And OWUI has a conversion issue:

In `openai.py`, for non-OpenAI models:
```python
elif 'api.openai.com' not in url:
    # Remove "max_completion_tokens" from the payload for backward compatibility
    if 'max_completion_tokens' in payload:
        payload['max_tokens'] = payload['max_completion_tokens']
        del payload['max_completion_tokens']
```

This means if a user sets `reasoning_effort` on an OWUI model configured with llama-server, it **will** reach llama-server correctly. But the `max_tokens` → `max_completion_tokens` ↔ `max_tokens` conversion dance could cause issues if both are present.

---

## 6. Summary: What Works vs What Doesn't Through OWUI → llama-server

| Parameter | CLI Default | API Body Field | Works through OWUI? |
|-----------|-------------|----------------|---------------------|
| `reasoning_effort` | `--reasoning-effort LEVEL` | `reasoning_effort` in body | ✅ **YES** — forwarded as-is by OWUI |
| `enable_thinking` | (none) | `chat_template_kwargs.enable_thinking` | ❌ No UI for it |
| `reasoning_budget_tokens` | `--reasoning-budget N` | `reasoning_budget_tokens` in body | ❌ Not in OWUI's mappings |
| `reasoning_format` | `--reasoning-format FORMAT` | `reasoning_format` in body | ❌ Not in OWUI's mappings |
| `reasoning_control` | (none) | `reasoning_control` in body | ❌ Not in OWUI's mappings |
| `reasoning_budget_message` | `--reasoning-budget-message MSG` | `reasoning_budget_message` in body | ❌ Not in OWUI's mappings |
| `reasoning_budget_start_tag` | (auto from template) | auto-populated | N/A — templated |
| `reasoning_budget_end_tags` | (auto from template) | auto-populated | N/A — templated |

### The bottom line:

**`reasoning_effort` does work through OWUI → llama-server.** OWUI's `AdvancedParams.svelte` exposes it, it passes through the param mapping, and llama-server picks it up from the request body. This was the missing piece in my earlier analysis.

However, the richer reasoning control that llama-server offers (budget tokens, real-time control, format selection) is **not exposed** through OWUI's UI. If you need those, you'd need to set them via CLI args on the server or bypass OWUI entirely.
