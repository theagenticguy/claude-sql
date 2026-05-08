---
name: cl100k_base overcounts Anthropic tokens ~22%; use count_tokens or empirical cache-response feedback
description: A prompt measured at 1258 cl100k tokens was BELOW Sonnet 4.6's 1024-token cache minimum; an empirical 22% Anthropic-to-cl100k ratio is safer than assuming parity.
type: reference
---

OpenAI's `tiktoken` `cl100k_base` encoder is the default Python reference
for token counting, but it **overcounts Anthropic tokens by ~22%** on
English prose. A prompt measured at 1258 cl100k tokens (4929 chars) came
in at ~981 Anthropic tokens — below Sonnet 4.6's 1024-token cache
minimum. The `cache_control: ephemeral` marker fires silently with zero
cache create or read; a naive "my prompt is 1258 tokens" check misses it.

**Empirical ratio from 2026-05 real Bedrock traffic**:
`anthropic_tokens ≈ cl100k_tokens × 0.78`

Verification options (ordered by reliability):
  1. `anthropic.Anthropic().messages.count_tokens(model=..., system=..., messages=...)` — authoritative, requires an API key.
  2. Real Bedrock call's `usage` response — check `cache_creation_input_tokens` > 0 on first call and `cache_read_input_tokens` > 0 on subsequent calls within the 5-minute TTL.
  3. cl100k count × 0.78 as a conservative pre-flight estimate; pad to 1700+ cl100k to have a safety margin above Sonnet's 1024 Anthropic-token floor.

**Model-specific cache minimums** (AWS Bedrock prompt-caching docs, 2026-05):
  - Claude Opus 4 / Sonnet 3.7 / Sonnet 4.5 / Sonnet 4.6: **1024 tokens**
  - Claude Haiku 4.5: **4096 tokens**  (4× higher — a 1300-token prompt that caches on Sonnet will NOT cache on Haiku)

Payload shape (new):
```python
usage = {
    "input_tokens": ...,               # NEW tokens (not cached)
    "output_tokens": ...,
    "cache_creation_input_tokens": ...,  # ~1.25× input rate (5m)
    "cache_read_input_tokens": ...,     # 0.1× input rate
    "cache_creation": {                 # present with 1h TTL
        "ephemeral_5m_input_tokens": ...,
        "ephemeral_1h_input_tokens": ...,  # 2× input rate
    },
}
```

Total billed input is `input_tokens + cache_creation_input_tokens +
cache_read_input_tokens` — `input_tokens` alone is only the NEW tokens
after the last cache breakpoint.

**When you hit this**: you added `cache_control` but cache hit rate is
0%. Check the Bedrock usage response, not the tiktoken estimate.
