---
title: Bedrock has TWO structured-output shapes — pick output_config.format, not Converse, when matching the existing claude-sql pipeline
track: knowledge
category: api-patterns
module: src/claude_sql/llm_worker.py
component: boto3 bedrock-runtime
severity: warning
tags: [bedrock, sonnet, structured-output, invoke-model, converse, refusal]
applies_when:
  - "Adding a new Sonnet 4.6 (or any Anthropic-on-Bedrock) call site that expects structured JSON"
  - "Choosing between matching the existing claude-sql llm_worker pipeline vs starting Converse-fresh"
pattern: |
  Bedrock exposes two structured-output APIs that look similar but have DIFFERENT field paths
  AND different stop_reason vocabularies:

  1. **InvokeModel + Anthropic-native body** (what claude-sql llm_worker uses today):
     - body shape: `{"anthropic_version": "bedrock-2023-05-31", "output_config": {"format": {"type": "json_schema", "schema": <flat-dict>}}, ...}`
     - `stop_reason == "refusal"` is Anthropic-native — handle with `BedrockRefusalError`
     - Schema is a flat Python dict (model_json_schema → flatten via _bedrock_schema)
     - Adaptive thinking via `"thinking": {"type": "adaptive"}`
     - System prompt: list of content blocks; cache_control on the system block enables prompt caching

  2. **Converse API**:
     - request shape: `outputConfig.textFormat.structure.jsonSchema.{schema, name, description}` where `schema` is a STRINGIFIED JSON
     - stop_reason enum: `guardrail_intervened`, `content_filtered`, `malformed_model_output` — NO `refusal`
     - Different refusal-detection logic required

  When extending claude-sql, **stay on path 1**. Reasons:
    - Reuse `_build_bedrock_client`, `_invoke_classifier_sync`, `_parse_structured_payload`, `BedrockRefusalError` for free
    - Match the `output_config.format` shape used by classify / friction / trajectory / conflicts workers
    - Don't duplicate refusal-detection logic with two different stop_reason vocabularies

  If a future use case needs Converse (cross-provider, e.g. mixing Claude + Llama in one panel),
  document the refusal-handling divergence explicitly.
example_files:
  - src/claude_sql/llm_worker.py        # canonical InvokeModel + output_config.format pipeline
  - src/claude_sql/friction_worker.py   # reuses llm_worker private surface for the second consumer
  - src/claude_sql/review_sheet_worker.py  # third consumer; same pattern
  - src/claude_sql/judge_worker.py      # the ONE Converse consumer; cross-provider only
---

# Why this matters

A subagent reading "Bedrock structured output" docs in 2026 will land on Converse first
(it's the AWS-promoted API for new code). Following that path silently breaks `BedrockRefusalError`
and divorces the new worker from the rest of the pipeline's resilience patterns (shared client,
pool sizing, tenacity, refusal short-circuit).

The existing claude-sql llm_worker uses InvokeModel for a reason: prompt-caching support, adaptive
thinking, and the Anthropic-native `stop_reason: "refusal"` terminal signal. Converse doesn't have
the third, has a different system-prompt shape, and needs its own retry/refusal logic.

When a session researches "Bedrock structured output" from scratch, surface this finding so the
subagent doesn't burn a Bedrock pipeline rewrite for a third consumer.

# Example

```python
# CORRECT — matches existing pipeline
from claude_sql.llm_worker import _build_bedrock_client, _invoke_classifier_sync, BedrockRefusalError
from claude_sql.schemas import PR_REVIEW_SHEET_SCHEMA

client = _build_bedrock_client(settings)
try:
    payload = _invoke_classifier_sync(
        client,
        model_id="global.anthropic.claude-sonnet-4-6",
        schema=PR_REVIEW_SHEET_SCHEMA,
        user_text=transcript_text,
        max_tokens=2048,
        thinking_mode="adaptive",
        system=system_prompt_with_xml_tags,
    )
except BedrockRefusalError as e:
    return {"refused": True, "reason": str(e), "metadata": metadata}

# WRONG — Converse path is structurally incompatible:
# - schema goes under outputConfig.textFormat.structure.jsonSchema as a STRING
# - no anthropic-native refusal stop_reason
# - cannot reuse _parse_structured_payload, _classify_one, etc.
# response = client.converse(modelId=..., outputConfig={"textFormat": {"structure": {"jsonSchema": {"schema": json.dumps(SCHEMA), "name": "PRReviewSheet"}}}}, ...)
```
