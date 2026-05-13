---
title: Pass the per-pipeline accumulator key as an explicit kwarg, not via schema metadata
category: best-practices
tags: [observability, threadsafe-accumulator, classify_one, structured-output, silent-no-op]
modules: [src/claude_sql/llm_shared.py, src/claude_sql/trajectory_worker.py, src/claude_sql/conflicts_worker.py, src/claude_sql/friction_worker.py]
added: 2026-05-13
---

## What happened

`llm_shared.pipeline_cache_stats("trajectory")` registered a
`_CACHE_STATS["trajectory"]` bucket via the context manager. Inside
`classify_one`, the per-call cache-hit accumulator was being looked up
by `schema.get("title")` — which on the trajectory pipeline resolved
to `"TrajectoryArrayResult"`, not `"trajectory"`. The accumulator
silently no-op'd for hours: `_CACHE_STATS["TrajectoryArrayResult"]`
didn't exist, so every increment hit the `bucket = _CACHE_STATS.get(...)`
None branch and returned. Cache-hit telemetry showed zeros across the
v1.0 windowed run despite Bedrock `usage` clearly reporting
`cache_read_input_tokens > 0`.

## Why it happened

The accumulator's *registration* key (the string the caller chose) and
its *lookup* key (whatever the call site happened to derive from data
in scope) were different shapes. The registration call had access to
the pipeline name as a string literal. The increment call site only
had `schema` and `client` — and the schema's `title` field is a
pydantic-generated artifact (`f"{ClassName}"`), not the pipeline
identity. They aliased correctly for the single-class pipelines
(classifier, friction) by accident, then drifted the moment we
introduced an array-wrapped output schema (`TrajectoryArrayResult`
wraps `windows: list[TrajectoryWindow]`).

The general anti-pattern: **inferring an identity key from incidental
metadata**. Whenever a caller registers state under one name and a
callee reads state under a different name, the link is fragile.
Schema titles, class names, and function names are all incidental —
they change for refactoring reasons that have nothing to do with the
accumulator's lifecycle.

## Fix

Thread the pipeline name as an explicit `pipeline: str` kwarg through
`classify_one` (and any helper between the context-manager boundary
and the increment call). Default to `"classifier"` for back-compat.
Each worker passes its own pipeline name explicitly:

```python
# llm_shared.py
async def classify_one(
    client, model_id, schema, text, *,
    max_tokens, thinking_mode, sem,
    system: str | None = None,
    pipeline: str = "classifier",  # <-- explicit, not derived
) -> dict:
    ...

# trajectory_worker.py
result = await classify_one(
    ...,
    pipeline="trajectory",  # matches pipeline_cache_stats("trajectory")
)
```

The increment site then looks up `_CACHE_STATS[pipeline]` with the
exact same string the context manager registered. The link is
typo-checkable at review time; there's no schema-metadata coupling.

## How to recall

- Symptom: per-pipeline observability counter reports zeros despite
  the underlying call clearly producing the events being counted.
- Symptom: the counter's accumulator dict has a key with a different
  shape than the lookup key (e.g., `"TrajectoryArrayResult"` vs
  `"trajectory"`).
- Trigger: any time you register state under a string and look it up
  somewhere else, ask "could the registration string and lookup
  string ever drift?" If the lookup derives from data shape (schema
  title, class name, type repr), assume they will drift.
- Search keywords: `_CACHE_STATS`, `pipeline_cache_stats`,
  `classify_one`, "silent no-op", "accumulator key mismatch".

## References

- src/claude_sql/llm_shared.py:147 (`_CACHE_STATS`), :207 (`_increment`),
  :259 (`pipeline_cache_stats`), :573 (`classify_one(pipeline=...)`).
- src/claude_sql/trajectory_worker.py:643 (`pipeline="trajectory"`).
- src/claude_sql/conflicts_worker.py + src/claude_sql/friction_worker.py
  (parallel call sites).
- Cross-ref: best-practices/anyio-structured-concurrency-for-blocking-io.md
  (the structured-concurrency boundary that owns the context manager's
  lifetime).
