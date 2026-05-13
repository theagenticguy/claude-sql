---
title: Test fakes that monkeypatch wide-fanout helpers must absorb future kwargs (`**_kw`)
category: conventions
tags: [testing, monkeypatch, fakes, signature-evolution, classify_one]
modules: [tests/test_analyze_chain.py, tests/test_trajectory_worker.py, tests/test_conflicts_worker.py, tests/test_friction_worker.py]
added: 2026-05-13
---

## What happened

The v1.0 windowed-pipelines change added a `pipeline: str = "classifier"`
kwarg to `claude_sql.llm_shared.classify_one`. Production callers
(trajectory, conflicts, friction workers) immediately started passing
`pipeline="<name>"`. Nine tests across three modules failed with
`TypeError: <fake>() got an unexpected keyword argument 'pipeline'`,
because each test's monkeypatch fake declared the *exact* historical
signature:

```python
async def _fake(client, model_id, schema, text, *, max_tokens, thinking_mode, sem, system=None):
    return {...}
monkeypatch.setattr(llm_shared, "classify_one", _fake)
```

Adding one kwarg to one helper exploded into a nine-file test fix.

## Why it happened

Tight signatures on test fakes are deliberately exact when the goal
is to verify caller behavior. But for **wide-fanout helpers**
(`classify_one` is called from every LLM-bearing worker) where the
fake exists only to short-circuit the network call, signature
exactness is anti-leverage: the production signature evolves more
often than the fake's behavior needs to change, and every evolution
forces a multi-file mechanical update.

The fix is a one-line discipline: **fakes for wide-fanout helpers
end with `**_kw: object`.** Future kwargs slot through silently,
the fake's behavior stays stable, the production change is
single-file.

## Fix

Apply the convention universally for any fake that monkeypatches
`classify_one`, the embed_worker fan-out helpers, or any other
helper called from 3+ production sites:

```python
# tests/test_analyze_chain.py and friends
async def _fake_classify_one(
    *_a: object,
    **_kw: object,  # <-- absorbs future kwargs (pipeline=, system=, etc.)
) -> dict[str, object]:
    return {"label": "noop", ...}
monkeypatch.setattr(llm_shared, "classify_one", _fake_classify_one)
```

Two corollaries:

1. **Don't `**_kw`-absorb tight call sites.** If a test verifies
   that the caller passes a specific kwarg shape (e.g., asserting
   `pipeline="trajectory"` was passed), use a `MagicMock` with
   `assert_called_with(...)` instead — or write an explicit fake
   with the kwarg named. The `**_kw` pattern is for the *body*
   tests (which only care that the fake returns a synthetic
   payload), not the *contract* tests.
2. **`*_a: object, **_kw: object`** beats `*args, **kwargs` for ty:
   strict mode flags the latter as missing annotations. The former
   is one keystroke worse and types cleanly.

`tests/test_analyze_chain.py` already used this pattern across
nine fakes (`_fake_skills_sync`, `_fake_ingest_count`, `_fake_embed`,
`_fake_cluster`, `_fake_classify`, `_fake_trajectory`, etc.) — the
v1.0 work just generalized the convention to every classify_one
fake in the repo.

## How to recall

- Symptom: `TypeError: <fake>() got an unexpected keyword argument
  '<new_kwarg>'` after adding a parameter to a wide-fanout helper.
- Symptom: a single production-side signature change requires
  edits to N test files, where N is the number of test modules
  that monkeypatch the helper.
- Trigger: when writing a fake that replaces a wide-fanout helper
  (called from 3+ sites), default to `*_a: object, **_kw: object`
  unless the test specifically needs to assert on a kwarg shape.
- Trigger: when reviewing a PR that adds a kwarg to such a helper,
  scan the test diff for `def _fake_*` blocks that grew a
  parameter — those tests are coupled to the historical signature
  and should be loosened in the same PR.
- Search keywords: `**_kw`, `monkeypatch.setattr`, `classify_one`,
  "stale-test-mock", "kwarg cascade".

## References

- tests/test_analyze_chain.py:238+ (canonical pattern: nine
  `*_a, **_kw`-shaped fakes for the analyze chain helpers).
- src/claude_sql/llm_shared.py:573 (`classify_one(..., pipeline=)`
  signature).
- Cross-ref: `best-practices/pipeline-accumulator-explicit-key.md`
  — the production change that motivated the test loosening.
