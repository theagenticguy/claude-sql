---
title: anyio dispatcher must not acquire the CapacityLimiter when the task body already does
category: api-patterns
tags: [anyio, structured-concurrency, capacity-limiter, fanout, classify_one]
modules: [src/claude_sql/trajectory_worker.py, src/claude_sql/conflicts_worker.py, src/claude_sql/friction_worker.py, src/claude_sql/llm_shared.py]
added: 2026-05-13
---

## What happened

The trajectory worker's per-session dispatcher looked like this:

```python
async def _process_session(sid, session_windows):
    async with sem:                       # <-- WRONG: outer hold
        for chunk in chunks_of(session_windows):
            await _classify_chunk(client, settings, sem, chunk=chunk, ...)

async with anyio.create_task_group() as tg:
    for sid, session_windows in by_session.items():
        tg.start_soon(_process_session, sid, session_windows)
```

`sem = anyio.CapacityLimiter(16)` — and yet `nvidia-smi`-equivalent
on Bedrock concurrency showed in-flight calls capped at **1**. The
dispatcher had the limiter held for the entire duration of one
session (which contains many sequential chunks), preventing any other
session from starting. Wall clock collapsed to single-threaded.

## Why it happened

`classify_one` already does `async with sem` internally (correctly —
the limiter scopes one Bedrock invocation). When the dispatcher
*also* takes the limiter and then loops over multiple `classify_one`
calls inside that scope, it both:

1. Holds the slot across N sequential calls (so nothing else can run).
2. Causes `classify_one` to block forever on `async with sem` because
   the outer scope already owns the only slot at concurrency=1
   (deadlock-adjacent — saved only by the limiter being >1).

The general rule: **CapacityLimiter is acquired at the BLOCKING-IO
boundary, not at the dispatch boundary.** The dispatch layer's job is
to start the tasks; the limiter's job is to throttle the actual
SDK calls. Putting the `async with sem` at the dispatch layer
double-counts and serializes.

## Fix

The dispatcher just `tg.start_soon`s; the per-session body iterates
over chunks awaiting them naturally; `_classify_chunk` (which calls
`classify_one`, which does `async with sem`) is the only place the
limiter is acquired:

```python
async def _process_session(sid, session_windows):
    # NO outer `async with sem` — let classify_one own the gate.
    for chunk in chunks_of(session_windows):
        await _classify_chunk(client, settings, sem, chunk=chunk, ...)

async with anyio.create_task_group() as tg:
    for sid, session_windows in by_session.items():
        tg.start_soon(_process_session, sid, session_windows)
```

Now N sessions can be in flight concurrently up to the limiter's
capacity; each session's chunks process serially within itself
(by design — chunks share state via the session checkpointer);
the Bedrock fan-out is throttled at exactly the limiter's capacity.

A sharper smell test: **count the `async with sem:` lines in your
worker.** There should be exactly one per limiter — at the SDK call
boundary. Anything else is a layering bug.

## How to recall

- Symptom: a worker with `CapacityLimiter(16)` shows in-flight
  concurrency of 1 (or any number << limiter capacity).
- Symptom: total throughput is exactly the per-call latency × N — no
  parallelism at all despite the task group.
- Trigger: any time you write `async with sem:` outside the function
  that actually awaits the blocking-IO call, ask "does the inner
  function also `async with sem:`?" If yes, delete the outer one.
- Trigger: any time a per-session loop iterates over multiple
  classify calls, the loop must NOT be inside `async with sem:` —
  the loop should be inside the task group, not inside the slot.
- Search keywords: `CapacityLimiter`, `tg.start_soon`,
  `async with sem`, "concurrency = 1", "no fan-out", "limiter
  double-acquire".

## References

- src/claude_sql/llm_shared.py:584 (`classify_one` is the sole
  `async with sem:` for Sonnet calls).
- src/claude_sql/trajectory_worker.py:895 (corrected dispatcher
  shape — `tg.start_soon` directly, no outer slot acquisition).
- Prior lesson: `best-practices/anyio-structured-concurrency-for-blocking-io.md`
  — the primer on `CapacityLimiter` + `to_thread.run_sync`. This
  lesson is the trap that lurks at the next layer up.
