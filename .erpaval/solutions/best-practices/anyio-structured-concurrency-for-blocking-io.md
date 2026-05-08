---
name: anyio.CapacityLimiter + to_thread.run_sync for blocking SDK calls in asyncio pipelines
description: Prefer anyio's CapacityLimiter + to_thread.run_sync over asyncio.Semaphore + asyncio.to_thread when wrapping blocking boto3/other SDK calls — it honors the enclosing cancellation scope.
type: best-practice
---

`asyncio.to_thread(fn, ...)` doesn't participate in structured
concurrency — if the enclosing task is cancelled, the thread silently
keeps running until the blocking call returns. At scale (16+ concurrent
boto3 `invoke_model` calls) this can orphan threads during interrupts
and cancellations.

**Preferred pattern**:

```python
import anyio
import anyio.to_thread

async def classify_one(client, text, limiter: anyio.CapacityLimiter):
    async with limiter:
        return await anyio.to_thread.run_sync(
            lambda: client.invoke_model(...),
        )

# Caller
limiter = anyio.CapacityLimiter(settings.llm_concurrency)
async with anyio.create_task_group() as tg:
    for text in texts:
        tg.start_soon(classify_one, client, text, limiter)
```

Why anyio over pure asyncio for blocking-IO workloads:
  - `CapacityLimiter` supports `async with`, same as `asyncio.Semaphore`, so swap is one-line.
  - `to_thread.run_sync` propagates cancellation to the thread pool shutdown, not just the await site.
  - Task groups give reliable error aggregation: one failure cancels peers cleanly.
  - anyio's `move_on_after` / `fail_after` give per-call timeouts without the `asyncio.wait_for` "cancel leaks CancelledError" issue.

**Compatibility**: `async with limiter:` works on both `asyncio.Semaphore`
and `anyio.CapacityLimiter`, so you can type the parameter as
`asyncio.Semaphore | anyio.CapacityLimiter` for a safe migration.

**When you hit this**: writing an async pipeline that hands blocking SDK
calls to a thread pool, and you want interrupts or timeouts to actually
cancel in-flight work — not just park the coroutine.
