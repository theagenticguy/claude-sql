---
title: Gate caches at both computation AND cache-replacement when a checkpointer admits growing sessions
track: knowledge
category: best-practices
tags: [checkpointer, sharded-parquet, rerun-semantics, trajectory-worker, dedup, idempotence]
modules:
  - src/claude_sql/trajectory_worker.py
  - src/claude_sql/parquet_shards.py
  - src/claude_sql/conflicts_worker.py
added: 2026-05-14
applies_when:
  - A worker writes sharded parquet caches per session via write_part
  - A session-level checkpointer gates computation on (latest_ts, message_count)
  - Sessions can grow between runs (active transcripts append turns)
  - The checkpointer re-admits growing sessions for full re-scoring
---

## What happened

`message_trajectory` accumulated duplicate `(session_id, prev_uuid,
curr_uuid)` rows every time an active session got re-classified. The
live corpus showed 4.6% of all rows were duplicates; one session had
132 rows for ~44 distinct pairs — exactly 3× multiplication matching
three same-day reruns. Every downstream analytic over
`message_trajectory` (frustration spikes, resolutions, sentiment
arcs) silently over-counted for growing sessions.

## Why it happened

The v1.0 trajectory pipeline has two independent gates that look
similar but serve different purposes:

1. **`checkpointer.filter_unchanged`** — a computation gate. Skips
   sessions whose `(latest_ts, message_count)` bounds haven't moved
   since the last run. Purpose: don't spend Bedrock tokens on sessions
   that haven't changed.

2. **The sharded parquet cache** — a storage gate. Should ensure each
   `(session_id, prev_uuid, curr_uuid)` appears at most once. Purpose:
   downstream SQL gets the invariant it expects.

The trajectory worker had #1 but not #2. When a session grew, #1
correctly admitted it for full re-scoring, `_process_session`
re-computed every window, and `write_part` dropped a fresh
`part-<ts_ns>.parquet` alongside the prior shard. The new rows stacked
on top of the old rows; nothing removed the old rows.

`_purge_old_shards` at `trajectory_worker.py:355` only deletes
**schema-stale** shards (the pre-v1.0 per-message shape). Current-schema
shards for the same session are invisible to it.

**Conflicts worker doesn't hit this** — it short-circuits at the
`already = set(done_df["session_id"].to_list())` check (`conflicts_worker.py:145-149`)
and skips any session already in the parquet entirely. Rerun with
growth isn't a thing for conflicts because the pipeline doesn't
re-score grown sessions in the first place.

**General anti-pattern:** a checkpointer that gates computation but
not cache replacement. The two concerns look twinned but drift the
moment the pipeline admits anything other than fresh sessions — a
retry, an advancing bound, a forced reclassify all surface the same
bug.

## Fix

A new helper in `parquet_shards.py` that drops rows for a set of
sessions across every shard, then a call from the worker's write
critical section to replace-then-append on rerun:

```python
# parquet_shards.py
def replace_sessions(
    target: Path,
    *,
    key_column: str,
    session_ids: Iterable[str],
) -> int:
    """Drop rows whose key_column is in session_ids across every shard.
    Rewrites mixed shards; unlinks shards that empty out. Returns
    removed row count. Idempotent on empty caches or empty id sets."""
    ids = set(session_ids)
    if not ids:
        return 0
    parts = iter_part_files(target)
    if not parts:
        return 0
    removed_total = 0
    for part in parts:
        try:
            df = pl.read_parquet(part)
        except (OSError, pl.exceptions.ComputeError) as exc:
            logger.warning("replace_sessions: unreadable shard {} ({}); skipping", part, exc)
            continue
        if key_column not in df.columns or df.height == 0:
            continue
        mask = df[key_column].is_in(list(ids))
        hit_count = int(mask.sum())
        if hit_count == 0:
            continue
        removed_total += hit_count
        kept = df.filter(~mask)
        if kept.height == 0:
            try:
                part.unlink()
            except OSError as exc:
                logger.warning("replace_sessions: failed to unlink empty shard {}: {}", part, exc)
            continue
        kept.write_parquet(part)
    return removed_total
```

Wire it into the trajectory writer's critical section:

```python
# trajectory_worker.py — inside _process_session
async with write_lock:
    replace_sessions(
        settings.trajectory_parquet_path,
        key_column="session_id",
        session_ids=[sid],
    )
    write_part(settings.trajectory_parquet_path, df)
    written_box[0] += len(all_rows)
    processed_sessions.add(sid)
```

Four load-bearing details:

1. **Serialized with the existing `write_lock`.** Concurrent sessions
   must not interleave replace + write across each other; the lock
   guarantees replace-then-append is atomic per session.
2. **Predicate uses polars `is_in` on the loaded DataFrame**, not
   parquet rowgroup statistics. Rowgroup `(min, max)` on UUID-shaped
   strings rarely prunes a shard, so a stats-based pre-filter would
   load every shard anyway. The polars read is the cheapest correct
   path.
3. **Unlink shards that empty out** rather than write a zero-row
   parquet. DuckDB's `read_parquet` glob binds empty parquets
   harmlessly but leaves the cache accumulating dead files.
4. **Legacy single-file branch falls out for free.** `iter_part_files`
   returns `[target]` for a `*.parquet` path; rewriting overwrites
   that file, and unlinking is the "now empty" case — which matches
   what the legacy branch wants.

## How to recall

- Symptom: downstream SQL shows `count(*) > count(DISTINCT (...))` for
  a parquet that should have unique pair keys.
- Symptom: row counts in a per-session parquet grow faster than the
  corresponding session's message count, especially for active
  sessions.
- Symptom: exact integer multiplication (2×, 3×, …) across
  `COUNT(*) / COUNT(DISTINCT …)` — tells you how many reruns stacked.
- Trigger: any time a session-level checkpointer admits the same
  session more than once (growth, retries, forced reclassify), ask
  "does the write path replace the prior cache rows for that
  session, or does it append?"
- Search keywords: `checkpointer`, `write_part`, `message_trajectory`,
  `replace_sessions`, "duplicate rows on rerun", "checkpointer appends
  shards".

## References

- GH issue #45 — bug(trajectory): duplicate rows on rerun —
  checkpointer appends shards instead of replacing.
- PR #54 — fix(trajectory): replace prior session shards on rerun.
- `src/claude_sql/trajectory_worker.py:696-700` (filter_unchanged call
  site) and `:882-903` (write critical section where the replace
  belongs).
- `src/claude_sql/parquet_shards.py::replace_sessions` — the helper.
- `src/claude_sql/conflicts_worker.py:145-149` — the `already` set
  that lets conflicts sidestep this bug entirely.
- Cross-ref:
  `.erpaval/solutions/best-practices/sharded-cache-field-name-stability.md`
  (same module, path-semantics choice that kept the API stable when
  the layout shifted to sharded dirs).
- Cross-ref:
  `.erpaval/solutions/best-practices/parquet-schema-migration-rip-and-replace.md`
  (different dimension of cache staleness — old-schema shards vs
  current-schema stale rows; they share the "check at worker startup,
  act before writing" shape).
