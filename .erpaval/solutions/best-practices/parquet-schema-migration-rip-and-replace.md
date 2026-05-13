---
title: Rip-and-replace parquet schema migrations via marker-column probe at startup
category: best-practices
tags: [parquet, schema-migration, pyarrow, sharded-cache, idempotent]
modules: [src/claude_sql/conflicts_worker.py, src/claude_sql/parquet_shards.py]
added: 2026-05-13
---

## What happened

v1.0 rewrote three parquet schemas: conflicts rekeyed from
`(session_id, conflict_idx)` to `(turn_a_uuid, turn_b_uuid)` and
dropped the `empty=True` sentinel; trajectory rekeyed from `uuid` +
`sentiment_delta` to `(prev_uuid, curr_uuid)` + sentiment-shift
classification; friction tightened its label set. Mixed directories
(legacy + new-schema shards) explode at view-registration time —
DuckDB's `read_parquet` glob unifies schemas across files, and a
column shape mismatch on a single shard fails the whole bind.

## Why it happened

Sharded parquet caches accumulate. A v0 run wrote 47 shards under
`~/.claude/conflicts/part-*.parquet`; a v1.0 run wants to write a
48th shard with a different column set. There's no in-place migration
path that's both fast and safe — you'd have to read every legacy
shard, project it into the new schema, write a fresh shard, delete
the old one, and survive a crash mid-migration. For analytics caches
(everything is recomputable from the JSONL corpus + Bedrock), the
cheaper-and-simpler shape is **delete and re-stamp**.

## Fix

At worker startup, probe each shard's `schema_arrow` for **legacy
marker columns** — names that exist only under the old schema. If
any shard carries a marker, `shutil.rmtree` the entire cache
directory. The next `read_all` returns an empty frame, and the worker
re-stamps from the JSONL corpus. Idempotent: on a fresh install (no
shards) or an already-migrated cache (no markers), the function is a
no-op.

```python
# claude_sql/conflicts_worker.py
import pyarrow.parquet as pq

_LEGACY_SCHEMA_MARKERS: frozenset[str] = frozenset({"conflict_idx", "empty"})

def _purge_legacy_shards(target: Path) -> None:
    parts = iter_part_files(target)
    if not parts:
        return
    legacy_hit = False
    for part in parts:
        try:
            schema = pq.ParquetFile(str(part)).schema_arrow  # zero rows read
        except (OSError, pa.ArrowInvalid):
            legacy_hit = True
            logger.warning("conflicts: unreadable shard {} — purging cache", part)
            break
        if set(schema.names) & _LEGACY_SCHEMA_MARKERS:
            legacy_hit = True
            logger.warning("conflicts: legacy shard {} — purging cache for v1.0 rekey", part)
            break
    if not legacy_hit:
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)
```

Three load-bearing details:

1. **`pq.ParquetFile.schema_arrow`** reads the parquet footer only —
   no row materialization. On a 47-shard / 200 MB cache the probe
   takes <100ms total. Don't reach for `pl.read_parquet` here; it
   reads data.
2. **Marker columns must be UNIQUE to the old schema.** Pick names
   that the new schema doesn't reintroduce, and that aren't generic
   (`session_id` is shared by every version — useless as a marker).
   `conflict_idx` was a v0-only sequence; `empty` was a v0-only
   sentinel. Both are safe.
3. **Break on first hit.** One legacy shard is enough to trigger the
   purge — no need to scan the rest. The whole directory dies anyway.

## How to recall

- Symptom: a worker that previously worked now explodes at
  view-registration time with a polars / DuckDB schema-unification
  error referencing a column from the old schema.
- Symptom: `read_parquet('cache/part-*.parquet')` fails on column
  count mismatch or type mismatch.
- Trigger: any time you change a parquet's column set (add, remove,
  rename, retype) in a worker that writes sharded caches under
  `~/.claude/`. Add markers to the OLD schema's distinctive columns
  and a `_purge_legacy_shards(...)` call at the worker entry point.
- Trigger: when reading a parquet footer, use `pq.ParquetFile(path).schema_arrow`,
  not `pl.read_parquet(path).schema` — the former reads zero rows.
- Search keywords: `_LEGACY_SCHEMA_MARKERS`, `schema_arrow`,
  `_purge_legacy_shards`, "parquet rekey", "schema migration",
  "rip and replace cache".

## References

- src/claude_sql/conflicts_worker.py:85 (`_LEGACY_SCHEMA_MARKERS`),
  :88 (`_purge_legacy_shards`), :143 (entry-point call).
- src/claude_sql/trajectory_worker.py:332+ (parallel migration for
  trajectory's `(uuid, sentiment_delta, is_transition, confidence)` →
  `(prev_uuid, curr_uuid, ...)` rekey).
- src/claude_sql/parquet_shards.py (`iter_part_files`, `read_all`).
- Cross-ref: `best-practices/sharded-cache-field-name-stability.md` —
  Settings field-name stability across the same migrations.
