---
title: Pair writer-side replace-then-append with reader-side dedup so the contract holds even when the writer breaks
track: knowledge
category: best-practices
tags: [sql_views, duckdb, read_parquet, qualify, row_number, dedup, defensive-design, sharded-parquet, trajectory, idempotence]
modules:
  - src/claude_sql/sql_views.py
  - src/claude_sql/trajectory_worker.py
  - src/claude_sql/parquet_shards.py
added: 2026-05-14
applies_when:
  - A pipeline writes sharded parquet caches that downstream SQL views read via read_parquet([...])
  - The writer side has (or will have) a per-session "replace prior rows then append" critical section
  - The view promises a partition-key uniqueness contract to downstream macros / queries
---

## What happened

PR #54 (v1.0.1) fixed `message_trajectory` row duplication by making the
trajectory worker replace prior `(session_id, …)` shards before
appending fresh rows on rerun. Issue #46 was the explicit
belt-and-suspenders companion: the *view* should also enforce
`(session_id, prev_uuid, curr_uuid)` uniqueness so accidental
duplicates from prior corpus state, future write-path bugs, or hand-
edited shards don't leak into downstream analytics.

The fix wraps the `read_parquet([...])` source in a `QUALIFY` filter:

```sql
CREATE OR REPLACE VIEW message_trajectory AS
SELECT *, curr_sentiment AS sentiment, is_transition AS transition
FROM read_parquet([<paths>])
QUALIFY row_number() OVER (
    PARTITION BY session_id, prev_uuid, curr_uuid
    ORDER BY classified_at DESC NULLS LAST
) = 1;
```

## Why it matters as a pattern

Two independent gates protect a partition-key uniqueness contract:

1. **Writer side** — replace prior rows for the affected session keys
   before writing fresh ones (`parquet_shards.replace_sessions` plus a
   call inside the worker's `write_lock`). Cost: one polars read +
   filter per shard per affected session per rerun. Lesson:
   `best-practices/checkpointer-doesnt-replace-sharded-cache-rows.md`.
2. **Reader side** — `QUALIFY row_number() OVER (PARTITION BY <key>
   ORDER BY <freshness> DESC NULLS LAST) = 1` in the view DDL. Cost:
   one window pass per query — negligible at 4.5K rows; worth
   benchmarking and possibly an ART index past 100K.

Each gate alone is insufficient:

- Writer-only: an old shard escaping cleanup (manual edit, partial
  migration, future bug) silently inflates downstream counts.
- Reader-only: every query pays the window-function cost, and the
  parquet directory grows monotonically because nothing prunes
  superseded rows on disk.

Pairing them gives belt-and-suspenders correctness *and* keeps
on-disk size bounded.

## DuckDB-specific implementation notes

Three load-bearing details that future contributors will trip on:

1. **`QUALIFY` is a filter clause, not a projection.** Unlike the
   `SELECT * EXCLUDE (_rn) FROM (... ROW_NUMBER() AS _rn ...) WHERE
   _rn = 1` pattern that other engines need, DuckDB's `QUALIFY`
   doesn't require a throwaway column in the projection. The view's
   DESCRIBE output stays clean — no `_rn` to hide. This matters for
   downstream macros that select-star.

2. **`ORDER BY <freshness> DESC NULLS LAST`** is belt-and-suspenders
   on top of DuckDB's default. DuckDB defaults `DESC` to `NULLS LAST`,
   but writing it explicitly documents the intent and survives a
   future engine change. With all-NULL freshness in a partition,
   `row_number()` still assigns a unique 1 deterministically.

3. **Parallel dict pattern in `register_analytics`.** The cleanest
   minimal change was a parallel `view_qualify: dict[str, str]`
   alongside the existing `view_projections: dict[str, str | None]`,
   then `qualify_clause = f" QUALIFY {qualify}" if qualify else ""`
   spliced into the DDL. The fallback `if projection != "*"` branch
   that catches legacy-schema shards intentionally drops the QUALIFY
   — legacy shards may not carry the freshness column. Don't collapse
   the two dicts; the asymmetry on the fallback path is the point.

## Performance plan equivalence

Pattern A (`SELECT * EXCLUDE (_rn) FROM (... row_number() AS _rn ...)
WHERE _rn = 1`) and pattern B (`QUALIFY row_number() … = 1`) lower to
identical execution plans in DuckDB. Confirmed via Context7 on
duckdb.org/docs/current/sql/query_syntax/qualify.html. Choose B
purely for readability — the EXCLUDE workaround in A only exists to
compensate for engines that lack QUALIFY.

## How to recall

- Symptom: a downstream SQL view promises partition-key uniqueness but
  reads from a sharded parquet cache that could (now or in future)
  carry duplicates.
- Symptom: a write-side fix lands for "duplicate rows on rerun" — ask
  immediately whether the read side should also enforce the invariant.
- Trigger: `CREATE OR REPLACE VIEW <v> AS SELECT … FROM
  read_parquet([...])` over a per-session sharded directory.
- Search keywords: `QUALIFY row_number()`, "read-side dedup",
  "belt-and-suspenders dedup", "read_parquet dedup view".

## References

- GH issue #46 — enhancement(sql_views): read-side dedup for
  message_trajectory.
- PR shipping this fix (sibling to #54).
- `src/claude_sql/sql_views.py` — `view_qualify` dict + `qualify_clause`
  splice in `register_analytics`.
- `tests/test_v2_analytics.py` — `_make_trajectory_parquet` helper +
  three dedup tests (`test_message_trajectory_dedups_within_shard`,
  `test_message_trajectory_dedups_across_shards`,
  `test_message_trajectory_clean_cache_passes_through`).
- Cross-ref:
  `.erpaval/solutions/best-practices/checkpointer-doesnt-replace-sharded-cache-rows.md`
  — the writer-side replace-then-append (the other half of the pair).
- Cross-ref:
  `.erpaval/solutions/api-patterns/duckdb-lag-named-window-vs-correlated-min.md`
  — same module, same "single FROM reference" theme as window-function
  refactors over `read_parquet`.
- DuckDB docs: QUALIFY clause —
  https://duckdb.org/docs/current/sql/query_syntax/qualify.html
