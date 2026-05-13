---
title: Prefer LAG over a named window over correlated-min on JSONL-backed read_json views
category: api-patterns
tags: [duckdb, read_json, named-window, lag, correlated-subquery, performance]
modules: [src/claude_sql/sql_views.py]
added: 2026-05-13
---

## What happened

`turn_window` — the v1.0 view that powers the friction and trajectory
pipelines — needs adjacent (prev, curr) message pairs per session.
The first cut used a correlated-min subquery:

```sql
SELECT
    a.session_id,
    (SELECT MAX(b.uuid) FROM messages_text b
     WHERE b.session_id = a.session_id AND b.ts < a.ts) AS prev_uuid,
    ...
FROM messages_text a
```

That syntax-cleanly does the "previous turn" lookup, but on a
4k-JSONL corpus it took ~3× as long to bind and run as the
LAG-over-named-window form. Profiling showed `messages_text` (which
sits on top of `read_json(..., union_by_name=true, sample_size=-1)`)
re-binding once per textual reference — three references per row in
the correlated form, one per row in the LAG form.

## Why it happened

Per the prior lesson `duckdb-read-json-rebinds-schema-on-describe.md`,
DuckDB does not cache `read_json` view bodies across statements OR
across textual references inside one statement. Every reference to
`messages_text` inside the SELECT list re-runs full JSON schema
inference at ~0.55s/bind on the live corpus. The correlated-min form
has three lexical references (`a.ts`, `b.session_id`, `b.ts`) plus the
implicit re-bind on the outer FROM — call it 3 binds per row in the
inner correlated subquery. The LAG form has exactly one (the FROM
clause); window-function evaluation runs over the already-bound
result set without re-touching the JSON view.

This isn't a generic SQL truth — on a regular materialized table the
two forms benchmark within a few percent. It's a DuckDB-specific
artifact of the `read_json` schema-rebinding behavior. Any view that
re-runs schema inference on every reference (TYPE LANCE views also
qualify) inherits the same speedup from collapsing references via
named windows.

## Fix

Use `LAG(...) OVER w` with a single `WINDOW w AS (PARTITION BY ...
ORDER BY ...)` clause. The window definition factors out the partition
spec — multiple LAG/LEAD/ROW_NUMBER calls share it for free, no
re-derivation per call:

```sql
CREATE OR REPLACE VIEW turn_window AS
SELECT
    session_id,
    LAG(uuid) OVER w  AS prev_uuid,
    LAG(role) OVER w  AS prev_role,
    LAG(ts)   OVER w  AS prev_ts,
    uuid              AS curr_uuid,
    role              AS curr_role,
    ts                AS curr_ts,
    date_diff('millisecond', LAG(ts) OVER w, ts) AS gap_ms,
    row_number() OVER w AS window_idx
FROM messages_text
WHERE is_compact_summary = false
WINDOW w AS (PARTITION BY session_id ORDER BY ts, uuid);
```

Three load-bearing details:

1. **Single FROM reference.** Everything is computed from the row's
   own columns plus the LAG offset — no inner subquery, no second
   reference to `messages_text`.
2. **Named window** (`WINDOW w AS ...`). Without it, every LAG call
   would inline its own `OVER (PARTITION BY ... ORDER BY ...)`. The
   parser still produces one window aggregation pass, but the named
   form documents intent and is one place to change the partition
   spec.
3. **Tie-break in the ORDER BY.** `ORDER BY ts, uuid` ensures
   deterministic LAG results when two messages share a millisecond
   timestamp — common on synthetic compact-summary rows.

## How to recall

- Symptom: a view over `read_json` (or any view-of-a-view chain that
  bottoms out in `read_json`) is 3-10× slower than expected for a
  syntactically simple SELECT.
- Symptom: changing a correlated-subquery shape to a window-function
  shape produces a wall-clock cliff, not a marginal improvement.
- Trigger: any time you reach for `(SELECT ... FROM same_view WHERE
  ... < a.ts)` over a JSONL-backed view, ask "can LAG/LEAD do this
  with one FROM reference?" Almost always yes for adjacency lookups.
- Search keywords: `LAG`, `WINDOW w AS`, `read_json` performance,
  `union_by_name=true sample_size=-1`, "correlated-min slow".

## References

- src/claude_sql/sql_views.py:759 (`turn_window` DDL).
- Prior lesson: `api-patterns/duckdb-read-json-rebinds-schema-on-describe.md`
  — root cause of the per-reference rebind cost.
- DuckDB docs: window functions
  https://duckdb.org/docs/sql/functions/window_functions
