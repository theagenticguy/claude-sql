---
title: DuckDB read_json view DESCRIBE re-runs schema inference
track: knowledge
category: api-patterns
module: claude_sql.sql_views
component: duckdb
severity: info
tags: [duckdb, read_json, describe, performance, schema-inference, introspection]
applies_when:
  - A view is built on top of `read_json(..., union_by_name=true, sample_size=-1)` over a non-trivial JSONL glob
  - The same view is referenced by many downstream CREATE OR REPLACE VIEW statements
  - You're calling `DESCRIBE <view>` (or `SELECT … FROM information_schema.columns`) in a loop for catalog introspection
pattern: |
  DuckDB does NOT cache the schema of a `read_json(...)` view across statements.
  Every CREATE OR REPLACE VIEW that transitively references the read_json view
  re-binds it (re-runs full JSON schema inference at ~0.55s/bind on a 4k-file
  corpus). And every DESCRIBE re-binds it too — which makes the per-view
  DESCRIBE loop a 27×0.55s = ~15s cost on top of the registration chain.

  Two cheap workarounds, in order of preference:

  1. **Bypass DuckDB introspection entirely.** Maintain a static
     `VIEW_SCHEMA: dict[str, tuple[tuple[str, str], ...]]` populated once
     from a `describe_all` run, plus a regression test that re-runs
     describe_all on a fixture corpus and asserts equality. The schema
     command then answers in <50ms with zero DuckDB connection.

  2. **Replace the DESCRIBE loop with one `duckdb_columns()` query.**
     `duckdb_columns()` reads the catalog directly without re-binding view
     bodies. ~1000× faster than the `for name in VIEW_NAMES: DESCRIBE name`
     pattern. Still requires `register_all` to populate the catalog first,
     so it doesn't fix the registration cost — only the DESCRIBE cost.

  The full bypass (option 1) is the right choice for an agent-facing CLI
  where `schema` is the most frequent introspection call and absolute
  latency matters. `duckdb_columns()` is a fine intermediate when full
  registration is needed for other reasons.
example_files:
  - src/claude_sql/sql_views.py
  - src/claude_sql/cli.py
---

# Why this matters

The first version of `claude-sql schema` took 41 seconds on a 921 MB / 3,845 JSONL corpus. Profiling broke down as: `register_raw` 1.7s, `register_views` 12.8s, `register_vss` 0.02s, `register_analytics` 2.0s, `register_macros` 7.6s, `describe_all` (DESCRIBE loop, 27 views) 14.5s. Both `register_views` and the DESCRIBE loop bottlenecked on the same root cause — `v_raw_events` (the `read_json(..., union_by_name=true, sample_size=-1)` view) re-runs full JSON schema inference on every bind.

`union_by_name=true` + `sample_size=-1` are load-bearing: the corpus has truncated/growing files and a heterogeneous schema that changes as Claude Code's transcript format evolves. There's no clean DuckDB-side workaround: you need full-scan schema inference for correctness; what you can change is how often you pay it.

The static-dict approach is the right shape for an agent-facing CLI: agents call `schema` first to discover view/column names before composing queries. Sub-50ms response over a static dict beats sub-second response that still requires opening a DuckDB connection.

# Example

```python
# sql_views.py — static catalog, hand-maintained alongside view DDL
VIEW_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "sessions": (
        ("session_id", "VARCHAR"),
        ("cwd", "VARCHAR"),
        # ...
    ),
    # ...
}

# tests/test_sql_views.py — drift test
def test_view_schema_matches_describe_all(tmp_corpus):
    con = duckdb.connect(":memory:")
    register_raw(con, glob=...)
    register_views(con)
    observed = describe_all(con)  # the slow path, run once in CI
    for view_name, expected_cols in VIEW_SCHEMA.items():
        assert tuple(observed[view_name]) == expected_cols
```
