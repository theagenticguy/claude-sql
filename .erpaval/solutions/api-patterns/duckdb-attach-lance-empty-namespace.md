---
title: DuckDB `ATTACH (TYPE LANCE)` is permissive — probe the namespace, not the filesystem
track: knowledge
category: api-patterns
module: src/claude_sql/sql_views.py
component: DuckDB lance core extension + LanceDB
severity: warning
tags: [duckdb, lance, attach, vss, empty-dataset]
applies_when:
  - Wiring DuckDB to a LanceDB local dataset via `INSTALL lance; LOAD lance; ATTACH '<dir>' (TYPE LANCE)`
  - Gating a fallback path on "is the embeddings table actually present"
  - Migrating from `embeddings/part-*.parquet` + `hnsw.duckdb` to LanceDB
pattern: |
  DuckDB's `ATTACH '<path>' (TYPE LANCE)` succeeds on **any** path —
  missing dir, empty dir, dir with stray files, dir with LanceDB metadata
  but no tables. The catalog error fires later, at downstream
  `SELECT/CREATE VIEW` time:

      Catalog Error: Table with name embeddings does not exist
      Did you mean "pg_catalog.pg_settings"?

  Filesystem heuristics like `not lance_uri.exists() or not
  any(lance_uri.iterdir())` are wrong: a directory with a stray file
  passes the gate, ATTACH succeeds, then the view bind explodes.

  The right gate is the LanceDB namespace itself:

  ```python
  from claude_sql import lance_store
  db = lance_store.connect_db(lance_uri)
  if not lance_store._has_table(db, lance_store.TABLE_NAME):
      # No embeddings table — create empty fallback so semantic_search binds.
      con.execute(f"CREATE OR REPLACE TABLE message_embeddings (...)")
      return False
  ```
example_files:
  - src/claude_sql/sql_views.py
  - tests/test_lance_store.py
---

# Why this matters

The bug that started a multi-hour spiral: `register_vss` checked
`if not lance_uri.exists() or not any(lance_uri.iterdir())` to detect
a fresh install. A directory with even *one* unrelated file passed
that check, dropped into the ATTACH path, and exploded at
`CREATE OR REPLACE VIEW message_embeddings AS SELECT ... FROM
lance_store.main.embeddings` time with the catalog error above.

DuckDB's lance extension intentionally treats ATTACH as best-effort
(matches Postgres's `IF NOT EXISTS` philosophy) — failing at attach
time would be hostile to fresh-install workflows. But that means
the application has to enforce the "is the table actually there"
invariant explicitly.

# Live verification

```python
>>> import duckdb, lancedb
>>> con = duckdb.connect(':memory:')
>>> con.execute('INSTALL lance; LOAD lance;')
>>> import tempfile, pathlib
>>> with tempfile.TemporaryDirectory() as d:
...     p = pathlib.Path(d) / 'empty'
...     p.mkdir()
...     con.execute(f"ATTACH '{p}' AS lance_store (TYPE LANCE);")  # OK
...     con.execute("SELECT count(*) FROM lance_store.main.embeddings")  # FAILS
CatalogException: Catalog Error: Table with name embeddings does not exist!
```

# Pinned by

`tests/test_lance_store.py::test_register_vss_empty_namespace_creates_fallback_table`
seeds a directory with a stray file and verifies the new gate falls
through to the empty-table fallback (returns False).

## 2026-05-13 update — TABLE vs VIEW shape switch on second `register_vss`

The empty-namespace fallback above creates `message_embeddings` as a
**TABLE** (an empty FLOAT[1024]-shaped table is what binds the
`semantic_search` macro when no Lance dataset exists yet). The
`analyze` chain shares one DuckDB connection across stages — `embed`
populates LanceDB, then `community` (or any later VSS reader) needs to
re-bind `message_embeddings` against the now-populated namespace as a
**VIEW** over `lance_store.main.embeddings`.

That second `register_vss` call hits a CatalogException:

    Catalog Error: Existing object 'message_embeddings' is of type Table,
    not View. Use DROP TABLE first.

`CREATE OR REPLACE VIEW` in DuckDB does NOT cross object types — and
`DROP VIEW IF EXISTS` does NOT match a Table either (`IF EXISTS` only
suppresses the not-found error, not the type-mismatch error). So the
re-bind helper has to drop **both** shapes proactively before re-issuing
`register_vss`:

```python
# claude_sql/cli.py::_rebind_vss
def _rebind_vss(con, settings, *, stage):
    try:
        con.execute("DROP VIEW IF EXISTS message_embeddings;")
    except duckdb.Error:
        # Existing object is a TABLE — drop it as TABLE.
        con.execute("DROP TABLE IF EXISTS message_embeddings;")
    with contextlib.suppress(duckdb.Error):
        con.execute("DETACH lance_store;")
    register_vss(con, ...)
```

The `DETACH lance_store` is also load-bearing: `register_vss` issues
its own `ATTACH` and DuckDB rejects re-attaching the same alias.
`contextlib.suppress(duckdb.Error)` covers the first-rebind case where
no prior ATTACH exists.

**When this fires:** any pipeline that opens one DuckDB connection,
runs `embed` (which writes to LanceDB out-of-band), and then runs a
later stage that reads `message_embeddings`. RFC 0002 §9.6 names this
"the analyze stale-connection bug". Without `_rebind_vss`, the later
stage reads zero rows even though Lance has the data.

**Pinned by:** `src/claude_sql/cli.py:398` (`_rebind_vss` helper) +
`src/claude_sql/cli.py` callers in the analyze chain after `embed`
and after `community`.
