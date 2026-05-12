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
