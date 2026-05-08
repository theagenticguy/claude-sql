---
title: ATTACH on a fresh DuckDB path leaves an empty file — gate rebuild on catalog, not file size
track: knowledge
category: api-patterns
module: src/claude_sql/sql_views.py
component: DuckDB ATTACH + experimental HNSW persistence
severity: info
tags: [duckdb, hnsw, attach, persistence, vss]
applies_when:
  - You ATTACH a database file that may or may not already contain tables
  - You want to skip rebuilding tables when the persisted store is fresh
  - You're using `hnsw_enable_experimental_persistence` or any other ATTACHed-store cache pattern
pattern: |
  DuckDB's `ATTACH '<path>' AS alias;` on a non-existent path silently
  creates a header-only file (~12 KB on DuckDB 1.5.x) before any tables
  exist. A naive freshness check that gates on file size or mtime alone
  will see the fresh file as "populated" and skip the CREATE TABLE step,
  leaving the catalog empty. The next query — including a
  ``CREATE OR REPLACE VIEW ... AS SELECT * FROM alias.main.tbl`` — fails
  with "Catalog Error: Table with name <tbl> does not exist".

  Fix: combine the filesystem-mtime check with a catalog probe. After
  ATTACH succeeds, query `duckdb_tables()` for the expected table; if
  it's absent, force the rebuild path even when the file size suggests
  the store is initialized.
example_files:
  - src/claude_sql/sql_views.py
---

# Why this matters

Without the catalog probe, the first ever invocation of
`register_vss(con, hnsw_db_path=<fresh path>)` writes a 12 KB empty
file via ATTACH, then short-circuits the rebuild because the size
exceeds the 1 KB heuristic. The CREATE OR REPLACE VIEW that wraps the
attached table fails because no such table was ever created. Symptom
is a ``CatalogException`` on the *first* run; subsequent runs work
because by then the rebuild stamp is correct. Easy to misdiagnose as
"experimental persistence is broken".

# Example

```python
def _attached_embeddings_table_present(con: duckdb.DuckDBPyConnection) -> bool:
    row = con.execute(
        """
        SELECT count(*)
        FROM duckdb_tables()
        WHERE database_name = 'hnsw_store'
          AND schema_name = 'main'
          AND table_name = 'message_embeddings';
        """
    ).fetchone()
    return bool(row and row[0])

# In register_vss:
con.execute(f"ATTACH '{persisted_path}' AS hnsw_store;")
rebuild = _hnsw_rebuild_needed(parquet, persisted_path)
rebuild = rebuild or not _attached_embeddings_table_present(con)
```
