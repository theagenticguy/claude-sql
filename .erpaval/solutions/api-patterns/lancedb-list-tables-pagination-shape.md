---
title: lancedb 0.30 `list_tables()` returns a pydantic model, not a list
track: knowledge
category: api-patterns
module: src/claude_sql/lance_store.py
component: LanceDB DBConnection
severity: info
tags: [lancedb, pydantic, pagination, api-shape]
applies_when:
  - Listing tables in a local LanceDB dataset (`lancedb.connect()`)
  - Migrating from `db.table_names()` (deprecated) to `db.list_tables()`
  - Writing compatibility wrappers across lancedb 0.x patch versions
pattern: |
  `db.list_tables()` returns a pydantic `ListTablesResponse(tables=[...],
  page_token=...)` — a typed model with two fields, NOT a flat list and
  NOT a paginated tuple sequence. The deprecated `db.table_names()`
  emits `DeprecationWarning` on every call.

  Don't write defensive compatibility shims that iterate the result and
  pattern-match on shape. Just read the `.tables` attribute:

  ```python
  def _has_table(db: lancedb.DBConnection, name: str) -> bool:
      return name in db.list_tables().tables
  ```

  Pagination via `page_token` exists for >1k-table namespaces; local
  installs never trigger it.
example_files:
  - src/claude_sql/lance_store.py
  - tests/test_lance_store.py
---

# Why this matters

The defensive wrapper that landed first iterated `list(db.list_tables())`
and pattern-matched on `[('tables', [...]), ('page_token', None)]`. It
worked **by accident** because pydantic models iterate as
`(field_name, value)` tuples — but that behavior is not part of the
public API contract and could change on any pydantic minor bump. The
actual API is just `.tables: list[str]`.

The second-order trap: `db.table_names()` looks like the right method,
but it's deprecated in 0.30+ and emits a warning. Use `list_tables()`
even though the name suggests pagination semantics that don't matter
for local namespaces.

# Example

```python
# Wrong (worked by accident on lancedb 0.30.2):
def _table_names(db) -> list[str]:
    list_tables = getattr(db, "list_tables", None)
    if list_tables is not None:
        result = list(list_tables())
        for key, value in result:
            if key == "tables" and isinstance(value, list):
                return [str(v) for v in value]
    return list(db.table_names())  # deprecated

# Right (verified against installed lancedb/db.py:167-193):
def _has_table(db: lancedb.DBConnection, name: str) -> bool:
    return name in db.list_tables().tables
```

# Verification

Live probe against `lancedb-0.30.2`:
```python
>>> db = lancedb.connect("/tmp/d", read_consistency_interval=timedelta(0))
>>> db.list_tables()
ListTablesResponse(tables=[], page_token=None)
>>> type(db.list_tables())
<class 'lance_namespace_urllib3_client.models.list_tables_response.ListTablesResponse'>
```

# References

- `tests/test_lance_store.py::test_has_table_returns_true_when_present` —
  pins the API shape against future lancedb versions
- `lancedb/db.py:167-193` — public `list_tables()` signature
- `lancedb/db.py:782-822` — deprecated `table_names()` with warning
