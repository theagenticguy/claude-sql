---
title: duckdb_functions() returns null parameters for table macros
track: knowledge
category: api-patterns
module: claude_sql.sql_views
component: duckdb
severity: info
tags: [duckdb, macros, table-macros, introspection, duckdb_functions, signature]
applies_when:
  - You define DuckDB macros with the table form (`CREATE OR REPLACE MACRO foo(a, b) AS TABLE (...)`)
  - You want to expose macro parameter names to downstream consumers (a CLI, an agent, a docs generator)
  - You're inspecting `duckdb_functions()` to enumerate macro signatures
pattern: |
  `duckdb_functions()` returns null in the `parameters` column for
  TABLE macros (`CREATE OR REPLACE MACRO foo(a, b) AS TABLE (...)`).
  Scalar macros (`CREATE OR REPLACE MACRO foo(a, b) AS (...)`) populate
  parameters correctly. The asymmetry is a DuckDB introspection
  limitation, not a bug in your DDL.

  Workaround: hand-maintain a `MACRO_SIGNATURES: dict[str, tuple[str, ...]]`
  next to the registration code. Add a drift test that regex-extracts the
  arg lists from `inspect.getsource(register_macros)` and asserts they
  match the dict.

  ```python
  import re
  CREATE_MACRO_RE = re.compile(
      r"CREATE OR REPLACE MACRO (\w+)\(([^)]*)\)", re.IGNORECASE
  )

  def test_macro_signatures_match_ddl():
      source = inspect.getsource(register_macros)
      observed = {
          name: tuple(p.strip() for p in args.split(",") if p.strip())
          for name, args in CREATE_MACRO_RE.findall(source)
      }
      assert observed == MACRO_SIGNATURES
  ```

  This shape catches the most common drift mode (you renamed a macro
  arg without updating the dict) at CI time, not at runtime.
example_files:
  - src/claude_sql/sql_views.py
  - tests/test_sql_views.py
---

# Why this matters

The schema command should expose macro signatures so agents can compose `query` calls without trial-and-error. `duckdb_functions()` looked like the obvious source — it lists every macro by name — but on inspection 11 of 16 macros came back with null parameters. All 11 were table macros (analytics aggregations); the 5 with populated parameters were scalar macros.

Hand-maintaining a `MACRO_SIGNATURES` dict adds maintenance overhead, but the regex drift test makes "you forgot one" a CI failure rather than a runtime mystery. Bonus: the dict is greppable and auto-completable in editors, which `duckdb_functions()` results aren't.

Worth checking on every DuckDB version bump: if a future release populates table-macro parameters, the dict can be retired. As of DuckDB 1.x (2026), it doesn't.

# Example

See `src/claude_sql/sql_views.py` for the dict shape (`MACRO_SIGNATURES`) and `tests/test_sql_views.py::test_macro_signatures_match_ddl` for the drift test.
