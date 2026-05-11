---
title: Parquet-existence gate over try/except + log demotion
track: knowledge
category: best-practices
module: claude_sql.sql_views
component: registration
severity: info
tags: [registration, ddl, parquet, log-levels, invariants, defense-in-depth]
applies_when:
  - You have DDL that registers macros/views referencing optional parquet caches
  - The "missing parquet" path is the *default* state on a fresh install (analytics not yet run)
  - The current code uses `try/except duckdb.Error` + `logger.warning` to swallow the missing-cache error
pattern: |
  When a DDL block references a parquet that may or may not exist on
  disk, encode the invariant in the registration code itself rather
  than in the log level.

  Two layers, one rule:

  1. **Gate the DDL on parquet existence.** Build a mapping
     `{macro_or_view_name: tuple_of_required_paths}` and check
     `_parquet_is_populated(path)` before each DDL. Skip the DDL
     entirely when the parquet is missing — no warning fires.

  2. **Keep `try/except + log.debug` as defense-in-depth backstop.**
     Catches the rare race where a parquet vanishes between gate-check
     and DDL-bind, or a future macro adds a different failure mode.
     DEBUG level (not WARNING) — this path shouldn't fire under normal
     operation, and if it does, it's diagnostic information, not
     actionable user-facing state.

  ```python
  _ANALYTICS_MACRO_REQUIREMENTS: dict[str, tuple[str, ...]] = {
      "friction_counts": ("user_friction_parquet_path",),
      "friction_rate":   ("user_friction_parquet_path",),
      "unused_skills":   ("user_friction_parquet_path", "skills_catalog_parquet_path"),
      # ...
  }

  for macro_name, ddl in analytics_macros:
      required = _ANALYTICS_MACRO_REQUIREMENTS.get(macro_name, ())
      if all(_parquet_is_populated(getattr(settings, f)) for f in required):
          _safe_macro(con, ddl)  # backstop: try/except + log.debug
      else:
          logger.debug("Skipped {} (parquet missing)", macro_name)
  ```

  CLAUDE.md's rule "warnings are reserved for genuinely actionable state"
  is the principle this encodes: a missing analytics parquet on a fresh
  install is the *expected* default, not a problem worth flooding stderr
  about. Gate at the right level and the warning class disappears.
example_files:
  - src/claude_sql/sql_views.py
  - tests/test_pr3_perf.py
---

# Why this matters

The original `register_macros` ran every analytics macro DDL unconditionally and relied on `_safe_macro` to swallow `duckdb.Error: Catalog Error: Table with name X does not exist`. Each swallow logged a WARNING. On a fresh install (no analytics run yet) this fired 10+ times per `claude-sql query "SELECT 1"` — flooding stderr and breaking `claude-sql schema --format json | jq` (the WARNINGs went to stderr but agents reading both streams saw them as noise).

Two failure modes get conflated when you swallow + warn:
- **Expected absence** (parquet not created yet) — should be silent.
- **Unexpected divergence** (DDL is wrong, schema mismatch, etc.) — should be loud.

The gate cleanly separates them. The backstop catches the rest at DEBUG level so a developer can find it via `--verbose` if a future macro misbehaves.

# Example

The full pattern lives in `src/claude_sql/sql_views.py` after the static-catalog refactor:
- `_ANALYTICS_MACRO_REQUIREMENTS` dict (line ~960)
- The gated loop in `register_macros` (line ~975)
- `_safe_macro`'s WARNING demoted to DEBUG (line ~802)

Tested by `tests/test_pr3_perf.py::test_register_macros_skips_friction_when_parquet_missing` which uses a custom loguru sink to assert zero WARNING records on a fresh-install settings object.
