---
title: SQLite WAL cold-start writers race on `PRAGMA journal_mode=WAL` itself
track: knowledge
category: best-practices
module: src/claude_sql/checkpointer.py
component: stdlib sqlite3 + WAL setup
severity: warning
tags: [sqlite, wal, threading, pragma, race-condition, cold-start]
applies_when:
  - Multiple threads/processes call `sqlite3.connect()` on the same fresh DB
  - Each connection runs `PRAGMA journal_mode=WAL` after open
  - You set `PRAGMA busy_timeout` to absorb writer-lock contention
pattern: |
  `PRAGMA journal_mode=WAL` itself acquires the SQLite writer lock to
  rewrite the database header. On cold-start with N concurrent
  connections, all N race to set WAL mode; the loser raises
  `sqlite3.OperationalError: database is locked` immediately, BEFORE
  the per-connection `busy_timeout` has a chance to absorb it. Same
  story for `CREATE TABLE IF NOT EXISTS` — idempotent at the SQL
  level, but still acquires the writer lock.

  Once WAL mode lands, it persists in the DB header, so subsequent
  connects don't re-trigger the race. The bug only fires on the
  *very first* concurrent open.

  Fix: hide the file-level setup behind a once-per-(process, path)
  lock alongside the schema bootstrap.

  ```python
  _SCHEMA_BOOTSTRAPPED: set[str] = set()
  _SCHEMA_BOOTSTRAP_LOCK = threading.Lock()

  def _connect(path: Path) -> sqlite3.Connection:
      con = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
      # Per-connection PRAGMAs — must run every open
      con.execute("PRAGMA busy_timeout=10000")
      con.execute("PRAGMA foreign_keys=ON")
      # File-level setup — once per (process, path)
      key = str(path.resolve())
      with _SCHEMA_BOOTSTRAP_LOCK:
          if key not in _SCHEMA_BOOTSTRAPPED:
              con.execute("PRAGMA journal_mode=WAL")
              con.execute("PRAGMA synchronous=NORMAL")
              con.execute(_CREATE_TABLES_SQL)
              _SCHEMA_BOOTSTRAPPED.add(key)
      con.isolation_level = "DEFERRED"
      return con
  ```
example_files:
  - src/claude_sql/checkpointer.py
  - tests/test_checkpointer_extras.py
---

# Why this matters

We migrated the checkpoint store from DuckDB to SQLite WAL specifically
to remove writer-lock retry storms. The first version of the migration
ran `PRAGMA journal_mode=WAL` and `CREATE TABLE IF NOT EXISTS` on every
`_connect()` call, gated only by the `_SCHEMA_BOOTSTRAPPED` cache for
the table DDL. The PRAGMAs ran unconditionally.

Result: the very test we wrote to prove the new system handled
concurrent writers (`test_connect_handles_concurrent_writers`) **failed
flakily** with `database is locked` — exactly the symptom we'd just
left behind on DuckDB. The race was on the WAL transition itself.

The lesson: in SQLite, **anything that mutates the DB header
(`journal_mode`, `synchronous`, `application_id`) is a writer-lock
operation**. Even when conceptually idempotent, multiple cold
writers will collide. Treat them as schema bootstrap, not as
per-connection setup.

# How we caught it

Writing a concurrency test that opens N connections in N threads from
a fresh `tmp_path` is the right shape — single-thread / single-process
tests would never have surfaced this. The test stayed flaky (fails
in isolation, passes after other tests had warmed the cache) until we
moved the PRAGMAs behind the schema lock.

# Pinned by

`tests/test_checkpointer_extras.py::test_connect_handles_concurrent_writers`
— two writer threads, fresh DB, asserts no `database is locked` error.
Repeatable in isolation post-fix.
