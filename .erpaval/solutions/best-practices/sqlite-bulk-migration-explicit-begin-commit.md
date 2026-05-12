---
title: stdlib sqlite3 autocommit + `executemany` fsyncs per row, not per batch
track: knowledge
category: best-practices
module: src/claude_sql/checkpointer.py
component: stdlib sqlite3 + WAL bulk insert
severity: warning
tags: [sqlite, sqlite3, executemany, fsync, transaction, autocommit, performance]
applies_when:
  - Bulk-loading rows via `connection.executemany()` after opening with `isolation_level=None`
  - Migrating data from one store to another in a single startup hook
  - Wondering why a "fast" SQLite write hangs on disks with high fsync latency
pattern: |
  Python's stdlib sqlite3 connection opened with `isolation_level=None`
  is in **autocommit mode**: every `execute()` and every row of
  `executemany()` is its own transaction. WAL mode means each
  transaction triggers a separate fsync of the WAL frame, so a 39k-row
  `executemany` becomes 39k fsyncs serialized on disk latency. On EFS
  / network-mounted home directories this stalls for many minutes;
  on local SSDs it's seconds-to-minutes.

  The `sqlite3.executemany` docstring suggests batch semantics, but
  in autocommit mode each row genuinely is its own transaction.

  Fix: wrap the bulk inserts in an explicit `BEGIN` / `COMMIT`. One
  transaction → one fsync at the end → 39k rows in <4 seconds:

  ```python
  con = sqlite3.connect(path, isolation_level=None, timeout=30.0)
  con.execute("PRAGMA journal_mode=WAL")
  con.execute("PRAGMA synchronous=NORMAL")
  con.execute(CREATE_TABLE_SQL)

  if rows:
      con.execute("BEGIN")
      try:
          con.executemany("INSERT INTO ...", rows)
          con.execute("COMMIT")
      except Exception:
          con.execute("ROLLBACK")
          raise
  ```

  Switching to `isolation_level="DEFERRED"` instead would also batch,
  but only AFTER the PRAGMAs run — and you can't set
  `journal_mode=WAL` from inside a transaction.
example_files:
  - src/claude_sql/checkpointer.py
---

# Why this matters

The first version of `_migrate_from_duckdb_if_present` opened the new
SQLite file in autocommit (so PRAGMAs could run) and `executemany`'d
39,425 checkpoint rows + 42 retry rows. On EFS-mounted `~/.claude/`,
the migration **hung indefinitely**. Killing the process left a
612 MB WAL file (every row had been written to WAL but never
committed). Subsequent calls retried from scratch — infinite
migration loop.

After the BEGIN/COMMIT fix: same migration, **3.76 seconds end-to-end**.

The trap: stdlib sqlite3's autocommit mode is the recommended setup
for setting up PRAGMAs that can't run inside a transaction
(`journal_mode=WAL` is the canonical example). It's natural to keep
the connection in autocommit and just `executemany` the data afterward.
That works for small batches; it falls off a cliff at scale on any
non-trivial fsync latency.

# Three modes, three behaviors

| `isolation_level` | `executemany(N rows)` behavior |
|---|---|
| `None` (autocommit) | N transactions, N fsyncs |
| `"DEFERRED"` | 1 implicit BEGIN + COMMIT around the executemany |
| explicit `BEGIN`/`COMMIT` | 1 transaction regardless of mode |

The "right" answer for our case: stay in autocommit for setup, wrap
bulk writes in explicit `BEGIN`/`COMMIT`. Switching to DEFERRED
permanently would be cleaner but breaks the PRAGMA-setup phase.

# Pinned by

Live observation, not a test: the WAL race fix
(`sqlite-wal-cold-start-pragma-race.md`) was masked by this hang
during the same session. Adding a unit test that times bulk-load
performance is brittle (depends on disk speed). The lesson lives
here so the pattern is recognized next time, not regression-tested.
