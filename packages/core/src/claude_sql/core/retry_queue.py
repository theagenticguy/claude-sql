"""Durable retry queue backed by SQLite WAL.

When a Bedrock call fails in a way that's worth retrying (parse failure,
throttle that outlived tenacity's stop-after-attempt, transient model
error), the unit of work gets enqueued here. A later run drains the
queue before starting fresh work, so a mid-run crash never costs us the
rows we already paid for.

One row per ``(pipeline, unit_id)``. ``unit_id`` is ``session_id`` for
``classify`` / ``conflicts`` / ``user_friction`` and the message ``uuid`` for
``trajectory``. Semantics are "upsert with attempt counter":

- First failure  → insert with attempts=1, next_attempt_at = now + 2 min.
- Retry failure  → update attempts += 1, next_attempt_at = now + 2^attempts min (cap 60).
- Retry success  → ``completed_at`` stamped; row stays as audit trail.

Lives in the same ``~/.claude/state.db`` as the checkpoint table so a
single SQLite file holds all durable worker state.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from claude_sql.core.checkpointer import (
    _SCHEMA_BOOTSTRAP_LOCK,
    _SCHEMA_BOOTSTRAPPED,
    PIPELINE_NAMES,
    _connect as _checkpoint_connect,
    _from_iso,
    _legacy_duckdb_path,
    _to_iso,
)

MAX_ATTEMPTS_DEFAULT: int = 5
_BACKOFF_CAP_MIN: int = 60

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS retry_queue (
    pipeline        TEXT    NOT NULL,
    unit_id         TEXT    NOT NULL,
    error           TEXT    NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    completed_at    TEXT,
    PRIMARY KEY (pipeline, unit_id)
);
"""


def _connect(path: Path, *, max_attempts: int = 20) -> sqlite3.Connection:
    """Open the queue DB. Reuses the checkpointer connection helper.

    Both tables live in the same SQLite file. The checkpointer's ``_connect``
    handles WAL pragma setup, schema bootstrap, and the one-time DuckDB
    migration. We additionally ensure the ``retry_queue`` table exists.
    """
    con = _checkpoint_connect(path)
    # Schema bootstrap once per process per path — see _SCHEMA_BOOTSTRAPPED in
    # checkpointer.py. Distinct sentinel so retry_queue's CREATE doesn't run
    # on every open. Locked to defeat the "two threads both win the
    # not-bootstrapped check" race that would double-issue DDL and trip the
    # writer lock.
    key = f"retry::{path.resolve()}"
    with _SCHEMA_BOOTSTRAP_LOCK:
        if key not in _SCHEMA_BOOTSTRAPPED:
            prior = con.isolation_level
            con.isolation_level = None  # autocommit for DDL — no implicit transaction wrap
            con.execute(_CREATE_TABLE_SQL)
            con.isolation_level = prior
            _SCHEMA_BOOTSTRAPPED.add(key)
    return con


def _backoff_delta(attempts: int) -> timedelta:
    """Exponential backoff in minutes: 2, 4, 8, 16, 32, capped at 60."""
    minutes = min(2**attempts, _BACKOFF_CAP_MIN)
    return timedelta(minutes=minutes)


def enqueue(
    db_path: Path,
    *,
    pipeline: str,
    unit_id: str,
    error: str,
    now: datetime | None = None,
) -> int:
    """Record a failure.  Increments ``attempts`` on repeat calls.

    Returns the resulting ``attempts`` value for logging.
    """
    if pipeline not in PIPELINE_NAMES:
        raise ValueError(f"unknown pipeline: {pipeline!r}")
    cur = now or datetime.now(UTC)
    cur_iso = _to_iso(cur)
    con = _connect(db_path)
    try:
        row = con.execute(
            "SELECT attempts FROM retry_queue WHERE pipeline = ? AND unit_id = ?",
            [pipeline, unit_id],
        ).fetchone()
        prev = int(row[0]) if row else 0
        attempts = prev + 1
        next_at = (cur if isinstance(cur, datetime) else datetime.now(UTC)) + _backoff_delta(
            attempts
        )
        next_at_iso = _to_iso(next_at)
        con.execute(
            "INSERT INTO retry_queue "
            "(pipeline, unit_id, error, attempts, next_attempt_at, created_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL) "
            "ON CONFLICT(pipeline, unit_id) DO UPDATE SET "
            "error = excluded.error, "
            "attempts = excluded.attempts, "
            "next_attempt_at = excluded.next_attempt_at, "
            "created_at = excluded.created_at, "
            "completed_at = NULL",
            [pipeline, unit_id, error[:2000], attempts, next_at_iso, cur_iso],
        )
        con.commit()
    finally:
        con.close()
    return attempts


def drain(
    db_path: Path,
    *,
    pipeline: str,
    now: datetime | None = None,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT,
    limit: int | None = None,
) -> list[str]:
    """Return unit_ids eligible for retry (not completed, attempts<max, due now)."""
    if not db_path.exists() and not _legacy_duckdb_path(db_path).exists():
        return []
    cur = now or datetime.now(UTC)
    cur_iso = _to_iso(cur)
    con = _connect(db_path)
    sql = (
        "SELECT unit_id FROM retry_queue "
        "WHERE pipeline = ? AND completed_at IS NULL "
        "  AND attempts < ? AND next_attempt_at <= ? "
        "ORDER BY next_attempt_at"
    )
    params: list[object] = [pipeline, max_attempts, cur_iso]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    try:
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    return [str(r[0]) for r in rows]


def mark_done(
    db_path: Path,
    *,
    pipeline: str,
    unit_ids: Iterable[str],
    now: datetime | None = None,
) -> int:
    """Mark the given unit_ids as completed. No-op if unknown."""
    ids = list(unit_ids)
    if not ids:
        return 0
    cur_iso = _to_iso(now or datetime.now(UTC))
    con = _connect(db_path)
    try:
        con.executemany(
            "UPDATE retry_queue SET completed_at = ? "
            "WHERE pipeline = ? AND unit_id = ? AND completed_at IS NULL",
            [(cur_iso, pipeline, uid) for uid in ids],
        )
        con.commit()
    finally:
        con.close()
    return len(ids)


def pending_count(db_path: Path, *, pipeline: str) -> int:
    """Count not-yet-completed rows for one pipeline."""
    if not db_path.exists() and not _legacy_duckdb_path(db_path).exists():
        return 0
    con = _connect(db_path)
    try:
        row = con.execute(
            "SELECT count(*) FROM retry_queue WHERE pipeline = ? AND completed_at IS NULL",
            [pipeline],
        ).fetchone()
    finally:
        con.close()
    return int(row[0]) if row else 0


# Re-export _from_iso for tests that need it
__all__ = [
    "MAX_ATTEMPTS_DEFAULT",
    "_from_iso",
    "drain",
    "enqueue",
    "mark_done",
    "pending_count",
]
