"""Tests for :mod:`claude_sql.retry_queue`."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from claude_sql.infrastructure.sqlite_state.retry_queue import (
    MAX_ATTEMPTS_DEFAULT,
    drain,
    enqueue,
    mark_done,
    pending_count,
)


def test_enqueue_rejects_unknown_pipeline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown pipeline"):
        enqueue(tmp_path / "q.duckdb", pipeline="bogus", unit_id="x", error="e")


def test_drain_missing_db_returns_empty(tmp_path: Path) -> None:
    assert drain(tmp_path / "nope.duckdb", pipeline="classify") == []


def test_pending_count_missing_db_is_zero(tmp_path: Path) -> None:
    assert pending_count(tmp_path / "nope.duckdb", pipeline="classify") == 0


def test_enqueue_increments_attempts(tmp_path: Path) -> None:
    db = tmp_path / "q.duckdb"
    a1 = enqueue(db, pipeline="classify", unit_id="s1", error="boom")
    a2 = enqueue(db, pipeline="classify", unit_id="s1", error="boom again")
    assert a1 == 1
    assert a2 == 2


def test_enqueue_backoff_increases(tmp_path: Path) -> None:
    db = tmp_path / "q.duckdb"
    t0 = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    enqueue(db, pipeline="classify", unit_id="s1", error="e", now=t0)
    enqueue(db, pipeline="classify", unit_id="s1", error="e", now=t0)
    con = sqlite3.connect(str(db))
    try:
        row = con.execute(
            "SELECT attempts, next_attempt_at FROM retry_queue WHERE unit_id = 's1'"
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    attempts, next_at_iso = row
    # Second failure → next_attempt = t0 + 2^2 min = t0 + 4 min.
    expected = (t0 + timedelta(minutes=4)).isoformat()
    assert attempts == 2
    assert next_at_iso == expected


def test_drain_respects_next_attempt_at(tmp_path: Path) -> None:
    db = tmp_path / "q.duckdb"
    t0 = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    enqueue(db, pipeline="classify", unit_id="s1", error="e", now=t0)
    # Too early — backoff from attempts=1 puts next_attempt_at at t0+2min.
    assert drain(db, pipeline="classify", now=t0) == []
    # Wait past the backoff.
    assert drain(db, pipeline="classify", now=t0 + timedelta(minutes=3)) == ["s1"]


def test_drain_respects_max_attempts(tmp_path: Path) -> None:
    db = tmp_path / "q.duckdb"
    t0 = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    for _ in range(MAX_ATTEMPTS_DEFAULT):
        enqueue(db, pipeline="classify", unit_id="doomed", error="e", now=t0)
    # After MAX_ATTEMPTS_DEFAULT failures the row is no longer drained.
    assert (
        drain(
            db, pipeline="classify", now=t0 + timedelta(hours=24), max_attempts=MAX_ATTEMPTS_DEFAULT
        )
        == []
    )


def test_drain_scoped_per_pipeline(tmp_path: Path) -> None:
    db = tmp_path / "q.duckdb"
    t0 = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    enqueue(db, pipeline="classify", unit_id="s1", error="e", now=t0)
    enqueue(db, pipeline="trajectory", unit_id="u1", error="e", now=t0)
    later = t0 + timedelta(minutes=5)
    assert drain(db, pipeline="classify", now=later) == ["s1"]
    assert drain(db, pipeline="trajectory", now=later) == ["u1"]


def test_mark_done_hides_entry_from_drain(tmp_path: Path) -> None:
    db = tmp_path / "q.duckdb"
    t0 = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    enqueue(db, pipeline="classify", unit_id="s1", error="e", now=t0)
    assert pending_count(db, pipeline="classify") == 1
    mark_done(db, pipeline="classify", unit_ids=["s1"])
    later = t0 + timedelta(minutes=10)
    assert drain(db, pipeline="classify", now=later) == []
    assert pending_count(db, pipeline="classify") == 0


def test_persists_across_connections(tmp_path: Path) -> None:
    db = tmp_path / "q.duckdb"
    t0 = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    enqueue(db, pipeline="classify", unit_id="s1", error="e", now=t0)
    # Simulating a fresh process is the same as opening another connection.
    assert pending_count(db, pipeline="classify") == 1


def test_enqueue_coexists_with_checkpoint_table(tmp_path: Path) -> None:
    """Both tables live in the same DB file; one shouldn't break the other."""
    from claude_sql.infrastructure.sqlite_state.checkpointer import mark_completed

    db = tmp_path / "shared.duckdb"
    t0 = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    mark_completed(db, pipeline="classify", rows=[("s1", t0, t0)])
    enqueue(db, pipeline="classify", unit_id="s1", error="e", now=t0)
    con = sqlite3.connect(str(db))
    try:
        tables = {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    finally:
        con.close()
    # Both core tables present; sqlite may also create internal pages.
    assert {"session_checkpoint", "retry_queue"}.issubset(tables)
