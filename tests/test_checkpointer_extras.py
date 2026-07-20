"""Coverage top-up for :mod:`claude_sql.checkpointer` (SQLite WAL).

The DuckDB-specific helpers (``_strip_tz``, ``_attach_tz``, the 20× lock-retry
loop) were dropped when the store moved to SQLite WAL: timestamps are now
ISO-8601 strings, and ``PRAGMA busy_timeout`` absorbs transient writer
contention. This file covers the replacement helpers and concurrency model.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from claude_sql.infrastructure.sqlite_state import checkpointer


def test_to_iso_none_returns_none() -> None:
    """``_to_iso(None)`` short-circuits."""
    assert checkpointer._to_iso(None) is None


def test_to_iso_emits_utc_string() -> None:
    dt = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    out = checkpointer._to_iso(dt)
    assert out is not None
    assert out.startswith("2026-04-20T12:00:00")
    assert out.endswith(("+00:00", "Z"))


def test_from_iso_none_returns_none() -> None:
    assert checkpointer._from_iso(None) is None


def test_from_iso_round_trips_to_aware_utc() -> None:
    dt = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    iso = checkpointer._to_iso(dt)
    assert iso is not None
    out = checkpointer._from_iso(iso)
    assert out is not None
    assert out.tzinfo is not None
    assert out == dt


def test_from_iso_attaches_utc_to_naive_input() -> None:
    """Naive ISO strings (legacy DuckDB rows) are interpreted as UTC."""
    out = checkpointer._from_iso("2026-04-20T12:00:00")
    assert out is not None
    assert out.tzinfo is UTC


def test_connect_creates_sqlite_wal_file(tmp_path: Path) -> None:
    """``_connect`` produces a real WAL-mode SQLite file."""
    db = tmp_path / "state.db"
    con = checkpointer._connect(db)
    try:
        row = con.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        assert row[0].lower() == "wal"
        # Schema is bootstrapped:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='session_checkpoint'"
        ).fetchone()
        assert rows is not None
    finally:
        con.close()


def test_connect_handles_concurrent_writers(tmp_path: Path) -> None:
    """Two threads writing to the same DB serialize cleanly under WAL.

    DuckDB's single-writer file lock used to force a 20× exponential-backoff
    retry loop here. SQLite WAL + ``busy_timeout`` absorbs the contention
    transparently — both writers complete without raising.
    """
    db = tmp_path / "state.db"
    errors: list[Exception] = []

    def writer(pipeline: str, n: int) -> None:
        try:
            for i in range(n):
                checkpointer.mark_completed(
                    db,
                    pipeline=pipeline,
                    rows=[(f"sess-{pipeline}-{i}", datetime.now(UTC), datetime.now(UTC))],
                )
        except Exception as exc:  # noqa: BLE001 — capture worker-thread failures for the main thread to assert on; KeyboardInterrupt/SystemExit must still propagate
            errors.append(exc)

    t1 = threading.Thread(target=writer, args=("classify", 20))
    t2 = threading.Thread(target=writer, args=("trajectory", 20))
    start = time.monotonic()
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    elapsed = time.monotonic() - start

    assert not errors, f"unexpected concurrent-writer failures: {errors}"
    assert elapsed < 5, f"concurrent writers took {elapsed:.2f}s — busy_timeout expected to absorb"
    assert checkpointer.count_rows(db) == 40


def test_stale_or_equal_handles_naive_inputs() -> None:
    """``_stale_or_equal`` accepts naive datetimes from legacy callers.

    Naive datetimes (no tzinfo) are interpreted as UTC by ``_stale_or_equal``
    so legacy callers that fetched raw DuckDB rows don't blow up. We
    construct aware datetimes and strip tzinfo to mirror that shape.
    """
    cur_naive = datetime(2026, 4, 20, 12, 0, tzinfo=UTC).replace(tzinfo=None)
    prev_naive = datetime(2026, 4, 20, 11, 0, tzinfo=UTC).replace(tzinfo=None)
    # current advanced past previous → not stale
    assert checkpointer._stale_or_equal(cur_naive, prev_naive) is False


def test_stale_or_equal_none_returns_false() -> None:
    cur = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    assert checkpointer._stale_or_equal(None, cur) is False
    assert checkpointer._stale_or_equal(cur, None) is False
    assert checkpointer._stale_or_equal(None, None) is False


def test_migrate_from_duckdb_sentinel_idempotent(tmp_path: Path) -> None:
    """When no legacy file exists, migration drops a sentinel and never re-runs."""
    db = tmp_path / "state.db"
    checkpointer._migrate_from_duckdb_if_present(db)
    sentinel = checkpointer._migration_sentinel_path(db)
    assert sentinel.exists()
    # Second call is a no-op (would otherwise re-import duckdb).
    checkpointer._migrate_from_duckdb_if_present(db)
    assert sentinel.exists()


@pytest.mark.parametrize(
    "pipeline",
    ["classify", "trajectory", "conflicts", "user_friction"],
)
def test_pipeline_names_includes_user_friction(pipeline: str) -> None:
    """``user_friction`` must be in PIPELINE_NAMES — the friction worker enqueues with it."""
    assert pipeline in checkpointer.PIPELINE_NAMES


def test_sqlite_db_is_actual_sqlite_file(tmp_path: Path) -> None:
    """Sanity: the file is SQLite, not DuckDB."""
    db = tmp_path / "state.db"
    con = checkpointer._connect(db)
    con.close()
    # Magic header for SQLite v3:
    assert db.read_bytes()[:16].startswith(b"SQLite format 3")


def test_concurrent_readers_under_writer(tmp_path: Path) -> None:
    """Many concurrent readers + one writer all succeed (WAL guarantee)."""
    db = tmp_path / "state.db"
    checkpointer.mark_completed(
        db,
        pipeline="classify",
        rows=[("seed", datetime.now(UTC), datetime.now(UTC))],
    )

    errors: list[Exception] = []

    def reader() -> None:
        try:
            checkpointer.load_as_map(db, "classify")
            checkpointer.count_rows(db)
        except Exception as exc:  # noqa: BLE001 — capture worker-thread failures; KeyboardInterrupt/SystemExit must still propagate
            errors.append(exc)

    def writer() -> None:
        try:
            for i in range(50):
                checkpointer.mark_completed(
                    db,
                    pipeline="classify",
                    rows=[(f"w-{i}", datetime.now(UTC), datetime.now(UTC))],
                )
        except Exception as exc:  # noqa: BLE001 — capture worker-thread failures; KeyboardInterrupt/SystemExit must still propagate
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    threads.append(threading.Thread(target=writer))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"unexpected reader/writer failures: {errors}"


def test_sqlite_module_attribute_exists() -> None:
    """``checkpointer.sqlite3`` is available for tests that monkeypatch the connect path."""
    assert checkpointer.sqlite3 is sqlite3
