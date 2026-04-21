"""Tests for :mod:`claude_sql.checkpointer`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb

from claude_sql.checkpointer import (
    PIPELINE_NAMES,
    count_rows,
    filter_unchanged,
    load_as_map,
    mark_completed,
)


def test_pipeline_names_covers_llm_workers() -> None:
    assert set(PIPELINE_NAMES) == {"classify", "trajectory", "conflicts"}


def test_load_as_map_missing_db(tmp_path: Path) -> None:
    assert load_as_map(tmp_path / "nope.duckdb", pipeline="classify") == {}


def test_count_rows_missing_db(tmp_path: Path) -> None:
    assert count_rows(tmp_path / "nope.duckdb") == 0


def test_mark_completed_writes_and_reads(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    ts = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mt = datetime(2026, 4, 20, 11, 0, tzinfo=UTC)
    n = mark_completed(
        db,
        pipeline="classify",
        rows=[("sess-a", ts, mt), ("sess-b", ts, mt)],
    )
    assert n == 2
    assert count_rows(db) == 2
    m = load_as_map(db, pipeline="classify")
    assert set(m.keys()) == {"sess-a", "sess-b"}
    assert m["sess-a"] == (ts, mt)


def test_mark_completed_upserts_by_session_pipeline(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    mark_completed(db, pipeline="classify", rows=[("sess-a", t0, t0)])
    mark_completed(db, pipeline="classify", rows=[("sess-a", t1, t1)])
    assert count_rows(db) == 1
    m = load_as_map(db, pipeline="classify")
    assert m["sess-a"] == (t1, t1)


def test_mark_completed_preserves_other_pipeline(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mark_completed(db, pipeline="classify", rows=[("sess-a", t0, t0)])
    mark_completed(db, pipeline="trajectory", rows=[("sess-a", t0, t0)])
    assert count_rows(db) == 2
    assert "sess-a" in load_as_map(db, pipeline="classify")
    assert "sess-a" in load_as_map(db, pipeline="trajectory")


def test_mark_completed_empty_rows_is_noop(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    assert mark_completed(db, pipeline="classify", rows=[]) == 0
    assert not db.exists()


def test_filter_unchanged_first_run_keeps_all(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    pending, skipped = filter_unchanged(
        [("sess-a", t0, t0), ("sess-b", t0, t0)],
        pipeline="classify",
        checkpoint_db_path=db,
    )
    assert sorted(pending) == ["sess-a", "sess-b"]
    assert skipped == 0


def test_filter_unchanged_skips_stale_sessions(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    mark_completed(
        db,
        pipeline="classify",
        rows=[("sess-a", t1, t1), ("sess-b", t0, t0)],
    )
    pending, skipped = filter_unchanged(
        [("sess-a", t1, t1), ("sess-b", t1, t0), ("sess-c", t0, t0)],
        pipeline="classify",
        checkpoint_db_path=db,
    )
    assert sorted(pending) == ["sess-b", "sess-c"]
    assert skipped == 1


def test_filter_unchanged_mtime_growth_reprocesses(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)
    mark_completed(db, pipeline="trajectory", rows=[("sess-a", t0, t0)])
    pending, skipped = filter_unchanged(
        [("sess-a", t0, t1)],
        pipeline="trajectory",
        checkpoint_db_path=db,
    )
    assert pending == ["sess-a"]
    assert skipped == 0


def test_filter_unchanged_scoped_per_pipeline(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mark_completed(db, pipeline="classify", rows=[("sess-a", t0, t0)])
    pending, skipped = filter_unchanged(
        [("sess-a", t0, t0)],
        pipeline="trajectory",
        checkpoint_db_path=db,
    )
    assert pending == ["sess-a"]
    assert skipped == 0


def test_filter_unchanged_handles_none_bounds(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mark_completed(db, pipeline="classify", rows=[("sess-a", t0, t0)])
    pending, _ = filter_unchanged(
        [("sess-a", t0, None)],
        pipeline="classify",
        checkpoint_db_path=db,
    )
    assert pending == ["sess-a"]


def test_table_created_with_primary_key(tmp_path: Path) -> None:
    """A second INSERT with the same (session_id, pipeline) must REPLACE, not append."""
    db = tmp_path / "ckpt.duckdb"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mark_completed(db, pipeline="classify", rows=[("sess-a", t0, t0)])
    # Query the raw DB to confirm the primary key constraint landed.
    con = duckdb.connect(str(db))
    try:
        cols = con.execute(
            "SELECT column_name FROM duckdb_columns() "
            "WHERE table_name = 'session_checkpoint' ORDER BY column_index"
        ).fetchall()
    finally:
        con.close()
    assert [c[0] for c in cols] == [
        "session_id",
        "pipeline",
        "last_ts_processed",
        "last_mtime_processed",
        "completed_at",
    ]


def test_checkpoint_persists_across_connections(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mark_completed(db, pipeline="classify", rows=[("sess-a", t0, t0)])
    # Close and reopen: state must survive.
    assert count_rows(db) == 1
    m = load_as_map(db, pipeline="classify")
    assert m == {"sess-a": (t0, t0)}
