"""Tests for :mod:`claude_sql.checkpointer`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from claude_sql.checkpointer import (
    PIPELINE_NAMES,
    filter_unchanged,
    load,
    load_as_map,
    mark_completed,
)


def test_pipeline_names_covers_llm_workers() -> None:
    assert set(PIPELINE_NAMES) == {"classify", "trajectory", "conflicts"}


def test_load_missing_returns_empty_with_schema(tmp_path: Path) -> None:
    df = load(tmp_path / "nope.parquet")
    assert df.is_empty()
    assert set(df.columns) == {
        "session_id",
        "pipeline",
        "last_ts_processed",
        "last_mtime_processed",
        "completed_at",
    }


def test_load_as_map_empty(tmp_path: Path) -> None:
    assert load_as_map(tmp_path / "nope.parquet", pipeline="classify") == {}


def test_mark_completed_writes_and_loads(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    ts = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mt = datetime(2026, 4, 20, 11, 0, tzinfo=UTC)
    n = mark_completed(
        path,
        pipeline="classify",
        rows=[("sess-a", ts, mt), ("sess-b", ts, mt)],
    )
    assert n == 2
    df = load(path)
    assert sorted(df["session_id"].to_list()) == ["sess-a", "sess-b"]
    assert df["pipeline"].unique().to_list() == ["classify"]


def test_mark_completed_upserts_by_session_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    mark_completed(path, pipeline="classify", rows=[("sess-a", t0, t0)])
    mark_completed(path, pipeline="classify", rows=[("sess-a", t1, t1)])
    df = load(path)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["last_ts_processed"] == t1


def test_mark_completed_upsert_preserves_other_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mark_completed(path, pipeline="classify", rows=[("sess-a", t0, t0)])
    mark_completed(path, pipeline="trajectory", rows=[("sess-a", t0, t0)])
    df = load(path).sort("pipeline")
    assert df["pipeline"].to_list() == ["classify", "trajectory"]


def test_mark_completed_empty_rows_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    assert mark_completed(path, pipeline="classify", rows=[]) == 0
    assert not path.exists()


def test_filter_unchanged_first_run_keeps_all(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    pending, skipped = filter_unchanged(
        [("sess-a", t0, t0), ("sess-b", t0, t0)],
        pipeline="classify",
        checkpoint_path=path,
    )
    assert sorted(pending) == ["sess-a", "sess-b"]
    assert skipped == 0


def test_filter_unchanged_skips_stale_sessions(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    mark_completed(
        path,
        pipeline="classify",
        rows=[("sess-a", t1, t1), ("sess-b", t0, t0)],
    )
    # sess-a's current state matches checkpoint exactly → skip.
    # sess-b grew (new max ts) → must re-process.
    # sess-c is brand new → must process.
    pending, skipped = filter_unchanged(
        [("sess-a", t1, t1), ("sess-b", t1, t0), ("sess-c", t0, t0)],
        pipeline="classify",
        checkpoint_path=path,
    )
    assert sorted(pending) == ["sess-b", "sess-c"]
    assert skipped == 1


def test_filter_unchanged_mtime_growth_reprocesses(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)
    mark_completed(path, pipeline="trajectory", rows=[("sess-a", t0, t0)])
    # ts unchanged but mtime advanced (e.g., tool-result rewrite) → reprocess.
    pending, skipped = filter_unchanged(
        [("sess-a", t0, t1)],
        pipeline="trajectory",
        checkpoint_path=path,
    )
    assert pending == ["sess-a"]
    assert skipped == 0


def test_filter_unchanged_scoped_per_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mark_completed(path, pipeline="classify", rows=[("sess-a", t0, t0)])
    # Checkpoint exists for classify, but we're asking about trajectory — must
    # not carry across pipelines.
    pending, skipped = filter_unchanged(
        [("sess-a", t0, t0)],
        pipeline="trajectory",
        checkpoint_path=path,
    )
    assert pending == ["sess-a"]
    assert skipped == 0


def test_filter_unchanged_handles_none_bounds(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mark_completed(path, pipeline="classify", rows=[("sess-a", t0, t0)])
    # Current candidate has no known mtime → never skip.
    pending, _ = filter_unchanged(
        [("sess-a", t0, None)],
        pipeline="classify",
        checkpoint_path=path,
    )
    assert pending == ["sess-a"]


def test_roundtrip_schema_matches_declared(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.parquet"
    t0 = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    mark_completed(path, pipeline="classify", rows=[("sess-a", t0, t0)])
    df = pl.read_parquet(path)
    dtypes = dict(zip(df.columns, df.dtypes, strict=True))
    assert dtypes["session_id"] == pl.Utf8
    assert dtypes["pipeline"] == pl.Utf8
    assert dtypes["last_ts_processed"] == pl.Datetime("us", "UTC")
    assert dtypes["last_mtime_processed"] == pl.Datetime("us", "UTC")
    assert dtypes["completed_at"] == pl.Datetime("us", "UTC")
