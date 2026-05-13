"""Tests for the v1.0 conflicts storage shape (RFC 0002 §3.4).

The v1.0 rewrite drops the ``empty=True`` sentinel scheme, rekeys the
parquet on ``(turn_a_uuid, turn_b_uuid)``, and adds two enums
(``conflict_kind`` and ``severity``).  These tests pin the behaviour
that:

* sessions with no conflicts produce ZERO parquet rows -- not one
  sentinel row;
* the natural key is the pair of opposing turn UUIDs;
* shards from the legacy ``(session_id, conflict_idx)`` schema are
  deleted on first run before the new shards land;
* ``conflicts_summary`` excludes sessions with zero conflicts (a
  ``LEFT JOIN`` against ``sessions`` is the caller's responsibility);
* both enums carry exactly the documented value sets.

All Bedrock calls are mocked.  The tests sit alongside the existing
``test_llm_worker_pipelines.py`` conflicts dry-run coverage.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import duckdb
import polars as pl
import pytest

from claude_sql import conflicts_worker, llm_shared
from claude_sql.config import Settings
from claude_sql.parquet_shards import iter_part_files, read_all
from claude_sql.schemas import SESSION_CONFLICTS_SCHEMA, ConflictPair, ConflictsResult
from claude_sql.sql_views import register_analytics, register_raw, register_views
from conftest import _seed_subagent_stub, make_user_msg, write_session_jsonl

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client_cache() -> Iterator[None]:
    """Wipe the boto3 client cache before/after each test."""
    llm_shared._CLIENT_CACHE.clear()
    yield
    llm_shared._CLIENT_CACHE.clear()


@pytest.fixture(autouse=True)
def _noop_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defang tenacity / time.sleep so retry paths run instantly."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_kw: None)
    try:
        import tenacity.nap

        monkeypatch.setattr(tenacity.nap.time, "sleep", lambda *_a, **_kw: None)
    except (ImportError, AttributeError):
        # tenacity.nap is private API.  Fall back to the top-level
        # time.sleep patch above when this submodule isn't available --
        # nothing else needs doing in either branch.
        pass


def _build_con(
    tmp_path: Path, sessions: list[tuple[str, list[dict[str, Any]]]]
) -> duckdb.DuckDBPyConnection:
    """Write ``sessions`` to a fresh corpus and return a registered connection."""
    proj = tmp_path / "projects" / "proj-c"
    for sid, msgs in sessions:
        write_session_jsonl(proj / f"{sid}.jsonl", messages=msgs)
    sa_glob, sa_meta = _seed_subagent_stub(tmp_path)
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    con = duckdb.connect(":memory:")
    register_raw(con, glob=glob, subagent_glob=sa_glob, subagent_meta_glob=sa_meta)
    register_views(con)
    return con


# ---------------------------------------------------------------------------
# 1. Empty conflicts list → zero rows (the load-bearing change)
# ---------------------------------------------------------------------------


def test_session_with_no_conflicts_writes_zero_rows(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``conflicts: []`` → no parquet rows for that session (no sentinel)."""
    sid = "sess-clean-zero-conflicts"
    con = _build_con(
        tmp_path,
        [
            (
                sid,
                [
                    make_user_msg(
                        "u-clean",
                        sid,
                        "let's pick a simple plan with no disagreement at all",
                        ts="2026-04-01T10:00:00.000Z",
                    )
                ],
            )
        ],
    )

    monkeypatch.setattr(
        conflicts_worker,
        "classify_one",
        AsyncMock(return_value={"conflicts": []}),
    )
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 1, "session should be marked done even though it had zero conflicts"

    df = read_all(tmp_settings.conflicts_parquet_path)
    # Either no parquet at all (preferred), or a parquet with zero rows -- both
    # represent "this session produced no conflict rows".  The legacy sentinel
    # would have shown up as one row; assert that does NOT happen.
    if df is not None:
        assert df.height == 0, "no-conflict session must not emit a sentinel row"
    con.close()


# ---------------------------------------------------------------------------
# 2. Two conflicts → two pair-keyed rows
# ---------------------------------------------------------------------------


def test_session_with_two_conflicts_writes_two_rows_keyed_on_pair(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two pairs returned → two parquet rows, keyed on ``(turn_a_uuid, turn_b_uuid)``."""
    sid = "sess-two-pairs"
    con = _build_con(
        tmp_path,
        [
            (
                sid,
                [
                    make_user_msg(
                        "u-rich",
                        sid,
                        "this is a substantive technical discussion that easily clears the filter",
                        ts="2026-04-02T10:00:00.000Z",
                    )
                ],
            )
        ],
    )

    payload = {
        "conflicts": [
            {
                "turn_a_uuid": "turn-001",
                "turn_b_uuid": "turn-002",
                "conflict_kind": "disagreement",
                "severity": "medium",
                "agent_position": "Use Sonnet for all classification.",
                "user_position": "Use Opus for the harder schemas.",
                "confidence": 0.82,
            },
            {
                "turn_a_uuid": "turn-007",
                "turn_b_uuid": "turn-009",
                "conflict_kind": "correction",
                "severity": "high",
                "agent_position": "Denormalize the table for the slow query.",
                "user_position": "Add a covering index instead.",
                "confidence": 0.95,
            },
        ]
    }

    monkeypatch.setattr(conflicts_worker, "classify_one", AsyncMock(return_value=payload))
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 1

    df = read_all(tmp_settings.conflicts_parquet_path)
    assert df is not None
    assert df.height == 2

    # Natural key shape — both pairs distinct on (turn_a_uuid, turn_b_uuid).
    keys = set(zip(df["turn_a_uuid"].to_list(), df["turn_b_uuid"].to_list(), strict=True))
    assert keys == {("turn-001", "turn-002"), ("turn-007", "turn-009")}

    # All v1.0 columns are present and the legacy ones are gone.
    expected_cols = {
        "session_id",
        "turn_a_uuid",
        "turn_b_uuid",
        "conflict_kind",
        "severity",
        "agent_position",
        "user_position",
        "confidence",
        "detected_at",
    }
    assert set(df.columns) == expected_cols
    assert "conflict_idx" not in df.columns
    assert "empty" not in df.columns
    assert "stance_a" not in df.columns
    assert "resolution" not in df.columns

    # Confidence round-trips as a float.
    assert df["confidence"].to_list() == pytest.approx([0.82, 0.95])
    con.close()


# ---------------------------------------------------------------------------
# 3. Legacy shards are purged on first v1.0 run
# ---------------------------------------------------------------------------


def test_old_schema_shards_deleted_on_first_run(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale shard with ``conflict_idx`` / ``empty`` columns is gone after the run."""
    sid = "sess-needs-purge"
    con = _build_con(
        tmp_path,
        [
            (
                sid,
                [
                    make_user_msg(
                        "u-purge",
                        sid,
                        "ordinary substantive message that clears the message-text filter",
                        ts="2026-04-03T10:00:00.000Z",
                    )
                ],
            )
        ],
    )

    # Seed a v0 shard manually.  Schema mirrors the pre-v1 ``conflicts_worker``
    # output: ``(session_id, conflict_idx, stance_a, stance_b, resolution,
    # detected_at, empty)``.
    cache_dir = tmp_settings.conflicts_parquet_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    legacy_df = pl.DataFrame(
        {
            "session_id": ["someone-else"],
            "conflict_idx": [0],
            "stance_a": [None],
            "stance_b": [None],
            "resolution": [None],
            "detected_at": [None],
            "empty": [True],
        },
        schema={
            "session_id": pl.Utf8,
            "conflict_idx": pl.Int32,
            "stance_a": pl.Utf8,
            "stance_b": pl.Utf8,
            "resolution": pl.Utf8,
            "detected_at": pl.Datetime("us", "UTC"),
            "empty": pl.Boolean,
        },
    )
    legacy_path = cache_dir / "part-legacy.parquet"
    legacy_df.write_parquet(legacy_path)
    assert legacy_path.exists()

    monkeypatch.setattr(
        conflicts_worker,
        "classify_one",
        AsyncMock(return_value={"conflicts": []}),
    )
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)

    # Legacy shard is gone; no v0 ``conflict_idx`` column survived in any
    # remaining parquet.
    assert not legacy_path.exists()
    for part in iter_part_files(cache_dir):
        df = pl.read_parquet(part)
        assert "conflict_idx" not in df.columns
        assert "empty" not in df.columns
    con.close()


# ---------------------------------------------------------------------------
# 4. conflicts_summary view excludes zero-conflict sessions
# ---------------------------------------------------------------------------


def test_conflicts_summary_view_excludes_zero_conflict_sessions(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session with no conflict rows simply doesn't appear in conflicts_summary."""
    sid_quiet = "sess-quiet-no-conflicts"
    sid_loud = "sess-loud-with-conflicts"
    con = _build_con(
        tmp_path,
        [
            (
                sid_quiet,
                [
                    make_user_msg(
                        "u-q",
                        sid_quiet,
                        f"MARKER_QUIET {sid_quiet} no disagreement here at all please",
                        ts="2026-04-04T10:00:00.000Z",
                    )
                ],
            ),
            (
                sid_loud,
                [
                    make_user_msg(
                        "u-l",
                        sid_loud,
                        f"MARKER_LOUD {sid_loud} substantive message about a real conflict",
                        ts="2026-04-05T10:00:00.000Z",
                    )
                ],
            ),
        ],
    )

    async def _by_marker(*args: Any, **kwargs: Any) -> dict[str, Any]:
        text = args[3] if len(args) >= 4 else kwargs.get("text", "")
        if "MARKER_LOUD" in text:
            return {
                "conflicts": [
                    {
                        "turn_a_uuid": "loud-a",
                        "turn_b_uuid": "loud-b",
                        "conflict_kind": "impasse",
                        "severity": "high",
                        "agent_position": "Ship the cleanup first.",
                        "user_position": "Ship the simple version now.",
                        "confidence": 0.9,
                    }
                ]
            }
        return {"conflicts": []}

    monkeypatch.setattr(conflicts_worker, "classify_one", AsyncMock(side_effect=_by_marker))
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 2

    register_analytics(con, settings=tmp_settings)

    rows = con.execute(
        "SELECT session_id, conflict_count FROM conflicts_summary ORDER BY session_id"
    ).fetchall()
    assert rows == [(sid_loud, 1)], "zero-conflict session must NOT appear in conflicts_summary"
    con.close()


# ---------------------------------------------------------------------------
# 5. Enum value sets are pinned
# ---------------------------------------------------------------------------


def test_conflict_kind_enum_values_are_four() -> None:
    """``conflict_kind`` enum is exactly {disagreement, correction, reversal, impasse}."""
    items_schema = SESSION_CONFLICTS_SCHEMA["properties"]["conflicts"]["items"]
    assert set(items_schema["properties"]["conflict_kind"]["enum"]) == {
        "disagreement",
        "correction",
        "reversal",
        "impasse",
    }


def test_severity_enum_values_are_three() -> None:
    """``severity`` enum is exactly {low, medium, high}."""
    items_schema = SESSION_CONFLICTS_SCHEMA["properties"]["conflicts"]["items"]
    assert set(items_schema["properties"]["severity"]["enum"]) == {
        "low",
        "medium",
        "high",
    }


# ---------------------------------------------------------------------------
# 6. Pydantic model round-trip (smoke check on the new schema names)
# ---------------------------------------------------------------------------


def test_conflict_pair_pydantic_round_trip() -> None:
    """``ConflictPair`` accepts a fully populated dict and serialises identically."""
    pair = ConflictPair(
        turn_a_uuid="t1",
        turn_b_uuid="t2",
        conflict_kind="reversal",
        severity="low",
        agent_position="Use the cached embeddings.",
        user_position="Rebuild from parquet every run.",
        confidence=0.6,
    )
    result = ConflictsResult(conflicts=[pair])
    assert len(result.conflicts) == 1
    assert result.conflicts[0].conflict_kind == "reversal"
    assert result.conflicts[0].severity == "low"
