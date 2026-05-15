"""Tests for :func:`claude_sql.session_text.session_bounds`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pytest

from claude_sql.core.session_text import session_bounds
from claude_sql.core.sql_views import register_raw, register_views


def _write_session_jsonl(
    path: Path,
    *,
    session_id: str,
    messages: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for msg in messages:
            fh.write(json.dumps(msg))
            fh.write("\n")


def _user_text_record(uuid: str, session_id: str, ts: str, text: str) -> dict:
    return {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": session_id,
        "version": "2.0.0",
        "gitBranch": "main",
        "cwd": "/home/u/proj",
        "userType": "external",
        "entrypoint": "cli",
        "permissionMode": "acceptEdits",
        "promptId": f"p-{uuid}",
        "message": {
            "id": f"m-{uuid}",
            "type": "message",
            "role": "user",
            "model": None,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


@pytest.fixture
def fixture_con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Write two session JSONLs and return a DuckDB connection with the views."""
    proj = tmp_path / "projects" / "proj-a"
    _write_session_jsonl(
        proj / "sess-one.jsonl",
        session_id="sess-one",
        messages=[
            _user_text_record(
                "u1",
                "sess-one",
                "2026-04-01T10:00:00.000Z",
                "hello this is a message long enough to pass the 32-char filter",
            ),
            _user_text_record(
                "u2",
                "sess-one",
                "2026-04-01T10:05:00.000Z",
                "world this is another message long enough to pass the filter too",
            ),
        ],
    )
    _write_session_jsonl(
        proj / "sess-two.jsonl",
        session_id="sess-two",
        messages=[
            _user_text_record(
                "v1",
                "sess-two",
                "2026-04-02T09:00:00.000Z",
                "second session has its own message clearing the 32-char floor",
            ),
        ],
    )
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    # Seed an empty subagent tree under tmp_path so the default globs (which
    # point at ~/.claude/...) don't crash DuckDB's read_json on a fresh host
    # (e.g. CI runners, first-install dev boxes without any subagents yet).
    subagent_dir = (
        tmp_path / "projects" / "proj-a" / ("00000000-0000-0000-0000-000000000000") / "subagents"
    )
    subagent_dir.mkdir(parents=True, exist_ok=True)
    # Write a single stub subagent record + meta so read_json can infer a
    # schema containing every column the downstream subagent views select
    # (timestamp, type, message.{role,model,usage,content} for v_raw_subagents;
    # agentType + description for v_raw_subagent_meta).  The row is filtered
    # out of all downstream queries by its non-matching parent session id.
    _write_session_jsonl(
        subagent_dir / "agent-placeholder.jsonl",
        session_id="placeholder",
        messages=[
            _user_text_record(
                "sa-stub",
                "placeholder",
                "2026-01-01T00:00:00.000Z",
                "subagent stub record to give read_json a schema to infer from",
            ),
        ],
    )
    (subagent_dir / "agent-placeholder.meta.json").write_text(
        json.dumps({"agentType": "stub", "description": "stub"}),
    )
    subagent_glob = str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl")
    subagent_meta_glob = str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.meta.json")
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=glob,
        subagent_glob=subagent_glob,
        subagent_meta_glob=subagent_meta_glob,
    )
    register_views(con)
    return con


def test_session_bounds_returns_last_ts_and_mtime(
    fixture_con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    bounds = session_bounds(fixture_con)
    assert set(bounds.keys()) == {"sess-one", "sess-two"}

    last_ts_one, mtime_one = bounds["sess-one"]
    assert last_ts_one is not None
    assert last_ts_one.isoformat().startswith("2026-04-01T10:05:00")
    # The mtime must land within the set of our fixture-file timestamps.
    one_path = tmp_path / "projects" / "proj-a" / "sess-one.jsonl"
    assert mtime_one is not None
    assert abs(mtime_one.timestamp() - one_path.stat().st_mtime) < 1.0


def test_session_bounds_advances_with_file_mtime(
    fixture_con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    before = session_bounds(fixture_con)
    mtime_before = before["sess-one"][1]
    assert mtime_before is not None

    # Advance the file mtime.
    one_path = tmp_path / "projects" / "proj-a" / "sess-one.jsonl"
    later = one_path.stat().st_mtime + 60
    os.utime(one_path, (later, later))

    after = session_bounds(fixture_con)
    mtime_after = after["sess-one"][1]
    assert mtime_after is not None
    assert mtime_after > mtime_before
