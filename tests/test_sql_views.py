"""Tests for sql_views.py -- view registration, macros, and pushdown behavior."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from claude_sql.sql_views import (
    describe_all,
    list_macros,
    register_macros,
    register_raw,
    register_views,
    register_vss,
)

# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


SESSION_IDS = [
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
    "33333333-3333-3333-3333-333333333333",
]


def _msg(
    uuid: str,
    session_id: str,
    ts: str,
    *,
    role: str,
    model: str | None,
    content: list[dict],
    usage: dict | None = None,
    type_: str = "assistant",
    parent: str | None = None,
) -> dict:
    """Build one transcript record in the Claude Code JSONL shape."""
    return {
        "parentUuid": parent,
        "isSidechain": False,
        "type": type_,
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": session_id,
        "version": "2.0.0",
        "gitBranch": "main",
        "cwd": "/home/alice/proj",
        "userType": "external",
        "entrypoint": "cli",
        "permissionMode": "acceptEdits",
        "promptId": f"p-{uuid}",
        "message": {
            "id": f"m-{uuid}",
            "type": "message",
            "role": role,
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": content,
            "usage": usage
            or {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    """Write 3 conversation JSONL files + 2 subagent pairs under tmp_path.

    Returns the project slug directory that the glob patterns anchor on.
    """
    proj = tmp_path / "projects" / "-home-alice-workplace-proj"
    proj.mkdir(parents=True)

    # Session 1: TodoWrite twice (evolves pending -> completed), opus, costly.
    # Also emits a user message to exercise the type IN ('user','assistant') filter.
    file1 = proj / f"{SESSION_IDS[0]}.jsonl"
    records1 = [
        _msg(
            "u-1",
            SESSION_IDS[0],
            "2026-04-01T10:00:00.000Z",
            role="user",
            model=None,
            type_="user",
            content=[{"type": "text", "text": "Please do X."}],
        ),
        _msg(
            "a-1",
            SESSION_IDS[0],
            "2026-04-01T10:00:10.000Z",
            role="assistant",
            model="claude-opus-4-7",
            content=[
                {"type": "text", "text": "Planning..."},
                {
                    "type": "tool_use",
                    "id": "tu-1",
                    "name": "TodoWrite",
                    "input": {
                        "todos": [
                            {
                                "content": "Task A",
                                "status": "pending",
                                "activeForm": "Doing A",
                            },
                            {
                                "content": "Task B",
                                "status": "pending",
                                "activeForm": "Doing B",
                            },
                        ]
                    },
                },
            ],
            usage={
                "input_tokens": 10_000,
                "output_tokens": 5_000,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        ),
        _msg(
            "a-2",
            SESSION_IDS[0],
            "2026-04-01T10:01:00.000Z",
            role="assistant",
            model="claude-opus-4-7",
            content=[
                {
                    "type": "tool_use",
                    "id": "tu-2",
                    "name": "TodoWrite",
                    "input": {
                        "todos": [
                            {
                                "content": "Task A",
                                "status": "completed",
                                "activeForm": "Doing A",
                            },
                            {
                                "content": "Task B",
                                "status": "in_progress",
                                "activeForm": "Doing B",
                            },
                        ]
                    },
                },
            ],
            usage={
                "input_tokens": 20_000,
                "output_tokens": 8_000,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        ),
    ]

    # Session 2: Task spawn + forward-compat extra field on a todo.
    file2 = proj / f"{SESSION_IDS[1]}.jsonl"
    records2 = [
        _msg(
            "a-3",
            SESSION_IDS[1],
            "2026-04-02T11:00:00.000Z",
            role="assistant",
            model="claude-sonnet-4-6",
            content=[
                {
                    "type": "tool_use",
                    "id": "tu-3",
                    "name": "Task",
                    "input": {
                        "subagent_type": "Explore",
                        "description": "Scan code",
                        "prompt": "Find all TODOs in the repo",
                    },
                },
                {
                    "type": "tool_use",
                    "id": "tu-4",
                    "name": "TodoWrite",
                    "input": {
                        "todos": [
                            {
                                "content": "Explore",
                                "status": "pending",
                                "activeForm": "Exploring",
                                "priority": "high",  # forward-compat extra field
                            },
                        ]
                    },
                },
            ],
        ),
    ]

    # Session 3: short haiku session, one todo.
    file3 = proj / f"{SESSION_IDS[2]}.jsonl"
    records3 = [
        _msg(
            "a-4",
            SESSION_IDS[2],
            "2026-04-03T12:00:00.000Z",
            role="assistant",
            model="claude-haiku-4-5",
            content=[
                {
                    "type": "tool_use",
                    "id": "tu-5",
                    "name": "TodoWrite",
                    "input": {
                        "todos": [
                            {
                                "content": "Solo task",
                                "status": "completed",
                                "activeForm": "Doing solo",
                            },
                        ]
                    },
                },
            ],
        ),
    ]

    for path, records in [(file1, records1), (file2, records2), (file3, records3)]:
        with path.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    # Subagent pairs under session 1's dir -- parent_session_id regex in
    # sql_views.py requires a full UUID directory name.
    sub_dir = proj / SESSION_IDS[0] / "subagents"
    sub_dir.mkdir(parents=True)
    for hex_id, desc in [
        ("aaaa1111aaaa1111", "Test agent 1"),
        ("bbbb2222bbbb2222", "Test agent 2"),
    ]:
        (sub_dir / f"agent-{hex_id}.meta.json").write_text(
            json.dumps({"agentType": "general-purpose", "description": desc})
        )
        jsonl = sub_dir / f"agent-{hex_id}.jsonl"
        with jsonl.open("w") as fh:
            fh.write(
                json.dumps(
                    _msg(
                        f"sub-u-{hex_id}",
                        f"sub-{hex_id}",
                        "2026-04-01T10:00:30.000Z",
                        role="user",
                        type_="user",
                        model=None,
                        content=[{"type": "text", "text": "go"}],
                    )
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    _msg(
                        f"sub-a-{hex_id}",
                        f"sub-{hex_id}",
                        "2026-04-01T10:00:35.000Z",
                        role="assistant",
                        model="claude-sonnet-4-6",
                        content=[{"type": "text", "text": "ok"}],
                    )
                )
                + "\n"
            )

    return proj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _connect(fixtures_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    glob = str(fixtures_dir / "*.jsonl")
    subagent_glob = str(fixtures_dir / "*/subagents/agent-*.jsonl")
    subagent_meta_glob = str(fixtures_dir / "*/subagents/agent-*.meta.json")
    register_raw(
        con,
        glob=glob,
        subagent_glob=subagent_glob,
        subagent_meta_glob=subagent_meta_glob,
    )
    register_views(con)
    # semantic_search macro references message_embeddings; register_vss with a
    # nonexistent parquet creates the empty table so macro creation resolves.
    register_vss(con, embeddings_parquet=fixtures_dir / "__no_embeddings__.parquet")
    register_macros(con)
    return con


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sessions_count(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    n = con.execute("SELECT count(*) FROM sessions").fetchone()[0]
    assert n == 3


def test_todo_events_count(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    # Hand count:
    #   Session 1: 2 TodoWrite calls, 2 todos each  = 4
    #   Session 2: 1 TodoWrite call, 1 todo         = 1
    #   Session 3: 1 TodoWrite call, 1 todo         = 1
    # Total = 6
    n = con.execute("SELECT count(*) FROM todo_events").fetchone()[0]
    assert n == 6


def test_todo_state_current_last_wins(fixtures_dir: Path) -> None:
    """Session 1's Task A: pending -> completed. Current state is completed."""
    con = _connect(fixtures_dir)
    row = con.execute(
        "SELECT status FROM todo_state_current WHERE session_id = ? AND subject = 'Task A'",
        [SESSION_IDS[0]],
    ).fetchone()
    assert row is not None
    assert row[0] == "completed"


def test_todo_events_forward_compat(fixtures_dir: Path) -> None:
    """Session 2 has a todo with an extra 'priority' field -- must not break todo_events."""
    con = _connect(fixtures_dir)
    rows = con.execute(
        "SELECT subject, status FROM todo_events WHERE session_id = ?",
        [SESSION_IDS[1]],
    ).fetchall()
    assert rows == [("Explore", "pending")]


def test_task_spawns_detected(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    rows = con.execute(
        "SELECT spawn_tool, subagent_type, description FROM task_spawns WHERE session_id = ?",
        [SESSION_IDS[1]],
    ).fetchall()
    assert rows == [("Task", "Explore", "Scan code")]


def test_subagent_sessions_count(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    n = con.execute(
        "SELECT count(*) FROM subagent_sessions WHERE parent_session_id = ?",
        [SESSION_IDS[0]],
    ).fetchone()[0]
    assert n == 2


def test_subagent_sessions_meta_join(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    rows = con.execute(
        "SELECT agent_type, description FROM subagent_sessions "
        "WHERE parent_session_id = ? ORDER BY agent_hex",
        [SESSION_IDS[0]],
    ).fetchall()
    assert rows == [
        ("general-purpose", "Test agent 1"),
        ("general-purpose", "Test agent 2"),
    ]


def test_macros_cost_estimate(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    cost = con.execute("SELECT cost_estimate(?)", [SESSION_IDS[0]]).fetchone()[0]
    # Session 1: opus @ (15, 75) per 1M tokens
    #   call 1: 10k in + 5k out  = 10_000 * 15 / 1e6 + 5_000 * 75 / 1e6 = 0.15 + 0.375 = 0.525
    #   call 2: 20k in + 8k out  = 20_000 * 15 / 1e6 + 8_000 * 75 / 1e6 = 0.30 + 0.600 = 0.900
    # Total ~ 1.425
    assert cost is not None
    assert 1.0 < cost < 2.0


def test_macros_todo_velocity(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    v = con.execute("SELECT todo_velocity(?)", [SESSION_IDS[0]]).fetchone()[0]
    # Session 1 current state: Task A=completed, Task B=in_progress -> 1/2 = 0.5
    assert v is not None
    assert abs(v - 0.5) < 1e-6


def test_macros_subagent_fanout(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    n = con.execute("SELECT subagent_fanout(?)", [SESSION_IDS[0]]).fetchone()[0]
    assert n == 2


def test_model_used_macro(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    m = con.execute("SELECT model_used(?)", [SESSION_IDS[0]]).fetchone()[0]
    assert m == "claude-opus-4-7"


def test_explain_has_pushdown_markers(fixtures_dir: Path) -> None:
    """EXPLAIN on a filtered messages query should surface READ_JSON/Filter markers."""
    con = _connect(fixtures_dir)
    rows = con.execute(
        f"EXPLAIN ANALYZE SELECT * FROM messages WHERE session_id = '{SESSION_IDS[0]}'"
    ).fetchall()
    plan = "\n".join(r[-1] for r in rows)
    # At least one marker should appear (DuckDB's exact wording varies across versions)
    markers = ("READ_JSON", "read_json", "Filter", "FILTER")
    assert any(m in plan for m in markers), plan


def test_describe_all_covers_every_view(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    views = describe_all(con)
    # v1 business views must always be present with at least one column.
    v1_views = {
        "sessions",
        "messages",
        "content_blocks",
        "messages_text",
        "tool_calls",
        "tool_results",
        "todo_events",
        "todo_state_current",
        "task_spawns",
        "subagent_sessions",
        "subagent_messages",
    }
    assert v1_views <= set(views.keys())
    for name in v1_views:
        assert views[name], f"v1 view {name} reported empty columns"
    # v2 analytics views appear in VIEW_NAMES but only materialize when their
    # parquets exist; this fixture uses a bare connection so describe_all
    # returns [] for them, which is expected.


def test_list_macros_includes_all(fixtures_dir: Path) -> None:
    con = _connect(fixtures_dir)
    macros = set(list_macros(con))
    expected = {
        "model_used",
        "cost_estimate",
        "tool_rank",
        "todo_velocity",
        "subagent_fanout",
        "semantic_search",
    }
    assert expected <= macros


def test_register_vss_without_parquet_creates_empty_table(tmp_path: Path) -> None:
    """register_vss with a nonexistent parquet should create an empty
    message_embeddings table and return False."""
    con = duckdb.connect(":memory:")
    ok = register_vss(con, embeddings_parquet=tmp_path / "nope.parquet")
    assert ok is False
    n = con.execute("SELECT count(*) FROM message_embeddings").fetchone()[0]
    assert n == 0
