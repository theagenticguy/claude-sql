"""Tests for ``skill_invocations``, ``skills_catalog``, and ``skill_usage``.

Covers the two invocation shapes (the ``Skill`` tool call and the user
``<command-name>/foo</command-name>`` slash-command text), catalog join
behavior, and the ``skill_rank`` / ``skill_source_mix`` / ``unused_skills``
macros.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from claude_sql.sql_views import (
    register_analytics,
    register_macros,
    register_raw,
    register_views,
    register_vss,
)

SESSION_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
SESSION_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _msg(
    uuid: str,
    session_id: str,
    ts: str,
    *,
    role: str,
    model: str | None,
    content: list[dict],
    type_: str = "assistant",
    parent: str | None = None,
) -> dict:
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
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


@pytest.fixture
def skills_fixture(tmp_path: Path) -> Path:
    """Build a project dir with two sessions exercising both invocation shapes.

    * Session A: assistant invokes ``Skill`` with ``skill='erpaval'``.
    * Session B: user types ``/personal-plugins:erpaval`` (slash command),
      plus a second user turn with ``/clear`` (built-in).
    """
    proj = tmp_path / "projects" / "-home-alice-workplace-proj"
    proj.mkdir(parents=True)

    file_a = proj / f"{SESSION_A}.jsonl"
    records_a = [
        _msg(
            "u-a-1",
            SESSION_A,
            "2026-04-01T10:00:00.000Z",
            role="user",
            model=None,
            type_="user",
            content=[{"type": "text", "text": "Do the thing."}],
        ),
        _msg(
            "a-a-1",
            SESSION_A,
            "2026-04-01T10:00:05.000Z",
            role="assistant",
            model="claude-opus-4-7",
            content=[
                {
                    "type": "tool_use",
                    "id": "tu-a-1",
                    "name": "Skill",
                    "input": {"skill": "erpaval", "args": "Land PR #42"},
                }
            ],
        ),
    ]
    file_a.write_text("\n".join(json.dumps(r) for r in records_a) + "\n")

    file_b = proj / f"{SESSION_B}.jsonl"
    slash_text = (
        "<command-name>/personal-plugins:erpaval</command-name>\n"
        "<command-message>personal-plugins:erpaval</command-message>\n"
        "<command-args>do X</command-args>"
    )
    # ``/clear`` comes through as a *bare string* in the older user-turn
    # shape, not a list of blocks -- exercise the VARCHAR-content branch
    # of ``skill_invocations`` explicitly.
    clear_text = "<command-name>/clear</command-name>\n<command-args></command-args>"
    u_b_2 = _msg(
        "u-b-2",
        SESSION_B,
        "2026-04-02T09:05:00.000Z",
        role="user",
        model=None,
        type_="user",
        content=[{"type": "text", "text": "placeholder"}],
    )
    # Overwrite ``message.content`` with the raw string Claude Code emits
    # for slash-command user turns.
    u_b_2["message"]["content"] = clear_text
    records_b = [
        _msg(
            "u-b-1",
            SESSION_B,
            "2026-04-02T09:00:00.000Z",
            role="user",
            model=None,
            type_="user",
            content=[{"type": "text", "text": slash_text}],
        ),
        _msg(
            "a-b-1",
            SESSION_B,
            "2026-04-02T09:00:05.000Z",
            role="assistant",
            model="claude-sonnet-4-6",
            content=[{"type": "text", "text": "ok"}],
        ),
        u_b_2,
    ]
    file_b.write_text("\n".join(json.dumps(r) for r in records_b) + "\n")

    # DuckDB's ``read_json`` errors on a glob with zero matches, so seed a
    # trivial subagent pair so the subagent globs resolve cleanly.  The
    # subagent content itself doesn't matter for these tests.
    sub_dir = proj / SESSION_A / "subagents"
    sub_dir.mkdir(parents=True)
    hex_id = "aaaa1111aaaa1111"
    (sub_dir / f"agent-{hex_id}.meta.json").write_text(
        json.dumps({"agentType": "general-purpose", "description": "seed"})
    )
    (sub_dir / f"agent-{hex_id}.jsonl").write_text(
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
    return proj


def _connect(proj: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    glob = str(proj / "*.jsonl")
    subagent_glob = str(proj / "*/subagents/agent-*.jsonl")
    subagent_meta_glob = str(proj / "*/subagents/agent-*.meta.json")
    register_raw(
        con,
        glob=glob,
        subagent_glob=subagent_glob,
        subagent_meta_glob=subagent_meta_glob,
    )
    register_views(con)
    register_vss(con, embeddings_parquet=proj / "__no_embeddings__.parquet")
    # Register with no analytics parquets -- skill_usage still works.
    register_analytics(
        con,
        classifications_parquet=proj / "__none__.parquet",
        trajectory_parquet=proj / "__none__.parquet",
        conflicts_parquet=proj / "__none__.parquet",
        clusters_parquet=proj / "__none__.parquet",
        cluster_terms_parquet=proj / "__none__.parquet",
        communities_parquet=proj / "__none__.parquet",
        user_friction_parquet=proj / "__none__.parquet",
        skills_catalog_parquet=proj / "__none__.parquet",
    )
    register_macros(con)
    return con


def _catalog_parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a minimal skills_catalog.parquet with the fields the view needs."""
    if not rows:
        # Empty frame needs an explicit schema so read_parquet knows the columns.
        df = pl.DataFrame(
            schema={
                "skill_id": pl.Utf8,
                "name": pl.Utf8,
                "plugin": pl.Utf8,
                "plugin_version": pl.Utf8,
                "source_kind": pl.Utf8,
                "description": pl.Utf8,
                "argument_hint": pl.Utf8,
                "source_path": pl.Utf8,
                "synced_at": pl.Datetime(time_unit="us", time_zone="UTC"),
            }
        )
    else:
        df = pl.DataFrame(rows)
    df.write_parquet(path)
    return path


def test_skill_invocations_tool(skills_fixture: Path) -> None:
    """Session A's ``Skill`` tool call becomes one row with source='tool'."""
    con = _connect(skills_fixture)
    rows = con.execute(
        "SELECT source, skill_id, args FROM skill_invocations WHERE source = 'tool' ORDER BY ts"
    ).fetchall()
    assert rows == [("tool", "erpaval", "Land PR #42")]


def test_skill_invocations_slash(skills_fixture: Path) -> None:
    """Session B's /personal-plugins:erpaval becomes a slash_command row."""
    con = _connect(skills_fixture)
    rows = con.execute(
        "SELECT source, skill_id, args FROM skill_invocations "
        "WHERE source = 'slash_command' ORDER BY ts"
    ).fetchall()
    assert rows == [
        ("slash_command", "personal-plugins:erpaval", "do X"),
        ("slash_command", "clear", None),
    ]


def test_skill_invocations_combined_count(skills_fixture: Path) -> None:
    con = _connect(skills_fixture)
    n = con.execute("SELECT count(*) FROM skill_invocations").fetchone()[0]
    assert n == 3


def test_skill_usage_without_catalog(skills_fixture: Path) -> None:
    """Missing catalog → skill_usage still serves with is_builtin=false everywhere."""
    con = _connect(skills_fixture)
    rows = con.execute(
        "SELECT skill_id, skill_name, is_builtin FROM skill_usage ORDER BY skill_id"
    ).fetchall()
    # 3 invocations: erpaval (tool), personal-plugins:erpaval (slash), clear (slash).
    assert len(rows) == 3
    assert all(not r[2] for r in rows)
    # skill_name falls back to skill_id when the catalog is missing.
    assert any(r[0] == "clear" and r[1] == "clear" for r in rows)


def test_skill_usage_with_catalog(skills_fixture: Path) -> None:
    """Catalog rows enrich skill_usage with plugin + is_builtin tags."""
    proj = skills_fixture
    catalog_path = proj / "skills_catalog.parquet"
    now = datetime(2026, 4, 23, tzinfo=UTC)
    rows = [
        {
            "skill_id": "erpaval",
            "name": "erpaval",
            "plugin": "personal-plugins",
            "plugin_version": "1.29.0",
            "source_kind": "plugin-skill",
            "description": "Evaluate, report, plan, act, validate.",
            "argument_hint": None,
            "source_path": "/fake/SKILL.md",
            "synced_at": now,
        },
        {
            "skill_id": "personal-plugins:erpaval",
            "name": "erpaval",
            "plugin": "personal-plugins",
            "plugin_version": "1.29.0",
            "source_kind": "plugin-skill",
            "description": "Evaluate, report, plan, act, validate.",
            "argument_hint": None,
            "source_path": "/fake/SKILL.md",
            "synced_at": now,
        },
        {
            "skill_id": "clear",
            "name": "clear",
            "plugin": None,
            "plugin_version": None,
            "source_kind": "builtin",
            "description": "Clear the current conversation context.",
            "argument_hint": None,
            "source_path": None,
            "synced_at": now,
        },
    ]
    _catalog_parquet(catalog_path, rows)

    con = duckdb.connect(":memory:")
    glob = str(proj / "*.jsonl")
    subagent_glob = str(proj / "*/subagents/agent-*.jsonl")
    subagent_meta_glob = str(proj / "*/subagents/agent-*.meta.json")
    register_raw(con, glob=glob, subagent_glob=subagent_glob, subagent_meta_glob=subagent_meta_glob)
    register_views(con)
    register_vss(con, embeddings_parquet=proj / "__none__.parquet")
    register_analytics(con, skills_catalog_parquet=catalog_path)
    register_macros(con)

    enriched = con.execute(
        "SELECT skill_id, skill_name, plugin, is_builtin FROM skill_usage ORDER BY skill_id"
    ).fetchall()
    assert ("clear", "clear", None, True) in enriched
    assert ("erpaval", "erpaval", "personal-plugins", False) in enriched
    assert ("personal-plugins:erpaval", "erpaval", "personal-plugins", False) in enriched


def test_skill_rank_macro(skills_fixture: Path) -> None:
    """skill_rank() returns rows for both invocation shapes."""
    con = _connect(skills_fixture)
    # 100 day window comfortably covers the April 2026 fixture timestamps.
    rows = con.execute("SELECT skill_id, n FROM skill_rank(10000) ORDER BY skill_id").fetchall()
    assert set(rows) == {("clear", 1), ("erpaval", 1), ("personal-plugins:erpaval", 1)}


def test_unused_skills_with_catalog(skills_fixture: Path) -> None:
    """Catalog entries with zero invocations surface in unused_skills()."""
    proj = skills_fixture
    catalog_path = proj / "skills_catalog.parquet"
    now = datetime(2026, 4, 23, tzinfo=UTC)
    _catalog_parquet(
        catalog_path,
        [
            # An unused plugin skill.
            {
                "skill_id": "browser-automation",
                "name": "browser-automation",
                "plugin": "personal-plugins",
                "plugin_version": "1.29.0",
                "source_kind": "plugin-skill",
                "description": "Drive a headless browser.",
                "argument_hint": None,
                "source_path": "/fake/SKILL.md",
                "synced_at": now,
            },
            # A used one — should NOT appear.
            {
                "skill_id": "erpaval",
                "name": "erpaval",
                "plugin": "personal-plugins",
                "plugin_version": "1.29.0",
                "source_kind": "plugin-skill",
                "description": "Erpaval loop.",
                "argument_hint": None,
                "source_path": "/fake/SKILL.md",
                "synced_at": now,
            },
        ],
    )

    con = duckdb.connect(":memory:")
    glob = str(proj / "*.jsonl")
    subagent_glob = str(proj / "*/subagents/agent-*.jsonl")
    subagent_meta_glob = str(proj / "*/subagents/agent-*.meta.json")
    register_raw(con, glob=glob, subagent_glob=subagent_glob, subagent_meta_glob=subagent_meta_glob)
    register_views(con)
    register_vss(con, embeddings_parquet=proj / "__none__.parquet")
    register_analytics(con, skills_catalog_parquet=catalog_path)
    register_macros(con)

    rows = con.execute("SELECT skill_id FROM unused_skills(10000) ORDER BY skill_id").fetchall()
    assert rows == [("browser-automation",)]
