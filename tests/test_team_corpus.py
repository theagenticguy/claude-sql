"""Tests for the team-corpus v0 surface (Settings.team_corpus_root).

Covers:
* The ``team_corpus_root`` field on :class:`claude_sql.config.Settings`
  rewriting the three transcript globs to ``<root>/<author>/projects/*``
  via the ``_derive_team_corpus_globs`` model_validator.
* User-pin precedence — an explicit per-glob override always wins over
  the team-corpus derivation.
* End-to-end behavior: a tmp_path 2-user fixture (alice has 2 sessions,
  bob has 1) registers via :func:`claude_sql.sql_views.register_all` and
  surfaces 3 distinct sessions through the standard ``sessions`` view.
* The ``CLAUDE_SQL_TEAM_CORPUS_ROOT`` env var path exercised via
  ``monkeypatch.setenv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from claude_sql.core.config import (
    Settings,
    _default_glob,
    _default_subagent_glob,
    _default_subagent_meta_glob,
)
from claude_sql.core.sql_views import register_all

# ---------------------------------------------------------------------------
# Fixture builder — duplicated from tests/test_sql_views.py to keep test
# files independent (house style: in-test materialization, no cross-test
# helper imports).
# ---------------------------------------------------------------------------


def _msg(
    uuid: str,
    session_id: str,
    ts: str,
    *,
    role: str,
    model: str | None,
    content: list[dict[str, object]],
    type_: str = "assistant",
) -> dict[str, object]:
    """Build one transcript record in the Claude Code JSONL shape."""
    return {
        "parentUuid": None,
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
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def _write_session(path: Path, session_id: str, ts: str, *, model: str) -> None:
    """Write a 3-message JSONL (user → assistant w/ tool_use → tool_result)."""
    records = [
        _msg(
            f"u-{session_id}",
            session_id,
            ts,
            role="user",
            model=None,
            type_="user",
            content=[{"type": "text", "text": "Please do X."}],
        ),
        _msg(
            f"a-{session_id}",
            session_id,
            ts,
            role="assistant",
            model=model,
            content=[
                {"type": "text", "text": "Working on it."},
                {
                    "type": "tool_use",
                    "id": f"tu-{session_id}",
                    "name": "Bash",
                    "input": {"command": "ls"},
                },
            ],
        ),
        _msg(
            f"r-{session_id}",
            session_id,
            ts,
            role="user",
            model=None,
            type_="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": f"tu-{session_id}",
                    "content": "ok",
                }
            ],
        ),
    ]
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _build_team_corpus(root: Path) -> Path:
    """Materialize a 2-user team corpus under ``<root>/team_corpus``.

    Layout::

        <root>/team_corpus/alice/projects/-home-alice-workplace-proj/sess-a1.jsonl
        <root>/team_corpus/alice/projects/-home-alice-workplace-proj/sess-a2.jsonl
        <root>/team_corpus/bob/projects/-home-bob-workplace-proj/sess-b1.jsonl
        <root>/team_corpus/alice/projects/-home-alice-workplace-proj/subagents/
            agent-aaaa1111aaaa1111.jsonl
            agent-aaaa1111aaaa1111.meta.json

    The subagent stub keeps the ``register_raw`` ``read_json`` glob from
    raising ``IOException`` on a no-match (DuckDB's behavior on an empty
    glob); it is otherwise inert because ``parent_session_id`` is extracted
    via a UUID-shaped directory regex that this layout intentionally does
    not satisfy.

    Returns the team-corpus root directory.
    """
    corpus = root / "team_corpus"
    alice = corpus / "alice" / "projects" / "-home-alice-workplace-proj"
    bob = corpus / "bob" / "projects" / "-home-bob-workplace-proj"
    alice.mkdir(parents=True)
    bob.mkdir(parents=True)
    _write_session(
        alice / "sess-a1.jsonl",
        "11111111-1111-1111-1111-111111111111",
        "2026-04-01T10:00:00.000Z",
        model="claude-sonnet-4-6",
    )
    _write_session(
        alice / "sess-a2.jsonl",
        "22222222-2222-2222-2222-222222222222",
        "2026-04-02T11:00:00.000Z",
        model="claude-opus-4-7",
    )
    _write_session(
        bob / "sess-b1.jsonl",
        "33333333-3333-3333-3333-333333333333",
        "2026-04-03T12:00:00.000Z",
        model="claude-sonnet-4-6",
    )
    sub_dir = alice / "subagents"
    sub_dir.mkdir()
    sub_meta = sub_dir / "agent-aaaa1111aaaa1111.meta.json"
    sub_meta.write_text(json.dumps({"agentType": "general-purpose", "description": "stub"}))
    sub_jsonl = sub_dir / "agent-aaaa1111aaaa1111.jsonl"
    with sub_jsonl.open("w") as fh:
        fh.write(
            json.dumps(
                _msg(
                    "sub-u-aaaa1111aaaa1111",
                    "sub-aaaa1111aaaa1111",
                    "2026-04-01T10:00:30.000Z",
                    role="user",
                    type_="user",
                    model=None,
                    content=[{"type": "text", "text": "go"}],
                )
            )
            + "\n"
        )
    return corpus


def _team_settings(corpus_root: Path, tmp_path: Path) -> Settings:
    """Build a :class:`Settings` rooted at ``corpus_root`` with all parquet
    caches redirected under ``tmp_path`` so the test never touches the
    user's real ``~/.claude/`` paths.
    """
    cache = tmp_path / "claude_cache"
    cache.mkdir(exist_ok=True)
    return Settings(
        team_corpus_root=corpus_root,
        embeddings_parquet_path=cache / "embeddings",
        classifications_parquet_path=cache / "session_classifications",
        trajectory_parquet_path=cache / "message_trajectory",
        conflicts_parquet_path=cache / "session_conflicts",
        clusters_parquet_path=cache / "clusters.parquet",
        cluster_terms_parquet_path=cache / "cluster_terms.parquet",
        communities_parquet_path=cache / "session_communities.parquet",
        user_friction_parquet_path=cache / "user_friction",
        skills_catalog_parquet_path=cache / "skills_catalog.parquet",
        user_skills_dir=cache / "skills",
        plugins_cache_dir=cache / "plugins",
        checkpoint_db_path=cache / "claude_sql.duckdb",
        duckdb_temp_dir=cache / "duckdb_tmp",
    )


# ---------------------------------------------------------------------------
# Settings-shape tests
# ---------------------------------------------------------------------------


def test_team_corpus_root_propagates_to_globs(tmp_path: Path) -> None:
    """Setting ``team_corpus_root`` rewrites all three transcript globs."""
    settings = Settings(team_corpus_root=tmp_path)
    resolved = tmp_path.expanduser().resolve()
    assert settings.default_glob == f"{resolved}/*/projects/*/*.jsonl"
    assert settings.subagent_glob == f"{resolved}/*/projects/*/subagents/agent-*.jsonl"
    assert settings.subagent_meta_glob == f"{resolved}/*/projects/*/subagents/agent-*.meta.json"


def test_team_corpus_root_unset_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``team_corpus_root`` unset, globs match the personal-corpus defaults."""
    monkeypatch.delenv("CLAUDE_SQL_TEAM_CORPUS_ROOT", raising=False)
    settings = Settings()
    assert settings.team_corpus_root is None
    assert settings.default_glob == _default_glob()
    assert settings.subagent_glob == _default_subagent_glob()
    assert settings.subagent_meta_glob == _default_subagent_meta_glob()


def test_team_corpus_user_pinned_glob_wins(tmp_path: Path) -> None:
    """An explicit per-glob override beats the team-corpus derivation."""
    custom = "/some/custom/path/*.jsonl"
    settings = Settings(team_corpus_root=tmp_path, default_glob=custom)
    assert settings.default_glob == custom
    # The other two also stay at defaults (we don't partially rewrite).
    assert settings.subagent_glob == _default_subagent_glob()
    assert settings.subagent_meta_glob == _default_subagent_meta_glob()


def test_team_corpus_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``CLAUDE_SQL_TEAM_CORPUS_ROOT`` env var routes through the same validator."""
    monkeypatch.setenv("CLAUDE_SQL_TEAM_CORPUS_ROOT", str(tmp_path))
    settings = Settings()
    resolved = tmp_path.expanduser().resolve()
    assert settings.team_corpus_root == tmp_path
    assert settings.default_glob == f"{resolved}/*/projects/*/*.jsonl"
    assert settings.subagent_glob == f"{resolved}/*/projects/*/subagents/agent-*.jsonl"
    assert settings.subagent_meta_glob == f"{resolved}/*/projects/*/subagents/agent-*.meta.json"


# ---------------------------------------------------------------------------
# End-to-end view registration over the synthetic team corpus
# ---------------------------------------------------------------------------


def test_views_resolve_across_authors(tmp_path: Path) -> None:
    """``register_all`` over a 2-user fixture surfaces all 3 sessions."""
    corpus = _build_team_corpus(tmp_path)
    settings = _team_settings(corpus, tmp_path)
    con = duckdb.connect(":memory:")
    register_all(con, settings=settings, include_analytics=False)
    n = con.execute("SELECT count(DISTINCT session_id) FROM sessions").fetchone()
    assert n is not None
    assert n[0] == 3


def test_team_corpus_macro_smoke(tmp_path: Path) -> None:
    """A v1 macro (``tool_rank``) returns rows over the team-corpus fixture.

    Uses ``last_n_days=365`` because the synthetic timestamps are a month
    or two in the past relative to the test clock; a 7-day window would
    legitimately return nothing and obscure the macro-binding signal we
    want here.
    """
    corpus = _build_team_corpus(tmp_path)
    settings = _team_settings(corpus, tmp_path)
    con = duckdb.connect(":memory:")
    register_all(con, settings=settings, include_analytics=False)
    rows = con.execute("SELECT * FROM tool_rank(365)").fetchall()
    # Each session has exactly one Bash tool_use → 3 calls total under one
    # tool name. The macro groups + orders by count desc.
    assert rows
    assert rows[0][0] == "Bash"
    assert rows[0][1] == 3
