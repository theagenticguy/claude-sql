"""Tests for the SQL stamp layer in :mod:`claude_sql.friction_worker`.

Three deterministic shapes the LLM doesn't need to see (RFC §4.3, §9.4):

* Rule 1: repeated user-message body within 10 turns → ``unmet_expectation``
  at 0.85 confidence.
* Rule 2: ≤30 chars + first-token ∈ {stop, redo, revert, rollback, undo,
  restart} → ``correction`` at 0.9 confidence.
* Rule 3: trailing ``?`` after an ``is_error=true`` tool_result →
  ``confusion`` at 0.85 confidence.

Source priority is regex > sql > llm — verified in
:func:`test_regex_takes_precedence_over_sql`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pytest

from claude_sql.analytics import friction_worker
from claude_sql.core.config import Settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from conftest import FakeBedrockClient


# ---------------------------------------------------------------------------
# Fixture builders — minimal messages_text + sessions tables, no JSONL chain.
# ---------------------------------------------------------------------------


def _seed_messages_text(con: duckdb.DuckDBPyConnection) -> None:
    """Build an empty messages_text + sessions pair the worker can read."""
    con.execute(
        """
        CREATE TABLE messages_text (
            uuid VARCHAR,
            session_id VARCHAR,
            ts TIMESTAMP,
            role VARCHAR,
            text_content VARCHAR
        );
        CREATE TABLE sessions (
            session_id VARCHAR,
            transcript_path VARCHAR
        );
        """
    )


def _insert_user_msg(
    con: duckdb.DuckDBPyConnection,
    *,
    uuid: str,
    session_id: str,
    ts: str,
    text: str,
    role: str = "user",
) -> None:
    con.execute(
        "INSERT INTO messages_text VALUES (?, ?, ?, ?, ?)",
        [uuid, session_id, ts, role, text],
    )


def _seed_session(con: duckdb.DuckDBPyConnection, session_id: str) -> None:
    # Path is non-existent on purpose; session_bounds tolerates an unreadable
    # transcript_path (returns mtime=None). Avoids ``/tmp`` to dodge S108.
    con.execute(
        "INSERT INTO sessions VALUES (?, ?)",
        [session_id, f"/nonexistent/{session_id}.jsonl"],
    )


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        user_friction_parquet_path=tmp_path / "user_friction",
        checkpoint_db_path=tmp_path / "checkpoint.duckdb",
        friction_max_chars=300,
        llm_concurrency=1,
        batch_size=4,
    )


# ---------------------------------------------------------------------------
# Rule 1 — repeated user message within 10 turns
# ---------------------------------------------------------------------------


def test_repeated_message_within_10_turns_marked_unmet_expectation() -> None:
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    base_ts = "2026-04-20T10:00:00Z"
    sid = "s-1"
    # Two identical user messages, 5 user-role turns apart (rn delta = 5).
    _insert_user_msg(con, uuid="u1", session_id=sid, ts=base_ts, text="run the tests please")
    for i in range(2, 6):
        _insert_user_msg(
            con, uuid=f"u{i}", session_id=sid, ts=f"2026-04-20T10:0{i}:00Z", text=f"filler {i}"
        )
    _insert_user_msg(
        con, uuid="u6", session_id=sid, ts="2026-04-20T10:06:00Z", text="run the tests please"
    )

    stamps = friction_worker.sql_stamp(con, ["u1", "u6"])
    assert "u6" in stamps
    label, conf, source = stamps["u6"]
    assert label == "unmet_expectation"
    assert conf == 0.85
    assert source == "sql"
    # The first occurrence has no prior match — must NOT be stamped.
    assert "u1" not in stamps


def test_repeated_message_outside_10_turns_falls_through_to_llm() -> None:
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    sid = "s-1"
    _insert_user_msg(con, uuid="u1", session_id=sid, ts="2026-04-20T10:00:00Z", text="run tests")
    # 12 user-role turns apart — rn delta = 12, outside the 10-turn window.
    for i in range(2, 13):
        _insert_user_msg(
            con, uuid=f"u{i}", session_id=sid, ts=f"2026-04-20T10:{i:02d}:00Z", text=f"step {i}"
        )
    _insert_user_msg(con, uuid="u13", session_id=sid, ts="2026-04-20T10:13:00Z", text="run tests")

    stamps = friction_worker.sql_stamp(con, ["u1", "u13"])
    assert "u13" not in stamps


def test_repeated_message_normalization_collapses_whitespace() -> None:
    """``RUN  THE\\nTESTS`` and ``run the tests`` must match — same normalized form."""
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    sid = "s-1"
    _insert_user_msg(
        con, uuid="u1", session_id=sid, ts="2026-04-20T10:00:00Z", text="run the tests"
    )
    _insert_user_msg(
        con, uuid="u2", session_id=sid, ts="2026-04-20T10:01:00Z", text="RUN  THE\nTESTS"
    )

    stamps = friction_worker.sql_stamp(con, ["u1", "u2"])
    assert stamps["u2"][0] == "unmet_expectation"


# ---------------------------------------------------------------------------
# Rule 2 — short imperative reverts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["undo", "stop", "redo", "revert", "rollback", "restart", "  undo  ", "Undo!"],
)
def test_short_imperative_marked_correction(text: str) -> None:
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    _insert_user_msg(con, uuid="u1", session_id="s-1", ts="2026-04-20T10:00:00Z", text=text)
    stamps = friction_worker.sql_stamp(con, ["u1"])
    assert "u1" in stamps
    label, conf, source = stamps["u1"]
    assert label == "correction"
    assert conf == 0.9
    assert source == "sql"


def test_long_imperative_falls_through() -> None:
    """``undo the last commit, please`` is >30 chars — must NOT be stamped."""
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    _insert_user_msg(
        con,
        uuid="u1",
        session_id="s-1",
        ts="2026-04-20T10:00:00Z",
        text="undo the last commit, please",  # 28 chars — actually short
    )
    _insert_user_msg(
        con,
        uuid="u2",
        session_id="s-1",
        ts="2026-04-20T10:01:00Z",
        text="undo the last three commits and rebase main",  # 44 chars — too long
    )
    stamps = friction_worker.sql_stamp(con, ["u1", "u2"])
    # u1 starts with 'undo' AND is ≤30 chars → stamped.
    assert stamps.get("u1", (None,))[0] == "correction"
    # u2 starts with 'undo' but is >30 chars → not stamped.
    assert "u2" not in stamps


def test_imperative_first_token_must_match() -> None:
    """``please undo`` does not start with an imperative — first token is ``please``."""
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    _insert_user_msg(
        con, uuid="u1", session_id="s-1", ts="2026-04-20T10:00:00Z", text="please undo"
    )
    stamps = friction_worker.sql_stamp(con, ["u1"])
    assert "u1" not in stamps


# ---------------------------------------------------------------------------
# Rule 3 — trailing '?' after error tool_result
# ---------------------------------------------------------------------------


def _seed_messages_with_error(
    con: duckdb.DuckDBPyConnection,
    *,
    error_ts: str,
    user_uuid: str,
    user_text: str,
    user_ts: str,
    session_id: str = "s-1",
) -> None:
    """Seed a ``messages`` view + ``messages_text`` row pair that exercises rule 3.

    The ``messages`` view is what rule 3 unnests for ``is_error``; it must
    expose a ``content_json`` column shaped like the production view.
    """
    error_block = json.dumps(
        [{"type": "tool_result", "tool_use_id": "tu-1", "content": "boom", "is_error": True}]
    )
    user_block = json.dumps([{"type": "text", "text": user_text}])
    con.execute(
        """
        CREATE TABLE messages (
            uuid VARCHAR,
            session_id VARCHAR,
            ts TIMESTAMP,
            role VARCHAR,
            content_json JSON
        );
        """
    )
    con.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
        ["err-1", session_id, error_ts, "user", error_block],
    )
    con.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
        [user_uuid, session_id, user_ts, "user", user_block],
    )
    _insert_user_msg(con, uuid=user_uuid, session_id=session_id, ts=user_ts, text=user_text)


def test_question_after_error_tool_result_marked_confusion() -> None:
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    _seed_messages_with_error(
        con,
        error_ts="2026-04-20T10:00:00Z",
        user_uuid="u1",
        user_text="what went wrong?",
        user_ts="2026-04-20T10:00:01Z",
    )
    stamps = friction_worker.sql_stamp(con, ["u1"])
    assert "u1" in stamps
    label, conf, source = stamps["u1"]
    assert label == "confusion"
    assert conf == 0.85
    assert source == "sql"


def test_question_without_preceding_error_falls_through() -> None:
    """A bare ``?`` user message with no preceding error must NOT be stamped."""
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    # Build the messages view but with NO error tool_result preceding the user.
    user_block = json.dumps([{"type": "text", "text": "what now?"}])
    con.execute(
        """
        CREATE TABLE messages (
            uuid VARCHAR,
            session_id VARCHAR,
            ts TIMESTAMP,
            role VARCHAR,
            content_json JSON
        );
        """
    )
    con.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
        ["u1", "s-1", "2026-04-20T10:00:00Z", "user", user_block],
    )
    _insert_user_msg(con, uuid="u1", session_id="s-1", ts="2026-04-20T10:00:00Z", text="what now?")
    stamps = friction_worker.sql_stamp(con, ["u1"])
    assert "u1" not in stamps


def test_question_with_event_between_error_and_user_falls_through() -> None:
    """An assistant turn between the error and the user '?' must break the chain."""
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    error_block = json.dumps(
        [{"type": "tool_result", "tool_use_id": "tu-1", "content": "boom", "is_error": True}]
    )
    user_block = json.dumps([{"type": "text", "text": "what now?"}])
    con.execute(
        """
        CREATE TABLE messages (
            uuid VARCHAR,
            session_id VARCHAR,
            ts TIMESTAMP,
            role VARCHAR,
            content_json JSON
        );
        """
    )
    con.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
        ["err-1", "s-1", "2026-04-20T10:00:00Z", "user", error_block],
    )
    # Intervening assistant message in messages_text — breaks the
    # "immediately preceding" chain.
    _insert_user_msg(
        con,
        uuid="a-1",
        session_id="s-1",
        ts="2026-04-20T10:00:01Z",
        text="working on it",
        role="assistant",
    )
    con.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
        ["u1", "s-1", "2026-04-20T10:00:02Z", "user", user_block],
    )
    _insert_user_msg(con, uuid="u1", session_id="s-1", ts="2026-04-20T10:00:02Z", text="what now?")
    stamps = friction_worker.sql_stamp(con, ["u1"])
    assert "u1" not in stamps


# ---------------------------------------------------------------------------
# Wire-up / priority tests
# ---------------------------------------------------------------------------


def test_sql_stamps_skip_llm_call(
    tmp_path: Path,
    fake_bedrock_client: Callable[..., FakeBedrockClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQL-stamped rows must NOT increment Bedrock invoke count.

    The fixture has one ``undo`` (rule 2 → correction) and one ordinary
    short message (would hit LLM if not for our zero-LLM mock setup).
    The test asserts the SQL-stamped row is written and Bedrock is
    never called for it.
    """
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    sid = "s-1"
    _seed_session(con, sid)
    _insert_user_msg(con, uuid="u1", session_id=sid, ts="2026-04-20T10:00:00Z", text="undo")
    _insert_user_msg(
        con, uuid="u2", session_id=sid, ts="2026-04-20T10:01:00Z", text="please review the PR"
    )

    fake = fake_bedrock_client(
        {"output": {"label": "none", "rationale": "ordinary", "confidence": 0.95}}
    )
    monkeypatch.setattr(friction_worker, "_build_bedrock_client", lambda _s: fake)

    settings = _build_settings(tmp_path)
    n = friction_worker.detect_user_friction(
        con, settings, since_days=None, limit=None, dry_run=False
    )
    assert n == 2  # one sql-stamp + one llm
    # Exactly ONE Bedrock invocation — the sql-stamped row was skipped.
    assert len(fake.captured) == 1
    # The body is for the LLM-pending row, not the imperative.
    sent_prompt = json.dumps(fake.captured[0]["body"])
    assert "review the PR" in sent_prompt
    assert "undo" not in sent_prompt or sent_prompt.count("undo") < 2


def test_regex_takes_precedence_over_sql(tmp_path: Path) -> None:
    """A message that satisfies BOTH regex (interruption: ``stop, please``)
    AND rule 2 (≤30 chars, starts with ``stop``) must be stamped by regex
    with ``source='regex'``, not ``source='sql'``."""
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    sid = "s-1"
    _seed_session(con, sid)
    # ``stop, please`` matches both:
    #   - regex interruption pattern (``stop`` + punctuation)
    #   - rule 2 imperative (≤30 chars, first-token ``stop`` after rtrim of punctuation)
    _insert_user_msg(con, uuid="u1", session_id=sid, ts="2026-04-20T10:00:00Z", text="stop, please")

    settings = _build_settings(tmp_path)
    # No Bedrock client needed — regex stamps it before LLM path.
    written = friction_worker.detect_user_friction(
        con, settings, since_days=None, limit=None, dry_run=False
    )
    assert written == 1

    # Read it back and verify ``source='regex'`` (not 'sql').
    import polars as pl

    from claude_sql.core.parquet_shards import read_all

    df = read_all(settings.user_friction_parquet_path)
    assert df is not None
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["source"] == "regex"
    assert row["label"] == "interruption"
    assert pl.DataFrame  # silence unused-import lint when read_all returns None


# ---------------------------------------------------------------------------
# sql_stamp graceful degradation when ``messages`` view is absent
# ---------------------------------------------------------------------------


def test_sql_stamp_skips_rule3_when_messages_view_missing() -> None:
    """Rule 3 requires ``messages`` view — when absent, sql_stamp returns
    rules 1/2 results and silently skips rule 3 (caught CatalogException)."""
    con = duckdb.connect(":memory:")
    _seed_messages_text(con)
    _insert_user_msg(con, uuid="u1", session_id="s-1", ts="2026-04-20T10:00:00Z", text="undo")
    # No `messages` table at all — rule 3 must catch CatalogException.
    stamps = friction_worker.sql_stamp(con, ["u1"])
    assert stamps["u1"][0] == "correction"
