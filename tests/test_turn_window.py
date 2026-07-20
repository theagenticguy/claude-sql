"""Tests for the v1.0 ``turn_window`` view and the ``messages_text``
``is_compact_summary`` projection / attachment filter.

The view is documented in RFC §3.2 / §9.1: an adjacent-turn window over
``messages_text`` per session, ordered by ``(ts, uuid)``, with compact-
summary rows excluded so the LAG() pointer never lands on a synthetic
checkpoint row.

Mock pattern follows ``tests/test_sql_views.py`` — write a small JSONL
fixture corpus under ``tmp_path``, register raw + derived views over it,
and query the view directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from claude_sql.infrastructure.duckdb_views import VIEW_SCHEMA, register_raw, register_views

# ---------------------------------------------------------------------------
# Fixture builder — local to this test module so we control every field
# (``isCompactSummary``, ``type='attachment'``) the spec exercises.
# ---------------------------------------------------------------------------


SID_SINGLE = "11111111-1111-1111-1111-111111111111"
SID_PAIR = "22222222-2222-2222-2222-222222222222"
SID_COMPACT = "33333333-3333-3333-3333-333333333333"
SID_ATTACH = "44444444-4444-4444-4444-444444444444"


def _record(
    *,
    uuid: str,
    session_id: str,
    ts: str,
    role: str = "user",
    type_: str | None = None,
    text: str = "hello there from a long enough message body",
    is_compact_summary: bool = False,
) -> dict[str, Any]:
    """Build one transcript record in the Claude Code v1 JSONL shape.

    ``isCompactSummary`` is only emitted when ``True`` so that the
    ``coalesce(isCompactSummary, false)`` projection is exercised on the
    common-case rows.  ``type`` defaults to ``role`` (so ``role='user'``
    yields ``type='user'``), but the caller can override to ``'attachment'``
    to exercise the messages_text attachment filter.
    """
    rec: dict[str, Any] = {
        "parentUuid": None,
        "isSidechain": False,
        "type": type_ or role,
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
            "role": role,
            "model": "claude-sonnet-4-6" if role == "assistant" else None,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }
    if is_compact_summary:
        rec["isCompactSummary"] = True
    return rec


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    """Four-session corpus tailored to the spec assertions.

    Sessions:

    * ``SID_SINGLE`` — one user message; first-row-per-session NULL prev.
    * ``SID_PAIR`` — user + assistant exactly 1 second apart; gap_ms=1000.
    * ``SID_COMPACT`` — user + compact-summary user + assistant; the
      compact-summary row must NOT appear as ``curr_uuid`` in turn_window.
    * ``SID_ATTACH`` — user + assistant with one ``type='attachment'`` row
      between them; the attachment row must be filtered from
      messages_text and therefore from turn_window.
    """
    proj = tmp_path / "projects" / "-home-u-proj"
    proj.mkdir(parents=True)

    files: dict[str, list[dict[str, Any]]] = {
        SID_SINGLE: [
            _record(
                uuid="single-1",
                session_id=SID_SINGLE,
                ts="2026-04-01T10:00:00.000Z",
                text="solitary user message that is plenty long to clear the 32-char floor",
            ),
        ],
        SID_PAIR: [
            _record(
                uuid="pair-1",
                session_id=SID_PAIR,
                ts="2026-04-02T10:00:00.000Z",
                role="user",
                text="opening user prompt long enough to clear the 32-char floor",
            ),
            _record(
                uuid="pair-2",
                session_id=SID_PAIR,
                ts="2026-04-02T10:00:01.000Z",  # exactly 1s after pair-1
                role="assistant",
                text="assistant reply that is comfortably above the 32-char filter",
            ),
        ],
        SID_COMPACT: [
            _record(
                uuid="compact-1",
                session_id=SID_COMPACT,
                ts="2026-04-03T09:00:00.000Z",
                role="user",
                text="real user request long enough to clear the 32-char floor",
            ),
            _record(
                uuid="compact-2",
                session_id=SID_COMPACT,
                ts="2026-04-03T09:00:30.000Z",
                role="user",
                text=(
                    "This session is being continued from a previous "
                    "conversation that ran out of context. Summary follows."
                ),
                is_compact_summary=True,
            ),
            _record(
                uuid="compact-3",
                session_id=SID_COMPACT,
                ts="2026-04-03T09:01:00.000Z",
                role="assistant",
                text="assistant resumes work after the synthetic compact-summary checkpoint",
            ),
        ],
        SID_ATTACH: [
            _record(
                uuid="attach-1",
                session_id=SID_ATTACH,
                ts="2026-04-04T08:00:00.000Z",
                role="user",
                text="user opening message long enough to clear the 32-char floor",
            ),
            _record(
                uuid="attach-2",
                session_id=SID_ATTACH,
                ts="2026-04-04T08:00:05.000Z",
                role="user",
                type_="attachment",
                text=(
                    "auto-injected attachment payload long enough to clear "
                    "the 32-char floor but not a real user turn"
                ),
            ),
            _record(
                uuid="attach-3",
                session_id=SID_ATTACH,
                ts="2026-04-04T08:00:10.000Z",
                role="assistant",
                text="assistant response that should follow the user message, not the attachment",
            ),
        ],
    }

    for sid, records in files.items():
        with (proj / f"{sid}.jsonl").open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    # Subagent globs must resolve to *something* even when empty — register_raw
    # would otherwise fail to bind v_raw_subagents.  An empty stub directory
    # plus a single placeholder file keeps schema inference happy.
    sub_dir = proj / SID_SINGLE / "subagents"
    sub_dir.mkdir(parents=True)
    (sub_dir / "agent-aaaa1111aaaa1111.meta.json").write_text(
        json.dumps({"agentType": "stub", "description": "stub"})
    )
    (sub_dir / "agent-aaaa1111aaaa1111.jsonl").write_text(
        json.dumps(
            _record(
                uuid="sub-1",
                session_id="sub-aaaa1111",
                ts="2026-04-01T11:00:00.000Z",
                role="user",
                text="subagent stub record so duckdb read_json infers a schema",
            )
        )
        + "\n"
    )

    return proj


def _connect(fixtures_dir: Path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with raw + derived views registered over the fixture corpus."""
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
    return con


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_messages_text_carries_is_compact_summary_column(fixtures_dir: Path) -> None:
    """``messages_text.is_compact_summary`` must exist and be BOOLEAN."""
    con = _connect(fixtures_dir)
    rows = con.execute("DESCRIBE messages_text").fetchall()
    cols = {str(r[0]): str(r[1]) for r in rows}
    assert "is_compact_summary" in cols, cols
    assert cols["is_compact_summary"] == "BOOLEAN", cols["is_compact_summary"]


def test_turn_window_excludes_compact_summary(fixtures_dir: Path) -> None:
    """A row with ``isCompactSummary:true`` must never appear as ``curr_uuid``."""
    con = _connect(fixtures_dir)
    rows = con.execute(
        "SELECT curr_uuid FROM turn_window WHERE session_id = ?",
        [SID_COMPACT],
    ).fetchall()
    curr_uuids = {r[0] for r in rows}
    assert "compact-2" not in curr_uuids, (
        f"compact-summary row leaked into turn_window: {curr_uuids}"
    )
    # The non-summary rows still surface.
    assert curr_uuids == {"compact-1", "compact-3"}


def test_turn_window_excludes_attachment_rows(fixtures_dir: Path) -> None:
    """``messages_text`` drops ``type='attachment'`` rows; turn_window inherits that."""
    con = _connect(fixtures_dir)

    # First, ``messages_text`` itself must not see the attachment row.
    text_uuids = {
        r[0]
        for r in con.execute(
            "SELECT uuid FROM messages_text WHERE session_id = ?",
            [SID_ATTACH],
        ).fetchall()
    }
    assert "attach-2" not in text_uuids, f"attachment row leaked into messages_text: {text_uuids}"

    # And therefore must not appear in turn_window.
    rows = con.execute(
        "SELECT curr_uuid FROM turn_window WHERE session_id = ?",
        [SID_ATTACH],
    ).fetchall()
    curr_uuids = {r[0] for r in rows}
    assert "attach-2" not in curr_uuids
    assert curr_uuids == {"attach-1", "attach-3"}


def test_turn_window_session_first_has_null_prev(fixtures_dir: Path) -> None:
    """First row per session has prev_uuid IS NULL and gap_ms IS NULL."""
    con = _connect(fixtures_dir)
    rows = con.execute(
        """
        SELECT curr_uuid, prev_uuid, gap_ms
        FROM turn_window
        WHERE session_id = ? AND window_idx = 1
        """,
        [SID_PAIR],
    ).fetchall()
    assert len(rows) == 1
    curr_uuid, prev_uuid, gap_ms = rows[0]
    assert curr_uuid == "pair-1"
    assert prev_uuid is None
    assert gap_ms is None


def test_turn_window_ordering_deterministic(fixtures_dir: Path) -> None:
    """Re-running the view yields byte-equal ``(prev_uuid, curr_uuid, window_idx)``."""
    con = _connect(fixtures_dir)
    sql = (
        "SELECT session_id, prev_uuid, curr_uuid, window_idx "
        "FROM turn_window ORDER BY session_id, window_idx"
    )
    first = con.execute(sql).fetchall()
    second = con.execute(sql).fetchall()
    assert first == second, (first, second)


def test_turn_window_gap_ms_is_bigint_milliseconds(fixtures_dir: Path) -> None:
    """For two rows exactly 1s apart, gap_ms = 1000."""
    con = _connect(fixtures_dir)
    row = con.execute(
        """
        SELECT gap_ms
        FROM turn_window
        WHERE session_id = ? AND curr_uuid = 'pair-2'
        """,
        [SID_PAIR],
    ).fetchone()
    assert row is not None
    gap_ms = row[0]
    assert gap_ms == 1000, gap_ms


def test_turn_window_view_schema_entry_present() -> None:
    """The hand-maintained VIEW_SCHEMA must list ``turn_window`` with the
    documented column shape from RFC §3.2.

    Catches the case where the view DDL ships without its schema entry and
    the agent-facing ``schema`` command would silently miss it.
    """
    cols = VIEW_SCHEMA.get("turn_window")
    assert cols is not None, "turn_window missing from VIEW_SCHEMA"
    names = tuple(name for name, _typ in cols)
    assert names == (
        "session_id",
        "prev_uuid",
        "prev_role",
        "prev_ts",
        "curr_uuid",
        "curr_role",
        "curr_ts",
        "gap_ms",
        "window_idx",
    )
