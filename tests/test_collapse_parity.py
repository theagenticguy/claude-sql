"""Drop-in parity proof: claude-sql's read seam == the consumer's collapse routine.

This ports the downstream consumer's own collapse test fixture verbatim
(``proj-a/sess-1/`` — two part files named out of lexical write order, plus a
``subagents/`` side transcript that must never be admitted) and asserts that
:meth:`DuckDbTranscriptReader.read_turn_text` produces the IDENTICAL string the
consumer's chronological-ordering test asserts. If this passes, the claude-sql
retrieval seam is a true drop-in for the downstream transcript reader contract.

The expected string below is copied verbatim from the consumer's own transcript
reader test suite — do not "fix" its spacing or timestamps; byte-parity with the
consumer is the whole point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claude_sql.domain.transcript import TranscriptRow, render_turn_text
from claude_sql.infrastructure.settings import Settings
from claude_sql.infrastructure.transcript_reader import DuckDbTranscriptReader
from conftest import _seed_subagent_stub

# --- The consumer's fixture, reproduced verbatim -----------------------------
# proj-a/sess-1 spans two part files; the later-epoch file was written first
# (lexically earlier name), so a reader that trusts read order would emit the
# turns out of order. The subagent sidecar must be excluded by the single-level
# part-*.jsonl glob.

_PART_LATER_EPOCH = "part-1700000002000-aaaaaa.jsonl"  # u2, u3, u4
_PART_EARLIER_EPOCH = "part-1700000001000-bbbbbb.jsonl"  # u1

_PART_LATER_ROWS = [
    {
        "uuid": "u2",
        "type": "assistant",
        "timestamp": "2023-11-14T22:13:22.000Z",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me list the directory."},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            ],
        },
    },
    {
        "uuid": "u3",
        "type": "user",
        "timestamp": "2023-11-14T22:13:23.000Z",
        "message": {"role": "user", "content": [{"type": "tool_result", "content": "file.txt"}]},
    },
    {
        "uuid": "u4",
        "type": "assistant",
        "timestamp": "2023-11-14T22:13:24.000Z",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "The directory contains file.txt."}],
        },
    },
]

_PART_EARLIER_ROWS = [
    {
        "uuid": "u1",
        "type": "user",
        "timestamp": "2023-11-14T22:13:21.000Z",
        "message": {"role": "user", "content": "What files are in this directory?"},
    },
]

_SUBAGENT_ROWS = [
    {
        "uuid": "sx",
        "type": "user",
        "timestamp": "2023-11-14T22:13:20.000Z",
        "message": {"role": "user", "content": "SUBAGENT_SIDE_TRANSCRIPT_MUST_NOT_APPEAR"},
    },
    {
        "uuid": "sy",
        "type": "assistant",
        "timestamp": "2023-11-14T22:13:20.500Z",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "SUBAGENT_REPLY_MUST_NOT_APPEAR"}],
        },
    },
]

# Copied verbatim from the consumer's transcript reader test for
# chronological ordering across part files.
_COLLAPSE_EXPECTED_LINES = [
    "[user 2023-11-14T22:13:21.000Z] What files are in this directory?",
    "[assistant 2023-11-14T22:13:22.000Z] Let me list the directory. [tool_use:Bash]",
    "[user 2023-11-14T22:13:23.000Z] [tool_result]",
    "[assistant 2023-11-14T22:13:24.000Z] The directory contains file.txt.",
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def _seed_collapse_fixture(tmp_path: Path) -> tuple[Settings, str]:
    """Materialize the consumer's fixture under a claude-sql part-dir layout.

    The consumer resolves ``{local_root}/{project_key}/{session_id}/part-*.jsonl``;
    claude-sql's part-dir layout is ``{root}/projects/*/{sid}/part-*.jsonl``. The
    only structural difference is the ``projects/`` segment, which does not enter
    the collapse — the four projected fields and the collapse math are identical.
    """
    sid = "sess-1"
    session_dir = tmp_path / "projects" / "proj-a" / sid
    _write_jsonl(session_dir / _PART_LATER_EPOCH, _PART_LATER_ROWS)
    _write_jsonl(session_dir / _PART_EARLIER_EPOCH, _PART_EARLIER_ROWS)
    # The subagent side transcript lives one segment deeper and must be excluded.
    _write_jsonl(session_dir / "subagents" / "agent-x.jsonl", _SUBAGENT_ROWS)

    sa_glob, sa_meta_glob = _seed_subagent_stub(tmp_path)
    settings = Settings(
        default_glob=str(tmp_path / "projects" / "*" / "*.jsonl"),
        subagent_glob=sa_glob,
        subagent_meta_glob=sa_meta_glob,
    )
    return settings, sid


def test_read_turn_text_matches_consumer_collapse(tmp_path: Path) -> None:
    """read_turn_text over the fixture == the consumer's expected string."""
    settings, sid = _seed_collapse_fixture(tmp_path)
    reader = DuckDbTranscriptReader(settings)
    try:
        text = reader.read_turn_text(sid)
    finally:
        reader.close()
    assert text.splitlines() == _COLLAPSE_EXPECTED_LINES
    assert text == "\n".join(_COLLAPSE_EXPECTED_LINES)


def test_subagent_side_transcript_is_excluded(tmp_path: Path) -> None:
    """The single-level part-*.jsonl glob must never admit ``subagents/*``."""
    settings, sid = _seed_collapse_fixture(tmp_path)
    reader = DuckDbTranscriptReader(settings)
    try:
        text = reader.read_turn_text(sid)
    finally:
        reader.close()
    assert "SUBAGENT" not in text
    assert "MUST_NOT_APPEAR" not in text


def test_pure_render_matches_consumer_collapse() -> None:
    """The pure domain renderer alone reproduces the consumer's collapse bytes.

    Same rows, JSON-encoded ``message`` (as DuckDB hands them back), fed
    straight to ``render_turn_text`` — proving the parity lives in the domain
    contract, not just the DuckDB read path.
    """
    rows = [
        TranscriptRow(
            uuid=r["uuid"],
            type=r["type"],
            timestamp=r["timestamp"],
            message=json.dumps(r["message"]),
        )
        for r in (*_PART_LATER_ROWS, *_PART_EARLIER_ROWS)
    ]
    assert render_turn_text(rows) == "\n".join(_COLLAPSE_EXPECTED_LINES)
