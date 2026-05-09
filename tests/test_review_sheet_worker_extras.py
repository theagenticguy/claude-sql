"""Coverage top-up for :mod:`claude_sql.review_sheet_worker`.

Targets the JSONL flattener edge cases (malformed line, truncation,
non-dict messages, exotic content blocks), the URI scheme guard, the
``transcript_uri_override=None`` branch, and the missing-file error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from claude_sql import review_sheet_worker
from claude_sql.config import Settings
from claude_sql.review_sheet_worker import (
    _flatten_jsonl_to_text,
    _render_content_block,
    _render_event_line,
    _render_message,
    _resolve_transcript_path,
    _resolve_uri_for_commit,
    _safe_preview,
    generate_review_sheet,
)

# ---------------------------------------------------------------------------
# URI scheme guard (line 185)
# ---------------------------------------------------------------------------


def test_resolve_transcript_path_rejects_non_file_scheme() -> None:
    """``s3://`` etc. raise ``ValueError`` — only ``file://`` is supported."""
    with pytest.raises(ValueError, match="not supported"):
        _resolve_transcript_path("s3://bucket/key")


def test_resolve_transcript_path_handles_percent_encoded_spaces(tmp_path: Path) -> None:
    """Round-trip: percent-encoded ``%20`` → real space in the resolved path."""
    p = tmp_path / "with space.jsonl"
    p.write_text("", encoding="utf-8")
    uri = p.resolve().as_uri()
    out = _resolve_transcript_path(uri)
    assert out == p.resolve()


# ---------------------------------------------------------------------------
# JSONL flattener edge cases (lines 211, 214-217, 220, 222-226, 245, 252-274)
# ---------------------------------------------------------------------------


def test_flatten_skips_blank_and_malformed_lines(tmp_path: Path) -> None:
    """Blank lines and JSONDecodeErrors are tolerated (lines 211, 214-217)."""
    jsonl = tmp_path / "x.jsonl"
    records = [
        "",  # blank line
        "this-is-not-json",  # malformed
        json.dumps(
            {
                "type": "user",
                "timestamp": "2026-05-09T10:00:00Z",
                "message": {"role": "user", "content": "hello"},
            }
        ),
    ]
    jsonl.write_text("\n".join(records) + "\n", encoding="utf-8")
    text = _flatten_jsonl_to_text(jsonl)
    # Only the one valid record produces output.
    assert "[user 2026-05-09T10:00:00Z] hello" in text


def test_flatten_truncates_at_total_max_chars(tmp_path: Path) -> None:
    """The truncation footer is emitted when accumulated chars exceed the cap."""
    jsonl = tmp_path / "big.jsonl"
    big_text = "x" * 200
    records = [
        json.dumps(
            {
                "type": "user",
                "timestamp": f"2026-05-09T10:00:0{i}Z",
                "message": {"role": "user", "content": big_text},
            }
        )
        for i in range(20)
    ]
    jsonl.write_text("\n".join(records) + "\n", encoding="utf-8")
    out = _flatten_jsonl_to_text(jsonl, total_max_chars=500)
    assert "transcript truncated at 500 chars" in out


def test_render_event_line_skips_unknown_record_type() -> None:
    """Records that aren't user/assistant return ``""`` (line 245)."""
    assert _render_event_line({"type": "system", "timestamp": "ts"}) == ""


def test_render_message_returns_empty_when_message_not_dict() -> None:
    """A record whose ``message`` is not a dict returns ``""`` (line 252)."""
    assert _render_message({"message": "string-not-dict"}, "ts") == ""


def test_render_message_string_content_path() -> None:
    """``message.content`` as a plain string → ``[role ts] body`` line."""
    record = {"message": {"role": "user", "content": "hello world"}}
    assert _render_message(record, "ts") == "[user ts] hello world"


def test_render_message_with_non_list_content_returns_empty() -> None:
    """``message.content`` that is neither str nor list → ``""`` (line 259)."""
    record = {"message": {"role": "user", "content": 42}}
    assert _render_message(record, "ts") == ""


def test_render_message_skips_non_dict_blocks() -> None:
    """Non-dict items in the content list are skipped (line 263)."""
    record = {
        "message": {
            "role": "assistant",
            "content": ["string-not-dict", {"type": "text", "text": "kept"}],
        }
    }
    assert "kept" in _render_message(record, "ts")


def test_render_message_returns_empty_when_no_blocks_render() -> None:
    """All blocks render to ``""`` → outer message returns ``""`` (line 268)."""
    record = {
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "text": "internal"}],
        }
    }
    assert _render_message(record, "ts") == ""


def test_render_message_multi_block_body_path() -> None:
    """Multiple text blocks join into one line (lines 273-274)."""
    record = {
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    }
    line = _render_message(record, "ts")
    assert line.startswith("[assistant ts] ")
    assert "first" in line and "second" in line


# ---------------------------------------------------------------------------
# Content-block edge cases (lines 281-284, 292-295)
# ---------------------------------------------------------------------------


def test_render_content_block_text_with_non_string_text_returns_empty() -> None:
    """A ``text`` block whose ``text`` field isn't str returns ``""`` (line 281-283)."""
    assert _render_content_block({"type": "text", "text": 42}, "ts") == ""


def test_render_content_block_thinking_returns_empty() -> None:
    """``thinking`` blocks are agent-internal and skipped (line 292-294)."""
    assert _render_content_block({"type": "thinking", "text": "x"}, "ts") == ""


def test_render_content_block_unknown_type_returns_empty() -> None:
    """Unknown block types fall through to ``""`` (line 295)."""
    assert _render_content_block({"type": "unknown_block_type"}, "ts") == ""


def test_render_content_block_tool_use_with_default_name() -> None:
    """A tool_use block without a name still renders with the ``tool`` default."""
    line = _render_content_block({"type": "tool_use", "input": {"k": "v"}}, "ts")
    assert "[tool_use:tool ts]" in line


# ---------------------------------------------------------------------------
# _safe_preview (lines 301, 306-307)
# ---------------------------------------------------------------------------


def test_safe_preview_handles_none() -> None:
    """``None`` short-circuits to empty string (line 301)."""
    assert _safe_preview(None) == ""


def test_safe_preview_falls_back_to_str_on_unserializable() -> None:
    """Non-JSON-serializable values fall through to ``str(value)`` (lines 306-307).

    ``json.dumps(default=str)`` calls ``str(value)`` for any value that
    isn't natively serializable. A bare class instance has no ``__dict__``
    JSON path, so the ``default=str`` fallback fires and the function
    returns a bounded-length string — exactly the line range we want to
    cover.
    """

    class HardToSerialize:
        pass

    h = HardToSerialize()
    # The "default=str" parameter means Python calls str() on it as a
    # fallback. If that returns a string, json.dumps succeeds. We just
    # assert the function returns a string of bounded length without
    # raising — both branches are exercised in practice.
    out = _safe_preview(h)
    assert isinstance(out, str)
    assert len(out) <= 800


def test_safe_preview_string_passthrough() -> None:
    """Strings round-trip up to the preview cap."""
    assert _safe_preview("short string") == "short string"


# ---------------------------------------------------------------------------
# _resolve_uri_for_commit override branch + binding lookup (lines 336-337)
# ---------------------------------------------------------------------------


def test_resolve_uri_for_commit_uses_override_when_present() -> None:
    """When override is set, no git binding lookup happens (line 334-335)."""
    out = _resolve_uri_for_commit("abc123", transcript_uri_override="file:///tmp/x.jsonl")
    assert out == "file:///tmp/x.jsonl"


def test_resolve_uri_for_commit_falls_back_to_binding_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No override → binding lookup pulls the URI off the trailer (line 336-337)."""

    class _FakeBinding:
        uri = "file:///bound/path.jsonl"

    monkeypatch.setattr(
        review_sheet_worker,
        "resolve_commit_to_transcript",
        lambda commit_sha: _FakeBinding(),
    )
    assert (
        _resolve_uri_for_commit("abc123", transcript_uri_override=None)
        == "file:///bound/path.jsonl"
    )


# ---------------------------------------------------------------------------
# generate_review_sheet missing transcript file (line 390)
# ---------------------------------------------------------------------------


def test_generate_review_sheet_missing_file_raises(tmp_path: Path) -> None:
    """A URI pointing at a non-existent JSONL raises ``FileNotFoundError``."""
    missing = tmp_path / "nope.jsonl"
    settings = Settings(checkpoint_db_path=tmp_path / "ckpt.duckdb")
    with pytest.raises(FileNotFoundError, match="transcript JSONL not found"):
        generate_review_sheet(
            None,
            settings,
            commit_sha="deadbeef",
            transcript_uri_override=missing.as_uri(),
            dry_run=True,
        )


def test_generate_review_sheet_no_thinking_disables_thinking_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``no_thinking=True`` switch sets thinking_mode='disabled' on invoke."""
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "type": "user",
                "timestamp": "ts",
                "message": {"role": "user", "content": "hi"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_invoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["thinking_mode"] = kwargs.get("thinking_mode")
        return {
            "human_intent": "x",
            "agent_exploration": [],
            "corrections": [],
            "tools_used": [],
            "tools_refused": [],
            "diff_rationale": "y",
        }

    monkeypatch.setattr(review_sheet_worker, "_invoke_classifier_sync", fake_invoke)
    monkeypatch.setattr(review_sheet_worker, "_build_bedrock_client", lambda s: MagicMock())

    settings = Settings(checkpoint_db_path=tmp_path / "ckpt.duckdb")
    out = generate_review_sheet(
        None,
        settings,
        commit_sha="abc",
        transcript_uri_override=jsonl.as_uri(),
        dry_run=False,
        no_thinking=True,
    )
    assert captured["thinking_mode"] == "disabled"
    assert "sheet" in out
