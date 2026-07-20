"""Thin unit tests for the ``peek`` application use-case.

The CLI-level golden-output tests live in ``test_peek.py`` (they exercise
``cli.peek`` end to end). These tests pin the extracted
``application.use_cases.peek.peek_session`` entry point directly: the summary
dict for a known session, and ``None`` for an unknown one (the CLI maps ``None``
to a ``not_found`` catalog error / exit 65).
"""

from __future__ import annotations

from typing import Any

import duckdb

from claude_sql.application.use_cases.peek import peek_session
from claude_sql.infrastructure.duckdb_views import register_raw, register_views


def _connect(tmp_corpus: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
    )
    register_views(con)
    return con


def test_peek_session_returns_summary(tmp_corpus: dict[str, Any]) -> None:
    con = _connect(tmp_corpus)
    try:
        sid_one = tmp_corpus["session_ids"][0]
        payload = peek_session(con, sid_one, sample_chars=240, top_tools=10)
    finally:
        con.close()

    assert payload is not None
    assert payload["session_id"] == sid_one
    assert payload["total_lines"] == 4
    assert payload["roles"] == {"user": 3, "assistant": 1}
    assert payload["top_tools"] == [{"name": "Read", "count": 1}]
    assert payload["samples"]["first_user"] is not None
    # sid_one's assistant text is under the 32-char messages_text floor.
    assert payload["samples"]["first_assistant_text"] is None


def test_peek_session_unknown_returns_none(tmp_corpus: dict[str, Any]) -> None:
    con = _connect(tmp_corpus)
    try:
        payload = peek_session(
            con,
            "ffffffff-ffff-ffff-ffff-ffffffffffff",
            sample_chars=240,
            top_tools=10,
        )
    finally:
        con.close()
    assert payload is None


def test_peek_session_truncates_samples(tmp_corpus: dict[str, Any]) -> None:
    """A tiny sample_chars cap truncates the sample text with an ellipsis."""
    con = _connect(tmp_corpus)
    try:
        sid_one = tmp_corpus["session_ids"][0]
        payload = peek_session(con, sid_one, sample_chars=10, top_tools=10)
    finally:
        con.close()
    assert payload is not None
    first_user = payload["samples"]["first_user"]
    assert first_user is not None
    assert first_user["text"] is not None
    assert first_user["text"].endswith("…")
    assert len(first_user["text"]) == 10
