"""Tests for the ``--profile-json`` flag on ``claude-sql query`` / ``explain``.

DuckDB writes JSON profiling output itself when ``profiling_output`` points
at a file, so the helper's job is to compute the destination path, set the
PRAGMAs, and let DuckDB do the rest. These tests verify the path-naming
contract and that DuckDB actually populates the file when the next query
runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from claude_sql.cli import _capture_profile, _profile_path_for


def test_profile_path_lands_under_home_claude_profiling(tmp_path: Path, monkeypatch) -> None:
    """``_profile_path_for`` resolves under ``$HOME/.claude/profiling/``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    out = _profile_path_for("query")
    assert out.parent == tmp_path / ".claude" / "profiling"
    assert out.parent.exists()
    assert out.name.startswith("query-")
    assert out.suffix == ".json"


def test_profile_path_sanitizes_label(tmp_path: Path, monkeypatch) -> None:
    """Special characters in the label become hyphens in the filename."""
    monkeypatch.setenv("HOME", str(tmp_path))
    out = _profile_path_for("weird/label with spaces!")
    name_without_suffix = out.name[: -len(".json")]
    assert "/" not in name_without_suffix
    assert " " not in out.name
    assert "!" not in out.name


def test_capture_profile_writes_real_json_after_query(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: PRAGMAs in place, run a query, file appears with expected shape."""
    monkeypatch.setenv("HOME", str(tmp_path))
    con = duckdb.connect(":memory:")
    try:
        out = _capture_profile(con, label="unit-test")
        # DuckDB writes the JSON when the next query finishes.
        con.execute("SELECT 1 + 1").fetchall()
    finally:
        con.close()
    assert out.exists()
    payload = json.loads(out.read_text())
    # Top-level keys vary by DuckDB version; assert we got *something*
    # JSON-shaped that callers can ingest.
    assert isinstance(payload, dict)
    assert payload  # non-empty
