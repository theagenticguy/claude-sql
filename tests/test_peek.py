"""Tests for the ``claude-sql peek <session_id>`` subcommand.

Mirrors the in-process invocation pattern from ``test_cli.py`` --
``cli.peek(...)`` is called directly with a ``Common`` pointing at the
``tmp_corpus`` fixture; output is asserted via ``capsys``.

Cases:

* JSON happy path (a session with text, an assistant tool_use, and four
  message rows).
* No-tool-calls path (sid_two has no ``tool_use`` blocks; assistant text
  is below the 32-char ``messages_text`` floor so the
  ``first_assistant_text`` sample is ``null``).
* Unknown session_id exits 65 with ``kind="not_found"``.
* Table format renders the section headers.
* CSV format reuses the JSON-shaped payload (peek emits a structured
  blob, not a dataframe).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from claude_sql.app import cli
from claude_sql.app.cli import Common
from claude_sql.core.output import OutputFormat


@pytest.fixture(autouse=True)
def cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect every claude-sql cache path under ``tmp_path``."""
    cache = tmp_path / "claude_home"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "duckdb_tmp").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH", str(cache / "embeddings"))
    monkeypatch.setenv("CLAUDE_SQL_LANCE_URI", str(cache / "embeddings_lance"))
    monkeypatch.setenv("CLAUDE_SQL_CLASSIFICATIONS_PARQUET_PATH", str(cache / "classifications"))
    monkeypatch.setenv("CLAUDE_SQL_TRAJECTORY_PARQUET_PATH", str(cache / "trajectory"))
    monkeypatch.setenv("CLAUDE_SQL_CONFLICTS_PARQUET_PATH", str(cache / "conflicts"))
    monkeypatch.setenv("CLAUDE_SQL_USER_FRICTION_PARQUET_PATH", str(cache / "user_friction"))
    monkeypatch.setenv("CLAUDE_SQL_CLUSTERS_PARQUET_PATH", str(cache / "clusters.parquet"))
    monkeypatch.setenv(
        "CLAUDE_SQL_CLUSTER_TERMS_PARQUET_PATH", str(cache / "cluster_terms.parquet")
    )
    monkeypatch.setenv("CLAUDE_SQL_COMMUNITIES_PARQUET_PATH", str(cache / "communities.parquet"))
    monkeypatch.setenv("CLAUDE_SQL_INGEST_STAMPS_PARQUET_PATH", str(cache / "ingest_stamps"))
    monkeypatch.setenv("CLAUDE_SQL_CHECKPOINT_DB_PATH", str(cache / "claude_sql.duckdb"))
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_TEMP_DIR", str(cache / "duckdb_tmp"))
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_THREADS", "2")
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_MEMORY_LIMIT", "1GB")
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    (tmp_path / "fake_home" / ".claude").mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture(autouse=True)
def _purge_meta_glob_env() -> Iterator[None]:
    prior = os.environ.get("CLAUDE_SQL_SUBAGENT_META_GLOB")
    yield
    if prior is None:
        os.environ.pop("CLAUDE_SQL_SUBAGENT_META_GLOB", None)
    else:
        os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = prior


def _common(tmp_corpus: dict[str, Any], fmt: OutputFormat = OutputFormat.JSON) -> Common:
    """Local copy of ``test_cli._common`` -- mirrors the convention."""
    os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = tmp_corpus["subagent_meta_glob"]
    return Common(
        verbose=False,
        quiet=True,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        format=fmt,
    )


def test_peek_happy_path_json(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sid_one = tmp_corpus["session_ids"][0]
    cli.peek(sid_one, common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)

    assert payload["session_id"] == sid_one
    assert payload["total_lines"] == 4
    assert payload["first_ts"].startswith("2026-04-01")
    assert payload["last_ts"].startswith("2026-04-01")
    assert payload["roles"] == {"user": 3, "assistant": 1}
    assert payload["top_tools"] == [{"name": "Read", "count": 1}]
    assert payload["samples"]["first_user"] is not None
    assert "first message in session one" in payload["samples"]["first_user"]["text"]
    assert payload["samples"]["last_user"] is not None
    # sid_one's assistant text "ok let me check the file" is only 24 chars,
    # under the 32-char messages_text floor, so the slot is null. Documents
    # the visible-but-correct edge case.
    assert payload["samples"]["first_assistant_text"] is None
    assert payload["source_file"].endswith(f"{sid_one}.jsonl")


def test_peek_populated_first_assistant_text(
    tmp_corpus: dict[str, Any],
    make_session_jsonl: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Add a 3rd session whose assistant text clears the 32-char floor."""
    from conftest import make_assistant_msg, make_user_msg

    sid_three = "33333333-3333-3333-3333-333333333333"
    proj = tmp_corpus["root"]
    make_session_jsonl(
        proj / f"{sid_three}.jsonl",
        messages=[
            make_user_msg(
                "u-three",
                sid_three,
                "third session: long enough user prompt to clear the floor",
                ts="2026-04-03T08:00:00.000Z",
            ),
            make_assistant_msg(
                "a-three",
                sid_three,
                ts="2026-04-03T08:00:05.000Z",
                content=[
                    {
                        "type": "text",
                        "text": (
                            "here is a sufficiently long assistant reply "
                            "to clear the 32-char messages_text floor"
                        ),
                    }
                ],
            ),
        ],
    )
    cli.peek(sid_three, common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples"]["first_assistant_text"] is not None
    assert (
        "sufficiently long assistant reply" in (payload["samples"]["first_assistant_text"]["text"])
    )


def test_peek_session_with_no_tool_calls(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sid_two = tmp_corpus["session_ids"][1]
    cli.peek(sid_two, common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)

    assert payload["total_lines"] == 2
    assert payload["roles"] == {"user": 1, "assistant": 1}
    assert payload["top_tools"] == []
    assert payload["samples"]["first_user"] is not None
    # sid_two assistant text is "done" — below the 32-char messages_text floor,
    # so first_assistant_text falls through to None.
    assert payload["samples"]["first_assistant_text"] is None


def test_peek_unknown_session_exits_65(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.peek("ffffffff-ffff-ffff-ffff-ffffffffffff", common=_common(tmp_corpus))
    assert exc.value.code == 65
    err = capsys.readouterr().err
    payload = json.loads(err)
    assert payload["error"]["kind"] == "not_found"
    assert "not found" in payload["error"]["message"]


def test_peek_table_format(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    sid_one = tmp_corpus["session_ids"][0]
    cli.peek(sid_one, common=_common(tmp_corpus, fmt=OutputFormat.TABLE))
    out = capsys.readouterr().out
    assert "session:" in out
    assert "roles:" in out
    assert "top tools:" in out
    assert "samples:" in out
    assert "Read" in out


def test_peek_csv_format_falls_through_to_json(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sid_one = tmp_corpus["session_ids"][0]
    cli.peek(sid_one, common=_common(tmp_corpus, fmt=OutputFormat.CSV))
    out = capsys.readouterr().out.strip()
    # ``emit_json`` JSON-shapes structured payloads regardless of format
    # so peek's contract stays "always a structured blob" for non-table
    # output.
    assert out.startswith("{")
    payload = json.loads(out)
    assert payload["total_lines"] == 4
