"""Tests for the user-friction classifier.

Regex fast-path is deterministic and runs in-process; the LLM path is
exercised with a fake Bedrock client that captures the invoke_model body so
the tests never touch the network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from claude_sql import friction_worker
from claude_sql.config import Settings
from claude_sql.schemas import USER_FRICTION_SCHEMA
from claude_sql.sql_views import register_analytics

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------


def test_user_friction_schema_no_ref() -> None:
    text = json.dumps(USER_FRICTION_SCHEMA)
    assert "$ref" not in text
    assert "$defs" not in text
    assert USER_FRICTION_SCHEMA["type"] == "object"
    assert USER_FRICTION_SCHEMA["additionalProperties"] is False


def test_user_friction_schema_labels_frozen() -> None:
    """All seven labels (including 'none' sentinel) must stay in the enum."""
    label_schema = USER_FRICTION_SCHEMA["properties"]["label"]
    assert set(label_schema["enum"]) == {
        "status_ping",
        "unmet_expectation",
        "confusion",
        "interruption",
        "correction",
        "frustration",
        "none",
    }


# ---------------------------------------------------------------------------
# Regex fast-path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        ("how's it going?", "status_ping"),
        ("any update?", "status_ping"),
        ("where are we at with this?", "status_ping"),
        ("status update?", "status_ping"),
        ("still working on it?", "status_ping"),
        ("how long until done", "status_ping"),
        ("wait, stop!", "interruption"),
        ("hold on a sec", "interruption"),
        ("actually, let me rethink", "interruption"),
        ("nvm", "interruption"),
        ("no, not that one", "correction"),
        ("nope", "correction"),
        ("that's wrong", "correction"),
        ("try again please", "correction"),
    ],
)
def test_regex_hits(text: str, expected_label: str) -> None:
    match = friction_worker.regex_fast_path(text)
    assert match is not None, f"no regex hit for {text!r}"
    label, conf = match
    assert label == expected_label
    assert conf == 0.9


@pytest.mark.parametrize(
    "text",
    [
        "",
        "screenshot?",  # should fall through to LLM
        "tests?",
        "why did you do that?",
        "can you write a test for the update function",  # NOT status ping
        "what's the status column called in the users table?",  # NOT status ping
        "please review the PR",
    ],
)
def test_regex_misses(text: str) -> None:
    """Ambiguous or on-topic shapes must fall through to the LLM path."""
    assert friction_worker.regex_fast_path(text) is None


# ---------------------------------------------------------------------------
# LLM invoke path
# ---------------------------------------------------------------------------


class _FakeBody:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload


class _FakeBedrockClient:
    def __init__(self, response_payload: dict[str, Any]) -> None:
        self.response_payload = response_payload
        self.captured: dict[str, Any] = {}

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.captured["modelId"] = kwargs["modelId"]
        self.captured["body"] = json.loads(kwargs["body"])
        return {"body": _FakeBody(self.response_payload)}


def test_llm_invoke_body_uses_user_friction_schema() -> None:
    """The friction pipeline goes through llm_worker._invoke_classifier_sync
    with USER_FRICTION_SCHEMA; no tool_use, output_config.format only."""
    from claude_sql import llm_worker

    fake_payload = {
        "output": {
            "label": "unmet_expectation",
            "rationale": "bare artifact ping",
            "confidence": 0.85,
        }
    }
    fake = _FakeBedrockClient(fake_payload)
    result = llm_worker._invoke_classifier_sync(
        fake,
        "global.anthropic.claude-sonnet-4-6",
        USER_FRICTION_SCHEMA,
        "screenshot?",
        max_tokens=256,
        thinking_mode="disabled",
    )
    assert result["label"] == "unmet_expectation"
    body = fake.captured["body"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"] == USER_FRICTION_SCHEMA
    assert "tool_choice" not in body
    assert "tools" not in body


# ---------------------------------------------------------------------------
# register_analytics — user_friction view
# ---------------------------------------------------------------------------


def _make_user_friction_parquet(path: Path) -> None:
    now = datetime.now(UTC)
    rows = [
        {
            "uuid": "u-1",
            "session_id": "s-1",
            "ts": now,
            "text_snippet": "screenshot?",
            "label": "unmet_expectation",
            "rationale": "bare artifact ping",
            "source": "llm",
            "confidence": 0.85,
            "classified_at": now,
        },
        {
            "uuid": "u-2",
            "session_id": "s-1",
            "ts": now,
            "text_snippet": "how's it going?",
            "label": "status_ping",
            "rationale": "regex match",
            "source": "regex",
            "confidence": 0.9,
            "classified_at": now,
        },
        {
            "uuid": "u-3",
            "session_id": "s-2",
            "ts": now,
            "text_snippet": "please add a test",
            "label": "none",
            "rationale": "ordinary instruction",
            "source": "llm",
            "confidence": 0.95,
            "classified_at": now,
        },
    ]
    pl.DataFrame(
        rows,
        schema={
            "uuid": pl.Utf8,
            "session_id": pl.Utf8,
            "ts": pl.Datetime("us", "UTC"),
            "text_snippet": pl.Utf8,
            "label": pl.Utf8,
            "rationale": pl.Utf8,
            "source": pl.Utf8,
            "confidence": pl.Float32,
            "classified_at": pl.Datetime("us", "UTC"),
        },
    ).write_parquet(path)


def test_register_analytics_creates_user_friction_view(tmp_path: Path) -> None:
    uf_path = tmp_path / "user_friction.parquet"
    _make_user_friction_parquet(uf_path)
    con = duckdb.connect(":memory:")
    register_analytics(con, user_friction_parquet=uf_path)

    (n,) = con.execute("SELECT count(*) FROM user_friction").fetchone()
    assert n == 3
    (unmet,) = con.execute(
        "SELECT count(*) FROM user_friction WHERE label = 'unmet_expectation'"
    ).fetchone()
    assert unmet == 1


def test_register_analytics_skips_missing_friction_parquet(tmp_path: Path) -> None:
    """Missing parquet must not error -- the view simply is not created."""
    con = duckdb.connect(":memory:")
    register_analytics(con, user_friction_parquet=tmp_path / "nope.parquet")
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    assert "user_friction" not in {r[0] for r in tables}


# ---------------------------------------------------------------------------
# dry-run path
# ---------------------------------------------------------------------------


def test_dry_run_counts_short_user_messages(tmp_path: Path) -> None:
    """`--dry-run` must return 0 (no rows written) but still log a count."""
    con = duckdb.connect(":memory:")
    # Build a tiny messages_text fixture directly -- avoids the full JSONL
    # register_raw chain since we only need this one view.
    con.execute(
        """
        CREATE TABLE messages_text AS
        SELECT * FROM (VALUES
            ('u1', 's1', TIMESTAMPTZ '2026-04-20T10:00:00Z', 'user', 'screenshot?'),
            ('u2', 's1', TIMESTAMPTZ '2026-04-20T10:01:00Z', 'user', 'how''s it going?'),
            ('u3', 's1', TIMESTAMPTZ '2026-04-20T10:02:00Z', 'user', 'ok thanks'),
            ('u4', 's1', TIMESTAMPTZ '2026-04-20T10:03:00Z', 'assistant', 'running tests'),
            ('u5', 's2', TIMESTAMPTZ '2026-04-20T10:04:00Z', 'user',
             '''"""
        + "x" * 400
        + """'''
            )
        ) AS t(uuid, session_id, ts, role, text_content);
        """
    )

    settings = Settings(
        user_friction_parquet_path=tmp_path / "user_friction.parquet",
        checkpoint_db_path=tmp_path / "checkpoint.duckdb",
        friction_max_chars=300,
    )
    n = friction_worker.detect_user_friction(
        con,
        settings,
        since_days=None,
        limit=None,
        dry_run=True,
    )
    # Dry run returns 0 (nothing written); the log output is not asserted
    # here, but the short-message filter should have kept u1/u2/u3 in scope
    # and dropped u4 (assistant) and u5 (too long).
    assert n == 0
    assert not (tmp_path / "user_friction.parquet").exists()
