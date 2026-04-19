"""v2 analytics smoke tests.

Uses in-memory tmp_path fixtures - no real Bedrock calls.  The ``_invoke_*``
tests use a fake boto3 client to capture the request body shape and assert
that the Bedrock ``output_config.format`` contract is honored (no
``tool_use`` / ``tools`` / ``tool_choice`` fields ever appear).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from claude_sql.config import Settings
from claude_sql.schemas import (
    MESSAGE_TRAJECTORY_SCHEMA,
    SESSION_CLASSIFICATION_SCHEMA,
    SESSION_CONFLICTS_SCHEMA,
)
from claude_sql.sql_views import (
    register_analytics,
    register_macros,
    register_raw,
    register_views,
    register_vss,
)

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------


def test_schemas_no_ref() -> None:
    """Bedrock refuses $ref/$defs - assert the flattened schemas have neither."""
    for schema in (
        SESSION_CLASSIFICATION_SCHEMA,
        SESSION_CONFLICTS_SCHEMA,
        MESSAGE_TRAJECTORY_SCHEMA,
    ):
        text = json.dumps(schema)
        assert "$ref" not in text
        assert "$defs" not in text
        assert "definitions" not in text


def test_schemas_additional_properties_false() -> None:
    """Every object-typed schema node must set additionalProperties: false."""

    def _walk(s: dict[str, Any]) -> None:
        if s.get("type") == "object":
            assert s.get("additionalProperties") is False, (
                f"missing additionalProperties:false in {s}"
            )
        for v in s.values():
            if isinstance(v, dict):
                _walk(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _walk(item)

    for schema in (
        SESSION_CLASSIFICATION_SCHEMA,
        SESSION_CONFLICTS_SCHEMA,
        MESSAGE_TRAJECTORY_SCHEMA,
    ):
        _walk(schema)


# ---------------------------------------------------------------------------
# Bedrock invoke_model body shape
# ---------------------------------------------------------------------------


class _FakeBody:
    """Minimal stand-in for the ``StreamingBody`` returned by boto3."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload


class _FakeBedrockClient:
    """Captures ``invoke_model`` kwargs so tests can assert on request shape."""

    def __init__(self, response_payload: dict[str, Any]) -> None:
        self.response_payload = response_payload
        self.captured: dict[str, Any] = {}

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.captured["modelId"] = kwargs["modelId"]
        self.captured["body"] = json.loads(kwargs["body"])
        return {"body": _FakeBody(self.response_payload)}


def test_llm_worker_invoke_body_shape_adaptive_thinking() -> None:
    """Body must use output_config.format (not tool_use) and include thinking."""
    from claude_sql import llm_worker

    fake = _FakeBedrockClient({"output": {"autonomy_tier": "autonomous"}})
    result = llm_worker._invoke_classifier_sync(
        fake,
        "global.anthropic.claude-sonnet-4-6",
        SESSION_CLASSIFICATION_SCHEMA,
        "test text",
        max_tokens=2048,
        thinking_mode="adaptive",
    )

    assert result == {"autonomy_tier": "autonomous"}
    body = fake.captured["body"]
    assert "output_config" in body
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"] == SESSION_CLASSIFICATION_SCHEMA
    assert "tool_choice" not in body, "output_config replaces tool_choice"
    assert "tools" not in body
    assert body.get("thinking", {}).get("type") == "adaptive"


def test_llm_worker_no_thinking_flag() -> None:
    """thinking_mode='disabled' must omit the thinking field entirely."""
    from claude_sql import llm_worker

    fake = _FakeBedrockClient({"output": {"foo": "bar"}})
    llm_worker._invoke_classifier_sync(
        fake,
        "x",
        SESSION_CLASSIFICATION_SCHEMA,
        "t",
        max_tokens=2048,
        thinking_mode="disabled",
    )
    body = fake.captured["body"]
    assert "thinking" not in body


# ---------------------------------------------------------------------------
# register_analytics
# ---------------------------------------------------------------------------


def test_register_analytics_skips_missing_parquets(tmp_path: Any) -> None:
    """register_analytics must not error when parquets are missing."""
    from claude_sql.sql_views import register_analytics

    con = duckdb.connect(":memory:")
    # All paths point at nonexistent files -- every view creation should
    # be skipped with a warning, none should raise.
    register_analytics(
        con,
        classifications_parquet=tmp_path / "nope1.parquet",
        trajectory_parquet=tmp_path / "nope2.parquet",
        conflicts_parquet=tmp_path / "nope3.parquet",
        clusters_parquet=tmp_path / "nope4.parquet",
        cluster_terms_parquet=tmp_path / "nope5.parquet",
        communities_parquet=tmp_path / "nope6.parquet",
    )


def test_register_analytics_with_fixture_parquet(tmp_path: Any) -> None:
    """Given a real parquet, register_analytics creates a queryable view."""
    from claude_sql.sql_views import register_analytics

    cls_df = pl.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "autonomy_tier": ["autonomous", "manual"],
            "work_category": ["sde", "admin"],
            "success": ["success", "failure"],
            "goal": ["ship thing", "do admin"],
            "confidence": [0.9, 0.5],
            "classified_at": [None, None],
        }
    )
    cls_path = tmp_path / "cls.parquet"
    cls_df.write_parquet(cls_path)

    con = duckdb.connect(":memory:")
    register_analytics(con, classifications_parquet=cls_path)

    (n,) = con.execute("SELECT count(*) FROM session_classifications").fetchone()
    assert n == 2
    tiers = con.execute(
        "SELECT autonomy_tier FROM session_classifications ORDER BY session_id"
    ).fetchall()
    assert [r[0] for r in tiers] == ["autonomous", "manual"]
    # session_goals is derived from session_classifications -- should also
    # resolve and return both rows (both have goal set).
    (goals_n,) = con.execute("SELECT count(*) FROM session_goals").fetchone()
    assert goals_n == 2


# ---------------------------------------------------------------------------
# register_analytics via Settings + macros
# ---------------------------------------------------------------------------


def _make_classifications_parquet(path: Path, n: int = 10) -> None:
    now = datetime.now(UTC)
    rows = [
        {
            "session_id": f"sess-{i:04d}",
            "autonomy_tier": ["manual", "assisted", "autonomous"][i % 3],
            "work_category": ["sde", "admin", "strategy_business"][i % 3],
            "success": ["success", "partial", "failure"][i % 3],
            "goal": f"goal {i}",
            "confidence": 0.7 + 0.02 * i,
            "classified_at": now,
        }
        for i in range(n)
    ]
    pl.DataFrame(
        rows,
        schema={
            "session_id": pl.Utf8,
            "autonomy_tier": pl.Utf8,
            "work_category": pl.Utf8,
            "success": pl.Utf8,
            "goal": pl.Utf8,
            "confidence": pl.Float32,
            "classified_at": pl.Datetime("us", "UTC"),
        },
    ).write_parquet(path)


def _make_clusters_parquet(path: Path, n: int = 20) -> None:
    rows = [
        {
            "uuid": f"msg-{i:04d}",
            "cluster_id": i % 4,
            "x": float(i),
            "y": float(i * 0.5),
            "is_noise": False,
        }
        for i in range(n)
    ]
    pl.DataFrame(
        rows,
        schema={
            "uuid": pl.Utf8,
            "cluster_id": pl.Int32,
            "x": pl.Float32,
            "y": pl.Float32,
            "is_noise": pl.Boolean,
        },
    ).write_parquet(path)


def _make_cluster_terms_parquet(path: Path) -> None:
    rows = []
    for cid in range(4):
        for rank, term in enumerate(["alpha", "beta", "gamma"], start=1):
            rows.append({"cluster_id": cid, "term": term, "weight": 1.0 / rank, "rank": rank})
    pl.DataFrame(
        rows,
        schema={
            "cluster_id": pl.Int32,
            "term": pl.Utf8,
            "weight": pl.Float32,
            "rank": pl.Int32,
        },
    ).write_parquet(path)


@pytest.fixture
def analytics_settings(tmp_path: Path) -> Settings:
    _make_classifications_parquet(tmp_path / "cls.parquet")
    _make_clusters_parquet(tmp_path / "clu.parquet")
    _make_cluster_terms_parquet(tmp_path / "ter.parquet")
    return Settings(
        classifications_parquet_path=tmp_path / "cls.parquet",
        clusters_parquet_path=tmp_path / "clu.parquet",
        cluster_terms_parquet_path=tmp_path / "ter.parquet",
        # Leave trajectory/conflicts/communities pointing at nonexistent paths
        trajectory_parquet_path=tmp_path / "nope_traj.parquet",
        conflicts_parquet_path=tmp_path / "nope_conf.parquet",
        communities_parquet_path=tmp_path / "nope_comm.parquet",
    )


def test_register_analytics_creates_available_views(analytics_settings: Settings) -> None:
    con = duckdb.connect(":memory:")
    register_analytics(con, settings=analytics_settings)
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert "session_classifications" in names
    assert "session_goals" in names
    assert "message_clusters" in names
    assert "cluster_terms" in names
    # Missing parquets -> views not created
    assert "message_trajectory" not in names
    assert "session_conflicts" not in names
    assert "session_communities" not in names


def test_register_analytics_count_classifications(analytics_settings: Settings) -> None:
    con = duckdb.connect(":memory:")
    register_analytics(con, settings=analytics_settings)
    count = con.execute("SELECT count(*) FROM session_classifications").fetchone()[0]
    assert count == 10


def test_analytics_macros_register_safely(analytics_settings: Settings, tmp_path: Path) -> None:
    """register_macros should succeed even when trajectory/conflicts/communities
    parquets are missing (those macros are wrapped in _safe_macro).  v1 macros
    still require the transcript-derived views + message_embeddings table."""
    con = duckdb.connect(":memory:")
    # v1 macros reference messages, tool_calls, todo_state_current,
    # subagent_sessions, message_embeddings -- stand up an empty transcript
    # dir so register_raw / register_views / register_vss succeed.
    # register_raw + register_views need non-empty globs with enough shape
    # for DuckDB to infer every column the DDL references.
    proj = tmp_path / "projects" / "-x"
    proj.mkdir(parents=True)
    parent_uuid = "99999999-9999-9999-9999-999999999999"
    sid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    session_record = {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "uuid": "u-m1",
        "timestamp": "2026-04-01T10:00:00.000Z",
        "sessionId": sid,
        "version": "2.0.0",
        "gitBranch": "main",
        "cwd": "/x",
        "userType": "external",
        "entrypoint": "cli",
        "permissionMode": "acceptEdits",
        "promptId": "p-u-m1",
        "message": {
            "id": "m-u-m1",
            "type": "message",
            "role": "user",
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": "seed text for macros"}],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }
    (proj / f"{sid}.jsonl").write_text(json.dumps(session_record) + "\n")
    sub_dir = proj / parent_uuid / "subagents"
    sub_dir.mkdir(parents=True)
    (sub_dir / "agent-deadbeef.meta.json").write_text(
        json.dumps({"agentType": "general-purpose", "description": "placeholder"})
    )
    sub_record = dict(session_record, sessionId=f"sub-{parent_uuid}", uuid="sub-u-1")
    (sub_dir / "agent-deadbeef.jsonl").write_text(json.dumps(sub_record) + "\n")
    register_raw(
        con,
        glob=str(proj / "*.jsonl"),
        subagent_glob=str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl"),
        subagent_meta_glob=str(
            tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.meta.json"
        ),
    )
    register_views(con)
    register_vss(con, embeddings_parquet=tmp_path / "__no_embeddings__.parquet")
    register_analytics(con, settings=analytics_settings)
    register_macros(con, settings=analytics_settings)
    # Verify that cluster_top_terms is callable (its backing view exists)
    rows = con.execute("SELECT term FROM cluster_top_terms(0, 5) ORDER BY rank").fetchall()
    assert len(rows) == 3  # our fixture has 3 terms per cluster
    assert [r[0] for r in rows] == ["alpha", "beta", "gamma"]
