"""v2 analytics smoke tests.

Uses in-memory tmp_path fixtures - no real Bedrock calls.  The ``_invoke_*``
tests use a fake boto3 client to capture the request body shape and assert
that the Bedrock ``output_config.format`` contract is honored (no
``tool_use`` / ``tools`` / ``tool_choice`` fields ever appear).
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
import polars as pl

from claude_sql.schemas import (
    MESSAGE_TRAJECTORY_SCHEMA,
    SESSION_CLASSIFICATION_SCHEMA,
    SESSION_CONFLICTS_SCHEMA,
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
