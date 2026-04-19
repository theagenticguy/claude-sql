"""Unit tests for llm_worker -- uses a mock Bedrock client to verify request shape."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from claude_sql.llm_worker import _invoke_classifier_sync
from claude_sql.schemas import SESSION_CLASSIFICATION_SCHEMA


def _make_mock_client(return_payload: dict) -> MagicMock:
    """Return a MagicMock that mimics boto3 bedrock-runtime.invoke_model."""
    client = MagicMock()
    body = json.dumps(return_payload).encode()
    client.invoke_model.return_value = {
        "body": SimpleNamespace(read=lambda: body),
    }
    return client


def _captured_body(client: MagicMock) -> dict:
    assert client.invoke_model.called
    kwargs = client.invoke_model.call_args.kwargs
    return json.loads(kwargs["body"])


def test_invoke_body_uses_output_config_not_tool_use() -> None:
    client = _make_mock_client(
        {
            "output": {
                "autonomy_tier": "autonomous",
                "work_category": "sde",
                "success": "success",
                "goal": "Classify sessions.",
                "confidence": 0.9,
            }
        }
    )
    _invoke_classifier_sync(
        client,
        "global.anthropic.claude-sonnet-4-6",
        SESSION_CLASSIFICATION_SCHEMA,
        "hello world",
        max_tokens=2048,
        thinking_mode="adaptive",
    )
    body = _captured_body(client)

    assert "tool_choice" not in body
    assert "tools" not in body
    assert body["output_config"]["format"]["type"] == "json_schema"
    # Schema passed through unchanged
    assert body["output_config"]["format"]["schema"] == SESSION_CLASSIFICATION_SCHEMA


def test_invoke_body_has_adaptive_thinking_by_default() -> None:
    client = _make_mock_client({"output": {"k": "v"}})
    _invoke_classifier_sync(
        client,
        "m",
        {},
        "x",
        max_tokens=1024,
        thinking_mode="adaptive",
    )
    body = _captured_body(client)
    assert body.get("thinking") == {"type": "adaptive"}


def test_invoke_body_drops_thinking_when_disabled() -> None:
    client = _make_mock_client({"output": {"k": "v"}})
    _invoke_classifier_sync(
        client,
        "m",
        {},
        "x",
        max_tokens=1024,
        thinking_mode="disabled",
    )
    body = _captured_body(client)
    assert "thinking" not in body


def test_invoke_parses_output_field() -> None:
    payload = {"output": {"autonomy_tier": "manual"}}
    client = _make_mock_client(payload)
    result = _invoke_classifier_sync(
        client,
        "m",
        {},
        "x",
        max_tokens=1024,
        thinking_mode="disabled",
    )
    assert result == {"autonomy_tier": "manual"}


def test_invoke_falls_back_to_text_block() -> None:
    """If the response doesn't put structured output under 'output', the code
    should parse a text content block as JSON."""
    payload: dict[str, Any] = {
        "content": [
            {"type": "text", "text": json.dumps({"work_category": "sde"})},
        ],
    }
    client = _make_mock_client(payload)
    result = _invoke_classifier_sync(
        client,
        "m",
        {},
        "x",
        max_tokens=1024,
        thinking_mode="disabled",
    )
    assert result == {"work_category": "sde"}
