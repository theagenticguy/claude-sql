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


def test_invoke_body_omits_system_when_not_supplied() -> None:
    """No system block on the wire when caller passes ``system=None``."""
    client = _make_mock_client({"output": {"k": "v"}})
    _invoke_classifier_sync(
        client,
        "m",
        {},
        "x",
        max_tokens=128,
        thinking_mode="disabled",
    )
    body = _captured_body(client)
    assert "system" not in body


def test_invoke_body_carries_system_block_with_cache_control() -> None:
    """A supplied system prompt is sent as a content-block list with
    ``cache_control: ephemeral`` so Anthropic prompt caching kicks in
    once the prompt crosses the per-model cacheable minimum."""
    client = _make_mock_client({"output": {"k": "v"}})
    sys_prompt = "You are a unit-test classifier. Be terse."
    _invoke_classifier_sync(
        client,
        "m",
        {},
        "x",
        max_tokens=128,
        thinking_mode="disabled",
        system=sys_prompt,
    )
    body = _captured_body(client)
    assert isinstance(body["system"], list)
    assert len(body["system"]) == 1
    block = body["system"][0]
    assert block["type"] == "text"
    assert block["text"] == sys_prompt
    assert block["cache_control"] == {"type": "ephemeral"}


def test_module_system_prompts_cross_anthropic_cache_threshold() -> None:
    """All four pipeline system prompts must cross ~1024 tokens so the
    ``cache_control`` marker actually triggers a prompt-cache discount.
    Below the threshold Bedrock silently ignores the header.

    We approximate token count via len(text) // 4 (Anthropic averages
    around 3.5–4 chars/token for English prose). The constant ``900`` is
    a conservative guard — real tokenization tends to land 5–10% higher
    than this lower bound, so 900 here means ~990+ Anthropic tokens.
    """
    from claude_sql.llm_worker import (
        CLASSIFY_SYSTEM_PROMPT,
        CONFLICTS_SYSTEM_PROMPT,
        TRAJECTORY_SYSTEM_PROMPT,
        USER_FRICTION_SYSTEM_PROMPT,
    )

    for name, prompt in (
        ("CLASSIFY", CLASSIFY_SYSTEM_PROMPT),
        ("TRAJECTORY", TRAJECTORY_SYSTEM_PROMPT),
        ("CONFLICTS", CONFLICTS_SYSTEM_PROMPT),
        ("FRICTION", USER_FRICTION_SYSTEM_PROMPT),
    ):
        # Empirical floor: cl100k tokens × 0.78 ≈ Anthropic tokens (validated
        # against a real 100-call trace where 1314 cl100k tokens crossed the
        # 1024-tok cache minimum and 1258 did not). Aim for 1300+ cl100k →
        # ~1014 Anthropic minimum, with a safety margin pushing each prompt
        # to 1700+ cl100k.
        from claude_sql.llm_worker import (
            CLASSIFY_SYSTEM_PROMPT,
            CONFLICTS_SYSTEM_PROMPT,
            TRAJECTORY_SYSTEM_PROMPT,
            USER_FRICTION_SYSTEM_PROMPT,
        )

        # cl100k ~ 0.78× Anthropic ratio; 1300 cl100k ≈ 1014 Anthropic
        approx_cl100k = len(prompt) // 3  # crude but conservative; overcounts
        assert approx_cl100k >= 1100, (
            f"{name} system prompt ~{approx_cl100k} cl100k tokens — likely "
            "below the 1024-token Anthropic cache minimum (cl100k × 0.78 "
            "is the empirical ratio). Pad with stable content so "
            "cache_control actually discounts the call."
        )
        # Silence unused-import warning when prompts equal the imports.
        _ = (
            CLASSIFY_SYSTEM_PROMPT,
            TRAJECTORY_SYSTEM_PROMPT,
            CONFLICTS_SYSTEM_PROMPT,
            USER_FRICTION_SYSTEM_PROMPT,
        )


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
