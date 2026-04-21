"""Tests for :func:`claude_sql.llm_worker._parse_structured_payload`.

The live Sonnet 4.6 structured-output shape has drifted across Bedrock
releases; these exercises the four shapes we've seen in the wild so the
parser never silently errors into the retry queue.
"""

from __future__ import annotations

import json

import pytest

from claude_sql.llm_worker import _parse_structured_payload


def test_parses_top_level_output_dict() -> None:
    """Earliest GA shape: structured object lives at payload['output']."""
    out = _parse_structured_payload({"output": {"autonomy_tier": "autonomous"}})
    assert out == {"autonomy_tier": "autonomous"}


def test_parses_content_text_block_as_json() -> None:
    """Anthropic message shape with a JSON-encoded text block."""
    payload = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": json.dumps({"work_category": "sde"})}],
        "stop_reason": "end_turn",
    }
    out = _parse_structured_payload(payload)
    assert out == {"work_category": "sde"}


def test_parses_content_output_block() -> None:
    """Current GA shape: a content block with type='output' holding the JSON."""
    payload = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output", "output": {"success": "success"}},
        ],
        "stop_reason": "end_turn",
    }
    out = _parse_structured_payload(payload)
    assert out == {"success": "success"}


def test_parses_content_output_block_with_json_key() -> None:
    payload = {
        "content": [
            {"type": "output", "json": {"goal": "Classify sessions."}},
        ],
    }
    out = _parse_structured_payload(payload)
    assert out == {"goal": "Classify sessions."}


def test_parses_content_output_block_with_string_json() -> None:
    payload = {
        "content": [
            {"type": "output", "output": json.dumps({"autonomy_tier": "assisted"})},
        ],
    }
    out = _parse_structured_payload(payload)
    assert out == {"autonomy_tier": "assisted"}


def test_parses_fenced_json_text_block() -> None:
    """Occasional shape: JSON wrapped in a ```json code fence."""
    payload = {
        "content": [
            {"type": "text", "text": '```json\n{"confidence": 0.9}\n```'},
        ],
    }
    out = _parse_structured_payload(payload)
    assert out == {"confidence": 0.9}


def test_skips_thinking_then_parses_text() -> None:
    """Adaptive thinking prepends a thinking block before the real output."""
    payload = {
        "content": [
            {"type": "thinking", "thinking": "reasoning about..."},
            {"type": "text", "text": json.dumps({"success": "partial"})},
        ],
    }
    out = _parse_structured_payload(payload)
    assert out == {"success": "partial"}


def test_observed_failure_shape_does_not_crash() -> None:
    """The shape that caused the 3 production errors on 2026-04-21.

    ``keys = ['model','id','type','role','content','stop_reason',
    'stop_sequence','usage']`` with a single text block holding the JSON.
    """
    payload = {
        "model": "global.anthropic.claude-sonnet-4-6",
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": json.dumps({"autonomy_tier": "autonomous"})}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    out = _parse_structured_payload(payload)
    assert out == {"autonomy_tier": "autonomous"}


def test_raises_on_truly_unknown_shape() -> None:
    with pytest.raises(RuntimeError, match="Unexpected response shape"):
        _parse_structured_payload({"mystery": "data"})
