"""Tests for pydantic v2 -> Bedrock JSON Schema derivation."""

from __future__ import annotations

import json

from claude_sql.schemas import (
    MESSAGE_TRAJECTORY_SCHEMA,
    SESSION_CLASSIFICATION_SCHEMA,
    SESSION_CONFLICTS_SCHEMA,
    Conflict,
    MessageTrajectory,
    SessionClassification,
    SessionConflicts,
)


def _walk(schema: dict) -> list[dict]:
    """Yield every dict in a nested schema (for assertions)."""
    out: list[dict] = [schema]
    for v in schema.values():
        if isinstance(v, dict):
            out.extend(_walk(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    out.extend(_walk(item))
    return out


def test_session_classification_schema_flat() -> None:
    # No $ref / $defs anywhere
    s = SESSION_CLASSIFICATION_SCHEMA
    dumped = json.dumps(s)
    assert "$ref" not in dumped
    assert "$defs" not in dumped


def test_session_classification_required_fields() -> None:
    s = SESSION_CLASSIFICATION_SCHEMA
    assert set(s["required"]) >= {
        "autonomy_tier",
        "work_category",
        "success",
        "goal",
        "confidence",
    }
    assert s["additionalProperties"] is False


def test_session_classification_enum_values() -> None:
    props = SESSION_CLASSIFICATION_SCHEMA["properties"]
    assert set(props["autonomy_tier"]["enum"]) == {"manual", "assisted", "autonomous"}
    assert set(props["success"]["enum"]) == {"success", "partial", "failure", "unknown"}


def test_message_trajectory_schema_flat() -> None:
    s = MESSAGE_TRAJECTORY_SCHEMA
    dumped = json.dumps(s)
    assert "$ref" not in dumped
    assert s["additionalProperties"] is False
    assert set(s["properties"]["sentiment_delta"]["enum"]) == {
        "positive",
        "neutral",
        "negative",
    }


def test_session_conflicts_nested_flattened() -> None:
    """SessionConflicts has a nested list[Conflict] -- verify the Conflict shape
    got inlined under items without a $ref."""
    s = SESSION_CONFLICTS_SCHEMA
    assert "$ref" not in json.dumps(s)
    assert "$defs" not in s
    # conflicts.items should be the Conflict schema inlined
    items = s["properties"]["conflicts"]["items"]
    assert items["additionalProperties"] is False
    assert set(items["required"]) >= {"stance_a", "stance_b", "resolution"}
    assert set(items["properties"]["resolution"]["enum"]) == {
        "resolved",
        "unresolved",
        "abandoned",
    }


def test_all_object_subschemas_forbid_additional() -> None:
    """Every object-typed subschema must have additionalProperties: false."""
    for root in (
        SESSION_CLASSIFICATION_SCHEMA,
        MESSAGE_TRAJECTORY_SCHEMA,
        SESSION_CONFLICTS_SCHEMA,
    ):
        for node in _walk(root):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, (
                    f"object node missing additionalProperties: {node}"
                )


def test_pydantic_models_validate_happy_path() -> None:
    # Round-trip: build a valid instance, dump, reload
    sc = SessionClassification(
        autonomy_tier="autonomous",
        work_category="sde",
        success="success",
        goal="Ship v2 analytics.",
        confidence=0.9,
    )
    assert sc.work_category == "sde"

    mt = MessageTrajectory(sentiment_delta="positive", is_transition=False, confidence=0.8)
    assert mt.is_transition is False

    sx = SessionConflicts(
        conflicts=[Conflict(stance_a="cosine", stance_b="euclidean", resolution="resolved")]
    )
    assert len(sx.conflicts) == 1
