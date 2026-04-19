"""Pydantic v2 classification schemas wired for Bedrock output_config.format.

Each model exposes a ``SCHEMA_DICT`` module constant -- the ``model_json_schema()``
output flattened (no ``$ref``/``$defs``) and with ``additionalProperties: false``
injected at every object level, so it passes Bedrock's JSON Schema Draft 2020-12
subset validator.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _bedrock_schema(model: type[BaseModel]) -> dict:
    """Turn a pydantic v2 BaseModel into a Bedrock-compatible JSON Schema.

    Bedrock's structured output only accepts JSON Schema Draft 2020-12 SUBSET.
    Pydantic's default ``model_json_schema()`` emits ``$ref``/``$defs`` for
    nested models; Bedrock 400s on those.  We inline refs and inject
    ``additionalProperties: false`` at every object level.

    Parameters
    ----------
    model : type[BaseModel]
        The pydantic model class to convert.

    Returns
    -------
    dict
        Flattened JSON Schema dict safe to pass to Bedrock's
        ``output_config.format.schema`` field.
    """
    raw = model.model_json_schema()
    return _flatten(raw)


def _flatten(schema: dict, defs: dict | None = None) -> dict:
    """Recursively inline $ref references and add additionalProperties: false.

    Uses ``$defs`` from the root schema as the lookup table for inlining.
    Walks every nested object/array and removes ``$defs``/``$ref`` keys.

    Parameters
    ----------
    schema : dict
        A (sub)schema dict produced by pydantic's ``model_json_schema``.
    defs : dict | None
        The ``$defs`` lookup table from the root schema.  Populated from
        ``schema`` on the first call.

    Returns
    -------
    dict
        The flattened schema with refs inlined and object-typed nodes
        forced to ``additionalProperties: false``.
    """
    if defs is None:
        defs = schema.get("$defs") or schema.get("definitions") or {}
    out: dict = {}
    for k, v in schema.items():
        if k in ("$defs", "definitions"):
            continue
        if k == "$ref":
            # e.g. "#/$defs/Conflict"
            name = v.rsplit("/", 1)[-1]
            target = defs.get(name, {})
            return _flatten(target, defs)  # replace the whole object
        if isinstance(v, dict):
            out[k] = _flatten(v, defs)
        elif isinstance(v, list):
            out[k] = [_flatten(i, defs) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    # Ensure object-typed schemas reject unknown fields.
    if out.get("type") == "object" and "additionalProperties" not in out:
        out["additionalProperties"] = False
    # Bedrock's JSON Schema subset rejects numeric range constraints on
    # number/integer types; strip them.  Pydantic still enforces them at
    # response-parse time, so we keep the ge/le annotations in the models.
    if out.get("type") in ("number", "integer"):
        for banned in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
            out.pop(banned, None)
    # String length constraints are also rejected by Bedrock's subset.
    if out.get("type") == "string":
        for banned in ("minLength", "maxLength", "pattern", "format"):
            out.pop(banned, None)
    # Array item/length constraints.
    if out.get("type") == "array":
        for banned in ("minItems", "maxItems", "uniqueItems"):
            out.pop(banned, None)
    # Title keys just add noise; Bedrock accepts them but they clutter the
    # schema for the model's pattern-matching.  Keep descriptions, drop titles.
    out.pop("title", None)
    return out


class SessionClassification(BaseModel):
    """Classify an entire Claude Code session.

    One row per session, written to ``session_classifications.parquet``.
    Used to build the ``session_classifications`` DuckDB view.
    """

    model_config = ConfigDict(extra="forbid")

    autonomy_tier: Literal["manual", "assisted", "autonomous"] = Field(
        ...,
        description=(
            "How much the agent drove the work.  "
            "'manual': the user typed every instruction and confirmed each step. "
            "'assisted': the agent took initiative but the user course-corrected often. "
            "'autonomous': the agent ran multi-step work end-to-end with minimal user "
            "intervention -- tier-3 work."
        ),
    )
    work_category: Literal[
        "sde",
        "admin",
        "strategy_business",
        "events",
        "thought_leadership",
        "other",
    ] = Field(
        ...,
        description=(
            "Dominant activity.  "
            "'sde': software engineering, debugging, code review, tests, CI. "
            "'admin': expense reports, scheduling, low-signal email triage, routine ops. "
            "'strategy_business': business analysis, competitive research, strategic memos, "
            "proposals. "
            "'events': event planning/logistics, speaker prep, agenda building. "
            "'thought_leadership': writing for external audiences (blog posts, conference "
            "abstracts, LinkedIn). "
            "'other': use only when nothing else fits."
        ),
    )
    success: Literal["success", "partial", "failure", "unknown"] = Field(
        ...,
        description=(
            "Did the session complete its stated goal? "
            "'success': the user's goal was clearly met. "
            "'partial': the goal was reached with caveats or leftover TODOs. "
            "'failure': the session ended without achieving the goal. "
            "'unknown': insufficient signal to judge."
        ),
    )
    goal: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description=(
            "ONE sentence summarizing the user's goal, inferred from the opening user "
            "messages and overall arc.  Present tense, <= 280 chars.  Example: "
            '"Refactor the auth middleware to use the new token rotator."'
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Classifier self-assessed confidence 0.0-1.0. Use <0.5 for genuinely "
            "ambiguous sessions."
        ),
    )


SESSION_CLASSIFICATION_SCHEMA: dict = _bedrock_schema(SessionClassification)


class MessageTrajectory(BaseModel):
    """Per-message sentiment + transition-filler flag.

    Applied only to messages that pass the cheap regex heuristic pre-filter.
    """

    model_config = ConfigDict(extra="forbid")

    sentiment_delta: Literal["positive", "neutral", "negative"] = Field(
        ...,
        description=(
            "Emotional polarity vs the prior message.  "
            "'positive': excitement, approval, momentum. "
            "'neutral': factual, procedural, no affect. "
            "'negative': frustration, pushback, blocked."
        ),
    )
    is_transition: bool = Field(
        ...,
        description=(
            "True if this message is pure transition/filler (e.g. 'Ok now let me check that', "
            "'Running the tests', 'Clean.').  Use True for acknowledgement-only messages that "
            "don't carry substantive content."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classifier self-confidence 0.0-1.0.",
    )


MESSAGE_TRAJECTORY_SCHEMA: dict = _bedrock_schema(MessageTrajectory)


class Conflict(BaseModel):
    """A single stance conflict within a session."""

    model_config = ConfigDict(extra="forbid")

    stance_a: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description="One-sentence summary of the first stance.",
    )
    stance_b: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description="One-sentence summary of the conflicting stance.",
    )
    resolution: Literal["resolved", "unresolved", "abandoned"] = Field(
        ...,
        description=(
            "How the conflict ended.  "
            "'resolved': the session converged on one stance. "
            "'unresolved': both stances were still live at end of session. "
            "'abandoned': the topic was dropped without resolution."
        ),
    )


class SessionConflicts(BaseModel):
    """All stance conflicts detected in a session.  Empty list if none found."""

    model_config = ConfigDict(extra="forbid")

    conflicts: list[Conflict] = Field(
        default_factory=list,
        description=(
            "Zero or more stance conflicts.  A conflict is a pair of mutually-exclusive "
            "positions on the same question that surface during the session.  Report "
            "only substantive technical or strategic conflicts -- skip trivial style "
            "disagreements."
        ),
    )


SESSION_CONFLICTS_SCHEMA: dict = _bedrock_schema(SessionConflicts)


__all__ = [
    "MESSAGE_TRAJECTORY_SCHEMA",
    "SESSION_CLASSIFICATION_SCHEMA",
    "SESSION_CONFLICTS_SCHEMA",
    "Conflict",
    "MessageTrajectory",
    "SessionClassification",
    "SessionConflicts",
]
