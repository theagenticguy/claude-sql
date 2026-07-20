"""Bedrock structured-output schema flattening (infrastructure adapter).

Bedrock's ``output_config.format`` accepts only a JSON Schema Draft 2020-12
SUBSET: no ``$ref``/``$defs``, ``additionalProperties: false`` required at every
object level, and none of the numeric/string/array range constraints pydantic
emits. This module owns the flattening pass that turns a pydantic v2 model into
that subset, plus the four live ``*_SCHEMA`` constants the classifier workers
pass to :func:`claude_sql.infrastructure.bedrock.client.classify_one`.

The flattening pass was split out here in T-2-1; in the v2 hexagonal final cut
(T-8-2) the pydantic *models* moved to :mod:`claude_sql.domain.models` (pure
domain types) and this module imports the model classes from there. The
flattening is a Bedrock wire-format concern, so it belongs in the Bedrock
infrastructure adapter rather than in the shared schema module. These four
``*_SCHEMA`` constants are the live schemas the classifier workers pass to
:func:`claude_sql.infrastructure.bedrock.client.classify_one`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from claude_sql.domain.models import (
    ConflictsResult,
    SessionClassification,
    TrajectoryArrayResult,
    UserFrictionSignal,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


def _bedrock_schema(model: type[BaseModel]) -> dict[str, Any]:
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


def _flatten(schema: dict[str, Any], defs: dict[str, Any] | None = None) -> dict[str, Any]:
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
    out: dict[str, Any] = {}
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


SESSION_CLASSIFICATION_SCHEMA: dict[str, Any] = _bedrock_schema(SessionClassification)
TRAJECTORY_ARRAY_SCHEMA: dict[str, Any] = _bedrock_schema(TrajectoryArrayResult)
SESSION_CONFLICTS_SCHEMA: dict[str, Any] = _bedrock_schema(ConflictsResult)
USER_FRICTION_SCHEMA: dict[str, Any] = _bedrock_schema(UserFrictionSignal)


__all__ = [
    "SESSION_CLASSIFICATION_SCHEMA",
    "SESSION_CONFLICTS_SCHEMA",
    "TRAJECTORY_ARRAY_SCHEMA",
    "USER_FRICTION_SCHEMA",
    "_bedrock_schema",
    "_flatten",
]
