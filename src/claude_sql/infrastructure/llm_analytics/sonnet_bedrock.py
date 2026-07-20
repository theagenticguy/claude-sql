"""Sonnet 4.6 on Bedrock: the base-install ``LlmAnalyticsProvider`` adapter.

A thin wrapper over the EXISTING structured-output path in
:mod:`claude_sql.infrastructure.bedrock.client`, the raw ``invoke_model`` +
``output_config.format`` call (:func:`~claude_sql.infrastructure.bedrock.client.classify_one`),
its tenacity retry, the 1h system-prompt cache, and the
:class:`~claude_sql.domain.errors.BedrockRefusalError` terminal contract. No
new dependency (``boto3`` is already core) and NO behavior change: selecting this
provider reproduces v1.2.1 trajectory analytics exactly.

The adapter converts the port's Pydantic-schema seam onto the legacy dict seam:
it flattens the schema for Bedrock's Draft-2020-12 subset via the same
:func:`~claude_sql.infrastructure.bedrock.structured_output._bedrock_schema`
helper the workers used directly, calls ``classify_one``, then validates the
returned dict back into the domain model with ``schema.model_validate``. So the
domain contract is single-sourced by the Pydantic model while the wire format
stays identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio

from claude_sql.infrastructure.bedrock.client import _build_bedrock_client, classify_one
from claude_sql.infrastructure.bedrock.structured_output import _bedrock_schema

if TYPE_CHECKING:
    from claude_sql.domain.ports import SchemaT
    from claude_sql.infrastructure.settings import Settings


class SonnetBedrockAnalytics:
    """``LlmAnalyticsProvider`` over Sonnet 4.6 on Bedrock (the default).

    Owns the cached ``bedrock-runtime`` client and a per-instance
    :class:`anyio.CapacityLimiter` sized to ``settings.llm_concurrency`` so
    concurrent worker chunks share exactly the limiter the raw path used. The
    system prompt, max tokens, and adaptive-thinking mode ride through unchanged.
    """

    provider = "sonnet-bedrock"

    def __init__(self, settings: Settings, *, thinking_mode: str = "adaptive") -> None:
        self._settings = settings
        self._thinking_mode = thinking_mode
        self._client: Any = None  # lazily built on first call (cached by llm_shared)
        # One shared limiter across every chunk in a run, matches the
        # single ``anyio.CapacityLimiter(settings.llm_concurrency)`` the
        # trajectory worker built inline before the seam was extracted.
        self._sem = anyio.CapacityLimiter(settings.llm_concurrency)

    async def classify_structured(
        self, *, system: str, prompt: str, schema: type[SchemaT]
    ) -> SchemaT:
        """One Bedrock structured-output call → a validated ``schema`` instance.

        Raises :class:`~claude_sql.domain.errors.BedrockRefusalError` on a
        content-policy refusal (terminal, non-retryable, the worker stamps
        neutral placeholders) and lets any residual transport exception
        propagate to the worker's existing retry/skip logic, exactly as the raw
        ``classify_one`` call did.
        """
        if self._client is None:
            self._client = _build_bedrock_client(self._settings)
        raw = await classify_one(
            self._client,
            self._settings.sonnet_model_id,
            _bedrock_schema(schema),
            prompt,
            max_tokens=self._settings.classify_max_tokens,
            thinking_mode=self._thinking_mode,
            sem=self._sem,
            system=system,
            pipeline="trajectory",
        )
        return schema.model_validate(raw)


__all__ = ["SonnetBedrockAnalytics"]
