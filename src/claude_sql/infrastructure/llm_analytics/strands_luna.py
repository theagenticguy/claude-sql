"""GPT-5.6-Luna on Bedrock Mantle: the opt-in ``LlmAnalyticsProvider`` adapter
([llm-analytics] extra: strands-agents[openai]).

Luna is served ONLY on the Bedrock Mantle endpoint via the OpenAI-compatible
Responses API. It is NOT available on ``bedrock-runtime``, NOT via ``Converse``,
NOT via ``InvokeModel``, so it CANNOT reuse claude-sql's existing
``invoke_model`` + ``output_config.format`` Sonnet path. This adapter drives Luna
through the Strands Agents SDK's
:class:`~strands.models.openai_responses.OpenAIResponsesModel` with a
``bedrock_mantle_config``, which:

* mints a fresh ``aws_bedrock_token_generator.provide_token`` bearer on every
  request (no stale-token risk for a long-running backfill);
* auto-derives the base URL ``https://bedrock-mantle.{region}.api.aws/openai/v1``
  from the ``openai.gpt-5.*`` model-id prefix (Luna's required ``openai/v1``
  path); and
* runs native structured output via ``responses.parse`` so the model is forced
  to emit a validated Pydantic instance.

Mirrors the consumer's structured-output adapter (a Mantle/Responses adapter):
lazy + cached model build, a FRESH ``Agent`` per call (a reused agent leaks prior
turns into ``agent.messages``), and a ``.structured_output`` None-guard.

FAIL-OPEN posture (mirrors a fail-open reviewer adapter): the LLM-analytics
plane is OPT-IN, so a Luna outage must NEVER crash the core SQL / embedding
flows. Any Strands / transport / structured-output failure is logged at WARNING
and re-raised as the terminal :class:`~claude_sql.domain.errors.LlmAnalyticsUnavailable`,
which the trajectory worker treats as "provider unavailable" (enqueue for a
later run) exactly like any other recoverable per-chunk failure.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from loguru import logger

from claude_sql.domain.errors import LlmAnalyticsUnavailable

if TYPE_CHECKING:
    from pydantic import BaseModel
    from strands.models.openai_responses import OpenAIResponsesModel

    from claude_sql.domain.ports import SchemaT
    from claude_sql.infrastructure.settings import Settings


@lru_cache(maxsize=4)
def _get_luna_model(model_id: str, region: str) -> OpenAIResponsesModel:
    """Lazily build + cache the Strands Responses model pointed at Bedrock Mantle.

    Cached on ``(model_id, region)`` because the ``OpenAIResponsesModel`` owns the
    connection setup, so IT is the cacheable unit; a FRESH ``Agent`` is built per
    call over it. ``bedrock_mantle_config`` derives the ``/openai/v1`` base URL
    for the ``openai.gpt-5.*`` id and mints a fresh ``provide_token`` bearer per
    request. Reasoning effort / text verbosity ride in ``params`` (kept ``low``,
    Luna's whole point is the fast/cheap tier).

    The ``strands`` import is deferred to here so a bare import of this module
    (or the CLI fast path) never needs the optional extra. A missing extra
    surfaces a clear install hint rather than a bare ``ModuleNotFoundError``.
    """
    try:
        from strands.models.openai_responses import OpenAIResponsesModel
    except ImportError as exc:
        raise ImportError(
            "strands-luna LLM analytics requires the optional extra: "
            "install claude-sql[llm-analytics] "
            "(e.g. `uv add 'claude-sql[llm-analytics]'` or "
            "`pip install 'claude-sql[llm-analytics]'`)."
        ) from exc
    return OpenAIResponsesModel(
        bedrock_mantle_config={"region": region},
        model_id=model_id,
        params={"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
    )


class StrandsLunaAnalytics:
    """``LlmAnalyticsProvider`` over GPT-5.6-Luna on Bedrock Mantle (Strands Responses).

    Model id and region come from ``settings`` (``luna_model_id`` /
    ``luna_region``). The ``strands`` import is deferred into :meth:`_get_model`
    (which caches the model) so constructing this adapter is cheap and
    import-safe; the first ``classify_structured`` call is what actually needs
    the extra installed.
    """

    provider = "strands-luna"

    def __init__(self, settings: Settings) -> None:
        self._model_id = settings.luna_model_id
        self._region = settings.luna_region

    def _get_model(self) -> OpenAIResponsesModel:
        """Return the cached Strands model, raising the install hint if the extra is absent."""
        return _get_luna_model(self._model_id, self._region)

    async def classify_structured(
        self, *, system: str, prompt: str, schema: type[SchemaT]
    ) -> SchemaT:
        """One Luna structured-output call → a validated ``schema`` instance.

        Builds a FRESH :class:`~strands.Agent` per call over the cached model so
        the invocation stays stateless, hands it ``schema`` as the forced
        structured-output model, and returns ``result.structured_output``.

        FAIL-OPEN: any Strands / transport / structured-output error, an empty
        (``None``) structured output, or a wrong-typed payload is logged at
        WARNING and re-raised as :class:`LlmAnalyticsUnavailable` so an opt-in
        run degrading never crashes the pipeline. The install-hint ``ImportError``
        from a missing extra propagates unchanged (it is a configuration error,
        not a transient outage).
        """
        # Resolve the model first so the friendly install-hint ImportError from
        # a missing [llm-analytics] extra wins over a bare ``from strands ...``
        # ModuleNotFoundError. Both raise ImportError; this one names the fix.
        model = self._get_model()
        from strands import Agent

        try:
            agent = Agent(model=model, system_prompt=system)  # fresh agent per call
            result: Any = await agent.invoke_async(prompt, structured_output_model=schema)
        except Exception as exc:
            # Botocore / OpenAI transport, StructuredOutputException, or any
            # residual Strands failure after its own retries. Fail open.
            logger.warning(
                "strands-luna: structured output for {} unavailable ({}: {})",
                schema.__name__,
                type(exc).__name__,
                exc,
            )
            raise LlmAnalyticsUnavailable(
                f"Luna structured output for {schema.__name__} unavailable: {exc}"
            ) from exc
        parsed = result.structured_output
        model_cls: type[BaseModel] = schema
        if not isinstance(parsed, model_cls):
            stop_reason = getattr(result, "stop_reason", None)
            logger.warning(
                "strands-luna: no {} in the model response (stop_reason={!r})",
                schema.__name__,
                stop_reason,
            )
            raise LlmAnalyticsUnavailable(
                f"Luna returned no structured {schema.__name__} (stop_reason={stop_reason!r})"
            )
        return parsed


__all__ = ["StrandsLunaAnalytics", "_get_luna_model"]
