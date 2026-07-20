"""Pluggable LLM-analytics infrastructure: the factory + backend adapters.

The :class:`~claude_sql.domain.ports.LlmAnalyticsProvider` port, its
:data:`~claude_sql.domain.ports.SchemaT` type var, and the terminal
:class:`~claude_sql.domain.errors.LlmAnalyticsUnavailable` error are pure and
live in ``domain``. This package holds the concrete adapters and the
:func:`build_llm_analytics_provider` factory.

Two adapters sit behind the port:

* ``sonnet-bedrock`` (default): wraps the existing raw ``invoke_model`` +
  ``output_config.format`` path in :mod:`claude_sql.infrastructure.bedrock.client`.
  Selecting it reproduces the v1.2.1 behavior exactly.
* ``strands-luna`` (opt-in): GPT-5.6-Luna on the Bedrock Mantle endpoint via
  the OpenAI-compatible Responses API, driven through the Strands Agents SDK,
  behind the optional ``[llm-analytics]`` extra.

Both concrete adapters are imported LAZILY inside
:func:`build_llm_analytics_provider` so a bare
``import claude_sql.infrastructure.llm_analytics`` (or the CLI fast path) never
drags in ``strands`` / ``openai`` / ``botocore``. This keeps the base wheel lean
and the CLI cold-start fast (pinned by ``test_cli_import_is_lean`` and the
forbidden-eager-import guard in ``tests/test_pr3_perf.py``).

Rehomed from ``core/llm_analytics/`` in the v2 hexagonal final cut (T-8-2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from claude_sql.domain.errors import LlmAnalyticsUnavailable
from claude_sql.domain.ports import LlmAnalyticsProvider

if TYPE_CHECKING:
    from claude_sql.infrastructure.settings import Settings


def build_llm_analytics_provider(
    settings: Settings, *, thinking_mode: str = "adaptive"
) -> LlmAnalyticsProvider:
    """Return the ``LlmAnalyticsProvider`` adapter selected by ``settings``.

    Switches on ``settings.llm_analytics_provider``. Adapter modules are
    imported lazily inside each branch so selecting one backend never imports
    the other's heavy dependency, and a bare import of this module stays free of
    ``strands`` / ``botocore``.

    ``thinking_mode`` is fixed for a whole run (the trajectory worker resolves it
    once from ``--no-thinking`` / ``settings.trajectory_thinking``), so it is
    construction-time config rather than a per-call parameter. The Sonnet adapter
    threads it into ``output_config`` adaptive thinking; the Luna adapter ignores
    it (Luna carries reasoning effort in its Responses ``params`` instead).

    The default ``sonnet-bedrock`` reproduces the v1.2.1 behavior exactly.
    """
    provider = settings.llm_analytics_provider
    if provider == "sonnet-bedrock":
        from claude_sql.infrastructure.llm_analytics.sonnet_bedrock import SonnetBedrockAnalytics

        return SonnetBedrockAnalytics(settings, thinking_mode=thinking_mode)
    if provider == "strands-luna":
        from claude_sql.infrastructure.llm_analytics.strands_luna import StrandsLunaAnalytics

        return StrandsLunaAnalytics(settings)
    raise ValueError(
        f"Unknown llm_analytics provider {provider!r}; "
        "expected one of 'sonnet-bedrock', 'strands-luna'."
    )


__all__ = [
    "LlmAnalyticsProvider",
    "LlmAnalyticsUnavailable",
    "build_llm_analytics_provider",
]
