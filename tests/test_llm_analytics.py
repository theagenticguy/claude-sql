"""Adapter-parity tests for the pluggable ``LlmAnalyticsProvider`` port (Wave D).

Every path here is hermetic: NO live Strands / Bedrock / Luna call, ever:

* the factory returns the right adapter per ``settings.llm_analytics_provider``;
* the ``sonnet-bedrock`` adapter round-trips a structured output against a mocked
  ``llm_shared`` (the shared ``FakeBedrockClient`` from conftest);
* the ``strands-luna`` adapter is driven with a monkeypatched ``Agent`` /
  ``OpenAIResponsesModel`` whose ``invoke_async`` returns a fake object exposing
  ``.structured_output`` = a valid ``TrajectoryArrayResult``, guarded by
  ``pytest.importorskip("strands")`` so the base env (no [llm-analytics] extra)
  skips cleanly;
* fail-open: when the mocked Strands raises, the adapter surfaces the terminal
  ``LlmAnalyticsUnavailable`` domain error rather than crashing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from claude_sql.domain.models import TrajectoryArrayResult
from claude_sql.infrastructure.llm_analytics import (
    LlmAnalyticsProvider,
    LlmAnalyticsUnavailable,
    build_llm_analytics_provider,
)
from claude_sql.infrastructure.settings import Settings

# A minimal valid TrajectoryArrayResult payload reused across the suite.
_ONE_WINDOW = {
    "windows": [
        {
            "prev_uuid": None,
            "curr_uuid": "c1",
            "prev_sentiment": None,
            "curr_sentiment": "neutral",
            "delta": None,
            "is_transition": False,
            "transition_kind": "none",
            "confidence": 0.9,
        }
    ]
}


# ---------------------------------------------------------------------------
# (a) factory returns the right adapter per settings
# ---------------------------------------------------------------------------


def test_factory_returns_sonnet_bedrock_by_default() -> None:
    """Default settings → the sonnet-bedrock adapter (unchanged v1.2.1 behavior)."""
    provider = build_llm_analytics_provider(Settings())
    assert isinstance(provider, LlmAnalyticsProvider)
    assert provider.provider == "sonnet-bedrock"


def test_factory_returns_strands_luna_when_selected() -> None:
    """``llm_analytics_provider='strands-luna'`` → the Luna adapter.

    Construction is cheap and import-safe (the ``strands`` import is deferred to
    the first call), so this passes even without the [llm-analytics] extra.
    """
    provider = build_llm_analytics_provider(Settings(llm_analytics_provider="strands-luna"))
    assert isinstance(provider, LlmAnalyticsProvider)
    assert provider.provider == "strands-luna"


def test_factory_rejects_unknown_provider() -> None:
    """An unknown provider name is a hard error (guarded by the Settings Literal,
    but the factory also raises if handed one directly)."""
    settings = Settings()
    object.__setattr__(settings, "llm_analytics_provider", "nope")
    with pytest.raises(ValueError, match="Unknown llm_analytics provider"):
        build_llm_analytics_provider(settings)


# ---------------------------------------------------------------------------
# (b) sonnet-bedrock adapter round-trips against a mocked llm_shared
# ---------------------------------------------------------------------------


def test_sonnet_bedrock_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sonnet-bedrock adapter returns a validated TrajectoryArrayResult.

    The boto3 client is the conftest ``FakeBedrockClient`` returning the GA
    ``{"output": {...}}`` structured-output shape; no network.
    """
    from claude_sql.infrastructure.llm_analytics import sonnet_bedrock
    from conftest import FakeBedrockClient

    fake = FakeBedrockClient({"output": _ONE_WINDOW})
    monkeypatch.setattr(sonnet_bedrock, "_build_bedrock_client", lambda _s: fake)

    provider = build_llm_analytics_provider(Settings(), thinking_mode="disabled")
    result = asyncio.run(
        provider.classify_structured(system="sys", prompt="user", schema=TrajectoryArrayResult)
    )
    assert isinstance(result, TrajectoryArrayResult)
    assert len(result.windows) == 1
    assert result.windows[0].curr_uuid == "c1"
    # The call went out under the trajectory pipeline tag with the sonnet model.
    assert fake.captured[0]["modelId"] == Settings().sonnet_model_id


def test_sonnet_bedrock_propagates_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Bedrock content-policy refusal surfaces as the terminal BedrockRefusalError."""
    from claude_sql.domain.errors import BedrockRefusalError
    from claude_sql.infrastructure.llm_analytics import sonnet_bedrock
    from conftest import FakeBedrockClient

    fake = FakeBedrockClient({"stop_reason": "refusal"})
    monkeypatch.setattr(sonnet_bedrock, "_build_bedrock_client", lambda _s: fake)

    provider = build_llm_analytics_provider(Settings(), thinking_mode="disabled")
    with pytest.raises(BedrockRefusalError):
        asyncio.run(
            provider.classify_structured(system="sys", prompt="user", schema=TrajectoryArrayResult)
        )


# ---------------------------------------------------------------------------
# (c) + (d) strands-luna adapter: monkeypatched Strands surface, no live call
# ---------------------------------------------------------------------------


class _FakeAgentResult:
    """Stand-in for a Strands AgentResult: exposes ``.structured_output``."""

    def __init__(self, structured_output: Any = None, stop_reason: str = "end_turn") -> None:
        self.structured_output = structured_output
        self.stop_reason = stop_reason


def _install_fake_strands(
    monkeypatch: pytest.MonkeyPatch, *, result: Any = None, exc: BaseException | None = None
) -> dict[str, Any]:
    """Monkeypatch the adapter's ``Agent`` + model build so no live Strands runs.

    Returns a ledger dict recording constructions so a test can assert a fresh
    ``Agent`` was built with the system prompt and the schema was handed to
    ``invoke_async``.
    """
    from claude_sql.infrastructure.llm_analytics import strands_luna

    ledger: dict[str, Any] = {"agents": [], "invocations": []}

    class _FakeAgent:
        def __init__(self, *, model: Any, system_prompt: str) -> None:
            ledger["agents"].append({"model": model, "system_prompt": system_prompt})

        async def invoke_async(self, prompt: str, *, structured_output_model: Any) -> Any:
            ledger["invocations"].append({"prompt": prompt, "schema": structured_output_model})
            if exc is not None:
                raise exc
            return _FakeAgentResult(structured_output=result)

    # Patch the model build to a sentinel (no OpenAIResponsesModel / bearer),
    # and the ``from strands import Agent`` lookup by pre-seeding sys.modules
    # with a fake ``strands`` module exposing ``Agent``.
    monkeypatch.setattr(strands_luna.StrandsLunaAnalytics, "_get_model", lambda self: object())
    import sys
    import types

    fake_strands = types.ModuleType("strands")
    setattr(fake_strands, "Agent", _FakeAgent)  # noqa: B010 (dynamic module attr for the fake)
    monkeypatch.setitem(sys.modules, "strands", fake_strands)
    return ledger


def test_strands_luna_returns_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Luna adapter returns the ``.structured_output`` TrajectoryArrayResult."""
    pytest.importorskip("strands", reason="[llm-analytics] extra not installed")
    expected = TrajectoryArrayResult.model_validate(_ONE_WINDOW)
    ledger = _install_fake_strands(monkeypatch, result=expected)

    provider = build_llm_analytics_provider(Settings(llm_analytics_provider="strands-luna"))
    result = asyncio.run(
        provider.classify_structured(
            system="sys-prompt", prompt="user-payload", schema=TrajectoryArrayResult
        )
    )
    assert result is expected
    # A fresh Agent was built with the system prompt and handed the schema.
    assert ledger["agents"][0]["system_prompt"] == "sys-prompt"
    assert ledger["invocations"][0]["schema"] is TrajectoryArrayResult
    assert ledger["invocations"][0]["prompt"] == "user-payload"


def test_strands_luna_fails_open_on_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Strands/transport error surfaces as the terminal LlmAnalyticsUnavailable."""
    pytest.importorskip("strands", reason="[llm-analytics] extra not installed")
    _install_fake_strands(monkeypatch, exc=RuntimeError("mantle gateway timeout"))

    provider = build_llm_analytics_provider(Settings(llm_analytics_provider="strands-luna"))
    with pytest.raises(LlmAnalyticsUnavailable, match="unavailable"):
        asyncio.run(
            provider.classify_structured(system="sys", prompt="user", schema=TrajectoryArrayResult)
        )


def test_strands_luna_fails_open_on_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None structured output surfaces as LlmAnalyticsUnavailable (not a crash)."""
    pytest.importorskip("strands", reason="[llm-analytics] extra not installed")
    _install_fake_strands(monkeypatch, result=None)

    provider = build_llm_analytics_provider(Settings(llm_analytics_provider="strands-luna"))
    with pytest.raises(LlmAnalyticsUnavailable, match="no structured"):
        asyncio.run(
            provider.classify_structured(system="sys", prompt="user", schema=TrajectoryArrayResult)
        )
