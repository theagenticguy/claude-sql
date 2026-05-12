"""Cache-control + per-pipeline cache-stat tests for ``llm_worker``.

Covers Act-5 of the v1.0 milestone:

* ``cache_control`` ttl=1h on the system block (AWS Bedrock User Guide,
  2026-05; Sonnet 4.5+ on InvokeModel via global CRIS).
* ``cacheable_text_block`` helper for downstream stage-builders.
* ``pipeline_cache_stats`` context manager, threadsafe accumulator, and
  the loguru INFO summary line on exit.
* Tolerance for the legacy usage shape (no ``cache_creation`` sub-object,
  only ``cache_creation_input_tokens``) and the current shape (with
  ``ephemeral_5m_input_tokens`` / ``ephemeral_1h_input_tokens``).

No Bedrock calls — the wire-level invocation tests in
``test_llm_worker.py`` already cover the request shape against a mock
client.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from loguru import logger as loguru_logger

from claude_sql import llm_worker
from claude_sql.llm_worker import (
    _accumulate_cache_stats,
    _extract_usage_metrics,
    _invoke_classifier_sync,
    cacheable_text_block,
    pipeline_cache_stats,
    pipeline_finalize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _capture_loguru(level: str = "INFO") -> tuple[list[str], int]:
    """Attach a sink that appends formatted records and return ``(buffer, sink_id)``.

    Caller is responsible for ``loguru_logger.remove(sink_id)`` in a
    ``try/finally`` to avoid leaking sinks across tests.
    """
    buf: list[str] = []
    sink_id = loguru_logger.add(lambda msg: buf.append(str(msg)), level=level)
    return buf, sink_id


# ---------------------------------------------------------------------------
# 1. System block carries ttl="1h"
# ---------------------------------------------------------------------------


def test_system_block_carries_ttl_1h() -> None:
    """The system content block must carry ``cache_control.ttl == "1h"``.

    1h TTL costs 2× input rate to write the cache but pays 0.1× per read
    for an hour — on a backfill that runs end-to-end inside an hour, 1h
    is the cheaper choice (RFC §4.6).
    """
    client = _make_mock_client({"output": {"k": "v"}})
    sys_prompt = "You are a unit-test classifier. Be terse."
    _invoke_classifier_sync(
        client,
        "global.anthropic.claude-sonnet-4-6",
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
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# ---------------------------------------------------------------------------
# 2. cacheable_text_block helper
# ---------------------------------------------------------------------------


def test_cacheable_text_block_default_5m() -> None:
    """Default TTL is 5m — the right choice for content stable only
    within a single pipeline run."""
    block = cacheable_text_block("schema reminder")
    assert block == {
        "type": "text",
        "text": "schema reminder",
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    }


def test_cacheable_text_block_explicit_1h() -> None:
    """Explicit ttl="1h" produces the expected shape."""
    block = cacheable_text_block("session header", ttl="1h")
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# ---------------------------------------------------------------------------
# 3. _extract_usage_metrics handles both old and new shapes
# ---------------------------------------------------------------------------


def test_extract_usage_metrics_handles_ephemeral_1h_subobject() -> None:
    """When ``cache_creation`` carries 5m + 1h splits, both attribute
    correctly to their respective accumulator buckets."""
    payload = {
        "usage": {
            "input_tokens": 50,
            "output_tokens": 10,
            "cache_read_input_tokens": 1234,
            "cache_creation_input_tokens": 300,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 100,
                "ephemeral_1h_input_tokens": 200,
            },
        }
    }
    metrics = _extract_usage_metrics(payload)
    assert metrics == {
        "calls": 1,
        "input_tokens": 50,
        "output_tokens": 10,
        "cache_read_input_tokens": 1234,
        "cache_creation_5m_input_tokens": 100,
        "cache_creation_1h_input_tokens": 200,
    }


def test_extract_usage_metrics_falls_back_to_legacy_shape() -> None:
    """Legacy responses (no ``cache_creation`` sub-object, only the flat
    ``cache_creation_input_tokens``) attribute the whole creation total
    to the 5m bucket — 1h TTL post-dates that response shape."""
    payload = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 750,
        }
    }
    metrics = _extract_usage_metrics(payload)
    assert metrics["cache_creation_5m_input_tokens"] == 750
    assert metrics["cache_creation_1h_input_tokens"] == 0
    assert metrics["input_tokens"] == 100


def test_extract_usage_metrics_handles_missing_usage() -> None:
    """A response with no ``usage`` key returns all-zero metrics with
    ``calls=1`` so the call still counts toward the total."""
    metrics = _extract_usage_metrics({})
    assert metrics == {
        "calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_5m_input_tokens": 0,
        "cache_creation_1h_input_tokens": 0,
    }


# ---------------------------------------------------------------------------
# 4. pipeline_cache_stats accumulates and emits a summary line
# ---------------------------------------------------------------------------


def test_pipeline_cache_stats_accumulates_across_calls() -> None:
    """Within a ``pipeline_cache_stats`` block, three responses' usage
    rows must sum into one INFO summary line on exit."""
    buf, sink_id = _capture_loguru("INFO")
    try:
        with pipeline_cache_stats("test"):
            for n in (1, 2, 3):
                _accumulate_cache_stats(
                    "test",
                    {
                        "usage": {
                            "input_tokens": 100 * n,
                            "output_tokens": 10 * n,
                            "cache_read_input_tokens": 200 * n,
                            "cache_creation": {
                                "ephemeral_5m_input_tokens": 50 * n,
                                "ephemeral_1h_input_tokens": 25 * n,
                            },
                        }
                    },
                )
    finally:
        loguru_logger.remove(sink_id)

    summary_lines = [m for m in buf if "pipeline=test" in m]
    assert len(summary_lines) == 1
    line = summary_lines[0]
    # 100+200+300 = 600 input
    assert "input=600" in line
    # 200+400+600 = 1200 cache_read → "1K" via the K formatter
    assert "cache_read=1K" in line
    # 50+100+150 = 300 cache_create_5m, 25+50+75 = 150 cache_create_1h
    assert "cache_create_5m=300" in line
    assert "cache_create_1h=150" in line
    # 10+20+30 = 60 output
    assert "output=60" in line
    assert "calls=3" in line


def test_pipeline_cache_stats_handles_missing_cache_creation_subobject() -> None:
    """A legacy 5m-only response (no ``cache_creation`` sub-object,
    just ``cache_creation_input_tokens``) accumulates into the 5m
    bucket without crashing."""
    buf, sink_id = _capture_loguru("INFO")
    try:
        with pipeline_cache_stats("legacy"):
            _accumulate_cache_stats(
                "legacy",
                {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 50,
                        "cache_read_input_tokens": 100,
                        "cache_creation_input_tokens": 800,
                    }
                },
            )
    finally:
        loguru_logger.remove(sink_id)

    summary_lines = [m for m in buf if "pipeline=legacy" in m]
    assert len(summary_lines) == 1
    line = summary_lines[0]
    assert "cache_create_5m=800" in line
    assert "cache_create_1h=0" in line
    assert "input=500" in line


def test_pipeline_cache_stats_resets_on_enter() -> None:
    """Two back-to-back blocks for the same pipeline name must NOT share
    state — entry resets the bucket."""
    buf, sink_id = _capture_loguru("INFO")
    try:
        with pipeline_cache_stats("reset_test"):
            _accumulate_cache_stats(
                "reset_test",
                {"usage": {"input_tokens": 10_000}},
            )
        # Exit cleared the bucket. Enter again.
        with pipeline_cache_stats("reset_test"):
            _accumulate_cache_stats(
                "reset_test",
                {"usage": {"input_tokens": 7}},
            )
    finally:
        loguru_logger.remove(sink_id)

    summaries = [m for m in buf if "pipeline=reset_test" in m]
    assert len(summaries) == 2
    # First block had 10_000 input; second had 7. The second summary
    # must NOT carry the first block's totals.
    assert "input=10K" in summaries[0]
    assert "input=7" in summaries[1]


def test_pipeline_cache_stats_no_op_when_inactive() -> None:
    """Accumulating against a pipeline name with no active context
    manager is a silent no-op — used by ``_maybe_log_bedrock_call`` so
    untracked calls don't crash."""
    # No pipeline_cache_stats block in scope.
    _accumulate_cache_stats(
        "never_registered",
        {"usage": {"input_tokens": 999}},
    )
    # And finalize against a name we never entered emits zeros.
    _, sink_id = _capture_loguru("INFO")
    try:
        totals = pipeline_finalize("still_never_registered")
    finally:
        loguru_logger.remove(sink_id)
    assert totals["calls"] == 0
    assert totals["input_tokens"] == 0


def test_maybe_log_bedrock_call_feeds_accumulator(monkeypatch) -> None:
    """``_maybe_log_bedrock_call`` is the integration point — verify it
    feeds the accumulator even when ``CLAUDE_SQL_BEDROCK_TRACE`` is unset."""
    monkeypatch.setattr(llm_worker, "_BEDROCK_TRACE_PATH", None)
    buf, sink_id = _capture_loguru("INFO")
    try:
        with pipeline_cache_stats("integration"):
            llm_worker._maybe_log_bedrock_call(
                "integration",
                "model-x",
                {
                    "usage": {
                        "input_tokens": 42,
                        "output_tokens": 7,
                        "cache_read_input_tokens": 1000,
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 0,
                            "ephemeral_1h_input_tokens": 5000,
                        },
                    }
                },
                12.5,
            )
    finally:
        loguru_logger.remove(sink_id)
    summary = next(m for m in buf if "pipeline=integration" in m)
    assert "calls=1" in summary
    assert "cache_read=1K" in summary
    assert "cache_create_1h=5K" in summary


def test_pipeline_cache_stats_threadsafe() -> None:
    """The lock-protected dict survives concurrent accumulation from
    multiple threads (mirrors the anyio.to_thread.run_sync hot path
    where workers land in ``_maybe_log_bedrock_call`` from the thread
    pool)."""
    n_threads = 8
    n_per_thread = 50
    buf, sink_id = _capture_loguru("INFO")

    def hammer() -> None:
        for _ in range(n_per_thread):
            _accumulate_cache_stats(
                "race",
                {"usage": {"input_tokens": 1, "output_tokens": 1}},
            )

    try:
        with pipeline_cache_stats("race"):
            threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    finally:
        loguru_logger.remove(sink_id)

    summary = next(m for m in buf if "pipeline=race" in m)
    assert f"calls={n_threads * n_per_thread}" in summary
    assert f"input={n_threads * n_per_thread}" in summary
