"""Bedrock transport for the structured-output classifiers (infrastructure).

This module owns everything that talks to ``bedrock-runtime``:

* Client construction + caching (``_build_bedrock_client``) and the retryable
  ``invoke_model`` wrapper (``_invoke_classifier_sync`` / ``classify_one``).
* Prompt-cache content-block helpers (``cacheable_text_block`` /
  ``build_system_content_block``).
* The per-pipeline cache-stat accumulator + ``pipeline_cache_stats`` context
  manager, and the ``CLAUDE_SQL_BEDROCK_TRACE`` JSONL tracer.
* Structured-payload parsing (``_parse_structured_payload``).

Rehomed here in T-2-1 from ``core/llm_shared.py``. The task-framing system
prompts moved to ``application/prompts.py`` and the schema flattening to
``infrastructure/bedrock/structured_output.py``; ``core/llm_shared.py`` stays a
shim re-exporting every name from its new home so old import paths keep working.

Import hygiene: this module pulls in boto3/botocore at module top, so it must
never be reachable from a module-top path under ``import claude_sql.interfaces.cli.app``
(pinned by ``tests/test_pr3_perf.py``). The per-stage workers import it lazily.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ClientError,
    ConnectionError as BotoConnectionError,
    EndpointConnectionError,
    ReadTimeoutError,
    SSLError,
)
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from claude_sql.domain.errors import BedrockRefusalError
from claude_sql.infrastructure.logging_setup import loguru_before_sleep

if TYPE_CHECKING:
    from claude_sql.infrastructure.settings import Settings


_RETRY_CODES: set[str] = {
    # Standard Bedrock throttle + transient-service errors.
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "ModelErrorException",
    # Bedrock-specific on-demand capacity errors (per AWS re:Post
    # "Troubleshoot Bedrock on-demand 429 Throttling", 2026-05-08).
    "ProvisionedThroughputExceededException",
    "TooManyRequestsException",
    # 5xx spikes on CRIS routing during global region failover — these
    # are idempotent for structured-output invocations so retry is safe.
    "InternalServerException",
    "InternalFailure",
}


#: When set, every classifier call appends a JSONL trace row to this path
#: capturing model id, input/output token counts, prompt-cache hits, and
#: wall-clock ms. Used to verify that ``cache_control`` on the system block
#: actually triggers Anthropic prompt caching and to compare the real
#: token mix against the static dry-run estimates. No-op in normal use.
_BEDROCK_TRACE_PATH = os.environ.get("CLAUDE_SQL_BEDROCK_TRACE")


def cacheable_text_block(text: str, ttl: str = "5m") -> dict[str, Any]:
    """Return a content block with ephemeral ``cache_control`` attached.

    Helper for per-stage request-body builders that want to mark a stable
    content block (e.g. a schema reminder, a session header) for prompt
    caching. Defaults to the 5m TTL — the right choice for content that
    only repeats within a single pipeline run. Pass ``ttl="1h"`` for
    pieces that are stable across runs, but note Anthropic's ordering
    rule: 1h breakpoints must precede 5m breakpoints in the prefix.

    See AWS Bedrock User Guide ("Prompt caching", 2026-05) for the per-
    model cache minimums and TTL semantics. The Anthropic docs page
    incorrectly claims Bedrock does not support 1h TTL — that is stale;
    1h is supported on Sonnet 4.5+ via global CRIS profiles.
    """
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": ttl}}


def build_system_content_block(text: str, *, ttl: str = "1h") -> dict[str, Any]:
    """Return the system-block dict carried in the Bedrock ``system`` field.

    Same shape as :func:`cacheable_text_block` but defaults to ``ttl="1h"``
    because system prompts are stable across an entire pipeline run — the
    1h cache write costs 2× input but pays 0.1× per read for an hour,
    which beats the 5m default for any backfill that runs end-to-end
    inside the hour. Sonnet 4.5+ on Bedrock supports 1h via global CRIS
    profiles on InvokeModel (verified against the Bedrock User Guide,
    2026-05).
    """
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": ttl}}


# ---------------------------------------------------------------------------
# Per-pipeline cache-stat accumulator
# ---------------------------------------------------------------------------
#
# Each Bedrock response carries a ``usage`` object with input / output
# token counts and prompt-cache stats. Aggregating those across a whole
# pipeline run is the only way to verify that the cache_control shape on
# the system block + first cacheable content block actually translates
# into a discount on the live corpus. See RFC §4.6 / §9.5.
#
# Threadsafe accumulation. ``classify_one`` dispatches blocking
# ``invoke_model`` calls via ``anyio.to_thread.run_sync`` (per the
# anyio-structured-concurrency lesson), so two worker threads can land
# in ``maybe_log_bedrock_call`` concurrently. We protect the dict with
# a ``threading.Lock`` rather than an ``anyio.CapacityLimiter`` because:
#
# * A Lock is the right primitive for a critical section that mutates
#   a shared dict — it costs nothing and works from any thread, including
#   the test fixtures that drive the accumulator without an event loop.
# * ``anyio.CapacityLimiter`` is a *concurrency* primitive (cap N
#   simultaneous tasks) — it doesn't serialize a critical section; you
#   still need a lock inside it. Using one here would conflate "how
#   many requests in flight" with "who owns the dict slot", and the
#   former is already governed upstream by the per-pipeline
#   ``settings.llm_concurrency`` limiter.
# * The hot path is two integer reads + a few dict ``+=`` operations
#   per response. Lock contention is a non-issue at our concurrency
#   ceiling (default 2, max ~16).

_CACHE_STATS_LOCK = threading.Lock()
_CACHE_STATS: dict[str, dict[str, int]] = {}


def _empty_cache_stats() -> dict[str, int]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_5m_input_tokens": 0,
        "cache_creation_1h_input_tokens": 0,
    }


def extract_usage_metrics(payload: dict[str, Any]) -> dict[str, int]:
    """Pull the six accumulator fields out of one Bedrock response.

    Handles both the legacy shape (``cache_creation_input_tokens`` only,
    no ``cache_creation`` sub-object — implicitly 5m TTL since 1h didn't
    exist) and the current shape (``cache_creation`` carrying explicit
    ``ephemeral_5m_input_tokens`` and ``ephemeral_1h_input_tokens``).

    Missing fields default to 0 so the accumulator never sees ``None``.
    """
    usage = payload.get("usage") or {}
    cache_creation = usage.get("cache_creation")
    if isinstance(cache_creation, dict):
        five_m = int(cache_creation.get("ephemeral_5m_input_tokens") or 0)
        one_h = int(cache_creation.get("ephemeral_1h_input_tokens") or 0)
    else:
        # Legacy shape: only cache_creation_input_tokens present, no
        # 5m/1h split. Attribute the whole thing to the 5m bucket since
        # 1h TTL post-dates this response shape.
        five_m = int(usage.get("cache_creation_input_tokens") or 0)
        one_h = 0
    return {
        "calls": 1,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "cache_creation_5m_input_tokens": five_m,
        "cache_creation_1h_input_tokens": one_h,
    }


def _accumulate_cache_stats(pipeline: str, payload: dict[str, Any]) -> None:
    """Add this response's usage to the accumulator under ``pipeline``.

    No-op when no ``pipeline_cache_stats`` block is active for this
    name (i.e. the pipeline isn't registered in ``_CACHE_STATS``).
    Failures are swallowed — accumulation must never break a real run.
    """
    if not pipeline:
        return
    try:
        metrics = extract_usage_metrics(payload)
    except (TypeError, ValueError):
        # Malformed usage payload — accumulation is best-effort.
        return
    with _CACHE_STATS_LOCK:
        bucket = _CACHE_STATS.get(pipeline)
        if bucket is None:
            # No active context manager for this pipeline; drop the
            # sample silently. The accumulator is opt-in per RFC §4.6.
            return
        for key, val in metrics.items():
            bucket[key] += val


def _format_token_count(n: int) -> str:
    """Return ``n`` as ``1.2M`` / ``950K`` / ``42`` for compact log lines."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def pipeline_finalize(pipeline: str) -> dict[str, int]:
    """Emit one INFO log line summarizing the accumulator for ``pipeline``.

    Returns the totals dict (post-clear, so callers can assert on it
    in tests). Safe to call when the pipeline has no entries — emits a
    log line with zeroes and clears nothing. Used by the
    ``pipeline_cache_stats`` context manager on exit.
    """
    with _CACHE_STATS_LOCK:
        totals = _CACHE_STATS.pop(pipeline, _empty_cache_stats())
    cache_read = totals["cache_read_input_tokens"]
    fresh_input = totals["input_tokens"]
    if fresh_input > 0:
        ratio = (cache_read + fresh_input) / fresh_input
        ratio_str = f"{ratio:.2f}x"
    else:
        ratio_str = "n/a"
    logger.info(
        "pipeline={}  calls={}  input={}  cache_read={} (ratio={})  "
        "cache_create_5m={}  cache_create_1h={}  output={}",
        pipeline,
        totals["calls"],
        _format_token_count(fresh_input),
        _format_token_count(cache_read),
        ratio_str,
        _format_token_count(totals["cache_creation_5m_input_tokens"]),
        _format_token_count(totals["cache_creation_1h_input_tokens"]),
        _format_token_count(totals["output_tokens"]),
    )
    return totals


@contextmanager
def pipeline_cache_stats(pipeline: str) -> Iterator[None]:
    """Reset, accumulate, then emit-and-clear the cache stats for ``pipeline``.

    Usage::

        with pipeline_cache_stats("trajectory"):
            await _trajectory_async(...)

    On entry the per-pipeline bucket is reset so a previous (e.g.
    crashed) run can't leak into this one. Every Bedrock response goes
    through ``maybe_log_bedrock_call`` which feeds the accumulator. On
    exit (success or exception) one summary line is emitted at INFO
    and the bucket is dropped from the registry.

    Not yet wired into the per-stage workers — exposed at module level
    for downstream agents to wrap the trajectory / classify / conflicts /
    friction loops in a follow-up PR per RFC §4.6.
    """
    with _CACHE_STATS_LOCK:
        _CACHE_STATS[pipeline] = _empty_cache_stats()
    try:
        yield
    finally:
        pipeline_finalize(pipeline)


def maybe_log_bedrock_call(
    pipeline: str, model_id: str, payload: dict[str, Any], elapsed_ms: float
) -> None:
    """Append a single trace row when ``CLAUDE_SQL_BEDROCK_TRACE`` is set
    and feed the per-pipeline cache-stat accumulator.

    Anthropic returns prompt-cache stats under ``payload["usage"]``; we
    capture the full shape so downstream cost accounting can split
    5-minute-TTL writes (1.25× input rate) from 1-hour-TTL writes
    (2× input rate) and cache reads (0.1× input rate). See Anthropic's
    prompt-caching docs for the schema, and AWS's prompt-caching page
    for the per-model cache minimums. Failures are swallowed — tracing
    must never break a real run.
    """
    # Always feed the accumulator first — independent of trace path.
    _accumulate_cache_stats(pipeline, payload)
    if not _BEDROCK_TRACE_PATH:
        return
    try:
        usage = payload.get("usage") or {}
        cache_creation = usage.get("cache_creation") or {}
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "pipeline": pipeline,
            "model": model_id,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            # New-shape fields (present when the model returns the
            # ``cache_creation`` sub-object; older responses omit them).
            "ephemeral_5m_input_tokens": cache_creation.get("ephemeral_5m_input_tokens"),
            "ephemeral_1h_input_tokens": cache_creation.get("ephemeral_1h_input_tokens"),
            "stop_reason": payload.get("stop_reason"),
            "elapsed_ms": round(elapsed_ms, 1),
        }
        path = Path(_BEDROCK_TRACE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        # Tracing must never break a real run.
        pass


def _is_retryable(exc: BaseException) -> bool:
    """Return True if ``exc`` is a Bedrock error worth retrying.

    Same policy as ``embed_worker._is_retryable`` -- throttle/service errors
    via ``ClientError`` plus SSL / connection / read-timeout exceptions.
    """
    if isinstance(exc, SSLError | BotoConnectionError | EndpointConnectionError | ReadTimeoutError):
        return True
    if not isinstance(exc, ClientError):
        return False
    code = exc.response.get("Error", {}).get("Code")
    return code in _RETRY_CODES


_CLIENT_LOCK = threading.Lock()
_CLIENT_CACHE: dict[tuple[str, int], Any] = {}


def _build_bedrock_client(settings: Settings) -> Any:
    """Return a process-wide ``bedrock-runtime`` client keyed on region + pool size.

    Per boto3's "Multithreading with clients" guide (2026-05-08) a single
    ``client`` instance is thread-safe and intended to be shared across
    workers; creating one per request wastes the TCP pool. We cache by
    ``(region, pool_size)`` so changes to ``llm_concurrency`` at runtime
    still produce a fresh client with the right ``max_pool_connections``.

    Config choices (sources in docstrings of the retry decorator and
    ``maybe_log_bedrock_call``):

    * ``max_pool_connections`` — botocore default is 10, which starves any
      concurrency >10. AWS's Bedrock scale guide recommends 50 for high
      throughput; we size to at least ``2 × llm_concurrency`` with a
      floor of 32 so embed + friction + trajectory can share without
      contention.
    * ``connect_timeout=10`` — aggressive enough to fail fast on network
      hiccups without swamping short backfills.
    * ``read_timeout=600`` — Sonnet 4.6 with adaptive thinking + 1M
      context can hold the connection past the 60-second botocore
      default. 10 minutes is a safe upper bound for any single call.
    * ``retries.mode='adaptive'`` + ``max_attempts=0`` — botocore's
      adaptive client-side token bucket absorbs short throttle bursts
      at the SDK layer while this module's tenacity decorator owns the
      semantic retry policy (refusal short-circuit, error
      classification). ``max_attempts=0`` disables botocore's own
      retry loop so tenacity sees errors immediately.
    """
    pool_size = max(
        32,
        max(settings.embed_concurrency, settings.llm_concurrency) * 2,
    )
    key = (settings.region, pool_size)
    with _CLIENT_LOCK:
        client = _CLIENT_CACHE.get(key)
        if client is None:
            boto_cfg = BotoConfig(
                region_name=settings.region,
                retries={"max_attempts": 0, "mode": "adaptive"},
                max_pool_connections=pool_size,
                connect_timeout=10,
                read_timeout=600,
            )
            client = boto3.client("bedrock-runtime", config=boto_cfg)
            _CLIENT_CACHE[key] = client
        return client


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception(_is_retryable),
    before_sleep=loguru_before_sleep("WARNING"),
    reraise=True,
)
def _invoke_classifier_sync(
    client: Any,
    model_id: str,
    schema: dict[str, Any],
    user_text: str,
    *,
    max_tokens: int,
    thinking_mode: str,
    system: str | None = None,
    pipeline: str = "classifier",
) -> dict[str, Any]:
    """One Bedrock ``invoke_model`` call with ``output_config.format`` structured output.

    Parameters
    ----------
    client
        A boto3 ``bedrock-runtime`` client.
    model_id
        Sonnet 4.6 CRIS profile ID (or any model that supports output_config).
    schema
        Flattened JSON Schema dict (see ``schemas.py``).
    user_text
        The full user-role message body (session text or single message).
    max_tokens
        Hard cap on response tokens.
    thinking_mode
        ``"adaptive"`` enables reasoning (higher quality, slower);
        ``"disabled"`` is the escape hatch if Bedrock rejects thinking
        combined with ``output_config``.
    system
        Optional system prompt. Pipelines pass a task-specific framing
        (what's being classified, what each label means, when to abstain)
        so the schema descriptions don't have to carry the whole load.

    Returns
    -------
    dict
        The structured-output JSON object that matches ``schema``.
    """
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "output_config": {
            "format": {"type": "json_schema", "schema": schema},
        },
        "messages": [{"role": "user", "content": user_text}],
    }
    if system:
        # Mark the system block with prompt caching so Anthropic reuses the
        # encoded prefix across calls. Sonnet 4.6 raised the cacheable
        # minimum from 1024 to 2048 input tokens (AWS Bedrock User Guide,
        # 2026-05) — below that the cache_control header is ignored
        # silently. Once the per-pipeline system prompts cross the
        # threshold, the discount kicks in automatically.
        #
        # ``ttl="1h"`` (vs the default 5m) costs 2× input rate to write
        # the cache but pays 0.1× input rate per read for an hour. For a
        # backfill that runs through the corpus in tens of minutes, 1h
        # is correctly the cheaper choice — the system prompt is stable
        # across the whole pipeline run. Per AWS ordering rule, 1h
        # breakpoints must precede 5m breakpoints in the prefix; the
        # system block is always first so this composes cleanly with
        # any per-stage 5m breakpoints further down. The Anthropic docs
        # claim "Bedrock does not support 1h TTL" — that page is stale;
        # 1h is supported on Sonnet 4.5+ via global CRIS profiles on
        # InvokeModel (verified against the Bedrock User Guide,
        # 2026-05).
        body["system"] = [build_system_content_block(system, ttl="1h")]
    if thinking_mode == "adaptive":
        body["thinking"] = {"type": "adaptive"}
    t0 = time.monotonic()
    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    payload = json.loads(resp["body"].read())
    maybe_log_bedrock_call(
        pipeline=pipeline,
        model_id=model_id,
        payload=payload,
        elapsed_ms=elapsed_ms,
    )
    return _parse_structured_payload(payload)


def _parse_structured_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull the structured JSON object out of a Bedrock response.

    Four shapes observed in production (2026-04):

    1. ``payload["output"]`` is a dict — early GA shape, straight return.
    2. Content block with ``type == "output"`` (current GA shape for
       ``output_config.format``) — the structured object is the block
       itself, typically under ``"output"`` / ``"json"`` / ``"content"``.
    3. Anthropic message shape (``content`` is a list of blocks with
       ``type == "text"``) — parse the first text block as JSON.
    4. Bare dict that already matches the schema — return as-is if it
       looks nothing like a Bedrock envelope.

    A ``RuntimeError`` with the observed top-level keys is raised when
    no shape matches; the caller enqueues the unit on the retry queue.
    """
    if payload.get("stop_reason") == "refusal":
        raise BedrockRefusalError("Bedrock refused the input (stop_reason=refusal)")
    if "output" in payload and isinstance(payload["output"], dict):
        return payload["output"]
    content = payload.get("content")
    if isinstance(content, list):
        # Shape 2: structured-output block.
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output":
                for key in ("output", "json", "content"):
                    val = block.get(key)
                    if isinstance(val, dict):
                        return val
                    if isinstance(val, str):
                        try:
                            return json.loads(val)
                        except json.JSONDecodeError:
                            continue
        # Shape 3: text block whose body is the structured JSON.
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text", "")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                stripped = text.strip()
                if stripped.startswith("```"):
                    stripped = stripped.strip("`").lstrip("json").strip()
                    try:
                        return json.loads(stripped)
                    except json.JSONDecodeError:
                        # Stripped payload still isn't valid JSON — fall
                        # through to the RuntimeError below so the caller
                        # surfaces "unexpected response shape" with the
                        # raw payload instead of swallowing the parse miss.
                        pass
        # Shape 3b: message with only non-text blocks (thinking, tool_use)
        # but a stop_reason of end_turn — no structured payload to parse.
    if payload.keys() == {"output"} and isinstance(payload["output"], str):
        return json.loads(payload["output"])
    raise RuntimeError(f"Unexpected response shape: {sorted(payload.keys())}")


async def classify_one(
    client: Any,
    model_id: str,
    schema: dict[str, Any],
    text: str,
    *,
    max_tokens: int,
    thinking_mode: str,
    sem: asyncio.Semaphore | anyio.CapacityLimiter,
    system: str | None = None,
    pipeline: str = "classifier",
) -> dict[str, Any]:
    """Run one classification call under the concurrency limiter.

    ``sem`` accepts either an ``asyncio.Semaphore`` (legacy) or an
    ``anyio.CapacityLimiter`` (new default) — both support
    ``async with``. The boto3 ``invoke_model`` call is blocking, so we
    hand it to ``anyio.to_thread.run_sync`` which honors the enclosing
    structured-concurrency cancellation scope (if any) instead of
    silently detaching on ``asyncio.to_thread`` cancellation.
    """
    async with sem:
        return await anyio.to_thread.run_sync(
            lambda: _invoke_classifier_sync(
                client,
                model_id,
                schema,
                text,
                max_tokens=max_tokens,
                thinking_mode=thinking_mode,
                system=system,
                pipeline=pipeline,
            )
        )


__all__ = [
    "BedrockRefusalError",
    "build_system_content_block",
    "cacheable_text_block",
    "classify_one",
    "extract_usage_metrics",
    "maybe_log_bedrock_call",
    "pipeline_cache_stats",
    "pipeline_finalize",
]
