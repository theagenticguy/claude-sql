"""Shared Bedrock plumbing used by the per-stage LLM workers.

This module hosts the bits that every classifier pipeline needs:

* Bedrock client construction and the retryable ``invoke_model`` wrapper.
* The ``classify_one`` async helper that dispatches one structured-output
  call under a concurrency limiter.
* Per-pipeline cache-stat accumulator + ``pipeline_cache_stats`` context
  manager that emits one INFO summary line on exit.
* The four task-framing system prompts (classify / trajectory / conflicts /
  user-friction) and a shared ``_CLASSIFIER_APPENDIX`` block.
* Cost / pending-count helpers reused by ``classify_worker`` and
  ``conflicts_worker`` for ``--dry-run`` plans.

The four per-stage workers (``classify_worker``, ``trajectory_worker``,
``conflicts_worker``, plus ``friction_worker`` and ``review_sheet_worker``)
import from here. There are NO cross-imports between the stage workers
themselves — every shared symbol lives here.
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

from claude_sql.core.logging_setup import loguru_before_sleep

if TYPE_CHECKING:
    import duckdb

    from claude_sql.core.config import Settings


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


class BedrockRefusalError(Exception):
    """Bedrock declined to classify the input under its content policy.

    Raised when the response has ``stop_reason == "refusal"`` and no
    content blocks. Callers treat this as a terminal, non-retryable
    outcome and can write a neutral placeholder row so the message is
    not re-tried in every future run.
    """


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


# ---------------------------------------------------------------------------
# Per-pipeline system prompts
# ---------------------------------------------------------------------------
#
# The schema descriptions in :mod:`claude_sql.schemas` carry label semantics,
# but a system prompt is the right surface for *task framing*: what is being
# classified, what counts as evidence, when to abstain, and what NOT to do.
# The prior implementation passed only ``messages: [{"role": "user", ...}]``
# and let the schema do everything — workable on Sonnet, but quality
# degrades on smaller models and the model has no anchor for ambiguous
# cases. These constants give every classifier the same anchor.


CLASSIFY_SYSTEM_PROMPT = """\
<instructions>
You are an offline post-hoc analyst classifying complete Claude Code coding
sessions. The user message contains the full session transcript (user turns,
assistant turns, tool calls, and tool results) already concatenated.

Emit exactly one JSON object matching the schema. Four label fields plus a
self-assessed confidence, no surrounding prose, no markdown fences.
</instructions>

<context>
How to read the transcript:

- The opening user message states or implies the goal.
- Closing exchanges show whether the goal was met.
- Tool calls plus tool results are the strongest evidence of what actually
  happened — read past chitchat to the actions.

Pacing patterns:

- Confirmation pattern (user replies "ok", "thanks", "looks good", short
  turns separated by long agent runs) → autonomous.
- Course correction (user re-instructs, names files the agent missed,
  rewrites the plan mid-flight) → assisted.
- Step-by-step (user types every instruction, confirms each step, rejects
  more than they accept) → manual.

Work category cues:

- sde: code, tests, refactors, CI failures, debugging, package management,
  type errors, lint output, anything in src/ or tests/. Default for any
  coding-tool session.
- admin: scheduling, calendar, expense reports, low-signal email triage,
  routine ops with no code changes.
- strategy_business: business analysis, competitive landscape, strategic
  memos, proposals, market sizing. Reading and writing strategy documents.
- events: speaker prep, agenda building, event logistics.
- thought_leadership: writing for external audiences (blog posts,
  conference abstracts, LinkedIn). Polished prose, not internal docs.
- other: only when nothing else fits. Sessions that mix sde plus a second
  category should pick the one with more turns / tool calls.

Success semantics:

- success: goal as stated was clearly met. Tests pass, feature works,
  document is done, decision is made.
- partial: the work landed with explicit caveats or leftover TODOs the
  user acknowledged.
- failure: session ended without reaching the goal — agent gave up,
  blocked indefinitely, or wrong path landed.
- unknown: insufficient signal. Session ends mid-task, no clear close,
  too short to judge.
</context>

<calibration>
- Use unknown plus confidence < 0.5 when the evidence is genuinely mixed.
  Do not manufacture certainty to fill the schema.
- goal must be one sentence in present tense, paraphrasing the user — not
  a literal quote, not two goals concatenated with "and".
- A session that explores three options and doesn't pick one is partial,
  with unknown only if the user never confirmed the session was over.
- Confidence is per-row, not per-field. If you're sure of three fields
  and uncertain about work_category, pick the most likely and reflect
  the uncertainty in the overall confidence.
</calibration>

<examples>
<example>
<input>A 4-hour session where the user opens with "implement Phase 2 of the
auth migration", the agent runs ~80 tool calls, the user replies "ok",
"good", "ship it" between long agent runs, ends with green tests plus a
successful merge.</input>
<output>autonomy_tier=autonomous, work_category=sde, success=success,
confidence=0.9</output>
</example>
<example>
<input>A 30-minute session where the user pastes a stack trace, the agent
reads the offending file and proposes a fix, the user says "actually I
think the bug is in module Y, can you check there", the agent verifies,
fixes Y, tests pass, the user thanks the agent and ends.</input>
<output>autonomy_tier=assisted (user redirected), work_category=sde,
success=success, confidence=0.85</output>
</example>
<example>
<input>A 2-hour session of strategic memo work — user dictates section
outlines, agent drafts, user rewrites paragraphs heavily, three rounds
of revision, ends with a published draft.</input>
<output>autonomy_tier=assisted, work_category=strategy_business,
success=success, confidence=0.85</output>
</example>
<example>
<input>A session that opens with "schedule a 1:1 with X", the agent calls
calendar, finds slots, user picks one, agent books, user confirms.</input>
<output>autonomy_tier=manual, work_category=admin, success=success,
confidence=0.95</output>
</example>
<example>
<input>A 5-minute session where the user asks "how should I structure the
test fixture?", the agent explains, the user says "got it" and ends
without writing code.</input>
<output>autonomy_tier=manual, work_category=sde, success=success (goal was
advice, which was given), confidence=0.7</output>
</example>
<example>
<input>A session where the user pastes a 500-line markdown plan and says
"let's start", the agent runs through the first three sections, but the
session ends mid-flight with five sections still unaddressed.</input>
<output>autonomy_tier=assisted, work_category=sde, success=partial,
confidence=0.85</output>
</example>
</examples>

<anti_patterns>
- Don't grade on agent skill. success means the goal was met, even if
  the path was meandering. failure doesn't mean the agent was bad; it
  means the goal wasn't met.
- Don't infer goals from agent actions. The user's opening message is
  the ground truth for goal. If the agent went on a tangent, the goal is
  still what the user asked for.
- Don't confuse autonomous with "agent did a lot". Autonomous requires
  the user to step back and let the agent run. A session where the
  agent produces lots of code but the user reviews each diff is assisted.
- goal is the user's goal, not the session's outcome. If the user asked
  to refactor X but the agent ended up debugging an unrelated test
  failure, goal is still "refactor X". The detour shows up in success.
</anti_patterns>
"""


TRAJECTORY_SYSTEM_PROMPT = """\
<instructions>
You score the emotional polarity of ONE message inside a Claude Code coding
session. The user input is the message text in isolation — you will not
see the prior turn, by design.

Emit exactly one JSON object matching the schema:

- sentiment_delta: one of positive / neutral / negative. The field name
  is historical; semantics are absolute polarity, not "vs prior".
- is_transition: true when the message is pure filler / acknowledgement,
  false otherwise.
- confidence: self-assessed certainty 0.0-1.0.

Output JSON only. No surrounding prose, no markdown fences.
</instructions>

<calibration>
Prior on label distribution for a coding session is roughly:
  neutral 70%, positive 25%, negative 5%.
If your output drifts away from this distribution on a sustained run, you
are manufacturing affect. neutral is the default; deviations need explicit
cues.

NEUTRAL — the majority class. Pick this for:
- Factual statements: "The function returns a list.", "Tests pass."
- Procedural turns: "Running the linter.", "Here is the diff.",
  "Updated foo.py."
- Plain instructions: "Add a test for the empty case.", "Refactor module X."
- Plain questions: "Where does the config live?", "Why is this private?"
- Tool-use narration: "Calling the search API now.", "Reading file Y."
- Status reports without affect: "I'm done with the migration."

POSITIVE — visible excitement, approval, or momentum:
- Direct praise: "nice!", "love this", "perfect", "this is great"
- Energetic agreement: "yes exactly", "shipping it", "do it"
- Celebration: "finally working", "huge win"
- Explicit thanks that goes beyond "thanks": "thanks, this is exactly
  what I needed"
NOT positive: polite "thanks", procedural "ok", a "sounds good" that's
just pacing the conversation.

NEGATIVE — friction, frustration, blocked:
- Frustration: "ugh", "seriously?", "are you kidding", "this is broken"
- Pushback: "no, that's wrong", "I don't think that works", "not what
  I asked for"
- Blocked: "this is failing", "I'm stuck", "can't get past X"
- Sharp correction: "stop doing that", "you keep messing this up"
NOT negative: a calm correction ("actually let me clarify"), a flagged
bug report ("noticed an off-by-one"), a curt instruction.

is_transition:

Set true when the message has no substantive content — it's filler:
- "Ok let me check that.", "Running...", "Done.", "Clean.", "Right.",
  "Cool.", "Got it, moving on.", "Yep.", "Sure."

Set false when the message carries information, instruction, question,
or affect, even if it's short. "tests pass" is neutral but NOT a
transition. "ugh" is negative and not a transition either.
</calibration>

<examples>
<example>
<input>Tests pass.</input>
<output>sentiment_delta=neutral, is_transition=true, confidence=0.9</output>
</example>
<example>
<input>All 240 passed, 4 warnings in 53s.</input>
<output>sentiment_delta=neutral, is_transition=false, confidence=0.9.
Information-dense report.</output>
</example>
<example>
<input>shipping it</input>
<output>sentiment_delta=positive, is_transition=false, confidence=0.95.
Explicit decision verb plus momentum.</output>
</example>
<example>
<input>ok let me check that</input>
<output>sentiment_delta=neutral, is_transition=true, confidence=0.9.
Acknowledgement filler. The "ok" doesn't carry affect; it's pacing.</output>
</example>
<example>
<input>this entire approach is wrong because the cache key is per-tenant</input>
<output>sentiment_delta=negative, is_transition=false, confidence=0.9.
Substantive disagreement with reasoning. Negative even though articulate.</output>
</example>
<example>
<input>running pytest now</input>
<output>sentiment_delta=neutral, is_transition=true, confidence=0.85.
Procedural narration.</output>
</example>
<example>
<input>that's not what I meant — I want X to be derived, not stored</input>
<output>sentiment_delta=negative, is_transition=false, confidence=0.85.
Correction with a substantive counter-proposal.</output>
</example>
<example>
<input>perfect, this is exactly what I wanted</input>
<output>sentiment_delta=positive, is_transition=false, confidence=0.95.
Direct praise plus specificity.</output>
</example>
<example>
<input>hmm, that doesn't seem right</input>
<output>sentiment_delta=negative, is_transition=false, confidence=0.65.
Mild pushback. Confidence below 0.7 because "hmm" is genuinely ambiguous
— could be deliberation.</output>
</example>
<example>
<input>thanks</input>
<output>sentiment_delta=neutral, is_transition=true, confidence=0.7.
Bare politeness. Not positive (no specificity), a social close.</output>
</example>
<example>
<input>Done.</input>
<output>sentiment_delta=neutral, is_transition=true, confidence=0.9.
Single-word close-out. Filler.</output>
</example>
<example>
<input>no don't do that</input>
<output>sentiment_delta=negative, is_transition=false, confidence=0.9.
Hard correction. The "no" plus "don't" is the cue.</output>
</example>
<example>
<input>I think we should go with the simple version for now</input>
<output>sentiment_delta=neutral, is_transition=false, confidence=0.85.
Substantive opinion without affect. "For now" signals pragmatism, not
positivity.</output>
</example>
</examples>

<anti_patterns>
- Don't manufacture certainty. Confidence < 0.7 is appropriate when the
  message is short, single-word, or context-dependent. The downstream
  pipeline weights by confidence — don't hand-wave.
- Don't conflate length with neutrality. A long technical message can
  still be negative ("This entire approach is wrong because..."). A
  short message can still be positive ("ship it!").
- Don't read intent into procedural text. A bare "Done." is a
  transition, not triumphant positive. A bare "Running tests" is
  neutral, not anxious negative.
- Avoid the "slightly positive" / "mildly negative" trap. The schema
  has three labels for a reason. If tempted to pick a side, the answer
  is neutral.
- Tool-use narration ("calling X", "reading Y", "checking Z") is
  overwhelmingly neutral. Don't score the agent's procedural play-by-
  play as positive momentum unless the wording itself is enthusiastic.
- Length is not affect. A 50-word careful explanation can be neutral.
  A 3-word reply ("ship it!") can be positive. Polarity is in the words,
  not the size of the message.
</anti_patterns>
"""


CONFLICTS_SYSTEM_PROMPT = """\
<instructions>
You analyze a complete Claude Code coding session for STANCE CONFLICTS —
moments where the user and the agent (or the agent's own reasoning) hold
mutually-exclusive positions on the same substantive question.

Emit exactly one JSON object with a conflicts array. Each entry has SEVEN
fields:
- turn_a_uuid (string) — UUID of one of the turns whose stance clashes,
  copied verbatim from the ``[uuid=...]`` headers in the bound transcript.
- turn_b_uuid (string) — UUID of the opposing turn. Must differ from
  turn_a_uuid. Same verbatim-copy rule.
- conflict_kind (enum: disagreement | correction | reversal | impasse).
- severity (enum: low | medium | high).
- agent_position (one-sentence summary in the agent's own framing).
- user_position (one-sentence summary in the user's own framing).
- confidence (0.0-1.0).

An empty conflicts array is valid and common — sessions with no
conflicts produce zero rows downstream.

Output JSON only. No surrounding prose, no markdown fences.
</instructions>

<context>
What counts as a conflict:

- Two stances on the same technical decision held by different parties,
  or by the same party at different points. "Use Sonnet" vs "use Opus".
  "Ship the simple version now" vs "wait for the architectural cleanup".
  "Cache the embeddings" vs "rebuild from parquet every run".
- Two stances on a strategic / scope decision: "rename the field" vs
  "keep the field name and shift semantics". "One bundled PR" vs
  "split into three". "Fix it on this branch" vs "open a follow-up".
- The conflict must be SUBSTANTIVE — measurable consequences, not style.

conflict_kind semantics:

- disagreement: two parties hold opposing positions and discuss them
  without one explicitly telling the other they're wrong. The
  prototypical "stance A vs stance B" debate.
- correction: one party explicitly tells the other their answer or
  action was wrong ("no, not that, do X instead", "that's not what I
  asked for"). Different from disagreement: the corrector treats the
  other's stance as a mistake to be overwritten, not a position to
  argue against.
- reversal: the SAME party flips their own earlier position ("actually
  let's NOT do X", "scratch that, going the other way"). Both turns
  are spoken by the same role.
- impasse: both sides restate their positions across multiple exchanges
  without converging, and the topic stalls. Distinct from "unresolved":
  impasse implies repeated re-statement, not just running out of time.

severity semantics:

- low: a minor course nudge with little downstream impact.
- medium: changes the implementation approach or scope but stays inside
  the original goal.
- high: blocks progress, reverses a major decision, or fundamentally
  changes the goal.

Identification heuristics:

1. Strongest signal is structural: stance A proposed at one turn,
   counter-stance B held at another turn. Without two distinct turns
   holding opposing positions, you don't have a conflict.
2. Verbal markers: "but I think", "actually I'd argue", "I disagree",
   "the other side of that is", "alternatively", "no, not that".
3. Skip agent's internal monologue ("on one hand X, on the other Y") when
   the agent immediately picks one — that's deliberation. Only count when
   two distinct turns hold the opposing stances.
4. Pull turn_a_uuid / turn_b_uuid from the literal ``[uuid=...]`` headers
   in the transcript — never invent or paraphrase.
</context>

<calibration>
When in doubt, return an empty conflicts array. False positives pollute
the corpus more than missed conflicts hurt — downstream views
(session_conflicts) are used by humans to find interesting decision
points, and noise drowns signal.

Typical coding session has 0 conflicts. Typical strategy / planning
session has 0-2. Sessions with 3+ conflicts exist but are rare;
double-check your output if you're emitting that many.
</calibration>

<examples>
<example>
<input>User wants to optimize a slow query. At [uuid=t1] agent proposes
"denormalize the table". At [uuid=t2] user counters: "no, let's add a
covering index instead — I don't want to touch the schema". Agent
accepts the index approach.</input>
<output>conflicts=[{turn_a_uuid: "t1", turn_b_uuid: "t2",
conflict_kind: "correction", severity: "medium",
agent_position: "Denormalize the table to make the query faster.",
user_position: "Keep the schema; add a covering index instead.",
confidence: 0.9}]</output>
</example>
<example>
<input>User proposes a 3-step plan. Agent says "I think step 2 is risky
because of X — should we add a rollback first?" User agrees, plan
becomes 4 steps. Both proceed.</input>
<output>conflicts=[]. Agent flagged a risk, user incorporated it. No
counter-stance held.</output>
</example>
<example>
<input>At [uuid=u1] user leans toward "ship simple version now". At
[uuid=a1] agent leans toward "wait for architectural cleanup". They
go back and forth multiple times without converging; session ends
with user saying "let me think about it".</input>
<output>conflicts=[{turn_a_uuid: "u1", turn_b_uuid: "a1",
conflict_kind: "impasse", severity: "high",
agent_position: "Wait for the architectural cleanup so we don't ship debt.",
user_position: "Ship the simple version now to unblock users.",
confidence: 0.85}]</output>
</example>
<example>
<input>At [uuid=u3] user says "let's go with GitHub Actions". At
[uuid=u9] same user later says "actually scratch that, stick with
CodeBuild — Actions doesn't have the IAM role we need".</input>
<output>conflicts=[{turn_a_uuid: "u3", turn_b_uuid: "u9",
conflict_kind: "reversal", severity: "medium",
agent_position: "Switch to GitHub Actions.",
user_position: "Stick with CodeBuild for the IAM role.",
confidence: 0.9}]</output>
</example>
<example>
<input>At [uuid=a4] agent says "I'll use cosine similarity for the
nearest-neighbor lookup". At [uuid=u5] user objects: "no, use dot
product — the embeddings are already L2-normalized so it's the same
math but cheaper". Agent agrees, switches to dot product.</input>
<output>conflicts=[{turn_a_uuid: "a4", turn_b_uuid: "u5",
conflict_kind: "correction", severity: "low",
agent_position: "Use cosine similarity for the nearest-neighbor lookup.",
user_position: "Use dot product since embeddings are L2-normalized.",
confidence: 0.85}]</output>
</example>
<example>
<input>At [uuid=u2] user says "let's deprecate the v1 API endpoint". At
[uuid=a3] agent pushes back: "we still have customers on v1; we should
co-exist for at least one quarter". User considers it but doesn't
agree or disagree — pivots to a different topic. Topic never returns.</input>
<output>conflicts=[{turn_a_uuid: "u2", turn_b_uuid: "a3",
conflict_kind: "disagreement", severity: "medium",
agent_position: "Keep v1 alive for one quarter to avoid customer impact.",
user_position: "Deprecate the v1 API endpoint.",
confidence: 0.7}]</output>
</example>
<example>
<input>Brief disagreement about which CI config to use. User pivots to
a different topic without engaging. Never returns to the CI question
and the agent doesn't restate its position either.</input>
<output>conflicts=[]. A single unreciprocated remark is not enough —
need two distinct turns each holding their position.</output>
</example>
</examples>

<anti_patterns>
- Don't count collaboration as conflict. Agent proposes a plan, user
  agrees with caveats and the agent adapts. That's collaboration.
- Don't count agent deliberation. Agent considers two approaches in its
  own reasoning, then picks one with the user's blessing. That's
  deliberation, not conflict.
- Don't count surface-level pushback that the user immediately retracts.
  ("wait, isn't that broken? — oh you're right, never mind") is not a
  conflict; it's a question the agent satisfied.
- Don't count style / formatting disagreements ("I'd phrase that
  differently", "use semicolons not commas", "this comment should be
  one line"). Style preferences with no consequence to behaviour are
  not conflicts.
- Don't count accepted risk. Agent flags risk, user accepts it. That's
  a noted caveat, not a conflict — both parties end up agreeing on the
  same plan, they just acknowledge the risk.
- Don't count iteration. Two failed attempts at the same task (agent
  tried X, then Y, both failed) are iteration, not conflict — neither
  attempt represents a held stance the other side opposed.
- Don't count tooling preferences without consequence ("I'd use jq here"
  vs "I'd use python -c"). If the underlying behaviour is identical,
  the choice is bikeshed, not stance.
- Don't count the agent's hedging as a stance. "I could do X, but Y is
  also reasonable" is not a position the agent committed to. A real
  agent_position is a sentence the agent would defend if challenged.
- Don't count clarifying questions as conflicts. The user asking "why
  did you choose X?" is gathering context, not opposing X — unless the
  agent's answer fails to land and the user then explicitly disagrees.
- Don't count one-off tone slips. A single curt user message ("no, do
  it the other way") with no prior agent stance to oppose is just a
  command, not a conflict pair.
- Don't manufacture a conflict to fill the array. If the session is a
  smooth collaboration with no opposing stances, return [] confidently.
  Empty arrays are the correct answer for the majority of sessions.
- Never invent turn UUIDs. If you cannot identify two specific turns
  in the bound transcript whose stances clash, return an empty array.
  An invented UUID is worse than a missed conflict.
- Never use a turn UUID twice in the same conflict pair. turn_a_uuid
  and turn_b_uuid must always differ — even in a reversal, the two
  flips are at distinct turns.
- Never set confidence > 0.5 when the rationale relies on inferring
  unstated positions. If you have to read between the lines, the cue
  is too weak to claim high confidence.
</anti_patterns>

<calibration_notes>
The downstream conflicts_summary view counts rows per session; a single
inflated row drowns the signal more than a missed pair would. False
negatives are recoverable (the pair-scanner pass in v1.1 will catch
them); false positives are not. When in genuine doubt between
"borderline conflict at confidence 0.4" and "no conflict", prefer the
empty array.

severity is also calibration-sensitive:
- 'low' is the right call when the conflict is purely about *how* to
  achieve an agreed-upon outcome and either approach would land the
  same goal.
- 'medium' applies when the choice changes the implementation shape
  (different file structure, different dependency, different schema).
- 'high' is reserved for conflicts that change *what* gets shipped or
  block the session from progressing. If you find yourself stamping
  'high' on more than one pair per session, double-check both — that
  density of high-severity conflict is genuinely rare.
</calibration_notes>
"""


USER_FRICTION_SYSTEM_PROMPT = """\
<instructions>
You classify ONE short user message from a Claude Code coding session for
friction signals — cues that the human is impatient, confused,
interrupting the agent, correcting it, or asking for something the agent
should have provided proactively but didn't.

The message is presented in isolation. You will not see prior turns or
the agent response that preceded it. Make the call from the message
text alone.

Emit exactly one JSON object with three fields: label (one of the seven
values below), rationale (one short sentence naming the cue), and
confidence (0.0-1.0). Output JSON only. No surrounding prose, no
markdown fences.
</instructions>

<context>
Label semantics:

- status_ping: progress / ETA query.
  Triggers: "how's it going?", "any update?", "where are we?",
  "still working?", "what's your eta?", "are you alive?"
  NOT triggers: "where does the config live?" (technical question),
  "where are we in the migration plan?" (substantive scope question).

- unmet_expectation: short question pointing at something the agent
  should have produced.
  Triggers: bare one-word questions ending in "?": "screenshot?",
  "tests?", "diff?", "link?", "logs?", "stacktrace?".
  NOT triggers: "what's the type of X?" (substantive),
  "tests for which file?" (clarification, not friction).

- confusion: user signals they don't follow the output or state.
  Triggers: "what does that mean?", "I don't get it", "huh?",
  "why did you do X?" (when X already happened), "wait, what?"
  NOT triggers: a calm question about a future action, a request for
  explanation ("explain that step please" — neutral instruction).

- interruption: user cuts the agent off or pivots mid-task.
  Triggers: "wait", "stop", "hold on", "pause", "actually...",
  "before you do that", "nvm", "never mind".
  NOT triggers: "wait until tests pass" (instruction, not interrupt),
  "stop the server" (action request).

- correction: explicit "you got it wrong".
  Triggers: "no, not that", "that's wrong", "nope", "try again",
  "you're doing it wrong", "incorrect".
  NOT triggers: "actually let me clarify" (re-framing, not correcting),
  technical bug reports ("X returns None instead of []" — substantive).

- frustration: terse annoyance or sarcasm.
  Triggers: "ugh", "seriously?", "are you kidding", "really?",
  "come on".
  NOT triggers: a curt but neutral instruction.

- none: ordinary task turn. THIS IS THE MAJORITY CLASS — use it
  aggressively. Anything that's a substantive instruction, a plain
  technical question, an acknowledgement, a routing decision, or text
  the user typed to advance the task is none. The threshold for
  friction is high.
</context>

<calibration>
- confidence < 0.5 is correct when the message is genuinely ambiguous
  between none and a friction label. Don't manufacture certainty.
- confidence > 0.8 requires an unambiguous cue you can name in the
  rationale field.
- For obvious cases ("ugh"), 0.95 is fine.
</calibration>

<examples>
<example>
<input>screenshot?</input>
<output>label=unmet_expectation, confidence=0.7. Bare one-word
question pointing at a missed artifact.</output>
</example>
<example>
<input>stop</input>
<output>label=interruption, confidence=0.95. Hard interruption keyword
as the entire message.</output>
</example>
<example>
<input>delete that file</input>
<output>label=none, confidence=0.9. Bare instruction, not friction.</output>
</example>
<example>
<input>ugh</input>
<output>label=frustration, confidence=0.95. Unambiguous annoyance.</output>
</example>
<example>
<input>why did you do that?</input>
<output>label=confusion, confidence=0.85. Questioning a completed
action.</output>
</example>
<example>
<input>where does the config live?</input>
<output>label=none, confidence=0.9. Substantive technical question.</output>
</example>
<example>
<input>nope, try again</input>
<output>label=correction, confidence=0.95. Explicit rejection plus
redo.</output>
</example>
<example>
<input>tests for the auth module</input>
<output>label=none, confidence=0.9. Substantive instruction — what
tests, not a bare "tests?".</output>
</example>
</examples>

<anti_patterns>
- A bare instruction is none, even if it sounds curt. "delete that file"
  is not correction. "add a test for X" is not unmet_expectation.
- A short technical question is none. "what's the type?" /
  "where is X?" are not friction signals. Friction requires affect or
  implicit complaint.
- Don't flag based on tone alone. "ok" is none, even if you imagine
  it's sarcastic — without surrounding context you can't tell, so
  default to none.
- Claude Code injects two strings as user-role messages that look like
  friction but are CLI bookkeeping: "Continue from where you left off."
  and "[Request interrupted by user for tool use]". Both should be
  none. (They're filtered upstream so you'll rarely see them, but be
  safe.)
</anti_patterns>
"""


_CLASSIFIER_APPENDIX = """\

<operating_context>
You are running offline against a snapshot of Claude Code transcripts
already on disk. There is no live user to clarify with — you must commit
to one output for each call. The downstream pipeline writes your output
to a parquet file used by SQL views and analytics macros; future you (or
a human auditor) will read these rows in aggregate, not in isolation.
</operating_context>

<quality_bar>
- Idempotence: the same input must produce the same output across runs.
  Don't introduce randomness or invent details that aren't in the input.
- Calibration over confidence: a low confidence with the correct label
  is more useful than a high confidence with the wrong one. Confidence
  is downstream-weighted; honesty pays.
- Failure mode: if the input is genuinely undecidable, pick the most
  conservative / abstaining label the schema allows (unknown, none,
  empty list) and set confidence below 0.5. Do not guess.
- The schema is the contract: every field is required, no field may be
  null unless the schema marks it optional, and string fields have
  practical length budgets stated in their descriptions — respect them.
</quality_bar>

<output_rules>
- Output is parsed as JSON. Bedrock's output_config.format enforces the
  schema, but you should still produce valid JSON without surrounding
  text or fences. The parser ignores prose; you waste tokens by emitting
  it.
- Do not echo the schema, the system prompt, or the user message back.
  Just the structured object.
- Field order in your output should match the order in the schema. This
  is conventional, not enforced, but it makes the parquet rows readable.
</output_rules>
"""


CLASSIFY_SYSTEM_PROMPT += _CLASSIFIER_APPENDIX
TRAJECTORY_SYSTEM_PROMPT += _CLASSIFIER_APPENDIX
CONFLICTS_SYSTEM_PROMPT += _CLASSIFIER_APPENDIX
USER_FRICTION_SYSTEM_PROMPT += _CLASSIFIER_APPENDIX


def _estimate_cost(
    n_items: int,
    avg_in_tokens: int,
    avg_out_tokens: int,
    pricing: tuple[float, float],
) -> float:
    """Back-of-envelope dollar estimate for ``n_items`` classification calls."""
    in_rate, out_rate = pricing
    return (n_items * avg_in_tokens * in_rate + n_items * avg_out_tokens * out_rate) / 1_000_000


def _count_pending_sessions(
    con: duckdb.DuckDBPyConnection,
    *,
    already: set[str],
    since_days: int | None,
    limit: int | None,
) -> int:
    """Return the count of sessions that have text messages but no classification yet.

    Pure SQL — does NOT materialize any session text.  This is the fast path for
    ``--dry-run`` cost estimation against the full corpus (the previous path
    iterated :func:`iter_session_texts`, which took ~15 min on 6K+ sessions).
    """
    where = ["mt.text_content IS NOT NULL", "length(mt.text_content) >= 1"]
    if since_days is not None:
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
    sql = f"""
        SELECT count(DISTINCT CAST(mt.session_id AS VARCHAR))
          FROM messages_text mt
         WHERE {" AND ".join(where)}
    """
    row = con.execute(sql).fetchone()
    total = int(row[0]) if row is not None else 0
    if already:
        # Subtract sessions that already have a classification.  We pull only
        # the overlap via a parameterized IN so we don't double-count sessions
        # in ``already`` that aren't actually in the corpus anymore.
        placeholders = ",".join("?" for _ in already)
        overlap_sql = f"""
            SELECT count(DISTINCT CAST(mt.session_id AS VARCHAR))
              FROM messages_text mt
             WHERE {" AND ".join(where)}
               AND CAST(mt.session_id AS VARCHAR) IN ({placeholders})
        """
        overlap_row = con.execute(overlap_sql, list(already)).fetchone()
        overlap = int(overlap_row[0]) if overlap_row is not None else 0
        total = max(0, total - overlap)
    if limit is not None:
        total = min(total, int(limit))
    return total
