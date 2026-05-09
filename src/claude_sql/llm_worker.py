"""Bedrock Sonnet 4.6 classification worker.

Uses ``invoke_model`` with ``output_config.format`` (GA structured output) --
NO ``tool_use`` / ``tool_choice`` machinery.  Pydantic v2 models in
``schemas.py`` supply the flattened JSON Schema dicts.

Three public pipelines
----------------------
classify_sessions(con, settings, *, since_days, limit, dry_run, no_thinking) -> int
trajectory_messages(con, settings, *, since_days, limit, dry_run, no_thinking) -> int
detect_conflicts(con, settings, *, since_days, limit, dry_run, no_thinking) -> int

Each pipeline discovers unfinished rows (anti-join against its parquet),
dispatches parallel Bedrock calls under a semaphore, and writes results in
chunks of ``max(batch_size * 4, 256)`` for crash-resilience.

Tenacity + botocore retry shape mirrors ``embed_worker._is_retryable`` exactly
so throttling behaves the same.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread
import boto3
import polars as pl
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

from claude_sql import checkpointer, retry_queue
from claude_sql.logging_setup import loguru_before_sleep
from claude_sql.parquet_shards import read_all, write_part
from claude_sql.schemas import (
    MESSAGE_TRAJECTORY_SCHEMA,
    SESSION_CLASSIFICATION_SCHEMA,
    SESSION_CONFLICTS_SCHEMA,
)
from claude_sql.session_text import iter_session_texts, session_bounds

if TYPE_CHECKING:
    import duckdb

    from claude_sql.config import Settings


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


def _maybe_log_bedrock_call(pipeline: str, model_id: str, payload: dict, elapsed_ms: float) -> None:
    """Append a single trace row when ``CLAUDE_SQL_BEDROCK_TRACE`` is set.

    Anthropic returns prompt-cache stats under ``payload["usage"]``; we
    capture the full shape so downstream cost accounting can split
    5-minute-TTL writes (1.25× input rate) from 1-hour-TTL writes
    (2× input rate) and cache reads (0.1× input rate). See Anthropic's
    prompt-caching docs for the schema, and AWS's prompt-caching page
    for the per-model cache minimums. Failures are swallowed — tracing
    must never break a real run.
    """
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
    ``_maybe_log_bedrock_call``):

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
    schema: dict,
    user_text: str,
    *,
    max_tokens: int,
    thinking_mode: str,
    system: str | None = None,
) -> dict:
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
        # encoded prefix across calls. Below the minimum-cacheable threshold
        # (~1024 tokens for Sonnet 4.6) the cache_control header is ignored
        # silently — no harm — and once the per-pipeline system prompts
        # cross the threshold, the discount kicks in automatically. We send
        # the system value as a content-block list so cache_control attaches
        # cleanly; Bedrock also accepts a bare string for non-cached calls.
        body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
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
    _maybe_log_bedrock_call(
        pipeline=schema.get("title", "classifier") if isinstance(schema, dict) else "classifier",
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


def _parse_structured_payload(payload: dict) -> dict:
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
                        pass
        # Shape 3b: message with only non-text blocks (thinking, tool_use)
        # but a stop_reason of end_turn — no structured payload to parse.
    if payload.keys() == {"output"} and isinstance(payload["output"], str):
        return json.loads(payload["output"])
    raise RuntimeError(f"Unexpected response shape: {sorted(payload.keys())}")


async def _classify_one(
    client: Any,
    model_id: str,
    schema: dict,
    text: str,
    *,
    max_tokens: int,
    thinking_mode: str,
    sem: asyncio.Semaphore | anyio.CapacityLimiter,
    system: str | None = None,
) -> dict:
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

Emit exactly one JSON object with a conflicts array. Each conflict has
three fields: stance_a, stance_b (one-sentence summaries) and resolution
(resolved / unresolved / abandoned). An empty list is valid and common.

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

Resolution semantics:

- resolved: the session converged on one stance with explicit agreement.
  Look for "ok let's do that", "you're right", "going with X".
- unresolved: both stances were still live at session end. User punted,
  agent didn't pick, or session ran out of time.
- abandoned: topic was dropped without a decision. Different from
  unresolved — abandoned means the conversation moved on, not that they
  failed to decide.

Identification heuristics:

1. Strongest signal is structural: stance A proposed, counter-stance B
   raised explicitly, then a decision made (or not). Without an explicit
   counter-stance, you don't have a conflict.
2. Verbal markers: "but I think", "actually I'd argue", "I disagree",
   "the other side of that is", "alternatively", "or we could".
3. Skip agent's internal monologue ("on one hand X, on the other Y") when
   the agent immediately picks one — that's deliberation. Only count when
   the user (or another party) holds the other stance.
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
<input>User wants to optimize a slow query. Agent proposes denormalizing
the table. User counters: "no, let's add a covering index instead — I
don't want to touch the schema". Agent accepts the index approach and
ships it.</input>
<output>conflicts=[{stance_a: "Denormalize the table to make the query
faster.", stance_b: "Keep the schema; add a covering index instead.",
resolution: "resolved"}]</output>
</example>
<example>
<input>User proposes a 3-step plan. Agent says "I think step 2 is risky
because of X — should we add a rollback first?" User agrees, plan
becomes 4 steps. Both proceed.</input>
<output>conflicts=[]. Agent flagged a risk, user incorporated it. No
counter-stance held.</output>
</example>
<example>
<input>Agent's reasoning shows "I could use approach A or approach B,
but A is simpler so I'll go with A". User says "ok".</input>
<output>conflicts=[]. Agent considered alternatives in its own
thinking. User didn't hold a counter-stance.</output>
</example>
<example>
<input>User says "wait, isn't that going to break X?" Agent explains
why not. User: "oh you're right, never mind."</input>
<output>conflicts=[]. Question surfaced, answered, retracted. No
sustained position.</output>
</example>
<example>
<input>User leans toward "ship simple version now", agent leans toward
"wait for architectural cleanup". Session ends without a decision; user
says "let me think about it".</input>
<output>conflicts=[{stance_a: "Ship the simple version now to unblock
users.", stance_b: "Wait for the architectural cleanup so we don't ship
debt.", resolution: "unresolved"}]</output>
</example>
<example>
<input>Brief disagreement about which CI config to use. User pivots to
a different topic. Never returns to the CI question.</input>
<output>conflicts=[{stance_a: "Use GitHub Actions for the new pipeline.",
stance_b: "Stick with the existing CodeBuild setup.",
resolution: "abandoned"}]</output>
</example>
</examples>

<anti_patterns>
- Don't count collaboration as conflict. Agent proposes a plan, user
  agrees with caveats and the agent adapts. That's collaboration.
- Don't count agent deliberation. Agent considers two approaches in its
  own reasoning, then picks one with the user's blessing. That's
  deliberation, not conflict.
- Don't count surface-level pushback that the user immediately retracts.
- Don't count style / formatting disagreements ("I'd phrase that
  differently", "use semicolons not commas").
- Don't count accepted risk. Agent flags risk, user accepts it. That's
  a noted caveat, not a conflict.
- Don't count iteration. Two failed attempts at the same task (agent
  tried X, then Y). That's iteration, not conflict.
- Don't count tooling preferences without consequence ("I'd use jq here"
  vs "I'd use python -c").
</anti_patterns>
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


# ---------------------------------------------------------------------------
# Pipeline 1: session classification
# ---------------------------------------------------------------------------


async def _classify_sessions_async(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None,
    limit: int | None,
    thinking_mode: str,
) -> int:
    """Async implementation behind :func:`classify_sessions`."""
    already: set[str] = set()
    done_df = read_all(settings.classifications_parquet_path)
    if done_df is not None and done_df.height > 0:
        already = set(done_df["session_id"].to_list())

    # Checkpoint skip: compare current (last_ts, mtime) against the last run.
    bounds = session_bounds(con, since_days=since_days, limit=limit)
    unchanged_pending, skipped = checkpointer.filter_unchanged(
        ((sid, lt, mt) for sid, (lt, mt) in bounds.items()),
        pipeline="classify",
        checkpoint_db_path=settings.checkpoint_db_path,
    )
    keep = set(unchanged_pending)

    # Retry queue: pull pending retries first so they're re-enqueued into
    # `keep` even when the checkpoint would otherwise skip them.
    retry_ids = set(retry_queue.drain(settings.checkpoint_db_path, pipeline="classify"))
    if retry_ids:
        logger.info("classify: draining {} retry-queue entries", len(retry_ids))
        keep |= retry_ids

    pending: list[tuple[str, str]] = []
    for sid, text in iter_session_texts(con, settings=settings, since_days=since_days, limit=limit):
        if sid in already and sid not in retry_ids:
            continue
        if sid not in keep:
            continue
        pending.append((sid, text))

    if not pending:
        logger.info("classify: no pending sessions (skipped={} via checkpoint)", skipped)
        return 0
    if skipped:
        logger.info("classify: skipped {} sessions via checkpoint", skipped)

    client = _build_bedrock_client(settings)
    sem = anyio.CapacityLimiter(settings.llm_concurrency)
    chunk_size = max(settings.batch_size * 4, 256)
    logger.info(
        "classify: {} pending, model={}, thinking={}, concurrency={}, chunks of {}",
        len(pending),
        settings.sonnet_model_id,
        thinking_mode,
        settings.llm_concurrency,
        chunk_size,
    )

    written = 0
    for i in range(0, len(pending), chunk_size):
        chunk = pending[i : i + chunk_size]
        t0 = time.monotonic()
        coros = [
            _classify_one(
                client,
                settings.sonnet_model_id,
                SESSION_CLASSIFICATION_SCHEMA,
                text,
                max_tokens=settings.classify_max_tokens,
                thinking_mode=thinking_mode,
                sem=sem,
                system=CLASSIFY_SYSTEM_PROMPT,
            )
            for _, text in chunk
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        elapsed = time.monotonic() - t0

        now = datetime.now(UTC)
        ok_rows: list[dict[str, Any]] = []
        errors = 0
        for (sid, _), res in zip(chunk, results, strict=True):
            if isinstance(res, BaseException):
                errors += 1
                logger.warning("classify: {} failed (queued for retry): {}", sid, res)
                retry_queue.enqueue(
                    settings.checkpoint_db_path,
                    pipeline="classify",
                    unit_id=sid,
                    error=str(res),
                )
                continue
            res_dict: dict[str, Any] = res
            ok_rows.append(
                {
                    "session_id": sid,
                    "autonomy_tier": res_dict.get("autonomy_tier"),
                    "work_category": res_dict.get("work_category"),
                    "success": res_dict.get("success"),
                    "goal": res_dict.get("goal"),
                    "confidence": float(res_dict.get("confidence", 0.0)),
                    "classified_at": now,
                }
            )

        if ok_rows:
            df = pl.DataFrame(
                ok_rows,
                schema={
                    "session_id": pl.Utf8,
                    "autonomy_tier": pl.Utf8,
                    "work_category": pl.Utf8,
                    "success": pl.Utf8,
                    "goal": pl.Utf8,
                    "confidence": pl.Float32,
                    "classified_at": pl.Datetime("us", "UTC"),
                },
            )
            write_part(settings.classifications_parquet_path, df)

        # Checkpoint the sessions we just classified — at their CURRENT bounds,
        # so a later re-run with no new messages is a no-op. Also clear those
        # sessions from the retry queue.
        if ok_rows:
            ok_sids = [row["session_id"] for row in ok_rows]
            checkpointer.mark_completed(
                settings.checkpoint_db_path,
                pipeline="classify",
                rows=[(sid, *bounds.get(sid, (None, None))) for sid in ok_sids],
            )
            retry_queue.mark_done(
                settings.checkpoint_db_path,
                pipeline="classify",
                unit_ids=ok_sids,
            )

        written += len(ok_rows)
        logger.info(
            "classify chunk {}/{}: {} ok, {} errors, {:.1f}s ({:.1f} sess/s)",
            i // chunk_size + 1,
            (len(pending) + chunk_size - 1) // chunk_size,
            len(ok_rows),
            errors,
            elapsed,
            len(ok_rows) / elapsed if elapsed > 0 else 0,
        )

    logger.info("classify: wrote {} total rows", written)
    return written


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


def classify_sessions(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
) -> int | dict[str, Any]:
    """Classify pending sessions and return count of successful classifications.

    In ``--dry-run`` mode, returns a plan dict with keys ``{pipeline,
    candidates, llm_calls, avg_input_tokens, avg_output_tokens,
    estimated_cost_usd, model, thinking, since_days, limit}`` instead of the
    row count, so the CLI can emit it as structured JSON.
    """
    thinking_mode = "disabled" if no_thinking else settings.classify_thinking

    if dry_run:
        already: set[str] = set()
        done_df = read_all(settings.classifications_parquet_path)
        if done_df is not None and done_df.height > 0:
            already = set(done_df["session_id"].to_list())
        pending_count = _count_pending_sessions(
            con, already=already, since_days=since_days, limit=limit
        )
        # Back-of-envelope: avg 8K input tokens, 300 output per session.
        cost = _estimate_cost(pending_count, 8000, 300, settings.sonnet_pricing)
        logger.info(
            "classify --dry-run: {} sessions pending.  Estimated cost ~${:.2f} "
            "(thinking={}, model={})",
            pending_count,
            cost,
            thinking_mode,
            settings.sonnet_model_id,
        )
        return {
            "pipeline": "classify",
            "candidates": pending_count,
            "llm_calls": pending_count,
            "avg_input_tokens": 8000,
            "avg_output_tokens": 300,
            "estimated_cost_usd": round(cost, 4),
            "model": settings.sonnet_model_id,
            "thinking": thinking_mode,
            "since_days": since_days,
            "limit": limit,
            "dry_run": True,
        }

    return asyncio.run(
        _classify_sessions_async(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            thinking_mode=thinking_mode,
        )
    )


# ---------------------------------------------------------------------------
# Pipeline 2: message trajectory
# ---------------------------------------------------------------------------

# Cheap prefilter: short + starts with acknowledgement pattern -> is_transition, skip LLM.
_TRANSITION_RE = re.compile(
    r"^\s*(ok|okay|alright|now|let me|great[,!]?|sure|got it|sounds good|perfect|clean)\b",
    re.IGNORECASE,
)


def _heuristic_trajectory(text: str) -> dict | None:
    """Fast path -- return a result dict if confident, else None."""
    if not text:
        return None
    if len(text) < 80 and _TRANSITION_RE.match(text):
        return {"sentiment_delta": "neutral", "is_transition": True, "confidence": 0.9}
    return None


async def _trajectory_async(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None,
    limit: int | None,
    thinking_mode: str,
) -> int:
    """Async implementation behind :func:`trajectory_messages`."""
    already: set[str] = set()
    done_df = read_all(settings.trajectory_parquet_path)
    if done_df is not None and done_df.height > 0:
        already = set(done_df["uuid"].to_list())

    # Session-level checkpoint: drop messages whose host session has not advanced
    # since the last trajectory run. This cuts the per-message SQL down before
    # the anti-join on uuid.
    bounds = session_bounds(con, since_days=since_days, limit=limit)
    unchanged_pending, skipped_sessions = checkpointer.filter_unchanged(
        ((sid, lt, mt) for sid, (lt, mt) in bounds.items()),
        pipeline="trajectory",
        checkpoint_db_path=settings.checkpoint_db_path,
    )
    active_sessions: set[str] = set(unchanged_pending)

    # Retry queue: drain pending failed uuids into the `already`-bypass set
    # so they get retried even though they landed in the parquet the first
    # time they were attempted.
    retry_uuids = set(retry_queue.drain(settings.checkpoint_db_path, pipeline="trajectory"))
    if retry_uuids:
        logger.info("trajectory: draining {} retry-queue entries", len(retry_uuids))
        already -= retry_uuids

    where = ["mt.text_content IS NOT NULL", "length(mt.text_content) >= 1"]
    if since_days is not None:
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
    if active_sessions:
        where.append(
            "CAST(mt.session_id AS VARCHAR) IN (SELECT unnest(?))",
        )
    sql = f"""
        SELECT CAST(mt.uuid AS VARCHAR) AS uuid,
               CAST(mt.session_id AS VARCHAR) AS sid,
               mt.text_content
          FROM messages_text mt
         WHERE {" AND ".join(where)}
         ORDER BY mt.ts
    """
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    params = [list(active_sessions)] if active_sessions else []
    rows_raw = con.execute(sql, params).fetchall() if active_sessions or not bounds else []
    rows = [(r[0], r[2]) for r in rows_raw if r[0] not in already]
    session_for_uuid = {r[0]: r[1] for r in rows_raw if r[0] not in already}
    if skipped_sessions:
        logger.info(
            "trajectory: skipped {} sessions via checkpoint",
            skipped_sessions,
        )
    logger.info("trajectory: {} pending messages", len(rows))

    if not rows:
        logger.info("trajectory: wrote 0 total rows (nothing pending)")
        return 0

    heuristic_rows: list[dict[str, Any]] = []
    llm_pending: list[tuple[str, str]] = []
    now = datetime.now(UTC)
    for uuid, text in rows:
        fast = _heuristic_trajectory(text)
        if fast is not None:
            heuristic_rows.append({"uuid": uuid, **fast, "classified_at": now})
        else:
            llm_pending.append((uuid, text))

    logger.info(
        "trajectory: {} heuristic, {} LLM",
        len(heuristic_rows),
        len(llm_pending),
    )

    if heuristic_rows:
        df = pl.DataFrame(
            heuristic_rows,
            schema={
                "uuid": pl.Utf8,
                "sentiment_delta": pl.Utf8,
                "is_transition": pl.Boolean,
                "confidence": pl.Float32,
                "classified_at": pl.Datetime("us", "UTC"),
            },
        )
        write_part(settings.trajectory_parquet_path, df)

    processed_sessions: set[str] = set()
    for row in heuristic_rows:
        sid = session_for_uuid.get(row["uuid"])
        if sid is not None:
            processed_sessions.add(sid)

    if not llm_pending:
        if processed_sessions:
            checkpointer.mark_completed(
                settings.checkpoint_db_path,
                pipeline="trajectory",
                rows=[(sid, *bounds.get(sid, (None, None))) for sid in processed_sessions],
            )
        logger.info("trajectory: wrote {} total rows", len(heuristic_rows))
        return len(heuristic_rows)

    client = _build_bedrock_client(settings)
    sem = anyio.CapacityLimiter(settings.llm_concurrency)
    chunk_size = max(settings.batch_size * 4, 256)
    written = len(heuristic_rows)

    for i in range(0, len(llm_pending), chunk_size):
        chunk = llm_pending[i : i + chunk_size]
        t0 = time.monotonic()
        coros = [
            _classify_one(
                client,
                settings.sonnet_model_id,
                MESSAGE_TRAJECTORY_SCHEMA,
                text,
                max_tokens=settings.classify_max_tokens,
                thinking_mode=thinking_mode,
                sem=sem,
                system=TRAJECTORY_SYSTEM_PROMPT,
            )
            for _, text in chunk
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        now = datetime.now(UTC)

        ok: list[dict[str, Any]] = []
        ok_uuids: list[str] = []
        refused_uuids: list[str] = []
        errors = 0
        for (uuid, _), res in zip(chunk, results, strict=True):
            if isinstance(res, BedrockRefusalError):
                # Terminal: Bedrock won't classify this body. Stamp a neutral
                # placeholder so the session moves on and the retry queue
                # doesn't cycle forever on the same refusal.
                logger.info("trajectory: {} refused by Bedrock — marking neutral", uuid)
                now = datetime.now(UTC)
                ok.append(
                    {
                        "uuid": uuid,
                        "sentiment_delta": "neutral",
                        "is_transition": False,
                        "confidence": 0.0,
                        "classified_at": now,
                    }
                )
                refused_uuids.append(uuid)
                continue
            if isinstance(res, BaseException):
                errors += 1
                logger.warning("trajectory: {} failed (queued for retry): {}", uuid, res)
                retry_queue.enqueue(
                    settings.checkpoint_db_path,
                    pipeline="trajectory",
                    unit_id=uuid,
                    error=str(res),
                )
                continue
            res_dict: dict[str, Any] = res
            ok.append(
                {
                    "uuid": uuid,
                    "sentiment_delta": res_dict.get("sentiment_delta"),
                    "is_transition": bool(res_dict.get("is_transition", False)),
                    "confidence": float(res_dict.get("confidence", 0.0)),
                    "classified_at": now,
                }
            )
            ok_uuids.append(uuid)
            sid = session_for_uuid.get(uuid)
            if sid is not None:
                processed_sessions.add(sid)
        if ok:
            df = pl.DataFrame(
                ok,
                schema={
                    "uuid": pl.Utf8,
                    "sentiment_delta": pl.Utf8,
                    "is_transition": pl.Boolean,
                    "confidence": pl.Float32,
                    "classified_at": pl.Datetime("us", "UTC"),
                },
            )
            write_part(settings.trajectory_parquet_path, df)
            # Clear retry queue for both successful uuids AND refusals we just
            # neutralised — the refusal placeholder lives in the parquet now,
            # so these uuids must not loop back through the queue.
            done_uuids = ok_uuids + refused_uuids
            if done_uuids:
                retry_queue.mark_done(
                    settings.checkpoint_db_path,
                    pipeline="trajectory",
                    unit_ids=done_uuids,
                )
            # Per-chunk checkpoint: stamp sessions we've fully processed so a
            # mid-run crash doesn't lose the whole trajectory run.
            chunk_sessions = {session_for_uuid[u] for u in ok_uuids if u in session_for_uuid}
            if chunk_sessions:
                checkpointer.mark_completed(
                    settings.checkpoint_db_path,
                    pipeline="trajectory",
                    rows=[(sid, *bounds.get(sid, (None, None))) for sid in chunk_sessions],
                )
        written += len(ok)
        logger.info(
            "trajectory chunk {}/{}: {} ok, {} errors, {:.1f}s",
            i // chunk_size + 1,
            (len(llm_pending) + chunk_size - 1) // chunk_size,
            len(ok),
            errors,
            time.monotonic() - t0,
        )

    if processed_sessions:
        checkpointer.mark_completed(
            settings.checkpoint_db_path,
            pipeline="trajectory",
            rows=[(sid, *bounds.get(sid, (None, None))) for sid in processed_sessions],
        )
    logger.info("trajectory: wrote {} total rows", written)
    return written


def trajectory_messages(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
) -> int | dict[str, Any]:
    """Per-message sentiment + transition classification.

    In ``--dry-run`` mode returns a plan dict (see :func:`classify_sessions`).
    """
    thinking_mode = "disabled" if no_thinking else settings.trajectory_thinking
    if dry_run:
        where = ["mt.text_content IS NOT NULL"]
        if since_days is not None:
            where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
        if limit is not None:
            sql = (
                f"SELECT least({int(limit)}, count(*)) "
                f"FROM messages_text mt WHERE {' AND '.join(where)}"
            )
        else:
            sql = f"SELECT count(*) FROM messages_text mt WHERE {' AND '.join(where)}"
        row = con.execute(sql).fetchone()
        n = int(row[0]) if row is not None else 0
        # Roughly half survive heuristic pre-filter.
        llm_n = n // 2
        cost = _estimate_cost(llm_n, 500, 50, settings.sonnet_pricing)
        logger.info(
            "trajectory --dry-run: {} messages, estimated LLM cost ~${:.2f}",
            n,
            cost,
        )
        return {
            "pipeline": "trajectory",
            "candidates": n,
            "llm_calls": llm_n,
            "avg_input_tokens": 500,
            "avg_output_tokens": 50,
            "estimated_cost_usd": round(cost, 4),
            "model": settings.sonnet_model_id,
            "thinking": thinking_mode,
            "since_days": since_days,
            "limit": limit,
            "dry_run": True,
        }
    return asyncio.run(
        _trajectory_async(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            thinking_mode=thinking_mode,
        )
    )


# ---------------------------------------------------------------------------
# Pipeline 3: conflict detection
# ---------------------------------------------------------------------------


async def _conflicts_async(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None,
    limit: int | None,
    thinking_mode: str,
) -> int:
    """Async implementation behind :func:`detect_conflicts`."""
    already: set[str] = set()
    done_df = read_all(settings.conflicts_parquet_path)
    if done_df is not None and done_df.height > 0:
        already = set(done_df["session_id"].to_list())

    bounds = session_bounds(con, since_days=since_days, limit=limit)
    unchanged_pending, skipped = checkpointer.filter_unchanged(
        ((sid, lt, mt) for sid, (lt, mt) in bounds.items()),
        pipeline="conflicts",
        checkpoint_db_path=settings.checkpoint_db_path,
    )
    keep = set(unchanged_pending)

    retry_ids = set(retry_queue.drain(settings.checkpoint_db_path, pipeline="conflicts"))
    if retry_ids:
        logger.info("conflicts: draining {} retry-queue entries", len(retry_ids))
        keep |= retry_ids

    pending: list[tuple[str, str]] = []
    for sid, text in iter_session_texts(con, settings=settings, since_days=since_days, limit=limit):
        if sid in already and sid not in retry_ids:
            continue
        if sid not in keep:
            continue
        pending.append((sid, text))

    if not pending:
        logger.info("conflicts: no pending sessions (skipped={} via checkpoint)", skipped)
        return 0
    if skipped:
        logger.info("conflicts: skipped {} sessions via checkpoint", skipped)

    client = _build_bedrock_client(settings)
    sem = anyio.CapacityLimiter(settings.llm_concurrency)
    chunk_size = max(settings.batch_size * 4, 256)
    logger.info("conflicts: {} pending sessions", len(pending))

    written = 0
    for i in range(0, len(pending), chunk_size):
        chunk = pending[i : i + chunk_size]
        t0 = time.monotonic()
        coros = [
            _classify_one(
                client,
                settings.sonnet_model_id,
                SESSION_CONFLICTS_SCHEMA,
                text,
                max_tokens=settings.classify_max_tokens,
                thinking_mode=thinking_mode,
                sem=sem,
                system=CONFLICTS_SYSTEM_PROMPT,
            )
            for _, text in chunk
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        now = datetime.now(UTC)

        rows: list[dict[str, Any]] = []
        errors = 0
        for (sid, _), res in zip(chunk, results, strict=True):
            if isinstance(res, BaseException):
                errors += 1
                logger.warning("conflicts: {} failed (queued for retry): {}", sid, res)
                retry_queue.enqueue(
                    settings.checkpoint_db_path,
                    pipeline="conflicts",
                    unit_id=sid,
                    error=str(res),
                )
                continue
            res_dict: dict[str, Any] = res
            conflicts = res_dict.get("conflicts") or []
            if not conflicts:
                # Write a sentinel row so we don't re-classify this session.
                rows.append(
                    {
                        "session_id": sid,
                        "conflict_idx": 0,
                        "stance_a": None,
                        "stance_b": None,
                        "resolution": None,
                        "detected_at": now,
                        "empty": True,
                    }
                )
                continue
            for idx, c in enumerate(conflicts):
                rows.append(
                    {
                        "session_id": sid,
                        "conflict_idx": idx,
                        "stance_a": c.get("stance_a"),
                        "stance_b": c.get("stance_b"),
                        "resolution": c.get("resolution"),
                        "detected_at": now,
                        "empty": False,
                    }
                )
        if rows:
            df = pl.DataFrame(
                rows,
                schema={
                    "session_id": pl.Utf8,
                    "conflict_idx": pl.Int32,
                    "stance_a": pl.Utf8,
                    "stance_b": pl.Utf8,
                    "resolution": pl.Utf8,
                    "detected_at": pl.Datetime("us", "UTC"),
                    "empty": pl.Boolean,
                },
            )
            write_part(settings.conflicts_parquet_path, df)
        ok_sids = {
            sid
            for (sid, _t), r in zip(chunk, results, strict=True)
            if not isinstance(r, BaseException)
        }
        if ok_sids:
            checkpointer.mark_completed(
                settings.checkpoint_db_path,
                pipeline="conflicts",
                rows=[(sid, *bounds.get(sid, (None, None))) for sid in ok_sids],
            )
            retry_queue.mark_done(
                settings.checkpoint_db_path,
                pipeline="conflicts",
                unit_ids=list(ok_sids),
            )
        written += len(ok_sids)
        logger.info(
            "conflicts chunk {}/{}: {} sessions processed, {} errors, {:.1f}s",
            i // chunk_size + 1,
            (len(pending) + chunk_size - 1) // chunk_size,
            len(chunk) - errors,
            errors,
            time.monotonic() - t0,
        )

    logger.info("conflicts: processed {} sessions", written)
    return written


def detect_conflicts(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
) -> int | dict[str, Any]:
    """Detect stance conflicts per session and return count processed.

    In ``--dry-run`` mode returns a plan dict (see :func:`classify_sessions`).
    """
    thinking_mode = "disabled" if no_thinking else settings.classify_thinking
    if dry_run:
        already: set[str] = set()
        done_df = read_all(settings.conflicts_parquet_path)
        if done_df is not None and done_df.height > 0:
            already = set(done_df["session_id"].to_list())
        pending_count = _count_pending_sessions(
            con, already=already, since_days=since_days, limit=limit
        )
        cost = _estimate_cost(pending_count, 6000, 400, settings.sonnet_pricing)
        logger.info(
            "conflicts --dry-run: {} sessions, estimated cost ~${:.2f}",
            pending_count,
            cost,
        )
        return {
            "pipeline": "conflicts",
            "candidates": pending_count,
            "llm_calls": pending_count,
            "avg_input_tokens": 6000,
            "avg_output_tokens": 400,
            "estimated_cost_usd": round(cost, 4),
            "model": settings.sonnet_model_id,
            "thinking": thinking_mode,
            "since_days": since_days,
            "limit": limit,
            "dry_run": True,
        }
    return asyncio.run(
        _conflicts_async(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            thinking_mode=thinking_mode,
        )
    )
