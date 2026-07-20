"""Per-session windowed trajectory pipeline (RFC 0002 §3.4 / §4.1).

Each session contributes one *window* per text turn — pairing the prior
turn (``prev_uuid``) with the current turn (``curr_uuid``). The first window
in a session has ``prev_uuid IS NULL`` plus a synthetic ``prev_sentiment``
of ``neutral`` so every text turn gets exactly one row in the output
parquet.

Sonnet 4.6 receives all windows for a session in one request body, batched
into chunks of ≤16 windows. For sessions longer than 16 text turns we
split into multiple chunks; consecutive chunks share an *anchor turn*
(chunk N's last window's ``curr_uuid`` equals chunk N+1's first window's
``prev_uuid``) so the model's context never breaks across chunks.

The model returns :class:`TrajectoryArrayResult` — an array of
:class:`TrajectoryWindow`. The host pipeline verifies completeness by
matching the returned ``(prev_uuid, curr_uuid)`` tuples against the
requested ones. Missing windows trigger ONE bounded retry of just the
missing windows; persistent misses are stamped with neutral
placeholder rows so a single refusing chunk never wedges the pipeline.

This is the v1.0 windowed rewrite — the v0.x per-message schema
``(uuid, sentiment_delta, is_transition, confidence, classified_at)`` is
gone. Stale per-message parquet shards under
``settings.trajectory_parquet_path`` are deleted on first run via an mtime
check so the analytics view always sees the new shape.

Moved here from ``analytics/trajectory_worker.py`` in the v2 hexagonal
reshape (MIGRATION Phase C). The pure delta/XML/window math lives in
:mod:`claude_sql.domain.trajectory`; all Bedrock plumbing — client
construction, retry, structured-output parsing, the per-pipeline cache-stat
accumulator — lives in :mod:`claude_sql.infrastructure.bedrock.client`.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import polars as pl
import pyarrow as pa
from loguru import logger

from claude_sql.domain.costs import estimate_cost as _estimate_cost
from claude_sql.domain.errors import BedrockRefusalError
from claude_sql.domain.models import TrajectoryArrayResult
from claude_sql.domain.trajectory import (
    _SENTIMENT_VAL,
    MAX_WINDOWS_PER_CHUNK,
    TRANSITION_KINDS,
    _build_row,
    _chunk_windows,
    _delta_value,
    _format_chunk_xml,
    _missing_keys,
    _placeholder_row,
    _xml_attr,
    _xml_text,
)
from claude_sql.infrastructure.adapters import build_cache, build_checkpoint, build_retry_queue
from claude_sql.infrastructure.bedrock.client import pipeline_cache_stats
from claude_sql.infrastructure.llm_analytics import build_llm_analytics_provider
from claude_sql.infrastructure.parquet_cache import iter_part_files
from claude_sql.infrastructure.session_text_loader import session_bounds
from claude_sql.infrastructure.sqlite_state import checkpointer

if TYPE_CHECKING:
    import duckdb

    from claude_sql.application.ports import (
        CachePort,
        CheckpointPort,
        LlmAnalyticsProvider,
        RetryQueuePort,
        TranscriptReaderPort,
    )
    from claude_sql.infrastructure.settings import Settings

# Re-exported pure helpers are listed here so the plain re-import above is
# marked used (ruff F401). The domain module is the source of truth; this
# module composes those helpers into the DuckDB/Bedrock orchestration and the
# The pre-hexagonal ``analytics.trajectory_worker`` shim (now deleted) aliased onto this module so the historic
# import path keeps resolving every one of these names.
__all__ = [
    "MAX_WINDOWS_PER_CHUNK",
    "TRAJECTORY_SYSTEM_PROMPT",
    "TRANSITION_KINDS",
    "_SENTIMENT_VAL",
    "_build_row",
    "_chunk_windows",
    "_classify_chunk",
    "_delta_value",
    "_format_chunk_xml",
    "_load_windows",
    "_missing_keys",
    "_placeholder_row",
    "_purge_old_shards",
    "_stale_old_shape",
    "_xml_attr",
    "_xml_text",
    "trajectory_messages",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Name of the transient Arrow relation that holds the in-window session-id
#: set for ``_load_windows``. Registered once per call and JOINed against
#: instead of binding the id list as a ``?`` parameter — DuckDB's Python
#: client probes ``import pandas`` ~twice per bound list element, and pandas
#: is not a dependency, so each probe is a full failed ``sys.path`` scan
#: (~3.4K failed imports on a cold-rebuild ~1.7K-session corpus). Mirrors the
#: ``session_text`` fix (#95). See ``_load_windows`` for the JOIN site.
_SID_FILTER_RELATION = "_trajectory_window_sid_filter"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
#
# Padded past the Sonnet 4.6 cache floor (2048 input tokens) so the
# ``cache_control: ttl=1h`` block on the system prompt actually triggers
# Anthropic prompt caching. cl100k_base × 0.78 ≈ Anthropic-token count;
# the constant is locked in at >2700 cl100k tokens (~2100 Anthropic),
# verified by ``test_trajectory_windowed.test_system_prompt_clears_cache_floor``.

TRAJECTORY_SYSTEM_PROMPT = """\
<instructions>
You score the emotional polarity arc across pairs of adjacent text turns
inside ONE Claude Code coding session. The user message contains the full
chunk: an ordered list of <window idx=N> XML blocks, each with a <prev>
text turn and a <curr> text turn from the same session.

Emit exactly one JSON object matching the schema:

{
  "windows": [
    {
      "prev_uuid": "<echo from <prev uuid='...'>; null on session-first window>",
      "curr_uuid": "<echo from <curr uuid='...'>",
      "prev_sentiment": "negative" | "neutral" | "positive" | null,
      "curr_sentiment": "negative" | "neutral" | "positive",
      "delta": -2 | -1 | 0 | 1 | 2 | null,
      "is_transition": true | false,
      "transition_kind": "frustration_spike" | "resolution" | "reset" | "drift" | "clarification" | "none",
      "confidence": 0.0..1.0
    }, ...
  ]
}

Output JSON only. No surrounding prose, no markdown fences. The host
pipeline parses your output with a strict JSON Schema validator —
missing fields, wrong types, or unknown enum values fail the row.
</instructions>

<context>
Each <window> represents two adjacent text turns from one session. The
ordering inside a chunk reflects chronological order. When a window has
no <prev> (the session-first window), set prev_uuid=null,
prev_sentiment=null, delta=null, transition_kind="none" unless curr
itself is a salient frustration_spike or resolution opening.

Sentiment labels (applied to a single turn):

- positive — excitement, approval, momentum, explicit thanks beyond
  politeness ("nice!", "love this", "shipping it", "huge win", "perfect,
  exactly what I needed").
- neutral — factual, procedural, acknowledgement, plain instruction,
  plain question. THIS IS THE MAJORITY CLASS. Coding sessions are
  ~70% neutral. "Tests pass.", "Run the linter.", "Where does X live?",
  "ok let me check", "running pytest" — all neutral.
- negative — frustration, pushback, blocked, sharp correction ("ugh",
  "seriously?", "this entire approach is wrong", "no don't do that",
  "you keep messing this up", "I'm stuck").

delta encoding (curr - prev, integer):

  prev          curr          delta
  --------      --------      -----
  negative      negative       0
  negative      neutral       +1
  negative      positive      +2
  neutral       negative      -1
  neutral       neutral        0
  neutral       positive      +1
  positive      negative      -2
  positive      neutral       -1
  positive      positive       0
  null  (session-first)        null

transition_kind labels (six):

- frustration_spike — prev is neutral/positive, curr is negative, AND
  the negative is salient (visible affect, not just a curt instruction).
  Example: agent reports "tests pass", user replies "no they don't,
  you're looking at the wrong file".
- resolution — prev is negative, curr is neutral or positive, AND there's
  evidence the underlying problem moved. Example: long debugging back-
  and-forth, user finally replies "got it, that fix works, thanks".
- reset — abrupt topic change, prev and curr discuss different subjects
  with no narrative bridge. Example: prev was about CI configuration,
  curr is "actually let's switch gears, draft a PR-FAQ for X".
- drift — same polarity, related sub-topic but a clear evolution.
  Example: prev was about adding test coverage to module A, curr is
  about extending coverage to a related module B.
- clarification — curr restates, refines, or narrows prev's substance.
  Example: user asked a vague question, then immediately re-asks with
  a concrete file path or constraint added.
- none — DEFAULT. Use this when no transition_kind clearly fits, when
  prev and curr are both routine procedural turns, or when the session-
  first window has no prior context. Most windows will be "none".

is_transition (per-row boolean): True when the *current* turn (curr) is
pure filler / acknowledgement with no substantive content. "ok", "running
tests", "done.", "got it, moving on" — all transitions. Independent of
transition_kind: a window can have transition_kind="none" and still have
is_transition=true if the curr turn is just filler.
</context>

<calibration>
- Use confidence < 0.5 when the cue is ambiguous (single-word turns,
  mixed signals, missing prev context).
- Use confidence > 0.85 only when an explicit affect cue (curse word,
  exclamation, "perfect", "ugh") makes the polarity unambiguous.
- The downstream pipeline weights by confidence — honesty pays.
- Do NOT manufacture "slightly positive" or "mildly negative" labels.
  Three-class output: pick the closest one, lower confidence on the
  borderline.
- "thanks" alone is neutral, not positive. Bare politeness is pacing.
- "ok" / "ok let me check" / "running" are neutral with is_transition=true.
- Tool-use narration ("calling X", "reading Y") is neutral.
- A long technical turn is not necessarily neutral — affect lives in
  the words, not the length. "this entire approach is wrong because…"
  stays negative even at 200 chars.
</calibration>

<examples>
<example>
<input>
<window idx=0>
<prev role="user" uuid="u1">tests pass</prev>
<curr role="user" uuid="u2">no they don't, you're reading the wrong file — look at tests/test_auth.py</curr>
</window>
</input>
<output>{"windows":[{"prev_uuid":"u1","curr_uuid":"u2","prev_sentiment":"neutral","curr_sentiment":"negative","delta":-1,"is_transition":false,"transition_kind":"frustration_spike","confidence":0.9}]}</output>
</example>
<example>
<input>
<window idx=0>
<prev role="user" uuid="u3">I'm stuck on this — the migration keeps failing the same way</prev>
<curr role="user" uuid="u4">got it, that's exactly what I needed — the rotator change is the missing piece</curr>
</window>
</input>
<output>{"windows":[{"prev_uuid":"u3","curr_uuid":"u4","prev_sentiment":"negative","curr_sentiment":"positive","delta":2,"is_transition":false,"transition_kind":"resolution","confidence":0.9}]}</output>
</example>
<example>
<input>
<window idx=0>
<prev role="user" uuid="u5">add a test for the empty-input case in module A</prev>
<curr role="user" uuid="u6">also add the same coverage to module B while you're at it</curr>
</window>
</input>
<output>{"windows":[{"prev_uuid":"u5","curr_uuid":"u6","prev_sentiment":"neutral","curr_sentiment":"neutral","delta":0,"is_transition":false,"transition_kind":"drift","confidence":0.85}]}</output>
</example>
<example>
<input>
<window idx=0>
<prev role="user" uuid="u7">running the linter</prev>
<curr role="user" uuid="u8">ok let me check that</curr>
</window>
</input>
<output>{"windows":[{"prev_uuid":"u7","curr_uuid":"u8","prev_sentiment":"neutral","curr_sentiment":"neutral","delta":0,"is_transition":true,"transition_kind":"none","confidence":0.9}]}</output>
</example>
<example>
<input>
<window idx=0>
<prev role="user" uuid=""></prev>
<curr role="user" uuid="u9">implement Phase 2 of the auth migration end-to-end</curr>
</window>
</input>
<output>{"windows":[{"prev_uuid":null,"curr_uuid":"u9","prev_sentiment":null,"curr_sentiment":"neutral","delta":null,"is_transition":false,"transition_kind":"none","confidence":0.9}]}</output>
</example>
<example>
<input>
<window idx=0>
<prev role="user" uuid="u10">where does the auth config live?</prev>
<curr role="user" uuid="u11">specifically the rotator config — under src/auth/?</curr>
</window>
</input>
<output>{"windows":[{"prev_uuid":"u10","curr_uuid":"u11","prev_sentiment":"neutral","curr_sentiment":"neutral","delta":0,"is_transition":false,"transition_kind":"clarification","confidence":0.85}]}</output>
</example>
<example>
<input>
<window idx=0>
<prev role="user" uuid="u12">we just shipped the rotator update — looks clean</prev>
<curr role="user" uuid="u13">switching gears: draft a PR-FAQ for the new dashboard launch</curr>
</window>
</input>
<output>{"windows":[{"prev_uuid":"u12","curr_uuid":"u13","prev_sentiment":"positive","curr_sentiment":"neutral","delta":-1,"is_transition":false,"transition_kind":"reset","confidence":0.85}]}</output>
</example>
</examples>

<anti_patterns>
- Do NOT echo back a prev_uuid that wasn't in the request. Bind exactly
  to the (prev_uuid, curr_uuid) tuples supplied in the <window> blocks.
  The host pipeline verifies completeness by uuid-pair echo; making up
  uuids breaks the verification step and triggers a costly retry.
- Do NOT skip windows. If the chunk has 12 <window> blocks, return 12
  TrajectoryWindow objects. Missing entries trigger a retry that costs
  another full Sonnet call.
- Do NOT pick transition_kind to inject narrative drama. Most windows
  are "none". A long session has at most a handful of frustration_spike
  / resolution windows; if you find yourself emitting frustration_spike
  on more than ~10% of windows, you're over-classifying.
- Do NOT confuse is_transition with transition_kind="none". They are
  independent axes: is_transition tags a filler/acknowledgement turn,
  transition_kind tags the prev→curr arc.
- Do NOT round confidence to 1.0. The downstream pipeline uses
  confidence as a weight; saturating at 1.0 erases the calibration
  signal. 0.9 means "very sure"; 0.95+ should be reserved for
  unambiguous explicit cues.
- Do NOT recompute delta in your head differently from the table above.
  delta is mechanical: encode prev (-1/0/+1), encode curr (-1/0/+1),
  subtract. The schema accepts {-2,-1,0,1,2,null}; anything else fails
  validation.
- Do NOT treat agent-role turns. Only user-role turns appear in the
  windows. The role attribute is informational; you score the same way
  regardless of role.
- Do NOT add commentary fields. The schema has exactly seven keys per
  window plus the outer "windows" array. Bedrock's structured-output
  validator rejects additional fields.
</anti_patterns>

<operating_context>
You run offline against a snapshot of Claude Code transcripts already on
disk. There is no live user to clarify with — commit to one output for
each chunk. The downstream pipeline writes your output to a parquet file
used by SQL views and analytics macros; future you (or a human auditor)
reads these rows in aggregate, not in isolation. Idempotence matters:
the same input must produce the same output across runs. Don't introduce
randomness or invent details that aren't in the input.

Failure mode: if a window's polarity is genuinely undecidable, set
curr_sentiment="neutral", transition_kind="none", and confidence below
0.5. Do not refuse the chunk — every requested (prev_uuid, curr_uuid)
must appear in your "windows" array, even if low-confidence.
</operating_context>
"""


# ---------------------------------------------------------------------------
# Old-schema cleanup
# ---------------------------------------------------------------------------


def _stale_old_shape(parts: list[Path]) -> bool:
    """Return True iff any part file uses the old per-message schema.

    The old schema had columns ``(uuid, sentiment_delta, is_transition,
    confidence, classified_at)`` — no ``session_id`` column, no
    ``curr_uuid``. We detect via parquet metadata, not by reading rows,
    so the check is cheap on the live corpus.
    """
    if not parts:
        return False
    try:
        import pyarrow.parquet as pq

        for p in parts:
            schema = pq.ParquetFile(str(p)).schema
            names = set(schema.names)
            # New shape MUST have curr_uuid; old shape has bare uuid + sentiment_delta.
            if "curr_uuid" in names:
                return False
            if "sentiment_delta" in names and "uuid" in names:
                return True
    except (OSError, ValueError) as exc:
        logger.warning("trajectory: parquet metadata probe failed ({}); treating as fresh", exc)
    return False


def _purge_old_shards(target: Path) -> int:
    """Delete every legacy per-message shard under ``target``.

    Called once per pipeline run. Returns the number of files removed.
    Idempotent — safe to call when nothing legacy remains.
    """
    parts = iter_part_files(target)
    if not _stale_old_shape(parts):
        return 0
    removed = 0
    for p in parts:
        try:
            p.unlink()
            removed += 1
        except OSError as exc:
            # If a path can't be removed we still continue; the new write
            # will land in a new part-<ts>.parquet alongside the stale one,
            # but the analytics view will fail to bind columns. Surface the
            # error so the operator notices.
            logger.warning("trajectory: failed to delete legacy shard {}: {}", p, exc)
    if removed:
        logger.info("trajectory: purged {} legacy per-message shard(s) under {}", removed, target)
    return removed


# ---------------------------------------------------------------------------
# Window assembly
# ---------------------------------------------------------------------------


def _load_windows(
    con: duckdb.DuckDBPyConnection,
    *,
    active_sessions: set[str] | None,
    since_days: int | None,
    limit: int | None,
) -> list[tuple[str, str | None, str, str | None, str, str | None, str]]:
    """Return windowed turns for the requested sessions.

    Each row is ``(session_id, prev_uuid, curr_uuid, prev_role, curr_role,
    prev_text, curr_text)`` ordered by ``(session_id, window_idx)``.
    The first window per session has ``prev_uuid IS NULL``.

    Reads ``turn_window`` (registered by ``register_views``) joined onto
    ``messages_text`` for the prev/curr text bodies. Compact-summary rows
    are excluded inside the view itself.
    """
    # Filter the session window via a JOIN onto an Arrow relation rather than
    # binding ``active_sessions`` as a ``?`` list parameter. On a cold/forced
    # rebuild ``active_sessions`` is the whole corpus (~1.7K+ ids), and a bound
    # Python list makes DuckDB's client probe ``import pandas`` ~twice per
    # element (a failed ``sys.path`` scan, since pandas is not a dependency) —
    # ~3.4K wasted imports here. Registering the ids once as an Arrow relation
    # and JOINing drops that to a single probe. Byte-identical output (ordered
    # and multiset) verified on the live corpus. Mirrors #95 in ``session_text``.
    where = ["1=1"]
    if since_days is not None:
        where.append(f"tw.curr_ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
    if active_sessions:
        where.append(f"CAST(tw.session_id AS VARCHAR) IN (SELECT sid FROM {_SID_FILTER_RELATION})")
    sql = f"""
        SELECT CAST(tw.session_id AS VARCHAR) AS session_id,
               tw.prev_uuid,
               tw.curr_uuid,
               tw.prev_role,
               tw.curr_role,
               mt_prev.text_content AS prev_text,
               mt_curr.text_content AS curr_text,
               tw.window_idx
          FROM turn_window tw
          LEFT JOIN messages_text mt_prev ON CAST(mt_prev.uuid AS VARCHAR) = tw.prev_uuid
          LEFT JOIN messages_text mt_curr ON CAST(mt_curr.uuid AS VARCHAR) = tw.curr_uuid
         WHERE {" AND ".join(where)}
         ORDER BY tw.session_id, tw.window_idx
    """
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    if active_sessions:
        sid_filter = pa.table({"sid": pa.array(list(active_sessions), type=pa.string())})
        con.register(_SID_FILTER_RELATION, sid_filter)
        try:
            rows = con.execute(sql).fetchall()
        finally:
            con.unregister(_SID_FILTER_RELATION)
    else:
        rows = con.execute(sql).fetchall()
    out: list[tuple[str, str | None, str, str | None, str, str | None, str]] = []
    for sid, prev_uuid, curr_uuid, prev_role, curr_role, prev_text, curr_text, _idx in rows:
        if curr_uuid is None or curr_text is None:
            continue
        out.append(
            (
                sid,
                prev_uuid,
                curr_uuid,
                prev_role,
                curr_role,
                prev_text,
                curr_text,
            )
        )
    return out


# Polars schema for the new parquet shape — load-bearing across reruns
# (parquet column types must match for the analytics view to bind).
_PARQUET_SCHEMA: dict[str, Any] = {
    "session_id": pl.Utf8,
    "prev_uuid": pl.Utf8,
    "curr_uuid": pl.Utf8,
    "prev_sentiment": pl.Utf8,
    "curr_sentiment": pl.Utf8,
    "delta": pl.Float64,
    "is_transition": pl.Boolean,
    "transition_kind": pl.Utf8,
    "confidence": pl.Float32,
    "classified_at": pl.Datetime("us", "UTC"),
}


# ---------------------------------------------------------------------------
# Sonnet dispatch
# ---------------------------------------------------------------------------


async def _classify_chunk(
    provider: LlmAnalyticsProvider,
    *,
    chunk: list[tuple[str, str | None, str, str | None, str, str | None, str]],
) -> dict[tuple[str | None, str], dict[str, Any]] | BedrockRefusalError | Exception:
    """Send one chunk through the LLM-analytics provider; return per-(prev,curr)
    result map or the raw exception.

    The structured-output call goes through ``provider.classify_structured``
    (Wave D): the SAME worker runs under either the default Sonnet-on-Bedrock
    adapter or the opt-in GPT-5.6-Luna adapter, so this function is
    provider-agnostic: it only knows the ``TrajectoryArrayResult`` schema.

    The schema reminder is concatenated into the user prompt alongside the XML
    payload. The system block (byte-stable task framing, 1h-cached on the Sonnet
    path / Mantle-cached on the Luna path) carries the heavy calibration; the
    reminder here just pins the output shape so the model doesn't drift to a
    single-window object.

    Terminal failures are RETURNED (not raised) so the caller's dispatch loop
    routes them: a :class:`BedrockRefusalError` (Sonnet content-policy refusal)
    becomes neutral placeholders; any other exception (including a Luna
    :class:`~claude_sql.domain.errors.LlmAnalyticsUnavailable` (fail-open
    "provider unavailable")) is enqueued for a later run.
    """
    schema_reminder = (
        "You will receive one or more <window> blocks. For EACH <window> block, "
        "emit one TrajectoryWindow object inside the top-level "
        '"windows" array. Echo the exact (prev_uuid, curr_uuid) from each '
        "window in your response — the host pipeline verifies completeness "
        "by uuid-pair match. Output JSON only, schema-conformant, no prose."
    )
    payload_xml = _format_chunk_xml(chunk)
    user_text = f"{schema_reminder}\n\n{payload_xml}"
    try:
        result = await provider.classify_structured(
            system=TRAJECTORY_SYSTEM_PROMPT,
            prompt=user_text,
            schema=TrajectoryArrayResult,
        )
    except BedrockRefusalError as exc:
        return exc
    except Exception as exc:  # noqa: BLE001 — propagate to caller's retry/skip logic; CancelledError still cancels the task group
        return exc
    indexed: dict[tuple[str | None, str], dict[str, Any]] = {}
    for win in result.windows:
        win_dict = win.model_dump()
        key = (win_dict.get("prev_uuid"), win_dict.get("curr_uuid"))
        if key[1] is None:
            continue
        indexed[key] = win_dict
    return indexed


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def _trajectory_async(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None,
    limit: int | None,
    thinking_mode: str,
    cache: CachePort,
    checkpoint: CheckpointPort,
    retry: RetryQueuePort,
    reader: TranscriptReaderPort | None,
) -> int:
    """Async implementation behind :func:`trajectory_messages`."""
    # Step 1: purge any stale per-message-shape shards left over from v0.x.
    _purge_old_shards(settings.trajectory_parquet_path)

    # Step 2: session-level checkpoint — only sessions whose transcripts
    # have advanced since the last run go through. ``session_bounds`` stays a
    # direct local call by default (monkeypatch seam).
    bounds = (
        reader.session_bounds(since_days=since_days, limit=limit)
        if reader is not None
        else session_bounds(con, since_days=since_days, limit=limit)
    )
    unchanged_pending, skipped_sessions = checkpointer.filter_unchanged(
        ((sid, lt, mt) for sid, (lt, mt) in bounds.items()),
        pipeline="trajectory",
        checkpoint_db_path=settings.checkpoint_db_path,
    )
    active_sessions: set[str] = set(unchanged_pending)

    # Retry queue: drain pending failed sessions so they get another shot.
    retry_sids = set(retry.drain(pipeline="trajectory"))
    if retry_sids:
        logger.info("trajectory: draining {} retry-queue entries", len(retry_sids))
        active_sessions |= retry_sids

    if skipped_sessions:
        logger.info("trajectory: skipped {} sessions via checkpoint", skipped_sessions)

    if not active_sessions and not bounds:
        logger.info("trajectory: no sessions in window")
        return 0

    # Step 3: load all windows for the active sessions.
    raw_rows = _load_windows(
        con,
        active_sessions=active_sessions,
        since_days=since_days,
        limit=limit,
    )
    if not raw_rows:
        logger.info("trajectory: 0 windows pending after filtering")
        return 0

    # Group by session to chunk per-session (anchor-sharing requires
    # contiguous windows from the same session in chunk order).
    by_session: dict[str, list[Any]] = defaultdict(list)
    for row in raw_rows:
        by_session[row[0]].append(row)

    logger.info(
        "trajectory: {} sessions, {} total windows pending",
        len(by_session),
        len(raw_rows),
    )

    # Build the structured-output provider ONCE per run (Wave D). The default
    # ``sonnet-bedrock`` adapter owns the cached bedrock-runtime client AND a
    # single ``anyio.CapacityLimiter(settings.llm_concurrency)`` shared across
    # every chunk, identical throttling to the pre-seam inline client+limiter.
    # The opt-in ``strands-luna`` adapter fails open (LlmAnalyticsUnavailable)
    # so a Luna outage degrades this run instead of crashing the pipeline.
    provider = build_llm_analytics_provider(settings, thinking_mode=thinking_mode)
    logger.info("trajectory: using llm-analytics provider {!r}", provider.provider)
    # Shared mutable state across concurrent session tasks. Mutations to
    # ``written`` / ``processed_sessions`` AND the parquet shard write are
    # serialized under ``write_lock``. The structured-output call itself is NOT
    # under this lock; the provider's own CapacityLimiter does the throttling,
    # capping in-flight chunks at ``settings.llm_concurrency``.
    write_lock = anyio.Lock()
    written_box = [0]  # boxed so the closure can mutate it
    processed_sessions: set[str] = set()

    # Step 4: per-session worker. One task per session; chunks within a
    # session run sequentially (per-chunk anchor caching makes
    # within-session sequential cheaper than within-session parallel —
    # RFC 0002 §4.1). The CapacityLimiter throttles per-CALL, so a
    # multi-chunk session contends for the same N slots as every other
    # in-flight chunk across the task group. Net concurrency for the
    # Bedrock layer is exactly ``settings.llm_concurrency``.
    async def _process_session(sid: str, session_windows: list[Any]) -> None:
        chunks = _chunk_windows(session_windows)
        all_rows: list[dict[str, Any]] = []
        session_failed = False

        try:
            for chunk_idx, chunk in enumerate(chunks):
                t0 = time.monotonic()
                res = await _classify_chunk(provider, chunk=chunk)
                now = datetime.now(UTC)

                if isinstance(res, BedrockRefusalError):
                    logger.info(
                        "trajectory: chunk {}/{} of session {} refused — neutral placeholders",
                        chunk_idx + 1,
                        len(chunks),
                        sid,
                    )
                    all_rows.extend(_placeholder_row(sid, row[1], row[2], now) for row in chunk)
                    continue
                if isinstance(res, Exception):
                    logger.warning(
                        "trajectory: chunk {}/{} of session {} failed ({}); enqueuing for retry",
                        chunk_idx + 1,
                        len(chunks),
                        sid,
                        res,
                    )
                    # retry.enqueue uses sqlite WAL — concurrent writers
                    # are safe per .erpaval/solutions/best-practices/
                    # sqlite-wal-cold-start-pragma-race.md.
                    retry.enqueue(pipeline="trajectory", unit_id=sid, error=str(res))
                    session_failed = True
                    break

                indexed = res
                missing = _missing_keys(chunk, indexed)

                # Bounded retry: re-request only the missing windows in one
                # additional structured-output call. Persistent misses become
                # placeholders.
                if missing:
                    logger.info(
                        "trajectory: chunk {}/{} of session {} missing {}/{} windows — retrying",
                        chunk_idx + 1,
                        len(chunks),
                        sid,
                        len(missing),
                        len(chunk),
                    )
                    retry_res = await _classify_chunk(provider, chunk=missing)
                    now = datetime.now(UTC)
                    if isinstance(retry_res, dict):
                        for key, win in retry_res.items():
                            indexed[key] = win
                    # Anything still missing after the retry → neutral placeholder.
                    still_missing = _missing_keys(chunk, indexed)
                    all_rows.extend(
                        _placeholder_row(sid, row[1], row[2], now) for row in still_missing
                    )
                    if still_missing:
                        logger.warning(
                            "trajectory: session {} chunk {}: {} window(s) "
                            "persistently missing — stamped neutral placeholders",
                            sid,
                            chunk_idx + 1,
                            len(still_missing),
                        )

                # Build rows for every window the model returned successfully.
                for row in chunk:
                    key = (row[1], row[2])
                    win = indexed.get(key)
                    if win is None:
                        continue
                    all_rows.append(_build_row(sid, win, now))

                logger.info(
                    "trajectory: session {} chunk {}/{} done in {:.1f}s "
                    "({} windows, {} placeholders)",
                    sid,
                    chunk_idx + 1,
                    len(chunks),
                    time.monotonic() - t0,
                    len(chunk),
                    sum(1 for r in all_rows if r["confidence"] == 0.0),
                )
        except Exception as exc:  # noqa: BLE001 — non-cancel exceptions go to retry; CancelledError still tears down the task group
            # Any exception escaping the loop (network blip post-retry,
            # parquet schema error, etc.) goes to the retry queue and is
            # swallowed so the task group keeps draining.
            logger.warning(
                "trajectory: session {} aborted ({}); enqueuing for retry",
                sid,
                exc,
            )
            retry.enqueue(pipeline="trajectory", unit_id=sid, error=str(exc))
            return

        if session_failed:
            return

        if all_rows:
            # Critical section: serialize parquet shard writes + counter
            # mutation. write_part itself produces a fresh part-<ts>.parquet
            # per call (see parquet_shards.write_part), so concurrent calls
            # don't collide on filenames — but we still keep the lock so the
            # in-memory ``written_box`` / ``processed_sessions`` set updates
            # in lockstep with the on-disk write.
            #
            # replace_sessions drops any prior rows for ``sid`` still sitting
            # in the cache from earlier runs. The checkpointer gates
            # computation on advancing (latest_ts, message_count) bounds but
            # does NOT touch the parquet cache; without this step a growing
            # active session duplicates its (prev_uuid, curr_uuid) pairs
            # on every rerun. See GH #45.
            df = pl.DataFrame(all_rows, schema=_PARQUET_SCHEMA)
            async with write_lock:
                cache.replace_sessions(key_column="session_id", session_ids=[sid])
                cache.write_part(df)
                written_box[0] += len(all_rows)
                processed_sessions.add(sid)

    async with anyio.create_task_group() as tg:
        for sid, session_windows in by_session.items():
            tg.start_soon(_process_session, sid, session_windows)

    written = written_box[0]

    if processed_sessions:
        checkpoint.mark_completed(
            pipeline="trajectory",
            rows=[(sid, *bounds.get(sid, (None, None))) for sid in processed_sessions],
        )
        retry.mark_done(pipeline="trajectory", unit_ids=list(processed_sessions))

    logger.info(
        "trajectory: wrote {} windows across {} sessions",
        written,
        len(processed_sessions),
    )
    return written


def trajectory_messages(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
    cache: CachePort | None = None,
    checkpoint: CheckpointPort | None = None,
    retry: RetryQueuePort | None = None,
    reader: TranscriptReaderPort | None = None,
) -> int | dict[str, Any]:
    """Per-session windowed sentiment + transition classification.

    In ``--dry-run`` mode returns a plan dict (per-session counts, not
    per-message). In real-run mode returns the count of windows written.
    The storage ports (``cache`` / ``checkpoint`` / ``retry``) and the optional
    ``reader`` default as in :func:`classify_sessions`.
    """
    thinking_mode = "disabled" if no_thinking else settings.trajectory_thinking
    if dry_run:
        # Per-session count is the right unit for the windowed pipeline.
        # Each session contributes ceil(text_turns / MAX_WINDOWS_PER_CHUNK)
        # Sonnet calls on a fresh run.
        where = ["text_content IS NOT NULL", "length(text_content) >= 1"]
        if since_days is not None:
            where.append(f"ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
        sql = (
            "SELECT count(DISTINCT CAST(session_id AS VARCHAR)) AS sessions, "
            "count(*) AS turns "
            f"FROM messages_text WHERE {' AND '.join(where)}"
        )
        row = con.execute(sql).fetchone()
        sessions = int(row[0]) if row is not None else 0
        turns = int(row[1]) if row is not None else 0
        if limit is not None:
            sessions = min(sessions, int(limit))
        # Each chunk is one Sonnet call. Per-chunk avg input tokens scale
        # with the windows-per-chunk count; budget conservatively for a
        # full 16-window chunk.
        avg_in = 4500
        avg_out = 800
        # Round up turns/16 per session for the LLM call estimate.
        llm_calls = (turns + MAX_WINDOWS_PER_CHUNK - 1) // MAX_WINDOWS_PER_CHUNK
        # The cost estimate uses the Sonnet pricing table (the only priced path
        # today); Luna is billed on the Mantle endpoint and reports its model id
        # for transparency. ``model`` tracks the selected provider so the plan
        # tells the operator which backend a --no-dry-run would actually call.
        cost = _estimate_cost(llm_calls, avg_in, avg_out, settings.sonnet_pricing)
        luna = settings.llm_analytics_provider == "strands-luna"
        plan_model = settings.luna_model_id if luna else settings.sonnet_model_id
        logger.info(
            "trajectory --dry-run: {} sessions, {} text turns, ~{} LLM calls, "
            "provider={}, est ${:.2f}",
            sessions,
            turns,
            llm_calls,
            settings.llm_analytics_provider,
            cost,
        )
        return {
            "pipeline": "trajectory",
            "candidates": sessions,
            "turns": turns,
            "llm_calls": llm_calls,
            "avg_input_tokens": avg_in,
            "avg_output_tokens": avg_out,
            "estimated_cost_usd": round(cost, 4),
            "provider": settings.llm_analytics_provider,
            "model": plan_model,
            "thinking": thinking_mode,
            "since_days": since_days,
            "limit": limit,
            "dry_run": True,
        }

    cache = cache if cache is not None else build_cache(settings.trajectory_parquet_path)
    checkpoint = checkpoint if checkpoint is not None else build_checkpoint(settings)
    retry = retry if retry is not None else build_retry_queue(settings)

    async def _run() -> int:
        return await _trajectory_async(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            thinking_mode=thinking_mode,
            cache=cache,
            checkpoint=checkpoint,
            retry=retry,
            reader=reader,
        )

    with pipeline_cache_stats("trajectory"):
        return asyncio.run(_run())
