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

All Bedrock plumbing — client construction, retry, structured-output
parsing, the per-pipeline cache-stat accumulator — lives in
:mod:`claude_sql.llm_shared`.
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
from loguru import logger

from claude_sql.core import checkpointer, retry_queue
from claude_sql.core.llm_shared import (
    BedrockRefusalError,
    _build_bedrock_client,
    _estimate_cost,
    classify_one,
    pipeline_cache_stats,
)
from claude_sql.core.parquet_shards import iter_part_files, replace_sessions, write_part
from claude_sql.core.schemas import TRAJECTORY_ARRAY_SCHEMA
from claude_sql.core.session_text import session_bounds

if TYPE_CHECKING:
    import duckdb

    from claude_sql.core.config import Settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Honest ceiling on windows per Sonnet request body. RFC §4.1 — empirically
#: the per-window output budget under thinking=disabled stays comfortably
#: below the 2048-token classify_max_tokens cap up to ~16 windows. Past that
#: the model truncates the array and we end up running the missing-windows
#: retry path on every chunk.
MAX_WINDOWS_PER_CHUNK: int = 16

#: Numeric encoding of the three-label sentiment for the ``delta`` column.
_SENTIMENT_VAL: dict[str, int] = {"negative": -1, "neutral": 0, "positive": 1}

#: The six transition_kind labels — pinned in code so the count never drifts
#: from the schema. Tested in ``test_transition_kind_enum_values_are_six``.
TRANSITION_KINDS: tuple[str, ...] = (
    "frustration_spike",
    "resolution",
    "reset",
    "drift",
    "clarification",
    "none",
)


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
    where = ["1=1"]
    params: list[object] = []
    if since_days is not None:
        where.append(f"tw.curr_ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
    if active_sessions:
        where.append("CAST(tw.session_id AS VARCHAR) IN (SELECT unnest(?))")
        params.append(list(active_sessions))
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
    rows = con.execute(sql, params).fetchall()
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


def _chunk_windows(
    windows: list[tuple[str, str | None, str, str | None, str, str | None, str]],
    *,
    chunk_size: int = MAX_WINDOWS_PER_CHUNK,
) -> list[list[tuple[str, str | None, str, str | None, str, str | None, str]]]:
    """Split ``windows`` (already filtered to one session) into anchor-sharing chunks.

    Chunks of size ``chunk_size`` overlap by one *anchor turn*: chunk N's
    last window's ``curr_uuid`` equals chunk N+1's first window's
    ``prev_uuid`` — the natural shape since adjacent windows already share
    that turn by construction. We do NOT duplicate the anchor row; we just
    let the natural overlap fall out of slicing on contiguous windows.

    Empty input returns an empty list.
    """
    if not windows:
        return []
    return [windows[i : i + chunk_size] for i in range(0, len(windows), chunk_size)]


def _format_chunk_xml(
    chunk: list[tuple[str, str | None, str, str | None, str, str | None, str]],
    *,
    max_text_chars: int = 2000,
) -> str:
    """Render one chunk of windows as XML for the user content block.

    Each window block has a <prev> and <curr> with the role + uuid as
    attributes and the text as the body. Body is clipped at
    ``max_text_chars`` per turn so a single jumbo prompt doesn't blow the
    request size. Truncation is footer-marked so the model knows the rest
    was elided.
    """
    out: list[str] = []
    for idx, (_sid, prev_uuid, curr_uuid, prev_role, curr_role, prev_text, curr_text) in enumerate(
        chunk
    ):
        prev_attr_uuid = "" if prev_uuid is None else prev_uuid
        prev_attr_role = "" if prev_role is None else prev_role
        prev_body = ""
        if prev_text is not None:
            prev_body = prev_text
            if len(prev_body) > max_text_chars:
                prev_body = prev_body[:max_text_chars] + "…(truncated)"
        curr_body = curr_text
        if len(curr_body) > max_text_chars:
            curr_body = curr_body[:max_text_chars] + "…(truncated)"
        out.append(
            f"<window idx={idx}>\n"
            f'<prev role="{_xml_attr(prev_attr_role)}" uuid="{_xml_attr(prev_attr_uuid)}">'
            f"{_xml_text(prev_body)}</prev>\n"
            f'<curr role="{_xml_attr(curr_role or "")}" uuid="{_xml_attr(curr_uuid)}">'
            f"{_xml_text(curr_body)}</curr>\n"
            f"</window>"
        )
    return "\n".join(out)


def _xml_attr(value: str) -> str:
    """Escape a string for use inside an XML attribute value."""
    return (
        value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _xml_text(value: str) -> str:
    """Escape a string for use inside an XML element body."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Sentiment / delta arithmetic
# ---------------------------------------------------------------------------


def _delta_value(prev: str | None, curr: str | None) -> float | None:
    """Compute the numeric delta (curr - prev) for the parquet row."""
    if prev is None or curr is None:
        return None
    return float(_SENTIMENT_VAL.get(curr, 0) - _SENTIMENT_VAL.get(prev, 0))


def _placeholder_row(
    session_id: str,
    prev_uuid: str | None,
    curr_uuid: str,
    classified_at: datetime,
) -> dict[str, Any]:
    """Build a neutral placeholder row when the model refuses or persistently misses a window."""
    return {
        "session_id": session_id,
        "prev_uuid": prev_uuid,
        "curr_uuid": curr_uuid,
        "prev_sentiment": None if prev_uuid is None else "neutral",
        "curr_sentiment": "neutral",
        "delta": None if prev_uuid is None else 0.0,
        "is_transition": False,
        "transition_kind": "none",
        "confidence": 0.0,
        "classified_at": classified_at,
    }


def _build_row(
    session_id: str,
    win: dict[str, Any],
    classified_at: datetime,
) -> dict[str, Any]:
    """Build one parquet row from a returned :class:`TrajectoryWindow` dict."""
    prev_uuid = win.get("prev_uuid")
    curr_uuid = win.get("curr_uuid")
    prev_sentiment = win.get("prev_sentiment")
    curr_sentiment = win.get("curr_sentiment") or "neutral"
    transition_kind = win.get("transition_kind") or "none"
    if transition_kind not in TRANSITION_KINDS:
        transition_kind = "none"
    # Trust the model's emitted delta when it parses to a number; otherwise
    # recompute from the prev/curr labels. Either path keeps the column
    # well-typed (Float64) and the recompute path is the audit trail when
    # the model returns garbage.
    raw_delta = win.get("delta")
    delta_val: float | None
    if raw_delta is None or prev_sentiment is None:
        delta_val = _delta_value(prev_sentiment, curr_sentiment) if prev_sentiment else None
    else:
        try:
            delta_val = float(raw_delta)
        except (TypeError, ValueError):
            delta_val = _delta_value(prev_sentiment, curr_sentiment)
    return {
        "session_id": session_id,
        "prev_uuid": prev_uuid,
        "curr_uuid": curr_uuid,
        "prev_sentiment": prev_sentiment,
        "curr_sentiment": curr_sentiment,
        "delta": delta_val,
        "is_transition": bool(win.get("is_transition", False)),
        "transition_kind": transition_kind,
        "confidence": float(win.get("confidence", 0.0)),
        "classified_at": classified_at,
    }


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
    client: Any,
    settings: Settings,
    sem: anyio.CapacityLimiter,
    *,
    chunk: list[tuple[str, str | None, str, str | None, str, str | None, str]],
    thinking_mode: str,
) -> dict[tuple[str | None, str], dict[str, Any]] | BedrockRefusalError | Exception:
    """Send one chunk to Sonnet; return per-(prev,curr) result map or the raw exception.

    The schema reminder is concatenated into the user_text alongside the
    XML payload. The system block (already 1h-cached by ``classify_one``)
    carries the heavy task framing — the schema reminder here just pins
    the output shape so the model doesn't drift to a single-window object.
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
        result = await classify_one(
            client,
            settings.sonnet_model_id,
            TRAJECTORY_ARRAY_SCHEMA,
            user_text,
            max_tokens=settings.classify_max_tokens,
            thinking_mode=thinking_mode,
            sem=sem,
            system=TRAJECTORY_SYSTEM_PROMPT,
            pipeline="trajectory",
        )
    except BedrockRefusalError as exc:
        return exc
    except Exception as exc:  # noqa: BLE001 — propagate to caller's retry/skip logic; CancelledError still cancels the task group
        return exc
    windows = result.get("windows") if isinstance(result, dict) else None
    if not isinstance(windows, list):
        return RuntimeError(f"trajectory: unexpected response shape {sorted(result)}")
    indexed: dict[tuple[str | None, str], dict[str, Any]] = {}
    for win in windows:
        if not isinstance(win, dict):
            continue
        key = (win.get("prev_uuid"), win.get("curr_uuid"))
        if key[1] is None:
            continue
        indexed[key] = win
    return indexed


def _missing_keys(
    chunk: list[tuple[str, str | None, str, str | None, str, str | None, str]],
    indexed: dict[tuple[str | None, str], dict[str, Any]],
) -> list[tuple[str, str | None, str, str | None, str, str | None, str]]:
    """Return the chunk rows whose (prev_uuid, curr_uuid) was NOT in the response."""
    out = []
    for row in chunk:
        prev_uuid, curr_uuid = row[1], row[2]
        if (prev_uuid, curr_uuid) not in indexed:
            out.append(row)
    return out


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
) -> int:
    """Async implementation behind :func:`trajectory_messages`."""
    # Step 1: purge any stale per-message-shape shards left over from v0.x.
    _purge_old_shards(settings.trajectory_parquet_path)

    # Step 2: session-level checkpoint — only sessions whose transcripts
    # have advanced since the last run go through.
    bounds = session_bounds(con, since_days=since_days, limit=limit)
    unchanged_pending, skipped_sessions = checkpointer.filter_unchanged(
        ((sid, lt, mt) for sid, (lt, mt) in bounds.items()),
        pipeline="trajectory",
        checkpoint_db_path=settings.checkpoint_db_path,
    )
    active_sessions: set[str] = set(unchanged_pending)

    # Retry queue: drain pending failed sessions so they get another shot.
    retry_sids = set(retry_queue.drain(settings.checkpoint_db_path, pipeline="trajectory"))
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
    by_session: dict[str, list] = defaultdict(list)
    for row in raw_rows:
        by_session[row[0]].append(row)

    logger.info(
        "trajectory: {} sessions, {} total windows pending",
        len(by_session),
        len(raw_rows),
    )

    client = _build_bedrock_client(settings)
    sem = anyio.CapacityLimiter(settings.llm_concurrency)
    # Shared mutable state across concurrent session tasks. Mutations to
    # ``written`` / ``processed_sessions`` AND the parquet shard write are
    # serialized under ``write_lock``. The Bedrock call itself (inside
    # ``classify_one``) is NOT under this lock — that's where the
    # CapacityLimiter does the throttling, capping in-flight chunks at
    # ``settings.llm_concurrency``.
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
                res = await _classify_chunk(
                    client,
                    settings,
                    sem,
                    chunk=chunk,
                    thinking_mode=thinking_mode,
                )
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
                    # retry_queue.enqueue uses sqlite WAL — concurrent writers
                    # are safe per .erpaval/solutions/best-practices/
                    # sqlite-wal-cold-start-pragma-race.md.
                    retry_queue.enqueue(
                        settings.checkpoint_db_path,
                        pipeline="trajectory",
                        unit_id=sid,
                        error=str(res),
                    )
                    session_failed = True
                    break

                indexed = res
                missing = _missing_keys(chunk, indexed)

                # Bounded retry: re-request only the missing windows in one
                # additional Sonnet call. Persistent misses become
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
                    retry_res = await _classify_chunk(
                        client,
                        settings,
                        sem,
                        chunk=missing,
                        thinking_mode=thinking_mode,
                    )
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
            retry_queue.enqueue(
                settings.checkpoint_db_path,
                pipeline="trajectory",
                unit_id=sid,
                error=str(exc),
            )
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
                replace_sessions(
                    settings.trajectory_parquet_path,
                    key_column="session_id",
                    session_ids=[sid],
                )
                write_part(settings.trajectory_parquet_path, df)
                written_box[0] += len(all_rows)
                processed_sessions.add(sid)

    async with anyio.create_task_group() as tg:
        for sid, session_windows in by_session.items():
            tg.start_soon(_process_session, sid, session_windows)

    written = written_box[0]

    if processed_sessions:
        checkpointer.mark_completed(
            settings.checkpoint_db_path,
            pipeline="trajectory",
            rows=[(sid, *bounds.get(sid, (None, None))) for sid in processed_sessions],
        )
        retry_queue.mark_done(
            settings.checkpoint_db_path,
            pipeline="trajectory",
            unit_ids=list(processed_sessions),
        )

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
) -> int | dict[str, Any]:
    """Per-session windowed sentiment + transition classification.

    In ``--dry-run`` mode returns a plan dict (per-session counts, not
    per-message). In real-run mode returns the count of windows written.
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
        cost = _estimate_cost(llm_calls, avg_in, avg_out, settings.sonnet_pricing)
        logger.info(
            "trajectory --dry-run: {} sessions, {} text turns, ~{} LLM calls, est ${:.2f}",
            sessions,
            turns,
            llm_calls,
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
            "model": settings.sonnet_model_id,
            "thinking": thinking_mode,
            "since_days": since_days,
            "limit": limit,
            "dry_run": True,
        }

    async def _run() -> int:
        return await _trajectory_async(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            thinking_mode=thinking_mode,
        )

    with pipeline_cache_stats("trajectory"):
        return asyncio.run(_run())
