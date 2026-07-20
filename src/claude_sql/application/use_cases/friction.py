"""Detect user-friction signals in short user-role messages.

A *friction signal* is anything in a user message that implies the agent's
last turn fell short of what the user expected:

- ``status_ping``          -- "how's it going?", "any update?", "status?"
- ``unmet_expectation``    -- "screenshot?", "tests?", "link?" (the agent
                              should have produced this proactively)
- ``confusion``            -- "what does that mean?", "why?", "I don't get it"
- ``interruption``         -- "wait", "stop", "actually...", "hold on"
- ``correction``           -- "no, not that", "that's wrong", "nope"
- ``frustration``          -- "ugh", "seriously?", terse annoyance
- ``none``                 -- ordinary task instruction (majority)

Pipeline shape
--------------
1. Pre-filter to user-role messages below ``settings.friction_max_chars``
   (default 300).  Long turns are almost always genuine instructions.
2. Regex fast-path for strong, unambiguous patterns (status pings, obvious
   interruption keywords).  Confidence 0.9 and skips the LLM. The pure regex
   bank lives in :mod:`claude_sql.domain.friction`.
3. Everything else goes to Sonnet 4.6 via ``invoke_model`` with
   ``output_config.format`` using :data:`USER_FRICTION_SCHEMA`.
4. Per-session checkpoint + per-uuid anti-join so reruns are free on
   untouched sessions.

Outputs ``user_friction.parquet`` with one row per analysed user message:

    {uuid, session_id, ts, text_snippet, label, rationale, source,
     confidence, classified_at}

``source`` is ``'regex'``, ``'sql'``, ``'llm'``, or ``'refused'`` so
downstream queries can filter to the high-recall LLM rows or audit the
fast paths separately.  The SQL stamp layer (RFC §4.3, §9.4) sits
between regex and LLM and catches three deterministic shapes — repeated
user messages, short imperative reverts, and trailing-? after error
tool_results — without paying Bedrock.

Moved here from ``analytics/friction_worker.py`` in the v2 hexagonal reshape
(MIGRATION Phase C). The pure regex fast-path moved to
:mod:`claude_sql.domain.friction`; the dollar estimate to
:mod:`claude_sql.domain.costs`.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import anyio
import duckdb
import polars as pl
from loguru import logger

from claude_sql.application.prompts import USER_FRICTION_SYSTEM_PROMPT
from claude_sql.domain.costs import estimate_cost as _estimate_cost
from claude_sql.domain.errors import BedrockRefusalError
from claude_sql.domain.friction import _REGEX_BANK, regex_fast_path
from claude_sql.infrastructure.adapters import build_cache, build_checkpoint, build_retry_queue
from claude_sql.infrastructure.bedrock.client import (
    _build_bedrock_client,
    classify_one,
    pipeline_cache_stats,
)
from claude_sql.infrastructure.bedrock.structured_output import USER_FRICTION_SCHEMA
from claude_sql.infrastructure.session_text_loader import session_bounds
from claude_sql.infrastructure.sqlite_state import checkpointer

if TYPE_CHECKING:
    from claude_sql.application.ports import (
        CachePort,
        CheckpointPort,
        RetryQueuePort,
        TranscriptReaderPort,
    )
    from claude_sql.infrastructure.settings import Settings


# ``_REGEX_BANK`` / ``regex_fast_path`` are re-imported from the domain module
# and listed in ``__all__`` so the shim (and the test suite that reads
# ``friction_worker._REGEX_BANK`` / ``regex_fast_path``) resolve them here.


# ---------------------------------------------------------------------------
# SQL stamp layer — pre-LLM, post-regex deterministic shapes (RFC §4.3, §9.4)
# ---------------------------------------------------------------------------
#
# Three rules catch deterministic friction shapes in pure SQL so we don't
# pay Bedrock for them.  The output is the same ``(label, confidence,
# source)`` triple the regex bank emits; ``source='sql'`` is the only new
# value.  Source priority is regex > sql > llm: regex hits always win,
# then SQL stamps, then anything still unstamped goes to Sonnet.
#
# Rule 1 — repeated user message body within 10 turns.  A user-role
#   message whose normalized text matches an earlier user-role message in
#   the same session, within 10 user-role turns prior, is stamped
#   ``unmet_expectation`` at confidence 0.85.  Re-asking the same
#   question is a strong signal the agent's first answer fell short.
#
# Rule 2 — short imperative reverts (≤30 chars, first token ∈
#   {stop, redo, revert, rollback, undo, restart}) → ``correction``,
#   confidence 0.9.  Ruff-style explicit corrections that bypass the
#   regex bank's punctuation requirements.
#
# Rule 3 — trailing ``?`` after an error tool_result → ``confusion``,
#   confidence 0.85.  We derive ``is_error`` from ``messages.content_json``
#   because the ``tool_results`` view does NOT expose it: ``content_blocks``
#   only extracts a fixed field set, and ``is_error`` lives at the same
#   level as ``$.content`` in the raw block JSON.  Re-deriving here keeps
#   the friction worker self-contained and dodges a sql_views.py change
#   that would touch every analytics caller.


_RULE1_REPEATED_TEMPLATE = """
    WITH candidates AS (
        SELECT mt.uuid, mt.session_id, mt.ts,
               lower(regexp_replace(mt.text_content, '\\s+', ' ', 'g')) AS norm,
               row_number() OVER (
                   PARTITION BY mt.session_id ORDER BY mt.ts, mt.uuid
               ) AS rn
        FROM messages_text mt
        WHERE mt.role = 'user'
    )
    SELECT cur.uuid AS uuid,
           'unmet_expectation' AS label,
           CAST(0.85 AS DOUBLE) AS confidence,
           'sql' AS source
      FROM candidates cur
      JOIN candidates prev
        ON cur.session_id = prev.session_id
       AND cur.norm = prev.norm
       AND prev.rn < cur.rn
       AND cur.rn - prev.rn <= 10
     WHERE cur.uuid IN (SELECT unnest(?))
"""


# Imperatives must be matched on a single first token, so an exact
# token-set membership beats a regex with anchored boundaries — and
# ``split_part(trim(...), ' ', 1)`` is portable across DuckDB versions.
_RULE2_IMPERATIVE_TEMPLATE = """
    SELECT mt.uuid AS uuid,
           'correction' AS label,
           CAST(0.9 AS DOUBLE) AS confidence,
           'sql' AS source
      FROM messages_text mt
     WHERE mt.role = 'user'
       AND length(mt.text_content) <= 30
       AND lower(rtrim(split_part(trim(mt.text_content), ' ', 1), '.,!?'))
           IN ('stop', 'redo', 'revert', 'rollback', 'undo', 'restart')
       AND mt.uuid IN (SELECT unnest(?))
"""


# Rule 3 needs ``is_error`` which is NOT exposed by the ``tool_results``
# view (see comment block above for why).  We unnest ``messages.content_json``
# to recover the boolean and join back to ``messages_text`` via session_id
# + a ts-ordered "no event in between" predicate.  The user message must
# end with ``?`` and be the immediate next event after the failing
# tool_result.
_RULE3_CONFUSION_TEMPLATE = """
    WITH error_results AS (
        SELECT m.session_id, m.ts
          FROM messages m,
               UNNEST(json_extract(m.content_json, '$[*]')) AS t(block)
         WHERE json_extract_string(block, '$.type') = 'tool_result'
           AND json_extract_string(block, '$.is_error') = 'true'
    )
    SELECT mt.uuid AS uuid,
           'confusion' AS label,
           CAST(0.85 AS DOUBLE) AS confidence,
           'sql' AS source
      FROM messages_text mt
      JOIN error_results er
        ON er.session_id = mt.session_id
       AND er.ts < mt.ts
     WHERE mt.role = 'user'
       AND mt.text_content LIKE '%?'
       AND mt.uuid IN (SELECT unnest(?))
       AND NOT EXISTS (
            SELECT 1
              FROM messages_text mt2
             WHERE mt2.session_id = mt.session_id
               AND mt2.ts > er.ts
               AND mt2.ts < mt.ts
       )
"""


def sql_stamp(
    con: duckdb.DuckDBPyConnection,
    candidate_uuids: list[str],
) -> dict[str, tuple[str, float, str]]:
    """Run the three deterministic SQL stamps over ``candidate_uuids``.

    Returns a ``{uuid: (label, confidence, source='sql')}`` map.  When two
    rules match the same uuid, the higher-confidence rule wins; rules 1+3
    tie at 0.85 so first-rule-wins (deterministic ordering: 1 → 2 → 3).

    Rule 3 may fail if the ``messages`` view is not registered (some test
    fixtures only create ``messages_text``).  That branch is caught and
    skipped so the SQL-stamp layer degrades gracefully — the LLM path
    will still classify those messages.
    """
    if not candidate_uuids:
        return {}
    out: dict[str, tuple[str, float, str]] = {}

    # Rules 1 and 2 — pure messages_text, always available.
    for sql_template in (_RULE1_REPEATED_TEMPLATE, _RULE2_IMPERATIVE_TEMPLATE):
        rows = con.execute(sql_template, [list(candidate_uuids)]).fetchall()
        for uuid, label, conf, source in rows:
            existing = out.get(uuid)
            if existing is None or conf > existing[1]:
                out[uuid] = (label, float(conf), source)

    # Rule 3 — needs the ``messages`` view (json content unnest).  Skip
    # silently when it's absent so test fixtures with a hand-built
    # ``messages_text`` table still exercise rules 1 and 2.
    try:
        rows = con.execute(_RULE3_CONFUSION_TEMPLATE, [list(candidate_uuids)]).fetchall()
    except duckdb.CatalogException:
        # ``messages`` view not registered — pure-messages_text test
        # fixture; rule 3 simply doesn't fire.  No-op.
        rows = []
    for uuid, label, conf, source in rows:
        existing = out.get(uuid)
        if existing is None or conf > existing[1]:
            out[uuid] = (label, float(conf), source)

    return out


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


_SCHEMA: dict[str, Any] = {
    "uuid": pl.Utf8,
    "session_id": pl.Utf8,
    "ts": pl.Datetime("us", "UTC"),
    "text_snippet": pl.Utf8,
    "label": pl.Utf8,
    "rationale": pl.Utf8,
    "source": pl.Utf8,  # 'regex' | 'llm' | 'refused'
    "confidence": pl.Float32,
    "classified_at": pl.Datetime("us", "UTC"),
}


# ---------------------------------------------------------------------------
# Candidate SQL
# ---------------------------------------------------------------------------


#: Claude Code injects these strings as user-role messages even though
#: they're system-generated bookkeeping. An audit of the live
#: ``user_friction.parquet`` showed they accounted for 279 of 298
#: LLM-classified rows (~94% of friction Bedrock calls). Filter at the
#: SQL boundary so they never reach Sonnet.
_SYSTEM_MARKER_TEXTS: tuple[str, ...] = (
    "Continue from where you left off.",
    "[Request interrupted by user for tool use]",
)


def _candidate_sql(max_chars: int, since_days: int | None) -> tuple[str, list[Any]]:
    """SQL pulling user-role messages under the char cutoff.

    Claude Code system markers (see ``_SYSTEM_MARKER_TEXTS``) are
    excluded here because they're CLI bookkeeping, not user-typed
    friction signals. Single-quotes inside markers are escaped per SQL
    rules.
    """
    quoted = ", ".join(f"'{m.replace(chr(39), chr(39) * 2)}'" for m in _SYSTEM_MARKER_TEXTS)
    where = [
        "mt.text_content IS NOT NULL",
        "length(mt.text_content) >= 1",
        f"length(mt.text_content) <= {int(max_chars)}",
        "mt.role = 'user'",
        f"trim(mt.text_content) NOT IN ({quoted})",
    ]
    if since_days is not None:
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
    sql = f"""
        SELECT CAST(mt.uuid AS VARCHAR)        AS uuid,
               CAST(mt.session_id AS VARCHAR)  AS session_id,
               mt.ts                           AS ts,
               mt.text_content                 AS text_content
          FROM messages_text mt
         WHERE {" AND ".join(where)}
         ORDER BY mt.ts
    """
    return sql, []


# ---------------------------------------------------------------------------
# Prompt wrapper
# ---------------------------------------------------------------------------
#
# Friction detection is context-free per message: we hand the LLM a single
# user utterance and a brief framing paragraph, and let the schema's
# field-level descriptions do the rest of the work.  No session history --
# that's what makes this cheap (hundreds of input tokens, not thousands).


_USER_PROMPT_TEMPLATE = """\
Classify the following SHORT USER MESSAGE from a Claude Code coding session.

You're looking for FRICTION SIGNALS — cues that the human is impatient,
confused, interrupting the agent, correcting it, or asking for something
the agent should have provided proactively but didn't.

Examples of NON-obvious friction:
- "screenshot?" → unmet_expectation (agent should have shared a screenshot)
- "tests?" → unmet_expectation (agent didn't run tests)
- "link?" → unmet_expectation (agent referenced a resource without linking)
- "why?" / "why did you do that?" → confusion
- "wait" / "actually..." → interruption
- "no not that" / "nope" → correction
- "are you there?" / "you alive?" → status_ping

The MAJORITY of short messages are ordinary task instructions and should
get label=none. Only flag a friction signal when the cue is clear.

USER MESSAGE:
```
{text}
```
"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def _classify_async(
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
    """Async body behind :func:`detect_user_friction`."""
    already: set[str] = set()
    done_df = cache.read_all(columns=["uuid"])
    if done_df is not None and done_df.height > 0:
        already = set(done_df["uuid"].to_list())

    # Session-level checkpoint: skip sessions unchanged since last run.
    # ``session_bounds`` stays a direct local call by default (monkeypatch seam).
    bounds = (
        reader.session_bounds(since_days=since_days, limit=limit)
        if reader is not None
        else session_bounds(con, since_days=since_days, limit=limit)
    )
    unchanged_pending, skipped_sessions = checkpointer.filter_unchanged(
        ((sid, lt, mt) for sid, (lt, mt) in bounds.items()),
        pipeline="user_friction",
        checkpoint_db_path=settings.checkpoint_db_path,
    )
    active_sessions: set[str] = set(unchanged_pending)

    retry_uuids = set(retry.drain(pipeline="user_friction"))
    if retry_uuids:
        logger.info("user_friction: draining {} retry-queue entries", len(retry_uuids))
        already -= retry_uuids

    sql, _ = _candidate_sql(settings.friction_max_chars, since_days)
    if active_sessions:
        sql = sql.replace(
            " WHERE ",
            " WHERE CAST(mt.session_id AS VARCHAR) IN (SELECT unnest(?)) AND ",
            1,
        )
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"

    params = [list(active_sessions)] if active_sessions else []
    rows_raw = con.execute(sql, params).fetchall() if active_sessions or not bounds else []
    candidates = [(r[0], r[1], r[2], r[3]) for r in rows_raw if r[0] not in already]
    session_for_uuid = {r[0]: r[1] for r in rows_raw if r[0] not in already}
    if skipped_sessions:
        logger.info("user_friction: skipped {} sessions via checkpoint", skipped_sessions)
    logger.info("user_friction: {} candidate user messages", len(candidates))

    if not candidates:
        logger.info("user_friction: nothing pending")
        return 0

    # 1. Regex fast-path.
    fast_rows: list[dict[str, Any]] = []
    sql_stamp_pending: list[tuple[str, str, Any, str]] = []
    now = datetime.now(UTC)
    for uuid, session_id, ts, text in candidates:
        hit = regex_fast_path(text or "")
        if hit is not None:
            label, conf = hit
            fast_rows.append(
                {
                    "uuid": uuid,
                    "session_id": session_id,
                    "ts": ts,
                    "text_snippet": (text or "")[:200],
                    "label": label,
                    "rationale": "regex match",
                    "source": "regex",
                    "confidence": conf,
                    "classified_at": now,
                }
            )
        else:
            sql_stamp_pending.append((uuid, session_id, ts, text or ""))

    # 2. SQL stamp layer (RFC §4.3, §9.4) — pre-LLM, post-regex.
    sql_stamps = sql_stamp(con, [c[0] for c in sql_stamp_pending])
    llm_pending: list[tuple[str, str, Any, str]] = []
    for uuid, session_id, ts, text in sql_stamp_pending:
        stamp = sql_stamps.get(uuid)
        if stamp is not None:
            label, conf, source = stamp
            fast_rows.append(
                {
                    "uuid": uuid,
                    "session_id": session_id,
                    "ts": ts,
                    "text_snippet": (text or "")[:200],
                    "label": label,
                    "rationale": "sql stamp",
                    "source": source,
                    "confidence": conf,
                    "classified_at": now,
                }
            )
        else:
            llm_pending.append((uuid, session_id, ts, text))

    logger.info(
        "user_friction: {} regex, {} sql-stamped, {} pending LLM",
        sum(1 for r in fast_rows if r["source"] == "regex"),
        sum(1 for r in fast_rows if r["source"] == "sql"),
        len(llm_pending),
    )

    if fast_rows:
        cache.write_part(pl.DataFrame(fast_rows, schema=_SCHEMA))

    processed_sessions: set[str] = {r["session_id"] for r in fast_rows}

    if not llm_pending:
        if processed_sessions:
            checkpoint.mark_completed(
                pipeline="user_friction",
                rows=[(sid, *bounds.get(sid, (None, None))) for sid in processed_sessions],
            )
        logger.info("user_friction: wrote {} total rows (regex only)", len(fast_rows))
        return len(fast_rows)

    # 2. LLM path.
    client = _build_bedrock_client(settings)
    sem = anyio.CapacityLimiter(settings.llm_concurrency)
    chunk_size = max(settings.batch_size * 4, 256)
    written = len(fast_rows)

    for i in range(0, len(llm_pending), chunk_size):
        chunk = llm_pending[i : i + chunk_size]
        t0 = time.monotonic()
        prompts = [_USER_PROMPT_TEMPLATE.format(text=text) for _, _, _, text in chunk]
        coros = [
            classify_one(
                client,
                settings.sonnet_model_id,
                USER_FRICTION_SCHEMA,
                prompt,
                max_tokens=settings.classify_max_tokens,
                thinking_mode=thinking_mode,
                sem=sem,
                system=USER_FRICTION_SYSTEM_PROMPT,
                pipeline="friction",
            )
            for prompt in prompts
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        now = datetime.now(UTC)

        ok_rows: list[dict[str, Any]] = []
        ok_uuids: list[str] = []
        refused_uuids: list[str] = []
        errors = 0
        for (uuid, session_id, ts, text), res in zip(chunk, results, strict=True):
            if isinstance(res, BedrockRefusalError):
                logger.info("user_friction: {} refused by Bedrock — marking none", uuid)
                ok_rows.append(
                    {
                        "uuid": uuid,
                        "session_id": session_id,
                        "ts": ts,
                        "text_snippet": text[:200],
                        "label": "none",
                        "rationale": "refused by bedrock",
                        "source": "refused",
                        "confidence": 0.0,
                        "classified_at": now,
                    }
                )
                refused_uuids.append(uuid)
                continue
            if isinstance(res, BaseException):
                errors += 1
                logger.warning("user_friction: {} failed (queued for retry): {}", uuid, res)
                retry.enqueue(pipeline="user_friction", unit_id=uuid, error=str(res))
                continue
            res_dict: dict[str, Any] = res
            ok_rows.append(
                {
                    "uuid": uuid,
                    "session_id": session_id,
                    "ts": ts,
                    "text_snippet": text[:200],
                    "label": res_dict.get("label", "none"),
                    "rationale": (res_dict.get("rationale") or "")[:200],
                    "source": "llm",
                    "confidence": float(res_dict.get("confidence", 0.0)),
                    "classified_at": now,
                }
            )
            ok_uuids.append(uuid)
            processed_sessions.add(session_id)

        if ok_rows:
            cache.write_part(pl.DataFrame(ok_rows, schema=_SCHEMA))
            done_uuids = ok_uuids + refused_uuids
            if done_uuids:
                retry.mark_done(pipeline="user_friction", unit_ids=done_uuids)
            chunk_sessions = {
                session_for_uuid[u] for u in ok_uuids + refused_uuids if u in session_for_uuid
            }
            if chunk_sessions:
                checkpoint.mark_completed(
                    pipeline="user_friction",
                    rows=[(sid, *bounds.get(sid, (None, None))) for sid in chunk_sessions],
                )

        written += len(ok_rows)
        logger.info(
            "user_friction chunk {}/{}: {} ok, {} errors, {:.1f}s",
            i // chunk_size + 1,
            (len(llm_pending) + chunk_size - 1) // chunk_size,
            len(ok_rows),
            errors,
            time.monotonic() - t0,
        )

    if processed_sessions:
        checkpoint.mark_completed(
            pipeline="user_friction",
            rows=[(sid, *bounds.get(sid, (None, None))) for sid in processed_sessions],
        )
    logger.info("user_friction: wrote {} total rows", written)
    return written


def detect_user_friction(
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
    """Classify short user messages for friction signals.

    See module docstring for category definitions and pipeline shape.

    Parameters
    ----------
    con
        DuckDB connection with ``messages_text`` registered.
    settings
        :class:`Settings` driving parquet path, char cutoff, concurrency.
    since_days
        Restrict to messages whose ``ts`` is within the last N days.  ``None``
        means the full corpus.
    limit
        Optional hard cap on candidate count.
    dry_run
        Count candidate messages and estimate LLM cost without calling Bedrock.
    no_thinking
        Force ``thinking_mode='disabled'``.  ``adaptive`` (default) gives
        better labels on edge cases like bare "screenshot?" where the model
        needs to reason about "did the agent just do something visual?".

    Returns
    -------
    int | dict
        Under ``dry_run=True`` a plan dict with ``{pipeline, candidates,
        llm_calls, estimated_cost_usd, ...}``; otherwise the count of rows
        written to the parquet.
    """
    thinking_mode = "disabled" if no_thinking else settings.friction_thinking

    if dry_run:
        sql, _ = _candidate_sql(settings.friction_max_chars, since_days)
        probe = f"SELECT count(*) FROM ({sql}) q"
        row = con.execute(probe).fetchone()
        n = int(row[0]) if row is not None else 0
        if limit is not None:
            n = min(n, int(limit))
        # Roughly half of short user messages survive the regex fast-path,
        # consistent with the trajectory pipeline's pre-filter survival rate.
        # Short-message prompt is ~200 input tokens (template + message),
        # output is ~60 tokens (label + rationale + confidence).
        llm_n = n // 2
        cost = _estimate_cost(llm_n, 200, 60, settings.sonnet_pricing)
        logger.info(
            "user_friction --dry-run: {} candidates (~{} hit LLM), estimated cost ~${:.2f}",
            n,
            llm_n,
            cost,
        )
        return {
            "pipeline": "friction",
            "candidates": n,
            "llm_calls": llm_n,
            "avg_input_tokens": 200,
            "avg_output_tokens": 60,
            "estimated_cost_usd": round(cost, 4),
            "model": settings.sonnet_model_id,
            "thinking": thinking_mode,
            "since_days": since_days,
            "limit": limit,
            "friction_max_chars": settings.friction_max_chars,
            "dry_run": True,
        }

    cache = cache if cache is not None else build_cache(settings.user_friction_parquet_path)
    checkpoint = checkpoint if checkpoint is not None else build_checkpoint(settings)
    retry = retry if retry is not None else build_retry_queue(settings)
    with pipeline_cache_stats("friction"):
        return asyncio.run(
            _classify_async(
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
        )


__all__ = [
    "_REGEX_BANK",
    "detect_user_friction",
    "regex_fast_path",
    "sql_stamp",
]
