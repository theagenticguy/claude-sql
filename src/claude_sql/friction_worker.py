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
   interruption keywords).  Confidence 0.9 and skips the LLM.
3. Everything else goes to Sonnet 4.6 via ``invoke_model`` with
   ``output_config.format`` using :data:`USER_FRICTION_SCHEMA`.
4. Per-session checkpoint + per-uuid anti-join so reruns are free on
   untouched sessions.

Outputs ``user_friction.parquet`` with one row per analysed user message:

    {uuid, session_id, ts, text_snippet, label, rationale, source,
     confidence, classified_at}

``source`` is ``'regex'`` or ``'llm'`` so downstream queries can filter to
the high-recall LLM rows or audit the fast-path separately.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from claude_sql import checkpointer, retry_queue
from claude_sql.llm_worker import (
    BedrockRefusalError,
    _build_bedrock_client,
    _classify_one,
    _estimate_cost,
)
from claude_sql.schemas import USER_FRICTION_SCHEMA
from claude_sql.session_text import session_bounds

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

    from claude_sql.config import Settings


# ---------------------------------------------------------------------------
# Regex fast-path
# ---------------------------------------------------------------------------
#
# These patterns catch the unambiguous cases so we don't pay Bedrock for them.
# Everything ambiguous falls through to the LLM, which is where the
# "screenshot?" / "tests?" class lives (those need semantic context to
# distinguish from genuine topic questions like "can you write tests?").


_REGEX_BANK: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Status pings — only multi-word phrasings that unambiguously ask about
    # progress.  Anything shorter or even slightly ambiguous ("status?",
    # "what's the status column called?") falls through to the LLM so it
    # can disambiguate via context.  The trailing-context guards below
    # require a question mark, end-of-string, or a progress-related word
    # to avoid matching "what's the status column called".
    (
        "status_ping",
        re.compile(
            r"""
            \bhow(?:'s|\s+is|\s+are|\s+it)?\s+(?:it|we|things|progress)\s+
                (?:going|coming|doing|looking|progressing|holding\s+up)\b
            | \bhow'?s?\s+progress\b(?=\s*[?.!]?\s*$)
            | \bany\s+update(?:s)?\b(?=\s*[?.!]?\s*$)
            | \bstatus\s+update\b
            | \bwhere\s+(?:are\s+we|we['\u2019]re)\s+(?:at|with)\b
            | \b(?:are\s+you\s+)?still\s+(?:working|going|running|on\s+it)\b
            | \bwhat'?s?\s+(?:the|your)\s+eta\b
            | \bhow\s+(?:much\s+)?long\s+(?:until|till|more|left)\b
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # Hard interruption keywords at the start of a message.
    (
        "interruption",
        re.compile(
            r"""
            ^\s*
            (?:
                wait(?:\s*[.,!]|\s+a\s+(?:sec|second|moment|minute)|[\s$])
              | stop(?:\s*[.,!]|\s+(?:right\s+)?there|[\s$])
              | hold\s+on
              | hold\s+up
              | hang\s+on
              | actually[,\s]
              | before\s+you\s+(?:do|go)
              | pause\b
              | nvm\b
              | never\s*mind\b
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # Explicit corrections.
    (
        "correction",
        re.compile(
            r"""
            ^\s*
            (?:
                no[,\s.!]
              | nope[,\s.!]?
              | nah[,\s.!]?
              | that'?s\s+(?:wrong|not\s+(?:right|it|what)|incorrect)
              | not\s+(?:that|what\s+i)
              | try\s+again
              | wrong\b
              | that'?s\s+not\s+
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
)


def regex_fast_path(text: str) -> tuple[str, float] | None:
    """Return ``(label, confidence)`` for a regex hit or ``None``.

    Confidence is a flat 0.9 for regex hits -- these are hand-picked,
    unambiguous phrasings.  Ambiguous shapes deliberately fall through to
    the LLM so a single misclassification in the bank does not poison the
    corpus.
    """
    if not text:
        return None
    probe = text[:512]
    for label, pat in _REGEX_BANK:
        if pat.search(probe):
            return label, 0.9
    return None


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


def _append_parquet(path: Path, df: pl.DataFrame) -> None:
    """Append-by-rewrite: load existing parquet, concat, write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 16:
        existing = pl.read_parquet(path)
        df = pl.concat([existing, df], how="diagonal_relaxed")
    df.write_parquet(path)


# ---------------------------------------------------------------------------
# Candidate SQL
# ---------------------------------------------------------------------------


def _candidate_sql(max_chars: int, since_days: int | None) -> tuple[str, list[Any]]:
    """SQL pulling user-role messages under the char cutoff."""
    where = [
        "mt.text_content IS NOT NULL",
        "length(mt.text_content) >= 1",
        f"length(mt.text_content) <= {int(max_chars)}",
        "mt.role = 'user'",
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
) -> int:
    """Async body behind :func:`detect_user_friction`."""
    out_path = settings.user_friction_parquet_path
    already: set[str] = set()
    if out_path.exists() and out_path.stat().st_size > 16:
        already = set(pl.read_parquet(out_path)["uuid"].to_list())

    # Session-level checkpoint: skip sessions unchanged since last run.
    bounds = session_bounds(con, since_days=since_days, limit=limit)
    unchanged_pending, skipped_sessions = checkpointer.filter_unchanged(
        ((sid, lt, mt) for sid, (lt, mt) in bounds.items()),
        pipeline="user_friction",
        checkpoint_db_path=settings.checkpoint_db_path,
    )
    active_sessions: set[str] = set(unchanged_pending)

    retry_uuids = set(retry_queue.drain(settings.checkpoint_db_path, pipeline="user_friction"))
    if retry_uuids:
        logger.info("user_friction: draining {} retry-queue entries", len(retry_uuids))
        already -= retry_uuids

    sql, _ = _candidate_sql(settings.friction_max_chars, since_days)
    if active_sessions:
        sql = sql.replace(
            "FROM messages_text mt",
            "FROM messages_text mt WHERE CAST(mt.session_id AS VARCHAR) IN (SELECT unnest(?))",
            1,
        ).replace(" WHERE ", " AND ", 1)
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
    llm_pending: list[tuple[str, str, Any, str]] = []
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
            llm_pending.append((uuid, session_id, ts, text or ""))

    logger.info(
        "user_friction: {} regex fast-path, {} pending LLM",
        len(fast_rows),
        len(llm_pending),
    )

    if fast_rows:
        _append_parquet(out_path, pl.DataFrame(fast_rows, schema=_SCHEMA))

    processed_sessions: set[str] = {r["session_id"] for r in fast_rows}

    if not llm_pending:
        if processed_sessions:
            checkpointer.mark_completed(
                settings.checkpoint_db_path,
                pipeline="user_friction",
                rows=[(sid, *bounds.get(sid, (None, None))) for sid in processed_sessions],
            )
        logger.info("user_friction: wrote {} total rows (regex only)", len(fast_rows))
        return len(fast_rows)

    # 2. LLM path.
    client = _build_bedrock_client(settings)
    sem = asyncio.Semaphore(settings.concurrency)
    chunk_size = max(settings.batch_size * 4, 256)
    written = len(fast_rows)

    for i in range(0, len(llm_pending), chunk_size):
        chunk = llm_pending[i : i + chunk_size]
        t0 = time.monotonic()
        prompts = [_USER_PROMPT_TEMPLATE.format(text=text) for _, _, _, text in chunk]
        coros = [
            _classify_one(
                client,
                settings.sonnet_model_id,
                USER_FRICTION_SCHEMA,
                prompt,
                max_tokens=settings.classify_max_tokens,
                thinking_mode=thinking_mode,
                sem=sem,
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
                retry_queue.enqueue(
                    settings.checkpoint_db_path,
                    pipeline="user_friction",
                    unit_id=uuid,
                    error=str(res),
                )
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
            _append_parquet(out_path, pl.DataFrame(ok_rows, schema=_SCHEMA))
            done_uuids = ok_uuids + refused_uuids
            if done_uuids:
                retry_queue.mark_done(
                    settings.checkpoint_db_path,
                    pipeline="user_friction",
                    unit_ids=done_uuids,
                )
            chunk_sessions = {
                session_for_uuid[u] for u in ok_uuids + refused_uuids if u in session_for_uuid
            }
            if chunk_sessions:
                checkpointer.mark_completed(
                    settings.checkpoint_db_path,
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
        checkpointer.mark_completed(
            settings.checkpoint_db_path,
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
) -> int:
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
    int
        Number of rows written (or candidate count under ``dry_run``).
    """
    thinking_mode = "disabled" if no_thinking else settings.classify_thinking

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
        return 0

    return asyncio.run(
        _classify_async(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            thinking_mode=thinking_mode,
        )
    )


__all__ = ["detect_user_friction", "regex_fast_path"]
