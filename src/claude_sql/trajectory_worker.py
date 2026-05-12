"""Per-message trajectory pipeline.

Scores each user-role message in the corpus for sentiment polarity
(positive / neutral / negative) plus an ``is_transition`` flag for filler /
acknowledgement turns. Short transition messages get a regex fast-path so
we don't pay Bedrock for them; everything else runs through Sonnet 4.6 with
:data:`TRAJECTORY_SYSTEM_PROMPT`.

All Bedrock plumbing — client construction, retry, structured-output
parsing, the per-pipeline cache-stat accumulator — lives in
:mod:`claude_sql.llm_shared`.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import anyio
import polars as pl
from loguru import logger

from claude_sql import checkpointer, retry_queue
from claude_sql.llm_shared import (
    TRAJECTORY_SYSTEM_PROMPT,
    BedrockRefusalError,
    _build_bedrock_client,
    _estimate_cost,
    classify_one,
)
from claude_sql.parquet_shards import read_all, write_part
from claude_sql.schemas import MESSAGE_TRAJECTORY_SCHEMA
from claude_sql.session_text import session_bounds

if TYPE_CHECKING:
    import duckdb

    from claude_sql.config import Settings


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
            classify_one(
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
