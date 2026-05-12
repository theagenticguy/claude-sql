"""Stance-conflict detection pipeline.

Reads complete Claude Code session transcripts and emits one row per
detected conflict (or a single sentinel ``empty=True`` row when the session
has none). Output schema mirrors :data:`SESSION_CONFLICTS_SCHEMA` and is
written to ``settings.conflicts_parquet_path``.

All Bedrock plumbing — client construction, retry, structured-output
parsing, the per-pipeline cache-stat accumulator — lives in
:mod:`claude_sql.llm_shared`.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import anyio
import polars as pl
from loguru import logger

from claude_sql import checkpointer, retry_queue
from claude_sql.llm_shared import (
    CONFLICTS_SYSTEM_PROMPT,
    _build_bedrock_client,
    _count_pending_sessions,
    _estimate_cost,
    classify_one,
)
from claude_sql.parquet_shards import read_all, write_part
from claude_sql.schemas import SESSION_CONFLICTS_SCHEMA
from claude_sql.session_text import iter_session_texts, session_bounds

if TYPE_CHECKING:
    import duckdb

    from claude_sql.config import Settings


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
            classify_one(
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
