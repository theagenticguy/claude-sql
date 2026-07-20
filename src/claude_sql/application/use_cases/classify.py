"""Session classification pipeline.

Reads complete Claude Code session transcripts and emits one row per session
into ``settings.classifications_parquet_path`` with autonomy_tier,
work_category, success, goal, and confidence fields. Pull-once / write-many
shape: anti-join against the parquet, dispatch parallel Bedrock calls under
``settings.llm_concurrency``, write results in chunks of
``max(batch_size * 4, 256)`` for crash-resilience.

Moved here from ``analytics/classify_worker.py`` in the v2 hexagonal reshape
(MIGRATION Phase C). All Bedrock plumbing — client construction, retry,
structured-output parsing, the per-pipeline cache-stat accumulator — lives in
:mod:`claude_sql.infrastructure.bedrock.client`; the pure dollar estimate lives in
:mod:`claude_sql.domain.costs`.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import anyio
import polars as pl
from loguru import logger

from claude_sql.application.prompts import CLASSIFY_SYSTEM_PROMPT
from claude_sql.application.use_cases._shared import _count_pending_sessions
from claude_sql.domain.costs import estimate_cost as _estimate_cost
from claude_sql.infrastructure.adapters import build_cache, build_checkpoint, build_retry_queue
from claude_sql.infrastructure.bedrock.client import (
    _build_bedrock_client,
    classify_one,
    pipeline_cache_stats,
)
from claude_sql.infrastructure.bedrock.structured_output import SESSION_CLASSIFICATION_SCHEMA
from claude_sql.infrastructure.session_text_loader import iter_session_texts, session_bounds
from claude_sql.infrastructure.sqlite_state import checkpointer

if TYPE_CHECKING:
    import duckdb

    from claude_sql.application.ports import (
        CachePort,
        CheckpointPort,
        RetryQueuePort,
        TranscriptReaderPort,
    )
    from claude_sql.infrastructure.settings import Settings


async def _classify_sessions_async(
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
    """Async implementation behind :func:`classify_sessions`."""
    already: set[str] = set()
    done_df = cache.read_all(columns=["session_id"])
    if done_df is not None and done_df.height > 0:
        already = set(done_df["session_id"].to_list())

    # Checkpoint skip: compare current (last_ts, mtime) against the last run.
    # ``session_bounds`` stays a direct local call by default so the existing
    # ``monkeypatch.setattr(worker, "session_bounds", ...)`` patches keep biting;
    # when a ``TranscriptReaderPort`` is injected it owns the connection.
    bounds = (
        reader.session_bounds(since_days=since_days, limit=limit)
        if reader is not None
        else session_bounds(con, since_days=since_days, limit=limit)
    )
    unchanged_pending, skipped = checkpointer.filter_unchanged(
        ((sid, lt, mt) for sid, (lt, mt) in bounds.items()),
        pipeline="classify",
        checkpoint_db_path=settings.checkpoint_db_path,
    )
    keep = set(unchanged_pending)

    # Retry queue: pull pending retries first so they're re-enqueued into
    # `keep` even when the checkpoint would otherwise skip them.
    retry_ids = set(retry.drain(pipeline="classify"))
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
            classify_one(
                client,
                settings.sonnet_model_id,
                SESSION_CLASSIFICATION_SCHEMA,
                text,
                max_tokens=settings.classify_max_tokens,
                thinking_mode=thinking_mode,
                sem=sem,
                system=CLASSIFY_SYSTEM_PROMPT,
                pipeline="classify",
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
                retry.enqueue(pipeline="classify", unit_id=sid, error=str(res))
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
            cache.write_part(df)

        # Checkpoint the sessions we just classified — at their CURRENT bounds,
        # so a later re-run with no new messages is a no-op. Also clear those
        # sessions from the retry queue.
        if ok_rows:
            ok_sids = [row["session_id"] for row in ok_rows]
            checkpoint.mark_completed(
                pipeline="classify",
                rows=[(sid, *bounds.get(sid, (None, None))) for sid in ok_sids],
            )
            retry.mark_done(pipeline="classify", unit_ids=ok_sids)

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


def classify_sessions(
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
    """Classify pending sessions and return count of successful classifications.

    In ``--dry-run`` mode, returns a plan dict with keys ``{pipeline,
    candidates, llm_calls, avg_input_tokens, avg_output_tokens,
    estimated_cost_usd, model, thinking, since_days, limit}`` instead of the
    row count, so the CLI can emit it as structured JSON.

    ``cache`` / ``checkpoint`` / ``retry`` are the storage ports (parquet
    cache, checkpoint db, retry queue); each defaults to the module-backed
    adapter built from ``settings``. ``reader`` is an optional
    :class:`TranscriptReaderPort` for ``session_bounds``; when ``None`` the
    direct ``session_bounds(con, ...)`` call is used (preserving the existing
    ``session_bounds`` monkeypatch seam).
    """
    thinking_mode = "disabled" if no_thinking else settings.classify_thinking
    cache = cache if cache is not None else build_cache(settings.classifications_parquet_path)

    if dry_run:
        already: set[str] = set()
        done_df = cache.read_all(columns=["session_id"])
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

    checkpoint = checkpoint if checkpoint is not None else build_checkpoint(settings)
    retry = retry if retry is not None else build_retry_queue(settings)
    with pipeline_cache_stats("classify"):
        return asyncio.run(
            _classify_sessions_async(
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
