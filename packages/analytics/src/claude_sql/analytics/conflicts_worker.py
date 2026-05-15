"""Stance-conflict detection pipeline.

Reads complete Claude Code session transcripts and emits one row per
detected conflict pair, keyed on ``(turn_a_uuid, turn_b_uuid)``. Sessions
with no conflicts produce ZERO rows -- the legacy ``empty=True`` sentinel
is gone (RFC 0002 §3.4).

The pair-scanner that emits one row per *adjacent turn pair* without an
LLM-per-session call is RFC §4.2 v1.1 work; for v1.0 the existing whole-
session prompt still runs but the schema and storage shape have been
rekeyed.

Output schema (per parquet shard)::

    session_id      VARCHAR
    turn_a_uuid     VARCHAR  (NOT NULL)
    turn_b_uuid     VARCHAR  (NOT NULL)
    conflict_kind   VARCHAR  (enum: disagreement|correction|reversal|impasse)
    severity        VARCHAR  (enum: low|medium|high)
    agent_position  VARCHAR
    user_position   VARCHAR
    confidence      DOUBLE
    detected_at     TIMESTAMP

All Bedrock plumbing -- client construction, retry, structured-output
parsing, the per-pipeline cache-stat accumulator -- lives in
:mod:`claude_sql.llm_shared`.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import anyio
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from claude_sql.core import checkpointer, retry_queue
from claude_sql.core.llm_shared import (
    CONFLICTS_SYSTEM_PROMPT,
    _build_bedrock_client,
    _count_pending_sessions,
    _estimate_cost,
    classify_one,
    pipeline_cache_stats,
)
from claude_sql.core.parquet_shards import iter_part_files, read_all, write_part
from claude_sql.core.schemas import SESSION_CONFLICTS_SCHEMA
from claude_sql.core.session_text import iter_session_texts, session_bounds

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

    from claude_sql.core.config import Settings


# v1.0 parquet schema — kept as a module constant so the worker, the test
# suite, and any future migration code share one source of truth.  The
# polars schema dict accepts both ``DataType`` instances and the class
# objects (``pl.Utf8``); the runtime accepts both, but the static type
# alias is the explicit ``DataType`` instance form.
_CONFLICTS_PARQUET_SCHEMA: dict[str, pl.DataType] = {
    "session_id": pl.Utf8(),
    "turn_a_uuid": pl.Utf8(),
    "turn_b_uuid": pl.Utf8(),
    "conflict_kind": pl.Utf8(),
    "severity": pl.Utf8(),
    "agent_position": pl.Utf8(),
    "user_position": pl.Utf8(),
    "confidence": pl.Float64(),
    "detected_at": pl.Datetime("us", "UTC"),
}

# Columns that prove a shard was written under the legacy ``(session_id,
# conflict_idx)`` schema. Encountering either marks the entire cache for
# deletion before the new run begins.
_LEGACY_SCHEMA_MARKERS: frozenset[str] = frozenset({"conflict_idx", "empty"})


def _purge_legacy_shards(target: Path) -> None:
    """Delete the conflicts cache when ANY shard carries the v0 schema.

    The v1.0 storage shape rekeys on ``(turn_a_uuid, turn_b_uuid)`` and
    drops the ``empty=True`` sentinel. A mixed directory (legacy +
    new-schema shards) would explode at view-registration time when
    ``read_parquet`` tries to unify schemas. We side-step that by nuking
    the whole cache the first time a legacy shard is detected.

    Idempotent: a fresh install (no shards) or an already-migrated cache
    (no legacy markers) is a no-op.
    """
    parts = iter_part_files(target)
    if not parts:
        return
    legacy_hit = False
    for part in parts:
        try:
            schema = pq.ParquetFile(str(part)).schema_arrow
        except (OSError, pa.ArrowInvalid):
            # A truncated or unreadable shard is worth dropping anyway —
            # the next run will re-stamp from the JSONL corpus.
            legacy_hit = True
            logger.warning("conflicts: unreadable shard {} — purging cache", part)
            break
        names = set(schema.names)
        if names & _LEGACY_SCHEMA_MARKERS:
            legacy_hit = True
            logger.warning(
                "conflicts: legacy shard {} carries {} — purging cache "
                "for v1.0 (turn_a_uuid, turn_b_uuid) rekey",
                part,
                sorted(names & _LEGACY_SCHEMA_MARKERS),
            )
            break
    if not legacy_hit:
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)


async def _conflicts_async(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None,
    limit: int | None,
    thinking_mode: str,
) -> int:
    """Async implementation behind :func:`detect_conflicts`."""
    # First run after the v0 → v1.0 rekey: drop any shard whose schema
    # carries the old ``conflict_idx`` / ``empty`` columns. Subsequent
    # runs see no markers and the call is a no-op.
    _purge_legacy_shards(settings.conflicts_parquet_path)

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

    written_pairs = 0
    processed_sessions = 0
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
                pipeline="conflicts",
            )
            for _, text in chunk
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        now = datetime.now(UTC)

        rows: list[dict[str, Any]] = []
        errors = 0
        ok_sids: set[str] = set()
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
            ok_sids.add(sid)
            conflicts = res_dict.get("conflicts") or []
            # Empty list -> zero rows for this session. The v0 sentinel
            # row is gone; the checkpointer + ``ok_sids`` below remember
            # we processed the session so we don't re-classify it.
            for c in conflicts:
                turn_a = c.get("turn_a_uuid")
                turn_b = c.get("turn_b_uuid")
                if not turn_a or not turn_b or turn_a == turn_b:
                    # Defensive: schema enforces non-empty distinct UUIDs,
                    # but a guard here keeps a buggy model output from
                    # poisoning the parquet.
                    logger.warning(
                        "conflicts: {} returned a degenerate pair "
                        "(turn_a={!r}, turn_b={!r}) — skipping",
                        sid,
                        turn_a,
                        turn_b,
                    )
                    continue
                rows.append(
                    {
                        "session_id": sid,
                        "turn_a_uuid": turn_a,
                        "turn_b_uuid": turn_b,
                        "conflict_kind": c.get("conflict_kind"),
                        "severity": c.get("severity"),
                        "agent_position": c.get("agent_position"),
                        "user_position": c.get("user_position"),
                        "confidence": float(c.get("confidence", 0.0)),
                        "detected_at": now,
                    }
                )
        if rows:
            df = pl.DataFrame(rows, schema=_CONFLICTS_PARQUET_SCHEMA)
            write_part(settings.conflicts_parquet_path, df)
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
        written_pairs += len(rows)
        processed_sessions += len(ok_sids)
        logger.info(
            "conflicts chunk {}/{}: {} sessions, {} pairs, {} errors, {:.1f}s",
            i // chunk_size + 1,
            (len(pending) + chunk_size - 1) // chunk_size,
            len(ok_sids),
            len(rows),
            errors,
            time.monotonic() - t0,
        )

    logger.info(
        "conflicts: processed {} sessions, wrote {} pair rows",
        processed_sessions,
        written_pairs,
    )
    return processed_sessions


def detect_conflicts(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
) -> int | dict[str, Any]:
    """Detect stance conflicts per session and return count of sessions processed.

    The returned int counts SESSIONS, not pair rows: a session that
    produced two conflict pairs and one that produced zero pairs both
    count as one toward the return value (and both are marked done in
    the checkpointer so a rerun is a no-op).

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
    with pipeline_cache_stats("conflicts"):
        return asyncio.run(
            _conflicts_async(
                con,
                settings,
                since_days=since_days,
                limit=limit,
                thinking_mode=thinking_mode,
            )
        )
