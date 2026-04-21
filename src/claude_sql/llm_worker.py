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
import logging
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import boto3
import polars as pl
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    ReadTimeoutError,
    SSLError,
)
from botocore.exceptions import (
    ConnectionError as BotoConnectionError,
)
from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from claude_sql import checkpointer, retry_queue
from claude_sql.schemas import (
    MESSAGE_TRAJECTORY_SCHEMA,
    SESSION_CLASSIFICATION_SCHEMA,
    SESSION_CONFLICTS_SCHEMA,
)
from claude_sql.session_text import iter_session_texts, session_bounds

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

    from claude_sql.config import Settings


_RETRY_CODES: set[str] = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "ModelErrorException",
}
_retry_logger = logging.getLogger("claude_sql.llm_worker")


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


def _build_bedrock_client(settings: Settings) -> Any:
    """Construct a ``bedrock-runtime`` client tuned for classification.

    ``retries={"max_attempts": 0}`` disables botocore's internal retry layer so
    tenacity sees throttling immediately.  ``read_timeout=300`` accommodates
    Sonnet 4.6 thinking which can hold the connection longer than the default.
    """
    boto_cfg = BotoConfig(
        region_name=settings.region,
        retries={"max_attempts": 0, "mode": "standard"},
        read_timeout=300,
        connect_timeout=10,
    )
    return boto3.client("bedrock-runtime", config=boto_cfg)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(_retry_logger, logging.WARNING),
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
    if thinking_mode == "adaptive":
        body["thinking"] = {"type": "adaptive"}
    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())
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
    sem: asyncio.Semaphore,
) -> dict:
    """Run one classification call under the concurrency semaphore."""
    async with sem:
        return await asyncio.to_thread(
            _invoke_classifier_sync,
            client,
            model_id,
            schema,
            text,
            max_tokens=max_tokens,
            thinking_mode=thinking_mode,
        )


def _estimate_cost(
    n_items: int,
    avg_in_tokens: int,
    avg_out_tokens: int,
    pricing: tuple[float, float],
) -> float:
    """Back-of-envelope dollar estimate for ``n_items`` classification calls."""
    in_rate, out_rate = pricing
    return (n_items * avg_in_tokens * in_rate + n_items * avg_out_tokens * out_rate) / 1_000_000


def _append_parquet(path: Path, df: pl.DataFrame) -> None:
    """Append-by-rewrite: load existing parquet, concat, write.

    Treats zero-byte / truncated files as absent so aborted runs don't lock
    the output into a corrupt state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 16:
        existing = pl.read_parquet(path)
        df = pl.concat([existing, df], how="diagonal_relaxed")
    df.write_parquet(path)


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
    if (
        settings.classifications_parquet_path.exists()
        and settings.classifications_parquet_path.stat().st_size > 16
    ):
        done_df = pl.read_parquet(settings.classifications_parquet_path)
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
    sem = asyncio.Semaphore(settings.concurrency)
    chunk_size = max(settings.batch_size * 4, 256)
    logger.info(
        "classify: {} pending, model={}, thinking={}, concurrency={}, chunks of {}",
        len(pending),
        settings.sonnet_model_id,
        thinking_mode,
        settings.concurrency,
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
            _append_parquet(settings.classifications_parquet_path, df)

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
) -> int:
    """Classify pending sessions and return count of successful classifications."""
    thinking_mode = "disabled" if no_thinking else settings.classify_thinking

    if dry_run:
        already: set[str] = set()
        if (
            settings.classifications_parquet_path.exists()
            and settings.classifications_parquet_path.stat().st_size > 16
        ):
            already = set(
                pl.read_parquet(settings.classifications_parquet_path)["session_id"].to_list()
            )
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
        return 0

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
    if (
        settings.trajectory_parquet_path.exists()
        and settings.trajectory_parquet_path.stat().st_size > 16
    ):
        already = set(pl.read_parquet(settings.trajectory_parquet_path)["uuid"].to_list())

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
        _append_parquet(settings.trajectory_parquet_path, df)

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
    sem = asyncio.Semaphore(settings.concurrency)
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
            _append_parquet(settings.trajectory_parquet_path, df)
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
) -> int:
    """Per-message sentiment + transition classification."""
    thinking_mode = "disabled" if no_thinking else settings.classify_thinking
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
        cost = _estimate_cost(n // 2, 500, 50, settings.sonnet_pricing)
        logger.info(
            "trajectory --dry-run: {} messages, estimated LLM cost ~${:.2f}",
            n,
            cost,
        )
        return 0
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
    if (
        settings.conflicts_parquet_path.exists()
        and settings.conflicts_parquet_path.stat().st_size > 16
    ):
        already = set(pl.read_parquet(settings.conflicts_parquet_path)["session_id"].to_list())

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
    sem = asyncio.Semaphore(settings.concurrency)
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
            _append_parquet(settings.conflicts_parquet_path, df)
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
) -> int:
    """Detect stance conflicts per session and return count processed."""
    thinking_mode = "disabled" if no_thinking else settings.classify_thinking
    if dry_run:
        already: set[str] = set()
        if (
            settings.conflicts_parquet_path.exists()
            and settings.conflicts_parquet_path.stat().st_size > 16
        ):
            already = set(pl.read_parquet(settings.conflicts_parquet_path)["session_id"].to_list())
        pending_count = _count_pending_sessions(
            con, already=already, since_days=since_days, limit=limit
        )
        cost = _estimate_cost(pending_count, 6000, 400, settings.sonnet_pricing)
        logger.info(
            "conflicts --dry-run: {} sessions, estimated cost ~${:.2f}",
            pending_count,
            cost,
        )
        return 0
    return asyncio.run(
        _conflicts_async(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            thinking_mode=thinking_mode,
        )
    )
