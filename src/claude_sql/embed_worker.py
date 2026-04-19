"""Cohere Embed v4 backfill worker for claude-sql.

Discovers messages with no embedding yet, invokes ``cohere.embed-v4:0`` on
Amazon Bedrock in parallel batches (up to 96 texts per call), and appends the
resulting vectors to a parquet file keyed by message ``uuid``.

The worker converts the int8 response to float on insert because DuckDB's VSS
HNSW index requires ``FLOAT[]`` columns (storage loss of ~4x is accepted in
v1 — see research notes).

Tenacity retries on transient Bedrock errors; ``before_sleep_log`` requires a
stdlib logger, so this module uses both loguru (primary) and a single stdlib
logger (retry hook only).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import duckdb
import polars as pl
from botocore.exceptions import ClientError
from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from claude_sql.config import Settings

#: Conservative per-text character cap before sending to Bedrock.  The real
#: model limit is 128K tokens per text; this cap keeps total payload below
#: the Bedrock 20 MB body ceiling even with a full batch of 96 large texts.
MAX_CHARS_PER_TEXT = 50_000

#: Bedrock error codes that tenacity should retry.
_RETRY_CODES: set[str] = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "ModelErrorException",
}

#: Stdlib logger used only by tenacity's ``before_sleep_log`` hook.  Everything
#: else in this module logs via loguru.
_retry_logger = logging.getLogger("claude_sql.embed_worker")


def _is_retryable(exc: BaseException) -> bool:
    """Return True if ``exc`` is a Bedrock error worth retrying."""
    if not isinstance(exc, ClientError):
        return False
    code = exc.response.get("Error", {}).get("Code")
    return code in _RETRY_CODES


def discover_unembedded(
    con: duckdb.DuckDBPyConnection,
    *,
    embeddings_parquet: Path,
    since_days: int | None = None,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """Return ``(uuid, text)`` pairs that have no embedding yet.

    Parameters
    ----------
    con
        An open DuckDB connection with the ``messages_text`` view registered.
    embeddings_parquet
        Path to the parquet of already-embedded rows; may not exist yet.
    since_days
        If given, only include messages with ``ts >= now() - since_days``.
    limit
        Optional row cap.

    Returns
    -------
    list of (uuid, text) tuples
        Messages needing embedding, in DuckDB's scan order.
    """
    if embeddings_parquet.exists():
        con.execute(
            "CREATE OR REPLACE TEMP VIEW _embedded AS SELECT uuid FROM read_parquet(?);",
            [str(embeddings_parquet)],
        )
        anti = "AND mt.uuid NOT IN (SELECT uuid FROM _embedded)"
    else:
        anti = ""

    where = ["mt.text_content IS NOT NULL", "length(mt.text_content) > 0"]
    params: list[Any] = []
    if since_days is not None:
        where.append("mt.ts >= current_timestamp - INTERVAL (?) DAY")
        params.append(since_days)

    sql = (
        f"SELECT mt.uuid, mt.text_content FROM messages_text mt WHERE {' AND '.join(where)} {anti}"
    )
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"

    rows = con.execute(sql, params).fetchall()
    return [(r[0], r[1]) for r in rows]


def _build_bedrock_client(settings: Settings) -> Any:
    """Construct a boto3 ``bedrock-runtime`` client from settings.

    Parameters
    ----------
    settings
        Application settings providing the target AWS region.

    Returns
    -------
    botocore client
        A low-level ``bedrock-runtime`` client.
    """
    return boto3.client("bedrock-runtime", region_name=settings.region)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(_retry_logger, logging.WARNING),
    reraise=True,
)
def _invoke_bedrock_sync(
    client: Any,
    model_id: str,
    texts: list[str],
    *,
    input_type: str,
    output_dimension: int,
    embedding_type: str,
) -> list[list[int]] | list[list[float]]:
    """Make one synchronous ``invoke_model`` call and return the vectors.

    Parameters
    ----------
    client
        A boto3 ``bedrock-runtime`` client.
    model_id
        Cohere Embed v4 model ID (direct or CRIS profile).
    texts
        Up to 96 strings; each is clipped to ``MAX_CHARS_PER_TEXT``.
    input_type
        Either ``"search_document"`` (corpus) or ``"search_query"``.
    output_dimension
        Target Matryoshka dimension: 256, 512, 1024, or 1536.
    embedding_type
        One of ``"int8"``, ``"float"``, ``"uint8"``, ``"binary"``, ``"ubinary"``.

    Returns
    -------
    list of list of int or float
        Flat list of vectors matching the order of ``texts``.
    """
    body = json.dumps(
        {
            "texts": [t[:MAX_CHARS_PER_TEXT] for t in texts],
            "input_type": input_type,
            "output_dimension": output_dimension,
            "embedding_types": [embedding_type],
            "truncate": "RIGHT",
        }
    )
    resp = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())
    return payload["embeddings"][embedding_type]


async def _embed_one_batch(
    client: Any,
    texts: list[str],
    model_id: str,
    *,
    input_type: str,
    output_dimension: int,
    embedding_type: str,
    sem: asyncio.Semaphore,
) -> list[list[int]] | list[list[float]]:
    """Embed a single batch under a concurrency-limiting semaphore."""
    async with sem:
        return await asyncio.to_thread(
            _invoke_bedrock_sync,
            client,
            model_id,
            texts,
            input_type=input_type,
            output_dimension=output_dimension,
            embedding_type=embedding_type,
        )


async def embed_documents_async(
    texts: list[str],
    *,
    settings: Settings,
) -> list[list[float]]:
    """Embed corpus documents in parallel and return ``float`` vectors.

    Uses ``input_type="search_document"``.  The Bedrock response (int8 or
    other type per ``settings.embedding_type``) is cast to ``float`` so the
    downstream parquet / DuckDB ``FLOAT[]`` column is directly consumable by
    the VSS HNSW index.

    Parameters
    ----------
    texts
        Full corpus to embed; this function handles batching and concurrency.
    settings
        Application settings (model, batch size, concurrency, output dim).

    Returns
    -------
    list of list of float
        One vector per input text, same order.
    """
    if not texts:
        return []

    client = _build_bedrock_client(settings)
    batch_size = settings.batch_size
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    sem = asyncio.Semaphore(settings.concurrency)

    logger.info(
        "Embedding {} texts in {} batches (batch_size={}, concurrency={}, model={})",
        len(texts),
        len(batches),
        batch_size,
        settings.concurrency,
        settings.active_model_id,
    )

    t0 = time.monotonic()
    coros = [
        _embed_one_batch(
            client,
            batch,
            settings.active_model_id,
            input_type="search_document",
            output_dimension=settings.output_dimension,
            embedding_type=settings.embedding_type,
            sem=sem,
        )
        for batch in batches
    ]
    results = await asyncio.gather(*coros)
    elapsed = time.monotonic() - t0

    vectors: list[list[float]] = [
        [float(x) for x in v] for batch_vecs in results for v in batch_vecs
    ]
    logger.info(
        "Embedded {} vectors across {} batches in {:.2f}s ({:.1f} vec/s)",
        len(vectors),
        len(batches),
        elapsed,
        len(vectors) / elapsed if elapsed > 0 else 0.0,
    )
    return vectors


def embed_query(text: str, *, settings: Settings) -> list[float]:
    """Embed a single query string for HNSW nearest-neighbor search.

    Uses ``input_type="search_query"`` and forces ``embedding_type="float"``
    regardless of ``settings.embedding_type`` because HNSW distance math
    needs float vectors.

    Parameters
    ----------
    text
        The user's natural-language query.
    settings
        Application settings (model, output dim, region).

    Returns
    -------
    list of float
        A single vector of length ``settings.output_dimension``.
    """
    client = _build_bedrock_client(settings)
    vectors = _invoke_bedrock_sync(
        client,
        settings.active_model_id,
        [text],
        input_type="search_query",
        output_dimension=settings.output_dimension,
        embedding_type="float",
    )
    return [float(x) for x in vectors[0]]


async def run_backfill(
    *,
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Discover unembedded messages, embed them, and append to parquet.

    Parameters
    ----------
    con
        An open DuckDB connection with ``messages_text`` registered.
    settings
        Application settings (model, batch size, concurrency, parquet path).
    since_days
        If given, only consider messages newer than this many days.
    limit
        Optional cap on number of messages to embed this run.
    dry_run
        If true, log the plan and return 0 without calling Bedrock.

    Returns
    -------
    int
        Count of newly embedded rows (0 for ``dry_run`` or when nothing is
        pending).
    """
    pending = discover_unembedded(
        con,
        embeddings_parquet=settings.embeddings_parquet_path,
        since_days=since_days,
        limit=limit,
    )
    if not pending:
        logger.info("No unembedded messages found - nothing to do")
        return 0

    n_batches = (len(pending) + settings.batch_size - 1) // settings.batch_size
    logger.info(
        "Backfill plan: {} messages, {} batches, concurrency={}, model={}",
        len(pending),
        n_batches,
        settings.concurrency,
        settings.active_model_id,
    )
    if dry_run:
        logger.info("dry_run=True - skipping Bedrock calls")
        return 0

    t0 = time.monotonic()
    texts = [p[1] for p in pending]
    vectors = await embed_documents_async(texts, settings=settings)
    elapsed = time.monotonic() - t0
    logger.info("Embedded {} vectors in {:.1f}s", len(vectors), elapsed)

    now = datetime.now(UTC)
    df = pl.DataFrame(
        {
            "uuid": [p[0] for p in pending],
            "model": [settings.active_model_id] * len(pending),
            "dim": [settings.output_dimension] * len(pending),
            "embedding": vectors,
            "embedded_at": [now] * len(pending),
        }
    )

    path = settings.embeddings_parquet_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pl.read_parquet(path)
        df = pl.concat([existing, df], how="diagonal_relaxed")
    df.write_parquet(path)
    logger.info("Wrote {} total embeddings to {}", len(df), path)

    return len(pending)
