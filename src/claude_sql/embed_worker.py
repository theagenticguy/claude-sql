"""Cohere Embed v4 backfill worker for claude-sql.

Discovers messages with no embedding yet, invokes ``cohere.embed-v4:0`` on
Amazon Bedrock in parallel batches (up to 96 texts per call), and appends the
resulting vectors to a parquet file keyed by message ``uuid``.

The worker converts the int8 response to float on insert because DuckDB's VSS
HNSW index requires ``FLOAT[]`` columns (storage loss of ~4x is accepted in
v1 — see research notes).

Tenacity retries on transient Bedrock errors via the loguru-native
``loguru_before_sleep`` helper from ``logging_setup`` — keeps every module
in claude-sql on a single logger (no stdlib ``logging`` imports).
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
import polars as pl
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ClientError,
    ConnectionError as BotoConnectionError,
    EndpointConnectionError,
    ReadTimeoutError,
    SSLError,
)
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from claude_sql import lance_store
from claude_sql.config import Settings
from claude_sql.logging_setup import loguru_before_sleep

if TYPE_CHECKING:
    import duckdb

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


def _is_retryable(exc: BaseException) -> bool:
    """Return True if ``exc`` is a Bedrock error worth retrying.

    Two buckets:
    * ``ClientError`` with a code in :data:`_RETRY_CODES` — service-level
      throttling and transient model failures.
    * Network-layer errors (SSL, connection, endpoint, read-timeout) that
      surface when long-running batches hit flaky TCP connections.
    """
    if isinstance(exc, SSLError | BotoConnectionError | EndpointConnectionError | ReadTimeoutError):
        return True
    if not isinstance(exc, ClientError):
        return False
    code = exc.response.get("Error", {}).get("Code")
    return code in _RETRY_CODES


def discover_unembedded(
    con: duckdb.DuckDBPyConnection,
    *,
    lance_uri: Path,
    since_days: int | None = None,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """Return ``(uuid, text)`` pairs that have no embedding yet.

    Parameters
    ----------
    con
        An open DuckDB connection with the ``messages_text`` view registered.
    lance_uri
        Path to the LanceDB local dataset directory backing the embeddings.
        May not exist yet — empty Lance == "no embeddings".
    since_days
        If given, only include messages with ``ts >= now() - since_days``.
    limit
        Optional row cap.

    Returns
    -------
    list of (uuid, text) tuples
        Messages needing embedding, in DuckDB's scan order.
    """
    # Read the already-embedded uuids straight from Lance via its Python API.
    # We don't go through the DuckDB ``message_embeddings`` view here because
    # the embed command runs with ``register_vss`` skipped (cli.py:1205-1213),
    # so the view isn't registered on this connection.
    # Materialise the anti-join in Python — at 22k rows the set fits
    # comfortably in memory, and DuckDB's NOT IN against a literal VALUES
    # list of that size is far slower than a Python set diff.
    embedded = lance_store.get_embedded_uuids(lance_uri)
    anti_in_python = bool(embedded)

    where = ["mt.text_content IS NOT NULL", "length(mt.text_content) > 0"]
    if since_days is not None:
        # DuckDB refuses to prepare an INTERVAL parameter; inline the coerced int.
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")

    sql = f"SELECT mt.uuid, mt.text_content FROM messages_text mt WHERE {' AND '.join(where)}"
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"

    rows = con.execute(sql).fetchall()
    # DuckDB returns UUIDs as uuid.UUID objects; polars wants str for pl.Utf8.
    pairs = [(str(r[0]), r[1]) for r in rows]
    if anti_in_python:
        pairs = [(u, t) for u, t in pairs if u not in embedded]
    return pairs


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
    # Disable botocore's internal retry layer so tenacity sees throttling
    # immediately — otherwise botocore silently absorbs 4 retries and our
    # retry policy never kicks in.  Also bump read_timeout for large batches.
    boto_cfg = BotoConfig(
        region_name=settings.region,
        retries={"max_attempts": 0, "mode": "standard"},
        read_timeout=60,
        connect_timeout=10,
    )
    return boto3.client("bedrock-runtime", config=boto_cfg)


@retry(
    # Cohere Embed v4 on Bedrock has a strict TPM bucket that replenishes over
    # tens of seconds; wait up to 60s between attempts and try up to 10 times
    # before surfacing the ThrottlingException.
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception(_is_retryable),
    before_sleep=loguru_before_sleep("WARNING"),
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
    sem = asyncio.Semaphore(settings.embed_concurrency)

    logger.info(
        "Embedding {} texts in {} batches (batch_size={}, concurrency={}, model={})",
        len(texts),
        len(batches),
        batch_size,
        settings.embed_concurrency,
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
) -> int | dict[str, Any]:
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
        If true, log the plan and return a plan dict without calling Bedrock.

    Returns
    -------
    int | dict
        Under ``dry_run=True``, a plan dict with ``{pipeline, candidates,
        batches, batch_size, concurrency, model, since_days, limit}``.
        Otherwise, count of newly embedded rows (0 when nothing is pending).
    """
    pending = discover_unembedded(
        con,
        lance_uri=settings.lance_uri,
        since_days=since_days,
        limit=limit,
    )
    if not pending:
        logger.info("No unembedded messages found - nothing to do")
        if dry_run:
            return {
                "pipeline": "embed",
                "candidates": 0,
                "batches": 0,
                "batch_size": settings.batch_size,
                "concurrency": settings.embed_concurrency,
                "model": settings.active_model_id,
                "since_days": since_days,
                "limit": limit,
                "dry_run": True,
            }
        return 0

    n_batches = (len(pending) + settings.batch_size - 1) // settings.batch_size
    logger.info(
        "Backfill plan: {} messages, {} batches, concurrency={}, model={}",
        len(pending),
        n_batches,
        settings.embed_concurrency,
        settings.active_model_id,
    )
    if dry_run:
        logger.info("dry_run=True - skipping Bedrock calls")
        return {
            "pipeline": "embed",
            "candidates": len(pending),
            "batches": n_batches,
            "batch_size": settings.batch_size,
            "concurrency": settings.embed_concurrency,
            "model": settings.active_model_id,
            "since_days": since_days,
            "limit": limit,
            "dry_run": True,
        }

    # Checkpoint every N messages so a throttling-induced timeout doesn't
    # discard work already embedded. chunk must be a multiple of batch_size.
    chunk_size = max(settings.batch_size * 4, 256)
    total_t0 = time.monotonic()
    written = 0
    db = lance_store.connect_db(settings.lance_uri)
    tbl = lance_store.open_or_create_table(db, dim=settings.output_dimension)
    chunks_since_optimize = 0
    for i in range(0, len(pending), chunk_size):
        slice_ = pending[i : i + chunk_size]
        logger.info(
            "Chunk {}/{}: embedding {} messages",
            i // chunk_size + 1,
            (len(pending) + chunk_size - 1) // chunk_size,
            len(slice_),
        )
        t0 = time.monotonic()
        texts = [p[1] for p in slice_]
        vectors = await embed_documents_async(texts, settings=settings)
        elapsed = time.monotonic() - t0
        logger.info(
            "Chunk done in {:.1f}s ({:.1f} vec/s)",
            elapsed,
            len(vectors) / elapsed if elapsed > 0 else 0.0,
        )

        now = datetime.now(UTC)
        # Force a fixed-size Array so to_arrow() produces pa.list_(pa.float32, dim)
        # which is what Lance requires for vector columns. A regular pl.List
        # becomes a variable-size list and Lance rejects it for indexing.
        df = pl.DataFrame(
            {
                "uuid": [p[0] for p in slice_],
                "model": [settings.active_model_id] * len(slice_),
                "dim": [settings.output_dimension] * len(slice_),
                "embedding": vectors,
                "embedded_at": [now] * len(slice_),
            },
            schema={
                "uuid": pl.Utf8,
                "model": pl.Utf8,
                "dim": pl.Int32,
                "embedding": pl.Array(pl.Float32, settings.output_dimension),
                "embedded_at": pl.Datetime("us", "UTC"),
            },
        )
        lance_store.add_chunk(tbl, df)
        written += len(slice_)
        chunks_since_optimize += 1
        logger.info("Checkpoint: {} rows -> {}", len(df), settings.lance_uri)

        # Compact periodically so fragment count stays bounded during long backfills.
        if chunks_since_optimize >= 8:
            lance_store.optimize_if_needed(tbl)
            chunks_since_optimize = 0

    # Final compaction + index ensure on the way out so the search command
    # sees an up-to-date index without paying brute-force scan latency.
    lance_store.optimize_if_needed(tbl)
    lance_store.ensure_index(tbl, metric=settings.hnsw_metric)

    total_elapsed = time.monotonic() - total_t0
    logger.info(
        "Backfill complete: {} embeddings in {:.1f}s ({:.1f} vec/s overall)",
        written,
        total_elapsed,
        written / total_elapsed if total_elapsed > 0 else 0.0,
    )
    return written
