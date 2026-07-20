"""Embedding backfill worker for claude-sql.

Discovers messages with no embedding yet, embeds them through the pluggable
:class:`claude_sql.domain.ports.EmbeddingProvider` port (Cohere on Bedrock by
default, Ollama or local ONNX BGE under optional extras), and appends the
resulting vectors to the LanceDB store keyed by message ``uuid``.

The embedder is built once per run via
:func:`claude_sql.infrastructure.embedding.build_embedder`; the document/query asymmetry,
batching, retry, and float-widening all live inside the adapters. The provider's
``model_id`` / ``dimension`` are stamped on every row and enforced against the
store's prior stamp (fail-loud on a provider switch: mixing vector spaces
silently corrupts kNN search).

Moved here from ``analytics/embed_worker.py`` in the v2 hexagonal reshape
(MIGRATION Phase C). Cohere internals (``_invoke_bedrock_sync``,
``_embed_one_batch``, ``_is_retryable``, ``MAX_CHARS_PER_TEXT``) are re-exported
from the cohere adapter for call-site stability.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from claude_sql.infrastructure.embedding import build_embedder, ensure_store_matches

# Re-exported for call-site + test stability: the Cohere invoke chain now lives
# in the cohere-bedrock adapter, but historic imports target this module.
from claude_sql.infrastructure.embedding.cohere_bedrock import (
    MAX_CHARS_PER_TEXT,
    _build_bedrock_client,
    _embed_one_batch,
    _invoke_bedrock_sync,
    _is_retryable,
)
from claude_sql.infrastructure.settings import Settings

if TYPE_CHECKING:
    import duckdb

    from claude_sql.application.ports import VectorStorePort

__all__ = [
    "MAX_CHARS_PER_TEXT",
    "_build_bedrock_client",
    "_embed_one_batch",
    "_invoke_bedrock_sync",
    "_is_retryable",
    "discover_unembedded",
    "embed_documents_async",
    "embed_query",
    "run_backfill",
]


def _ingest_stamps_view_present(con: duckdb.DuckDBPyConnection) -> bool:
    """Probe whether the ``ingest_stamps`` view is registered on ``con``.

    ``register_analytics_views`` only binds ``ingest_stamps`` when its
    parquet exists on disk (parquet-existence-gate pattern); on a fresh
    install the view is absent. Probing the catalog before injecting the
    LEFT JOIN avoids a binder error in that default state.

    The empty-stub-view alternative (always register ``ingest_stamps``,
    even as an empty stub, so the JOIN is always safe) lives in
    ``sql_views.py`` and is the cleaner long-term fix; until that lands,
    a local catalog probe keeps the join opt-in and the tests pin the
    fall-through behaviour.
    """
    row = con.execute(
        "SELECT count(*) FROM duckdb_views() WHERE view_name = 'ingest_stamps'"
    ).fetchone()
    return row is not None and int(row[0]) > 0


def discover_unembedded(
    con: duckdb.DuckDBPyConnection,
    *,
    lance_uri: Path,
    since_days: int | None = None,
    limit: int | None = None,
    store: VectorStorePort | None = None,
) -> list[tuple[str, str]]:
    """Return ``(uuid, text)`` pairs that have no embedding yet.

    When the ``ingest_stamps`` view is registered (i.e. the user has run
    ``claude-sql ingest``), the candidate set is left-joined against it
    and rows whose ``canonical_uuid`` points at a *different* uuid are
    dropped. The canonical row of each near-duplicate cluster is still
    embedded; queries against a near-dup's content fall back to the
    canonical's vector. ``LEFT JOIN`` semantics keep unstamped rows in
    the candidate set (NULL canonical_uuid → "embed it"), so a partial
    ingest run never starves the embed pipeline.

    Parameters
    ----------
    con
        An open DuckDB connection with the ``messages_text`` view registered.
        Optionally also has ``ingest_stamps`` registered (the dedup gate
        is opt-in via catalog probe).
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
    # Read the already-embedded uuids straight from Lance via the vector-store
    # port. Defaulting the adapter here keeps the ~2.6s lancedb import subtree
    # off the CLI's non-embed command paths (the adapter defers it). We don't go
    # through the DuckDB ``message_embeddings`` view because the embed command
    # runs with ``register_vss`` skipped (cli.py:1205-1213), so the view isn't
    # registered on this connection.
    # Materialise the anti-join in Python — at 22k rows the set fits
    # comfortably in memory, and DuckDB's NOT IN against a literal VALUES
    # list of that size is far slower than a Python set diff.
    if store is None:
        from claude_sql.infrastructure.adapters import LanceVectorStore

        store = LanceVectorStore(lance_uri)
    embedded = store.get_embedded_uuids()
    anti_in_python = bool(embedded)

    where = ["mt.text_content IS NOT NULL", "length(mt.text_content) > 0"]
    if since_days is not None:
        # DuckDB refuses to prepare an INTERVAL parameter; inline the coerced int.
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")

    # Inject the canonical-uuid skip only when ``ingest_stamps`` is bound;
    # otherwise the LEFT JOIN errors with a binder-time catalog miss. The
    # absent-view branch is the fresh-install default and must produce the
    # full unfiltered candidate set.
    if _ingest_stamps_view_present(con):
        join_clause = "LEFT JOIN ingest_stamps ist ON mt.uuid = ist.uuid"
        # NULL canonical_uuid (unstamped row) → keep; canonical pointing at
        # the row itself → keep; canonical pointing elsewhere → skip.
        where.append("(ist.canonical_uuid IS NULL OR ist.canonical_uuid = mt.uuid)")
    else:
        join_clause = ""

    sql = (
        f"SELECT mt.uuid, mt.text_content "
        f"FROM messages_text mt "
        f"{join_clause} "
        f"WHERE {' AND '.join(where)}"
    )
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"

    rows = con.execute(sql).fetchall()
    # DuckDB returns UUIDs as uuid.UUID objects; polars wants str for pl.Utf8.
    pairs = [(str(r[0]), r[1]) for r in rows]
    if anti_in_python:
        pairs = [(u, t) for u, t in pairs if u not in embedded]
    return pairs


async def embed_documents_async(
    texts: list[str],
    *,
    settings: Settings,
) -> list[list[float]]:
    """Embed corpus documents through the active :class:`EmbeddingProvider`.

    Thin orchestration shim over :func:`build_embedder`: the batching,
    concurrency, retry, and float-widening all live inside the selected adapter.
    Empty input short-circuits to ``[]`` without building an embedder client.

    Parameters
    ----------
    texts
        Full corpus to embed; the adapter handles batching and concurrency.
    settings
        Application settings (selects the provider + its config).

    Returns
    -------
    list of list of float
        One vector per input text, same order.
    """
    if not texts:
        return []
    embedder = build_embedder(settings)
    return await embedder.embed_documents(texts)


def embed_query(text: str, *, settings: Settings) -> list[float]:
    """Embed a single query string for nearest-neighbor search.

    Thin shim over :func:`build_embedder`: each adapter returns a float query
    vector of length ``embedder.dimension`` (Cohere forces ``float`` even when
    documents were stored int8; bge applies the query instruction prefix; Ollama
    is symmetric).

    Parameters
    ----------
    text
        The user's natural-language query.
    settings
        Application settings (selects the provider + its config).

    Returns
    -------
    list of float
        A single vector of length ``embedder.dimension``.
    """
    return build_embedder(settings).embed_query(text)


async def run_backfill(
    *,
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    store: VectorStorePort | None = None,
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
    store
        Optional :class:`VectorStorePort` for the embeddings store. Defaults to
        the module-backed :class:`~claude_sql.infrastructure.adapters.LanceVectorStore`
        over ``settings.lance_uri`` once the provider's ``dimension`` is known.

    Returns
    -------
    int | dict
        Under ``dry_run=True``, a plan dict with ``{pipeline, candidates,
        batches, batch_size, concurrency, model, since_days, limit}``.
        Otherwise, count of newly embedded rows (0 when nothing is pending).
    """
    plan_model = settings.expected_embedding_identity()[0]
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
                "model": plan_model,
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
        plan_model,
    )
    if dry_run:
        logger.info("dry_run=True - skipping embedding calls")
        return {
            "pipeline": "embed",
            "candidates": len(pending),
            "batches": n_batches,
            "batch_size": settings.batch_size,
            "concurrency": settings.embed_concurrency,
            "model": plan_model,
            "since_days": since_days,
            "limit": limit,
            "dry_run": True,
        }

    # Build the provider once for the whole run. dimension / model_id become the
    # single contract source (replacing the free settings.output_dimension).
    embedder = build_embedder(settings)
    model_id = embedder.model_id
    dim = embedder.dimension

    # Default the vector-store port now that ``dim`` is known (the adapter needs
    # it to open/create the table). Deferred build keeps the lancedb import off
    # the dry-run / nothing-pending paths above, which return before this point.
    if store is None:
        from claude_sql.infrastructure.adapters import build_vector_store

        store = build_vector_store(settings, dim=dim)

    # Fail-loud guard: if the store was written by a different provider/model,
    # refuse to append into it (mixing vector spaces silently corrupts kNN).
    identity = store.table_identity()
    if identity is not None:
        stored_model, stored_dim = identity
        ensure_store_matches(
            stored_model=stored_model,
            stored_dim=stored_dim,
            expected_model=model_id,
            expected_dim=dim,
        )

    # Checkpoint every N messages so a throttling-induced timeout doesn't
    # discard work already embedded. chunk must be a multiple of batch_size.
    chunk_size = max(settings.batch_size * 4, 256)
    total_t0 = time.monotonic()
    written = 0
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
        vectors = await embedder.embed_documents(texts)
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
                "model": [model_id] * len(slice_),
                "dim": [dim] * len(slice_),
                "embedding": vectors,
                "embedded_at": [now] * len(slice_),
            },
            schema={
                "uuid": pl.Utf8,
                "model": pl.Utf8,
                "dim": pl.Int32,
                "embedding": pl.Array(pl.Float32, dim),
                "embedded_at": pl.Datetime("us", "UTC"),
            },
        )
        store.add_chunk(df)
        written += len(slice_)
        chunks_since_optimize += 1
        logger.info("Checkpoint: {} rows -> {}", len(df), settings.lance_uri)

        # Compact periodically so fragment count stays bounded during long backfills.
        if chunks_since_optimize >= 8:
            store.optimize()
            chunks_since_optimize = 0

    # Final compaction + index ensure on the way out so the search command
    # sees an up-to-date index without paying brute-force scan latency.
    store.optimize()
    store.ensure_index(metric=settings.hnsw_metric)

    total_elapsed = time.monotonic() - total_t0
    logger.info(
        "Backfill complete: {} embeddings in {:.1f}s ({:.1f} vec/s overall)",
        written,
        total_elapsed,
        written / total_elapsed if total_elapsed > 0 else 0.0,
    )
    return written
