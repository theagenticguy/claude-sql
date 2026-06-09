"""Local LanceDB embeddings store.

Replaces the sharded ``embeddings/part-*.parquet`` write path AND the
DuckDB-backed ``hnsw.duckdb`` ATTACH path with a single LanceDB local
dataset. Lance is exactly the right shape for "append-only AI artifact
store with versioning + native vector search":

* writes are append-only fragments via ``tbl.add()``
* compaction is explicit via ``tbl.optimize()`` (replaces ``cache compact``)
* the vector index lives next to the data
* DuckDB reads it back via the lance core extension
  (``INSTALL lance; LOAD lance; ATTACH (TYPE LANCE)``)

Removed entirely:
* ``parquet_shards`` for the embeddings cache (other caches still use it)
* ``hnsw.duckdb`` ATTACH path with experimental persistence flag
* ``_hnsw_rebuild_needed`` mtime dance
* unlink-on-IOException destroy-and-rebuild on lock conflicts

Migration: on first connect, if the legacy ``embeddings/`` directory exists
and the new lance dataset doesn't, the old parquet shards are read once and
written into the lance dataset. The legacy directory is left in place for
one minor release of overlap; cleanup is manual.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import lancedb
import polars as pl
import pyarrow as pa
from loguru import logger

#: Lance table name inside the namespace.
TABLE_NAME = "embeddings"

#: Glob pattern matching legacy parquet shards (for one-time migration).
_LEGACY_PART_GLOB = "part-*.parquet"

# Process-local connection cache. lancedb.connect() is cheap (no I/O until a
# table is opened) but caching avoids re-running the consistency-interval setup.
_DB_CACHE: dict[str, lancedb.DBConnection] = {}


def _has_table(db: lancedb.DBConnection, name: str) -> bool:
    """True iff ``name`` exists in the LanceDB root namespace.

    ``db.list_tables()`` returns a ``ListTablesResponse`` (pydantic model)
    with a ``tables: list[str]`` field plus an opaque ``page_token``. The
    local namespace is small enough that pagination never kicks in, so
    we read ``.tables`` directly. ``db.table_names()`` is deprecated as
    of lancedb 0.30 (emits ``DeprecationWarning``) — don't reach for it.
    """
    return name in db.list_tables().tables


def lance_schema(dim: int) -> pa.Schema:
    """Pyarrow schema for the embeddings table.

    The ``embedding`` column is a FIXED-SIZE list of float32. Lance treats
    this as a vector column for indexing — a regular ``pa.list_(pa.float32())``
    won't work for ``create_index``.
    """
    return pa.schema(
        [
            pa.field("uuid", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("dim", pa.int32(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("embedded_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )


def connect_db(uri: Path | str) -> lancedb.DBConnection:
    """Open (or reuse) a LanceDB connection rooted at ``uri``.

    ``read_consistency_interval=timedelta(0)`` makes every read see writes
    from other processes immediately. Without this, readers get a stale
    manifest snapshot and miss recently appended fragments.
    """
    key = str(Path(uri).resolve())
    cached = _DB_CACHE.get(key)
    if cached is not None:
        return cached
    Path(uri).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(key, read_consistency_interval=timedelta(0))
    _DB_CACHE[key] = db
    return db


def open_or_create_table(db: lancedb.DBConnection, *, dim: int) -> Any:
    """Open the embeddings table, creating it with the right schema if missing."""
    if _has_table(db, TABLE_NAME):
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=lance_schema(dim), mode="create")


def add_chunk(tbl: Any, df: pl.DataFrame) -> None:
    """Append one chunk of embeddings.

    The polars DataFrame must have ``embedding: pl.Array(pl.Float32, dim)``
    — the fixed-size list shape Lance expects. ``to_arrow()`` preserves
    the distinction; a regular ``pl.List`` becomes a variable-size list and
    Lance will reject it as a vector column for indexing.
    """
    arrow_table = df.to_arrow()
    tbl.add(arrow_table)


def ensure_index(tbl: Any, *, metric: str = "cosine") -> None:
    """Create the IVF_HNSW_SQ vector index on the ``embedding`` column.

    No-op if an index already exists. Index name is exactly ``IVF_HNSW_SQ``
    (scalar-quantized HNSW) — there is no plain ``IVF_HNSW`` literal in
    LanceDB. SQ gives a good size/recall balance at our 22k-row scale.
    """
    try:
        existing = tbl.list_indices()
    except (AttributeError, RuntimeError):
        existing = []
    for idx in existing:
        if getattr(idx, "column", None) == "embedding":
            return
        if "embedding" in getattr(idx, "columns", []):
            return
    try:
        tbl.create_index(
            metric=metric,
            vector_column_name="embedding",
            index_type="IVF_HNSW_SQ",
        )
    except (RuntimeError, ValueError) as exc:
        # Index creation can fail on tiny tables (no vectors yet, or
        # k-means seeding hits a degenerate split). Log and continue —
        # cosine search still works without an index, just slower.
        logger.warning("LanceDB create_index failed ({}); falling back to brute-force scan", exc)


def optimize_if_needed(tbl: Any) -> None:
    """Compact accumulated fragments + clean up old versions.

    Lance accumulates one fragment per ``tbl.add()`` call. Past ~100
    fragments read latency starts to degrade; we trigger compaction much
    earlier (every 8 chunks during a backfill) because the embeddings
    table is small (~22k rows total). ``tbl.optimize()`` returns ``None``
    and mutates in place — no stats to log.
    """
    try:
        tbl.optimize()
    except (RuntimeError, AttributeError) as exc:
        logger.warning("Lance optimize failed: {}", exc)


def migrate_from_parquet_shards(
    *,
    legacy_dir: Path,
    lance_uri: Path,
    dim: int,
    delete_legacy: bool = False,
) -> int:
    """One-time copy from the legacy ``embeddings/part-*.parquet`` directory.

    Idempotent: if the lance table already has rows, returns 0 and skips.
    Returns the number of rows migrated. The legacy directory is left in
    place by default for rollback — pass ``delete_legacy=True`` only when
    the user has opted into cleanup.
    """
    if not legacy_dir.exists() or not legacy_dir.is_dir():
        return 0
    parts = sorted(legacy_dir.glob(_LEGACY_PART_GLOB))
    if not parts:
        return 0

    db = connect_db(lance_uri)
    if _has_table(db, TABLE_NAME):
        existing = db.open_table(TABLE_NAME).count_rows()
        if existing > 0:
            logger.debug(
                "Lance table {} already has {} rows; skipping legacy migration",
                TABLE_NAME,
                existing,
            )
            return 0

    df = pl.read_parquet([str(p) for p in parts])
    if df.is_empty():
        return 0

    # The legacy parquet shape used `dim: pl.UInt16`; Lance schema wants Int32.
    # Embedding column drift (e.g. List vs Array) should raise loudly at
    # `tbl.add()` rather than be silently re-cast here — masking it hid drift
    # bugs in the past. Drop the columns Lance doesn't know about (legacy may
    # carry extras) and let polars / Lance complain about type drift.
    if "dim" in df.columns and df.schema["dim"] != pl.Int32:
        df = df.with_columns(pl.col("dim").cast(pl.Int32))
    keep = ["uuid", "model", "dim", "embedding", "embedded_at"]
    df = df.select([c for c in keep if c in df.columns])

    tbl = open_or_create_table(db, dim=dim)
    add_chunk(tbl, df)
    optimize_if_needed(tbl)
    ensure_index(tbl)

    n_rows = len(df)
    logger.info(
        "Migrated {} embeddings from {} legacy shards at {} -> {}",
        n_rows,
        len(parts),
        legacy_dir,
        lance_uri,
    )
    if delete_legacy:
        import contextlib

        for p in parts:
            p.unlink()
        with contextlib.suppress(OSError):
            legacy_dir.rmdir()
    return n_rows


def count_rows(lance_uri: Path) -> int:
    """Return the row count of the embeddings Lance table, or 0 when missing."""
    if not lance_uri.exists():
        return 0
    db = connect_db(lance_uri)
    if not _has_table(db, TABLE_NAME):
        return 0
    return int(db.open_table(TABLE_NAME).count_rows())


def get_embedded_uuids(lance_uri: Path) -> set[str]:
    """Return the set of uuids that already have embeddings.

    Used by the embed worker's anti-join to discover unembedded messages.
    Reads only the ``uuid`` column via a column-projected scan so the
    ``FLOAT[dim]`` vector column is never decoded — ``to_arrow()`` has no
    projection pushdown and materializes the full N×dim float matrix before
    a ``.select(["uuid"])`` could prune it (measured: ~221 MB peak / 55 ms at
    20k×1024 for the old path vs ~0 MB / 18 ms for the projected scan). The
    explicit ``limit(count_rows())`` overrides LanceDB's default 10-row query
    cap so the scan returns every row.
    """
    n = count_rows(lance_uri)
    if n == 0:
        return set()
    db = connect_db(lance_uri)
    tbl = db.open_table(TABLE_NAME)
    arrow = tbl.search().select(["uuid"]).limit(n).to_arrow()
    return {str(u) for u in arrow.column("uuid").to_pylist()}


__all__ = [
    "TABLE_NAME",
    "add_chunk",
    "connect_db",
    "count_rows",
    "ensure_index",
    "get_embedded_uuids",
    "lance_schema",
    "migrate_from_parquet_shards",
    "open_or_create_table",
    "optimize_if_needed",
]
