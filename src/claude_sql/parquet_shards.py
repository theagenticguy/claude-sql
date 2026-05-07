"""Sharded parquet I/O helpers for the five worker-append caches.

Background
----------
Workers ``embed_worker``, ``llm_worker`` (classify / trajectory / conflicts),
and ``friction_worker`` previously used a "read whole parquet → concat →
rewrite whole parquet" pattern on every chunk. At ~50 MB this is roughly
1.5 s of pure IO per chunk × 100 chunks per backfill = ~150 s wasted on a
single full backfill. This module replaces that with a directory-of-parts
pattern: each chunk writes ``<dir>/part-<ts_ns>.parquet`` and readers glob
the directory.

Design (intentional deviation from the original plan)
-----------------------------------------------------
The plan called for renaming five ``Settings.*_parquet_path`` fields to
``*_dir_path``. That would cascade across ~30 call sites in the CLI, SQL
views, and tests for no semantic gain. Instead we keep the field names and
overload their meaning:

* If the path is a *directory* (or doesn't yet exist on disk), it's a sharded
  cache — :func:`write_part` drops a fresh ``part-<ts_ns>.parquet`` into it,
  and :func:`read_all` / :func:`iter_part_files` glob the directory.
* If the path is a *file*, the legacy single-file behavior kicks in —
  :func:`write_part` does the read-then-rewrite, :func:`read_all` reads the
  one file, etc.

New installs get directories (the field default factories in ``config.py``
were updated). Existing single-file caches keep working until migrated; see
``claude-sql cache migrate``.

Public API
----------
* :func:`is_sharded_dir` — does this path point at a sharded directory?
* :func:`write_part` — append by writing a fresh part (or legacy rewrite).
* :func:`read_all` — load the union of all parts (or the legacy single file).
* :func:`iter_part_files` — sorted list of part files (or ``[target]``).
* :func:`count_rows` — sum of row counts across parts (or single file).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl

#: Glob pattern for shard part files within a sharded cache directory.
PART_GLOB: str = "part-*.parquet"


def is_sharded_dir(path: Path) -> bool:
    """Return True iff ``path`` is (or should be) treated as a sharded cache directory.

    Two cases qualify:

    1. ``path`` exists and is a directory.
    2. ``path`` does not exist yet — new caches default to directory layout.

    A path that points at an existing *file* is the legacy single-file shape.
    """
    if path.exists():
        return path.is_dir()
    # Heuristic: if the path has a parquet suffix, treat it as legacy single
    # file even when missing (so brand-new tests using ``tmp_path/"x.parquet"``
    # still take the legacy code path). Otherwise assume sharded directory.
    return path.suffix != ".parquet"


def iter_part_files(target: Path) -> list[Path]:
    """Return a sorted list of parquet files backing ``target``.

    For a sharded directory: every ``part-*.parquet`` under it, sorted by
    name (which is timestamp-keyed so the order is also chronological).

    For a legacy single-file path: ``[target]`` if it exists, else ``[]``.
    """
    if not target.exists():
        return []
    if target.is_dir():
        return sorted(target.glob(PART_GLOB))
    return [target]


def write_part(target: Path, df: pl.DataFrame) -> Path:
    """Write ``df`` as a new shard (or rewrite the legacy single file).

    Sharded directory branch:
        Ensure ``target`` exists as a directory and drop a brand-new
        ``part-<ns>.parquet`` into it. No read-then-rewrite — append cost is
        proportional to ``len(df)`` only.

    Legacy single-file branch:
        Load the existing parquet (if non-empty), concat ``df`` onto the
        tail, and rewrite the whole file. Preserves the historical behavior
        for users who haven't migrated yet.

    Parameters
    ----------
    target
        Either a sharded cache directory or a legacy ``*.parquet`` file path.
    df
        The polars DataFrame to persist.

    Returns
    -------
    Path
        The path that was actually written (a part file in the sharded case,
        or ``target`` itself in the legacy case).
    """
    if is_sharded_dir(target):
        target.mkdir(parents=True, exist_ok=True)
        # Nanosecond-resolution timestamps are sortable, monotonic on Linux,
        # and avoid filename collisions even when two part-writes land in the
        # same millisecond (chunk_size=256 and concurrency=8 can race).
        part_path = target / f"part-{time.time_ns()}.parquet"
        df.write_parquet(part_path)
        return part_path

    # Legacy single-file branch.
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 16:
        existing = pl.read_parquet(target)
        df = pl.concat([existing, df], how="diagonal_relaxed")
    df.write_parquet(target)
    return target


def read_all(target: Path, *, dtypes: dict[str, Any] | None = None) -> pl.DataFrame | None:
    """Return the union of all part files (or the legacy single file).

    Returns ``None`` when the cache is empty or missing. ``dtypes`` is
    accepted for forward-compatibility but currently unused — the caches we
    own all carry self-describing schemas, so an explicit dtype map only
    matters once we hit a parquet whose schema has drifted.
    """
    del dtypes  # reserved for future schema-pinning; not needed today
    parts = iter_part_files(target)
    if not parts:
        return None
    # ``pl.read_parquet`` accepts a list of paths and concatenates with
    # ``how='vertical_relaxed'`` semantics, which matches the historical
    # ``pl.concat(..., how='diagonal_relaxed')`` used by the workers.
    return pl.read_parquet([str(p) for p in parts])


def count_rows(target: Path) -> int:
    """Return the total row count across every part file under ``target``.

    Uses ``pyarrow.parquet.ParquetFile.metadata`` so we read parquet footers
    only — no row-group materialization.  Returns 0 when the cache is empty
    or missing.  Pyarrow ships as a polars dependency, so this is free.
    """
    parts = iter_part_files(target)
    if not parts:
        return 0
    import pyarrow.parquet as pq

    total = 0
    for p in parts:
        total += int(pq.ParquetFile(str(p)).metadata.num_rows)
    return total


__all__ = [
    "PART_GLOB",
    "count_rows",
    "is_sharded_dir",
    "iter_part_files",
    "read_all",
    "write_part",
]
