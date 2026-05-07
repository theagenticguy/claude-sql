"""Tests for the sharded parquet helpers in :mod:`claude_sql.parquet_shards`.

The helpers back the directory-of-parts pattern used by the embed / classify /
trajectory / conflicts / friction caches.  Two layouts are first-class:

* sharded directory — ``write_part`` drops a fresh ``part-<ns>.parquet``
  on every call; ``read_all`` reads the union; ``iter_part_files`` lists
  the parts in sorted (== chronological) order.
* legacy single file — ``write_part`` keeps the historical read+concat
  behavior so existing single-file caches continue to work until migrated.

The legacy branch is what test_cluster_worker / test_v2_analytics / a few
others depend on with their ``tmp_path / "x.parquet"`` paths.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from claude_sql.parquet_shards import (
    count_rows,
    is_sharded_dir,
    iter_part_files,
    read_all,
    write_part,
)


def _df(n: int, *, base: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {"uuid": [f"u-{i + base}" for i in range(n)], "v": list(range(base, base + n))},
        schema={"uuid": pl.Utf8, "v": pl.Int64},
    )


def test_is_sharded_dir_true_for_missing_path_without_suffix(tmp_path: Path) -> None:
    """Missing paths default to sharded directory unless they look like ``*.parquet``."""
    assert is_sharded_dir(tmp_path / "embeddings") is True
    assert is_sharded_dir(tmp_path / "embeddings.parquet") is False


def test_is_sharded_dir_distinguishes_existing_files_from_dirs(tmp_path: Path) -> None:
    f = tmp_path / "a.parquet"
    f.write_bytes(b"x")
    d = tmp_path / "d"
    d.mkdir()
    assert is_sharded_dir(f) is False
    assert is_sharded_dir(d) is True


def test_write_part_creates_directory_and_part_file(tmp_path: Path) -> None:
    target = tmp_path / "embeddings"
    written = write_part(target, _df(3))
    assert target.is_dir()
    assert written.parent == target
    assert written.name.startswith("part-") and written.name.endswith(".parquet")
    parts = iter_part_files(target)
    assert parts == [written]


def test_write_part_appends_without_rewrite(tmp_path: Path) -> None:
    target = tmp_path / "embeddings"
    write_part(target, _df(2, base=0))
    # ``time.time_ns`` is monotonic on Linux, but sleep a hair to dodge a
    # same-nanosecond collision on virtualized CI — write_part picks a new
    # filename per call so the next part file MUST be distinct.
    time.sleep(0.001)
    write_part(target, _df(3, base=2))

    parts = iter_part_files(target)
    assert len(parts) == 2

    df = read_all(target)
    assert df is not None
    assert df.height == 5
    assert df["uuid"].to_list() == [f"u-{i}" for i in range(5)]


def test_iter_part_files_is_sorted(tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    # Write parts out of timestamp order.
    later = target / "part-2000000000000000000.parquet"
    earlier = target / "part-1000000000000000000.parquet"
    _df(1).write_parquet(later)
    _df(1).write_parquet(earlier)

    parts = iter_part_files(target)
    assert parts == [earlier, later]


def test_legacy_single_file_round_trip(tmp_path: Path) -> None:
    """A ``*.parquet`` path keeps the legacy read-concat-rewrite branch alive."""
    target = tmp_path / "legacy.parquet"
    written = write_part(target, _df(2, base=0))
    assert written == target
    written = write_part(target, _df(3, base=2))
    assert written == target  # still the same file — concat-rewrite

    df = read_all(target)
    assert df is not None
    assert df.height == 5
    # Legacy mode must NOT create a sibling directory.
    assert iter_part_files(target) == [target]
    assert not (tmp_path / "legacy").exists()


def test_read_all_returns_none_when_cache_is_empty(tmp_path: Path) -> None:
    assert read_all(tmp_path / "missing") is None
    assert read_all(tmp_path / "missing.parquet") is None
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert read_all(empty_dir) is None


def test_count_rows_sums_across_parts(tmp_path: Path) -> None:
    target = tmp_path / "out"
    write_part(target, _df(2))
    time.sleep(0.001)
    write_part(target, _df(5))
    assert count_rows(target) == 7


def test_count_rows_handles_legacy_and_missing(tmp_path: Path) -> None:
    legacy = tmp_path / "single.parquet"
    write_part(legacy, _df(4))
    assert count_rows(legacy) == 4
    assert count_rows(tmp_path / "no_such_thing") == 0
