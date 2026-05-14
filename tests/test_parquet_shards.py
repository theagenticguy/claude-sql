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
    replace_sessions,
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


# ---------------------------------------------------------------------------
# replace_sessions — the writer-side dedup helper (GH #45)
# ---------------------------------------------------------------------------


def _keyed_df(session_id: str, n: int, *, base: int = 0) -> pl.DataFrame:
    """DataFrame with a ``session_id`` key column for replace_sessions tests."""
    return pl.DataFrame(
        {
            "session_id": [session_id] * n,
            "uuid": [f"u-{session_id}-{i + base}" for i in range(n)],
            "v": list(range(base, base + n)),
        },
        schema={"session_id": pl.Utf8, "uuid": pl.Utf8, "v": pl.Int64},
    )


def test_replace_sessions_returns_zero_on_empty_cache(tmp_path: Path) -> None:
    assert (
        replace_sessions(
            tmp_path / "missing",
            key_column="session_id",
            session_ids=["S1"],
        )
        == 0
    )
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert replace_sessions(empty_dir, key_column="session_id", session_ids=["S1"]) == 0


def test_replace_sessions_returns_zero_when_no_ids(tmp_path: Path) -> None:
    target = tmp_path / "out"
    write_part(target, _keyed_df("S1", 3))
    assert replace_sessions(target, key_column="session_id", session_ids=[]) == 0
    df = read_all(target)
    assert df is not None
    assert df.height == 3


def test_replace_sessions_unlinks_shard_when_all_rows_match(tmp_path: Path) -> None:
    """A shard that contains only the target session becomes empty → unlink."""
    target = tmp_path / "out"
    write_part(target, _keyed_df("S1", 4))
    [before] = iter_part_files(target)
    removed = replace_sessions(target, key_column="session_id", session_ids=["S1"])
    assert removed == 4
    assert iter_part_files(target) == []
    assert not before.exists()


def test_replace_sessions_rewrites_shard_with_mixed_sessions(tmp_path: Path) -> None:
    """A shard with both target and other sessions gets rewritten minus the target."""
    target = tmp_path / "out"
    mixed = pl.concat([_keyed_df("S1", 2), _keyed_df("S2", 3)], how="vertical")
    write_part(target, mixed)
    removed = replace_sessions(target, key_column="session_id", session_ids=["S1"])
    assert removed == 2
    df = read_all(target)
    assert df is not None
    # Only S2 rows survive, and none for S1.
    assert df.height == 3
    assert set(df["session_id"].to_list()) == {"S2"}
    # The shard was rewritten in place — still exactly one part file.
    assert len(iter_part_files(target)) == 1


def test_replace_sessions_leaves_unrelated_shards_alone(tmp_path: Path) -> None:
    """Shards with no matching session_id are not rewritten."""
    target = tmp_path / "out"
    write_part(target, _keyed_df("S1", 2))
    time.sleep(0.001)
    write_part(target, _keyed_df("S2", 3))
    mtimes_before = {p: p.stat().st_mtime_ns for p in iter_part_files(target)}
    # No matching row on disk; helper returns 0 and leaves every part
    # file's mtime untouched.
    removed = replace_sessions(target, key_column="session_id", session_ids=["S_missing"])
    assert removed == 0
    mtimes_after = {p: p.stat().st_mtime_ns for p in iter_part_files(target)}
    assert mtimes_before == mtimes_after


def test_replace_sessions_across_multiple_shards(tmp_path: Path) -> None:
    """Replaces rows for one session spread across multiple shards."""
    target = tmp_path / "out"
    write_part(target, _keyed_df("S1", 2))
    time.sleep(0.001)
    write_part(target, _keyed_df("S2", 3))
    time.sleep(0.001)
    write_part(target, _keyed_df("S1", 1, base=2))

    removed = replace_sessions(target, key_column="session_id", session_ids=["S1"])
    assert removed == 3
    df = read_all(target)
    assert df is not None
    assert df.height == 3
    assert set(df["session_id"].to_list()) == {"S2"}
    # Only the surviving S2 shard remains (the two S1-only shards were unlinked).
    assert len(iter_part_files(target)) == 1


def test_replace_sessions_skips_shards_without_key_column(tmp_path: Path) -> None:
    """A shard that doesn't carry the key column is left untouched."""
    target = tmp_path / "out"
    target.mkdir()
    # A rogue shard shaped differently — no session_id column.
    foreign = target / "part-1000000000000000000.parquet"
    pl.DataFrame({"uuid": ["x"], "v": [1]}).write_parquet(foreign)
    time.sleep(0.001)
    write_part(target, _keyed_df("S1", 1))

    removed = replace_sessions(target, key_column="session_id", session_ids=["S1"])
    # S1 shard lost its one row and was unlinked; foreign shard survives.
    assert removed == 1
    surviving = iter_part_files(target)
    assert surviving == [foreign]


def test_replace_sessions_legacy_single_file(tmp_path: Path) -> None:
    """Legacy ``*.parquet`` paths filter in place, or unlink when emptied."""
    legacy = tmp_path / "out.parquet"
    mixed = pl.concat([_keyed_df("S1", 2), _keyed_df("S2", 3)], how="vertical")
    write_part(legacy, mixed)

    removed = replace_sessions(legacy, key_column="session_id", session_ids=["S1"])
    assert removed == 2
    assert legacy.exists()
    df = read_all(legacy)
    assert df is not None
    assert df.height == 3
    assert set(df["session_id"].to_list()) == {"S2"}

    # Now drop S2 too — the legacy file should be unlinked.
    removed = replace_sessions(legacy, key_column="session_id", session_ids=["S2"])
    assert removed == 3
    assert not legacy.exists()


def test_replace_sessions_is_idempotent_on_second_call(tmp_path: Path) -> None:
    """Calling twice with the same ids is safe — the second call is a no-op."""
    target = tmp_path / "out"
    mixed = pl.concat([_keyed_df("S1", 2), _keyed_df("S2", 3)], how="vertical")
    write_part(target, mixed)

    first = replace_sessions(target, key_column="session_id", session_ids=["S1"])
    second = replace_sessions(target, key_column="session_id", session_ids=["S1"])
    assert first == 2
    assert second == 0
    df = read_all(target)
    assert df is not None
    assert df.height == 3
