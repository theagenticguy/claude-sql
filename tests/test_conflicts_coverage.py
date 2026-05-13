"""Patch-coverage fillers for :mod:`claude_sql.conflicts_worker`.

The v1.0 storage shape and "happy path" are pinned by
``test_conflicts_storage_v2.py``. This module covers the dispatch / retry
/ purge / dry-run branches not exercised there:

* unreadable-shard purge (``_purge_legacy_shards`` ``OSError`` branch);
* legacy single-file path (``unlink`` branch instead of ``rmtree``);
* retry-queue drain logging when an entry is due;
* already-classified anti-join skip + empty-pending early return;
* per-session Sonnet exception → warn + ``retry_queue.enqueue`` + no row;
* degenerate-pair guard (``turn_a == turn_b`` and missing UUIDs);
* ``--dry-run`` plan dict shape against a populated corpus.

All Bedrock calls are mocked. The tests share corpus scaffolding with
``test_conflicts_storage_v2.py`` (see :func:`_build_con` below).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from claude_sql import checkpointer, conflicts_worker, llm_shared, retry_queue
from claude_sql.config import Settings
from claude_sql.parquet_shards import iter_part_files, read_all
from claude_sql.sql_views import register_raw, register_views
from conftest import _seed_subagent_stub, make_user_msg, write_session_jsonl

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client_cache() -> Iterator[None]:
    """Wipe the boto3 client cache before/after each test."""
    llm_shared._CLIENT_CACHE.clear()
    yield
    llm_shared._CLIENT_CACHE.clear()


@pytest.fixture(autouse=True)
def _noop_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defang tenacity / time.sleep so retry paths run instantly."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_kw: None)
    try:
        import tenacity.nap

        monkeypatch.setattr(tenacity.nap.time, "sleep", lambda *_a, **_kw: None)
    except (ImportError, AttributeError):
        # tenacity.nap is private API; fall back to the time.sleep patch
        # above when this submodule isn't present.
        pass


def _build_con(
    tmp_path: Path, sessions: list[tuple[str, list[dict[str, Any]]]]
) -> duckdb.DuckDBPyConnection:
    """Write ``sessions`` to a fresh corpus and return a registered connection."""
    proj = tmp_path / "projects" / "proj-cov"
    for sid, msgs in sessions:
        write_session_jsonl(proj / f"{sid}.jsonl", messages=msgs)
    sa_glob, sa_meta = _seed_subagent_stub(tmp_path)
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    con = duckdb.connect(":memory:")
    register_raw(con, glob=glob, subagent_glob=sa_glob, subagent_meta_glob=sa_meta)
    register_views(con)
    return con


def _one_session_con(tmp_path: Path, sid: str = "sess-cov-1") -> duckdb.DuckDBPyConnection:
    """One session, one substantive user message — clears the 32-char filter."""
    return _build_con(
        tmp_path,
        [
            (
                sid,
                [
                    make_user_msg(
                        "u-cov-1",
                        sid,
                        "substantive user message that easily clears the messages_text filter",
                        ts="2026-04-10T10:00:00.000Z",
                    )
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. _purge_legacy_shards — unreadable shard branch
# ---------------------------------------------------------------------------


def test_purge_legacy_shards_drops_unreadable_part(tmp_path: Path) -> None:
    """A 0-byte shard trips ``OSError``/``ArrowInvalid`` → cache wiped."""
    cache = tmp_path / "conflicts_dir"
    cache.mkdir()
    bad = cache / "part-1.parquet"
    bad.write_bytes(b"")  # 0 bytes — pyarrow raises ArrowInvalid/OSError
    assert bad.exists()

    # Sanity: confirm pyarrow really does fail to parse this file.
    with pytest.raises((OSError, pa.ArrowInvalid)):
        _ = pq.ParquetFile(str(bad)).schema_arrow

    conflicts_worker._purge_legacy_shards(cache)

    assert not cache.exists(), "purge should rmtree the cache directory"


# ---------------------------------------------------------------------------
# 2. _purge_legacy_shards — target is a *file*, not a dir → unlink branch
# ---------------------------------------------------------------------------


def test_purge_legacy_shards_unlinks_legacy_single_file(tmp_path: Path) -> None:
    """A legacy single-file cache with a v0 shard column is ``unlink``-ed."""
    legacy = tmp_path / "conflicts.parquet"
    df = pl.DataFrame(
        {
            "session_id": ["s"],
            "conflict_idx": [0],
            "empty": [True],
            "detected_at": [datetime.now(UTC)],
        },
        schema={
            "session_id": pl.Utf8,
            "conflict_idx": pl.Int32,
            "empty": pl.Boolean,
            "detected_at": pl.Datetime("us", "UTC"),
        },
    )
    df.write_parquet(legacy)
    assert legacy.exists() and legacy.is_file()

    conflicts_worker._purge_legacy_shards(legacy)

    assert not legacy.exists(), "legacy single-file cache must be unlinked, not rmtree'd"


# ---------------------------------------------------------------------------
# 3. _purge_legacy_shards — empty cache directory is a no-op
# ---------------------------------------------------------------------------


def test_purge_legacy_shards_noop_on_empty_dir(tmp_path: Path) -> None:
    """No shards → early return, no exceptions, directory left in place."""
    cache = tmp_path / "conflicts_empty"
    cache.mkdir()
    conflicts_worker._purge_legacy_shards(cache)
    assert cache.exists()


# ---------------------------------------------------------------------------
# 4. Already-classified session is skipped (line 166 anti-join)
# ---------------------------------------------------------------------------


def test_already_classified_session_skipped(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v1.0 shard already covering the session → no Sonnet call, return 0."""
    sid = "sess-already-done"
    con = _one_session_con(tmp_path, sid=sid)

    # Seed a v1.0 shard that already contains a row for this session_id.
    # ``_purge_legacy_shards`` won't fire (no legacy markers) so the shard
    # survives, populating ``already`` and triggering the anti-join skip.
    cache_dir = tmp_settings.conflicts_parquet_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed = pl.DataFrame(
        {
            "session_id": [sid],
            "turn_a_uuid": ["seed-a"],
            "turn_b_uuid": ["seed-b"],
            "conflict_kind": ["disagreement"],
            "severity": ["low"],
            "agent_position": ["A"],
            "user_position": ["B"],
            "confidence": [0.5],
            "detected_at": [datetime.now(UTC)],
        },
        schema=conflicts_worker._CONFLICTS_PARQUET_SCHEMA,
    )
    seed.write_parquet(cache_dir / "part-seeded.parquet")

    # Sonnet must NOT be called; if it is, the test fails loudly.
    fail_if_called = AsyncMock(side_effect=AssertionError("classify_one should not run"))
    monkeypatch.setattr(conflicts_worker, "classify_one", fail_if_called)
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 0
    fail_if_called.assert_not_called()
    con.close()


# ---------------------------------------------------------------------------
# 5. Empty corpus → "no pending sessions" early return
# ---------------------------------------------------------------------------


def test_empty_corpus_returns_zero(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corpus with no qualifying messages → 0 returned, no client built."""
    # Drop a single short user message (under the 32-char ``messages_text``
    # filter). ``register_raw`` succeeds, ``messages_text`` is empty, and
    # the worker takes the "no pending sessions" early return.
    proj = tmp_path / "projects" / "proj-empty"
    write_session_jsonl(
        proj / "tiny.jsonl",
        messages=[
            make_user_msg(
                "u-tiny",
                "sess-tiny",
                "short",  # <32 chars → filtered out of messages_text
                ts="2026-04-10T10:00:00.000Z",
            )
        ],
    )
    sa_glob, sa_meta = _seed_subagent_stub(tmp_path)
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    con = duckdb.connect(":memory:")
    register_raw(con, glob=glob, subagent_glob=sa_glob, subagent_meta_glob=sa_meta)
    register_views(con)

    fail_if_called = AsyncMock(side_effect=AssertionError("classify_one should not run"))
    monkeypatch.setattr(conflicts_worker, "classify_one", fail_if_called)

    def _no_client(_s: Settings) -> object:
        raise AssertionError("client must not be built when there are zero pending sessions")

    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", _no_client)

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 0
    fail_if_called.assert_not_called()
    con.close()


# ---------------------------------------------------------------------------
# 6. Sonnet raises → warn + retry_queue.enqueue + no parquet row
# ---------------------------------------------------------------------------


def test_classify_exception_enqueues_retry(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``classify_one`` raises → row queued for retry, parquet stays empty."""
    sid = "sess-blowup"
    con = _one_session_con(tmp_path, sid=sid)

    boom = AsyncMock(side_effect=RuntimeError("simulated bedrock failure"))
    monkeypatch.setattr(conflicts_worker, "classify_one", boom)
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 0  # nothing succeeded
    boom.assert_called()  # but Sonnet was invoked

    # Retry queue carries one row for this session.
    pending = retry_queue.pending_count(tmp_settings.checkpoint_db_path, pipeline="conflicts")
    assert pending >= 1

    # No parquet rows landed.
    df = read_all(tmp_settings.conflicts_parquet_path)
    if df is not None:
        assert df.height == 0
    con.close()


# ---------------------------------------------------------------------------
# 7. Retry-queue drain log line (lines 160-161)
# ---------------------------------------------------------------------------


def test_retry_queue_drain_path(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queued retry entry is drained back into ``keep`` and re-classified."""
    sid = "sess-retry-drain"
    con = _one_session_con(tmp_path, sid=sid)

    # Pre-seed the retry queue with this session_id so ``drain`` returns it.
    # ``enqueue`` schedules the next attempt 2 minutes out; pass a backdated
    # ``now`` so the entry is already due.
    backdate = datetime.now(UTC) - timedelta(hours=1)
    retry_queue.enqueue(
        tmp_settings.checkpoint_db_path,
        pipeline="conflicts",
        unit_id=sid,
        error="prior failure",
        now=backdate,
    )

    fake = AsyncMock(return_value={"conflicts": []})
    monkeypatch.setattr(conflicts_worker, "classify_one", fake)
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 1
    fake.assert_called_once()

    # The queued entry is marked done now.
    remaining = retry_queue.pending_count(tmp_settings.checkpoint_db_path, pipeline="conflicts")
    assert remaining == 0
    con.close()


# ---------------------------------------------------------------------------
# 8. Degenerate pairs (lines 231-238)
# ---------------------------------------------------------------------------


def test_degenerate_pair_same_uuid_skipped(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``turn_a == turn_b`` is dropped before reaching parquet."""
    sid = "sess-dupe-pair"
    con = _one_session_con(tmp_path, sid=sid)

    payload = {
        "conflicts": [
            {
                "turn_a_uuid": "x",
                "turn_b_uuid": "x",  # degenerate
                "conflict_kind": "disagreement",
                "severity": "low",
                "agent_position": "A",
                "user_position": "B",
                "confidence": 0.5,
            }
        ]
    }
    monkeypatch.setattr(conflicts_worker, "classify_one", AsyncMock(return_value=payload))
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 1  # session was processed even though row was dropped

    df = read_all(tmp_settings.conflicts_parquet_path)
    if df is not None:
        assert df.height == 0, "degenerate pair must not land in the parquet"
    con.close()


def test_degenerate_pair_missing_uuid_skipped(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``turn_b_uuid`` missing → guard fires, row dropped."""
    sid = "sess-missing-b"
    con = _one_session_con(tmp_path, sid=sid)

    payload = {
        "conflicts": [
            {
                "turn_a_uuid": "good-a",
                "turn_b_uuid": "",  # falsy → dropped
                "conflict_kind": "correction",
                "severity": "medium",
                "agent_position": "A",
                "user_position": "B",
                "confidence": 0.7,
            }
        ]
    }
    monkeypatch.setattr(conflicts_worker, "classify_one", AsyncMock(return_value=payload))
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 1

    df = read_all(tmp_settings.conflicts_parquet_path)
    if df is not None:
        assert df.height == 0
    con.close()


def test_mixed_valid_and_degenerate_only_keeps_valid(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One degenerate pair beside a valid one → only the valid pair lands."""
    sid = "sess-mixed-pairs"
    con = _one_session_con(tmp_path, sid=sid)

    payload = {
        "conflicts": [
            {
                "turn_a_uuid": "dupe",
                "turn_b_uuid": "dupe",  # degenerate
                "conflict_kind": "disagreement",
                "severity": "low",
                "agent_position": "x",
                "user_position": "y",
                "confidence": 0.4,
            },
            {
                "turn_a_uuid": "real-a",
                "turn_b_uuid": "real-b",  # valid
                "conflict_kind": "correction",
                "severity": "medium",
                "agent_position": "use Sonnet",
                "user_position": "use Opus",
                "confidence": 0.85,
            },
        ]
    }
    monkeypatch.setattr(conflicts_worker, "classify_one", AsyncMock(return_value=payload))
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 1

    df = read_all(tmp_settings.conflicts_parquet_path)
    assert df is not None
    assert df.height == 1
    pair = (df["turn_a_uuid"].to_list()[0], df["turn_b_uuid"].to_list()[0])
    assert pair == ("real-a", "real-b")
    con.close()


# ---------------------------------------------------------------------------
# 9. Dry-run plan dict (lines 306-319)
# ---------------------------------------------------------------------------


def test_dry_run_plan_shape_populated_corpus(
    tmp_path: Path,
    tmp_settings: Settings,
) -> None:
    """``dry_run=True`` returns a plan dict with non-zero candidates."""
    con = _one_session_con(tmp_path)

    plan = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=True)
    assert isinstance(plan, dict)
    assert plan["pipeline"] == "conflicts"
    assert plan["candidates"] >= 1
    assert plan["llm_calls"] == plan["candidates"]
    assert plan["dry_run"] is True
    assert plan["model"] == tmp_settings.sonnet_model_id
    assert plan["since_days"] is None
    assert plan["limit"] is None
    # Adaptive thinking is the Settings default, not "disabled".
    assert plan["thinking"] == tmp_settings.classify_thinking
    # Cost rounded to 4 decimals.
    assert isinstance(plan["estimated_cost_usd"], float)
    con.close()


def test_dry_run_no_thinking_flag_overrides_thinking(
    tmp_path: Path,
    tmp_settings: Settings,
) -> None:
    """``no_thinking=True`` resolves to ``thinking="disabled"``."""
    con = _one_session_con(tmp_path)

    plan = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=True, no_thinking=True)
    assert isinstance(plan, dict)
    assert plan["thinking"] == "disabled"
    con.close()


def test_dry_run_subtracts_already_classified(
    tmp_path: Path,
    tmp_settings: Settings,
) -> None:
    """A v1.0 shard covering the only session → ``candidates == 0`` in dry-run."""
    sid = "sess-already-classified-dry"
    con = _one_session_con(tmp_path, sid=sid)

    cache_dir = tmp_settings.conflicts_parquet_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "session_id": [sid],
            "turn_a_uuid": ["a"],
            "turn_b_uuid": ["b"],
            "conflict_kind": ["disagreement"],
            "severity": ["low"],
            "agent_position": ["A"],
            "user_position": ["B"],
            "confidence": [0.5],
            "detected_at": [datetime.now(UTC)],
        },
        schema=conflicts_worker._CONFLICTS_PARQUET_SCHEMA,
    ).write_parquet(cache_dir / "part-cov.parquet")

    plan = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=True)
    assert isinstance(plan, dict)
    assert plan["candidates"] == 0
    assert plan["llm_calls"] == 0
    con.close()


# ---------------------------------------------------------------------------
# 10. Cache directory still exists after a successful no-op run
# ---------------------------------------------------------------------------


def test_checkpoint_skipped_session_is_filtered(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session marked complete in the checkpoint is dropped, the other runs.

    Hits both the ``sid not in keep`` continue (line 168) and the
    ``if skipped:`` info log (line 175): one session is fresh + processed,
    the other is checkpoint-skipped without a parquet shard.
    """
    sid_done = "sess-ckpt-done"
    sid_fresh = "sess-ckpt-fresh"
    con = _build_con(
        tmp_path,
        [
            (
                sid_done,
                [
                    make_user_msg(
                        "u-done",
                        sid_done,
                        "MARKER_DONE substantive message clearing the 32-char filter",
                        ts="2026-04-11T10:00:00.000Z",
                    )
                ],
            ),
            (
                sid_fresh,
                [
                    make_user_msg(
                        "u-fresh",
                        sid_fresh,
                        "MARKER_FRESH substantive message clearing the 32-char filter",
                        ts="2026-04-12T10:00:00.000Z",
                    )
                ],
            ),
        ],
    )

    # Mark sid_done as fully processed in the checkpointer with bounds far in
    # the future, so ``filter_unchanged`` skips it. We deliberately do NOT
    # write a v1.0 shard — that means ``already`` is empty, the session
    # appears in ``iter_session_texts``, and only the ``sid not in keep``
    # guard (line 168) drops it.
    far_future = datetime(2099, 1, 1, tzinfo=UTC)
    checkpointer.mark_completed(
        tmp_settings.checkpoint_db_path,
        pipeline="conflicts",
        rows=[(sid_done, far_future, far_future)],
    )

    seen_sids: list[str] = []

    async def _capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        text = args[3] if len(args) >= 4 else kwargs.get("text", "")
        if "MARKER_DONE" in text:
            seen_sids.append(sid_done)
        if "MARKER_FRESH" in text:
            seen_sids.append(sid_fresh)
        return {"conflicts": []}

    monkeypatch.setattr(conflicts_worker, "classify_one", AsyncMock(side_effect=_capture))
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 1, "only the fresh session should reach Sonnet"
    assert sid_fresh in seen_sids
    assert sid_done not in seen_sids
    con.close()


def test_cache_dir_survives_when_no_pending(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing cache dir with a v1.0 shard isn't disturbed by a no-op run."""
    sid = "sess-already-noop"
    con = _one_session_con(tmp_path, sid=sid)

    cache_dir = tmp_settings.conflicts_parquet_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "session_id": [sid],
            "turn_a_uuid": ["a"],
            "turn_b_uuid": ["b"],
            "conflict_kind": ["disagreement"],
            "severity": ["low"],
            "agent_position": ["A"],
            "user_position": ["B"],
            "confidence": [0.5],
            "detected_at": [datetime.now(UTC)],
        },
        schema=conflicts_worker._CONFLICTS_PARQUET_SCHEMA,
    ).write_parquet(cache_dir / "part-keep.parquet")

    monkeypatch.setattr(
        conflicts_worker,
        "classify_one",
        AsyncMock(side_effect=AssertionError("must not be called")),
    )
    monkeypatch.setattr(conflicts_worker, "_build_bedrock_client", lambda _s: object())

    processed = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=False)
    assert processed == 0
    # Sanity: the seed row is still on disk.
    parts = iter_part_files(cache_dir)
    assert len(parts) == 1
    con.close()
