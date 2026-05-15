"""Coverage for the LLM pipelines: classify / trajectory / conflicts.

These tests exercise the full async pipeline machinery and the ``--dry-run``
plan paths without touching Bedrock. They sit alongside ``test_llm_worker``
(request-shape unit tests) and ``test_llm_worker_parser`` (response-shape
parser tests); together the three modules push pipeline coverage
from 25% to >=80%.

Tactics:

* Build the corpus from ``tmp_corpus`` (conftest factory) and stand up our
  own DuckDB connection with ``register_raw`` + ``register_views`` only.
  The shared ``registered_con`` fixture also runs ``register_macros``,
  which fails on a fresh connection because ``message_embeddings`` is not
  yet bound (a pre-existing fixture bug, scope-out for this change).
* Patch ``<worker>.classify_one`` (the module-level reference) with
  ``unittest.mock.AsyncMock`` so the async pipeline body runs without ever
  hitting ``boto3``. Side-effects are sequenced via ``side_effect=[...]``.
* Reset ``llm_shared._CLIENT_CACHE`` between tests via an autouse fixture
  to keep the boto3 client identity tests deterministic.
* Neutralize tenacity sleeps so retry paths run instantly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import duckdb
import polars as pl
import pytest

from claude_sql.analytics import (
    classify_worker,
    conflicts_worker,
    trajectory_worker,
)
from claude_sql.core import (
    llm_shared,
    retry_queue,
)
from claude_sql.core.config import Settings
from claude_sql.core.parquet_shards import iter_part_files
from claude_sql.core.sql_views import register_raw, register_views
from conftest import (
    _seed_subagent_stub,
    make_user_msg,
    write_session_jsonl,
)

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client_cache() -> Iterator[None]:
    """Wipe ``llm_shared._CLIENT_CACHE`` before/after every test."""
    llm_shared._CLIENT_CACHE.clear()
    yield
    llm_shared._CLIENT_CACHE.clear()


@pytest.fixture(autouse=True)
def _noop_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defang tenacity / time.sleep so retry paths don't burn wall time."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_kw: None)
    try:
        import tenacity.nap

        monkeypatch.setattr(tenacity.nap.time, "sleep", lambda *_a, **_kw: None)
    except (ImportError, AttributeError):
        # tenacity.nap is private API. ImportError covers a future tenacity
        # release dropping the submodule; AttributeError covers one that
        # stops re-exporting `time`. In both "not present in this tenacity"
        # branches the top-level time.sleep patch above is sufficient — no
        # further action needed, so we let the except body be a no-op.
        pass


@pytest.fixture
def basic_con(tmp_corpus: dict[str, Any]) -> Iterator[duckdb.DuckDBPyConnection]:
    """``register_raw`` + ``register_views`` only — skip the broken macro step.

    The shared ``registered_con`` fixture in conftest also runs
    ``register_macros``, which currently errors out because the
    ``semantic_search`` macro references ``message_embeddings`` (registered
    by ``register_vss``, not by us). Our pipelines only need the basic
    views, so we keep the setup minimal.
    """
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
    )
    register_views(con)
    try:
        yield con
    finally:
        con.close()


def _build_con(
    tmp_path: Path, sessions: list[tuple[str, list[dict[str, Any]]]]
) -> tuple[duckdb.DuckDBPyConnection, dict[str, Any]]:
    """Write ``sessions`` to a fresh corpus root and return ``(con, corpus)``."""
    proj = tmp_path / "projects" / "proj-x"
    for sid, msgs in sessions:
        write_session_jsonl(proj / f"{sid}.jsonl", messages=msgs)
    sa_glob, sa_meta = _seed_subagent_stub(tmp_path)
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    con = duckdb.connect(":memory:")
    register_raw(con, glob=glob, subagent_glob=sa_glob, subagent_meta_glob=sa_meta)
    register_views(con)
    return con, {
        "glob": glob,
        "subagent_glob": sa_glob,
        "subagent_meta_glob": sa_meta,
        "root": proj,
    }


# ===========================================================================
# 1. _count_pending_sessions
# ===========================================================================


def test_count_pending_sessions_empty_already(
    basic_con: duckdb.DuckDBPyConnection, tmp_corpus: dict[str, Any]
) -> None:
    """Empty ``already`` set → count of distinct sessions in ``messages_text``."""
    n = llm_shared._count_pending_sessions(basic_con, already=set(), since_days=None, limit=None)
    # The conftest corpus has two sessions, both with text messages clearing
    # the 32-char filter.
    assert n == 2
    # Sanity: the value matches the raw distinct-count too.
    (raw,) = basic_con.execute("SELECT count(DISTINCT session_id) FROM messages_text").fetchone()
    assert n == int(raw)


def test_count_pending_sessions_already_subtracts(
    basic_con: duckdb.DuckDBPyConnection, tmp_corpus: dict[str, Any]
) -> None:
    """``already`` containing one session → count is reduced by 1."""
    baseline = llm_shared._count_pending_sessions(
        basic_con, already=set(), since_days=None, limit=None
    )
    one_done = {tmp_corpus["session_ids"][0]}
    reduced = llm_shared._count_pending_sessions(
        basic_con, already=one_done, since_days=None, limit=None
    )
    assert reduced == baseline - 1


def test_count_pending_sessions_limit_cap(
    basic_con: duckdb.DuckDBPyConnection,
) -> None:
    """``limit`` caps the returned count regardless of corpus size."""
    capped = llm_shared._count_pending_sessions(basic_con, already=set(), since_days=None, limit=1)
    assert capped == 1


def test_count_pending_sessions_since_days_filter(
    tmp_path: Path,
) -> None:
    """``since_days=1`` filter cuts older sessions out."""
    proj = tmp_path / "projects" / "proj-y"
    sid_old = "old-session"
    sid_new = "new-session"
    # An "old" session — 30 days back, well outside any since_days=1 window.
    write_session_jsonl(
        proj / f"{sid_old}.jsonl",
        messages=[
            make_user_msg(
                "uo",
                sid_old,
                "old session content past the 32-char floor for filtering",
                ts="2025-01-01T00:00:00.000Z",
            )
        ],
    )
    # A "new" session — today's date, will pass since_days=1.
    from datetime import UTC, datetime, timedelta

    now_iso = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    write_session_jsonl(
        proj / f"{sid_new}.jsonl",
        messages=[
            make_user_msg(
                "un",
                sid_new,
                "fresh session content past the 32-char floor for filtering",
                ts=now_iso,
            )
        ],
    )
    sa_glob, sa_meta = _seed_subagent_stub(tmp_path)
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=str(tmp_path / "projects" / "*" / "*.jsonl"),
        subagent_glob=sa_glob,
        subagent_meta_glob=sa_meta,
    )
    register_views(con)

    all_n = llm_shared._count_pending_sessions(con, already=set(), since_days=None, limit=None)
    recent_n = llm_shared._count_pending_sessions(con, already=set(), since_days=1, limit=None)
    assert all_n == 2
    assert recent_n == 1
    con.close()


# ===========================================================================
# 2. classify_sessions(dry_run=True)
# ===========================================================================


def _empty_messages_text_con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Build a connection where read_json finds files but messages_text is empty.

    DuckDB ``read_json`` raises ``IOException: No files found ...`` if the
    glob has zero matches, so we always seed at least one record. We use a
    body shorter than the 32-char floor so the ``messages_text`` view drops
    it — effectively a "no candidates" corpus.
    """
    proj = tmp_path / "projects" / "proj-empty"
    write_session_jsonl(
        proj / "stub.jsonl",
        messages=[
            make_user_msg(
                "u-stub",
                "stub-sid",
                "tiny",  # < 32 chars → filtered out by messages_text
                ts="2026-04-01T10:00:00.000Z",
            )
        ],
    )
    sa_glob, sa_meta = _seed_subagent_stub(tmp_path)
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=str(tmp_path / "projects" / "*" / "*.jsonl"),
        subagent_glob=sa_glob,
        subagent_meta_glob=sa_meta,
    )
    register_views(con)
    return con


def test_classify_dry_run_empty_corpus(tmp_path: Path, tmp_settings: Settings) -> None:
    """Empty corpus → candidates=0, dry_run=True, pipeline='classify'."""
    con = _empty_messages_text_con(tmp_path)
    # Sanity guard: messages_text really is empty.
    (n,) = con.execute("SELECT count(*) FROM messages_text").fetchone()
    assert n == 0

    plan = classify_worker.classify_sessions(con, tmp_settings, dry_run=True)
    assert isinstance(plan, dict)
    assert plan["pipeline"] == "classify"
    assert plan["candidates"] == 0
    assert plan["dry_run"] is True
    assert plan["model"] == tmp_settings.sonnet_model_id
    con.close()


def test_classify_dry_run_populated_corpus(
    basic_con: duckdb.DuckDBPyConnection, tmp_settings: Settings
) -> None:
    """Populated corpus → candidates>0, estimated_cost_usd>0, model is sonnet id."""
    plan = classify_worker.classify_sessions(basic_con, tmp_settings, dry_run=True)
    assert isinstance(plan, dict)
    assert plan["pipeline"] == "classify"
    assert plan["candidates"] > 0
    assert plan["estimated_cost_usd"] > 0
    assert plan["model"] == tmp_settings.sonnet_model_id
    assert plan["dry_run"] is True
    # The plan also surfaces avg token assumptions used for the estimate.
    assert plan["avg_input_tokens"] == 8000
    assert plan["avg_output_tokens"] == 300


def test_classify_dry_run_anti_joins_done_sessions(
    basic_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
    tmp_corpus: dict[str, Any],
) -> None:
    """Pre-write a parquet shard for one session_id → candidates is reduced."""
    baseline = classify_worker.classify_sessions(basic_con, tmp_settings, dry_run=True)
    base_n = baseline["candidates"]
    # Stamp a fake row for one of the corpus sessions.
    from datetime import UTC, datetime

    done_sid = tmp_corpus["session_ids"][0]
    df = pl.DataFrame(
        [
            {
                "session_id": done_sid,
                "autonomy_tier": "manual",
                "work_category": "sde",
                "success": "success",
                "goal": "Test fixture row.",
                "confidence": 0.9,
                "classified_at": datetime.now(UTC),
            }
        ],
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
    tmp_settings.classifications_parquet_path.mkdir(parents=True, exist_ok=True)
    (tmp_settings.classifications_parquet_path / "part-1.parquet").write_bytes(
        b""
    )  # placeholder, overwritten below
    df.write_parquet(tmp_settings.classifications_parquet_path / "part-1.parquet")

    after = classify_worker.classify_sessions(basic_con, tmp_settings, dry_run=True)
    assert after["candidates"] == base_n - 1


# ===========================================================================
# 3. classify_sessions real run (dry_run=False)
# ===========================================================================


def test_classify_real_run_writes_parquet_then_idempotent(
    basic_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    fake_bedrock_client: Any,
) -> None:
    """Patch ``_build_bedrock_client`` and run the classify pipeline.

    First invocation: at least one ``part-*.parquet`` must land under the
    classifications directory. Second invocation: the already-set anti-join
    plus the per-session checkpoint must short-circuit it to zero rows.
    """
    fake = fake_bedrock_client(
        {
            "output": {
                "autonomy_tier": "autonomous",
                "work_category": "sde",
                "success": "success",
                "goal": "Do x.",
                "confidence": 0.9,
            }
        }
    )
    monkeypatch.setattr(classify_worker, "_build_bedrock_client", lambda _settings: fake)

    written = classify_worker.classify_sessions(basic_con, tmp_settings, dry_run=False)
    assert isinstance(written, int)
    assert written >= 1

    # Parquet directory should now have at least one part file.
    parts = iter_part_files(tmp_settings.classifications_parquet_path)
    assert len(parts) >= 1

    # Second pass is a no-op — anti-join + checkpoint skip everything.
    second = classify_worker.classify_sessions(basic_con, tmp_settings, dry_run=False)
    assert second == 0


# ===========================================================================
# 4. classify_sessions retry path
# ===========================================================================


def test_classify_retry_queue_captures_failures(
    basic_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``classify_one`` raises for every session → retry queue grows.

    We can't easily target one specific session because iteration order over
    ``iter_session_texts`` isn't part of the public contract — but
    ``pending_count`` should be at least 1 once at least one session fails.
    """
    monkeypatch.setattr(classify_worker, "_build_bedrock_client", lambda _settings: object())
    fail = AsyncMock(side_effect=RuntimeError("simulated parse failure"))
    monkeypatch.setattr(classify_worker, "classify_one", fail)

    written = classify_worker.classify_sessions(basic_con, tmp_settings, dry_run=False)
    assert written == 0  # every call failed, nothing landed in parquet
    pending = retry_queue.pending_count(tmp_settings.checkpoint_db_path, pipeline="classify")
    assert pending >= 1


# ===========================================================================
# 5. trajectory_messages(dry_run=True) — v1.0 windowed shape
# ===========================================================================
#
# The detailed real-run coverage moved to ``test_trajectory_windowed.py``
# alongside the windowed-pipeline rewrite (RFC 0002 §3.4 / §4.1). These
# two dry-run smoke tests stay here to round out the cross-pipeline
# matrix: classify / trajectory / conflicts / friction all share the
# dry-run contract.


def test_trajectory_dry_run_empty_corpus(tmp_path: Path, tmp_settings: Settings) -> None:
    """Empty corpus → candidates=0, llm_calls=0."""
    con = _empty_messages_text_con(tmp_path)

    plan = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=True)
    assert isinstance(plan, dict)
    assert plan["pipeline"] == "trajectory"
    assert plan["candidates"] == 0
    assert plan["llm_calls"] == 0
    con.close()


def test_trajectory_dry_run_populated_corpus(
    basic_con: duckdb.DuckDBPyConnection, tmp_settings: Settings
) -> None:
    """Populated corpus → candidates is the per-session count, not per-message.

    The v1.0 rewrite (RFC 0002 §3.4 / §4.1) reports per-session candidates;
    ``llm_calls`` is the chunk count (ceil(turns / 16)).
    """
    plan = trajectory_worker.trajectory_messages(basic_con, tmp_settings, dry_run=True)
    assert isinstance(plan, dict)
    assert plan["candidates"] > 0
    assert plan["llm_calls"] >= 1
    assert plan["model"] == tmp_settings.sonnet_model_id
    # The plan also exposes the raw text-turn count (windowed pipeline shape).
    assert "turns" in plan
    assert plan["turns"] >= plan["candidates"]


# ===========================================================================
# 8. detect_conflicts(dry_run=True)
# ===========================================================================


def test_conflicts_dry_run_empty_corpus(tmp_path: Path, tmp_settings: Settings) -> None:
    """Empty corpus → candidates=0."""
    con = _empty_messages_text_con(tmp_path)

    plan = conflicts_worker.detect_conflicts(con, tmp_settings, dry_run=True)
    assert isinstance(plan, dict)
    assert plan["pipeline"] == "conflicts"
    assert plan["candidates"] == 0
    con.close()


def test_conflicts_dry_run_populated_corpus(
    basic_con: duckdb.DuckDBPyConnection, tmp_settings: Settings
) -> None:
    """Populated corpus → candidates>0, llm_calls equal, model=sonnet."""
    plan = conflicts_worker.detect_conflicts(basic_con, tmp_settings, dry_run=True)
    assert isinstance(plan, dict)
    assert plan["candidates"] > 0
    assert plan["llm_calls"] == plan["candidates"]
    assert plan["model"] == tmp_settings.sonnet_model_id
    assert plan["avg_input_tokens"] == 6000
    assert plan["avg_output_tokens"] == 400


# ===========================================================================
# 9. detect_conflicts real-run coverage lives in test_conflicts_storage_v2.py
# ===========================================================================
#
# The legacy ``empty=True`` sentinel was removed in v1.0 (RFC 0002 §3.4).
# Real-run coverage for the new ``(turn_a_uuid, turn_b_uuid)`` schema lives
# in ``tests/test_conflicts_storage_v2.py``.


# ===========================================================================
# 10. _build_bedrock_client cache identity
# ===========================================================================


def test_build_bedrock_client_cache_identity(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same (region, pool_size) returns the same client; bumping concurrency does not.

    Real ``boto3.client('bedrock-runtime')`` instantiation reads the AWS
    credential chain and validates the endpoint; both add measurable wall
    time. We stub ``boto3.client`` with a sentinel factory so each unique
    (region, pool_size) gets a unique-but-cheap object — the test still
    exercises the cache key shape without paying the AWS SDK startup cost.
    """
    counter = {"n": 0}

    def _fake_boto_client(*_args: Any, **_kwargs: Any) -> object:
        counter["n"] += 1
        return object()

    monkeypatch.setattr("claude_sql.core.llm_shared.boto3.client", _fake_boto_client)

    a = llm_shared._build_bedrock_client(tmp_settings)
    b = llm_shared._build_bedrock_client(tmp_settings)
    assert a is b
    assert counter["n"] == 1  # second call hit the cache

    # Bumping llm_concurrency past the floor changes the pool size key.
    bumped = tmp_settings.model_copy(update={"llm_concurrency": 64})
    c = llm_shared._build_bedrock_client(bumped)
    assert c is not a
    assert counter["n"] == 2  # cache miss for the new key


# ===========================================================================
# 11. _maybe_log_bedrock_call tracing
# ===========================================================================


def test_maybe_log_bedrock_call_noop_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No trace path → function is a no-op."""
    monkeypatch.setattr(llm_shared, "_BEDROCK_TRACE_PATH", None)
    # Should not raise even with bogus payload structure.
    llm_shared.maybe_log_bedrock_call("classify", "m-id", {"usage": {}}, 12.5)


def test_maybe_log_bedrock_call_appends_jsonl_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Trace path set → exactly one JSONL row appended per call."""
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setattr(llm_shared, "_BEDROCK_TRACE_PATH", str(trace))
    payload = {
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 50,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 0,
            },
        },
        "stop_reason": "end_turn",
    }
    llm_shared.maybe_log_bedrock_call("classify", "model-x", payload, 42.123)

    lines = trace.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["pipeline"] == "classify"
    assert row["model"] == "model-x"
    assert row["input_tokens"] == 1200
    assert row["output_tokens"] == 50
    assert row["cache_read_input_tokens"] == 800
    assert row["elapsed_ms"] == pytest.approx(42.1)


# ===========================================================================
# 12. _estimate_cost arithmetic
# ===========================================================================


@pytest.mark.parametrize(
    ("n", "in_tok", "out_tok", "pricing", "expected"),
    [
        # Trivial: zero items zero cost.
        (0, 1000, 100, (3.0, 15.0), 0.0),
        # Sonnet 4.6 pricing, 10 items x 8K in x 300 out.
        # in_cost  = 10 * 8000 * 3.0  / 1e6 = 0.24, out_cost = 10 * 300 * 15.0 / 1e6 = 0.045.
        (10, 8000, 300, (3.0, 15.0), 0.285),
        # Cheap model — verify the arithmetic doesn't accidentally hard-code Sonnet.
        (5, 1000, 200, (0.80, 4.0), (5 * 1000 * 0.80 + 5 * 200 * 4.0) / 1_000_000),
    ],
)
def test_estimate_cost_arithmetic(
    n: int,
    in_tok: int,
    out_tok: int,
    pricing: tuple[float, float],
    expected: float,
) -> None:
    actual = llm_shared._estimate_cost(n, in_tok, out_tok, pricing)
    assert actual == pytest.approx(expected)
