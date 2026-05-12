"""Tests for the v1.0 windowed trajectory pipeline (RFC 0002 §3.4 / §4.1).

Covers:

* Window-count invariants — every text turn produces exactly one window
  (session-first turns get a synthetic neutral pair).
* Chunk splitting at the 16-window cap with anchor-turn sharing across
  consecutive chunks.
* Verification + bounded retry — array responses with missing windows
  trigger ONE retry of just the missing windows; persistent misses become
  neutral placeholders rather than wedging the pipeline.
* Old-shape parquet-shard cleanup — stale per-message shards are deleted
  on first run via parquet metadata probe.
* Pipeline-level cache stats — the ``pipeline_cache_stats`` context
  manager emits one summary line wrapping the windowed run.
* Schema invariants — ``transition_kind`` enum has exactly the six values
  promised by RFC §3.4; system prompt clears the Sonnet 2048-token cache
  floor.

Bedrock is fully mocked. Bedrock client construction is monkeypatched
to a sentinel ``object()`` so no boto3 wiring runs.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest
from loguru import logger

from claude_sql import llm_shared, trajectory_worker
from claude_sql.config import Settings
from claude_sql.parquet_shards import iter_part_files, read_all
from claude_sql.schemas import TRAJECTORY_ARRAY_SCHEMA, TrajectoryArrayResult, TrajectoryWindow
from claude_sql.sql_views import register_raw, register_views
from conftest import _seed_subagent_stub, make_user_msg, write_session_jsonl

# ---------------------------------------------------------------------------
# Test plumbing
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


def _build_corpus(
    tmp_path: Path,
    sessions: list[tuple[str, list[dict[str, Any]]]],
) -> duckdb.DuckDBPyConnection:
    """Write ``sessions`` to a fresh corpus root and return an open DuckDB con."""
    proj = tmp_path / "projects" / "proj-traj"
    for sid, msgs in sessions:
        write_session_jsonl(proj / f"{sid}.jsonl", messages=msgs)
    sa_glob, sa_meta = _seed_subagent_stub(tmp_path)
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    con = duckdb.connect(":memory:")
    register_raw(con, glob=glob, subagent_glob=sa_glob, subagent_meta_glob=sa_meta)
    register_views(con)
    return con


def _user(uuid: str, sid: str, text: str, *, off: int) -> dict[str, Any]:
    """User-role message with a deterministic timestamp offset (seconds)."""
    base = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
    ts = base.replace(microsecond=0).timestamp() + off
    iso = datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")
    return make_user_msg(uuid, sid, text, ts=iso)


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_transition_kind_enum_values_are_six() -> None:
    """RFC 0002 §3.4 promises exactly six transition_kind labels."""
    props = TRAJECTORY_ARRAY_SCHEMA["properties"]
    items = props["windows"]["items"]
    enum = set(items["properties"]["transition_kind"]["enum"])
    expected = {
        "frustration_spike",
        "resolution",
        "reset",
        "drift",
        "clarification",
        "none",
    }
    assert enum == expected, f"transition_kind enum drift: {enum} vs {expected}"
    # The TRANSITION_KINDS constant in the worker must agree.
    assert set(trajectory_worker.TRANSITION_KINDS) == expected


def test_pydantic_model_round_trip() -> None:
    """TrajectoryWindow + TrajectoryArrayResult validate the contract."""
    win = TrajectoryWindow(
        prev_uuid="u1",
        curr_uuid="u2",
        prev_sentiment="neutral",
        curr_sentiment="positive",
        delta=1.0,
        is_transition=False,
        transition_kind="resolution",
        confidence=0.8,
    )
    assert win.transition_kind == "resolution"
    arr = TrajectoryArrayResult(windows=[win])
    assert len(arr.windows) == 1


def test_system_prompt_clears_cache_floor() -> None:
    """Sonnet 4.6 raised the cache minimum from 1024→2048 input tokens.

    Padding the trajectory system prompt past that floor is load-bearing —
    below it the ``cache_control: ttl=1h`` annotation is silently ignored
    and the entire system prompt re-tokenizes per call. cl100k_base × 0.78
    is the published rule-of-thumb for converting to Anthropic-token count.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    n = len(enc.encode(trajectory_worker.TRAJECTORY_SYSTEM_PROMPT))
    estimated_anthropic = n * 0.78
    assert estimated_anthropic >= 2048, (
        f"trajectory system prompt is {n} cl100k / ~{estimated_anthropic:.0f} "
        f"Anthropic tokens — below the Sonnet 4.6 2048-token cache floor"
    )


# ---------------------------------------------------------------------------
# Per-session window invariants
# ---------------------------------------------------------------------------


def test_session_with_n_text_messages_produces_n_windows(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every text turn → one window. Session-first turn → synthetic neutral pair."""
    sid = "sess-counts"
    # Three text user turns. Expect 3 windows (1 session-first + 2 paired).
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [
                    _user(
                        "u1", sid, "first long opener message that clears the 32-char filter", off=0
                    ),
                    _user(
                        "u2", sid, "second substantive turn that exceeds the filter cutoff", off=1
                    ),
                    _user(
                        "u3", sid, "third substantive turn long enough to clear filtering", off=2
                    ),
                ],
            )
        ],
    )

    captured_windows: list[list[tuple[str | None, str]]] = []

    # Monkeypatch ``_classify_chunk`` directly so the test controls the
    # indexed-by-(prev_uuid, curr_uuid) return shape — no boto3 round-trip.
    async def fake_chunk(client, settings, sem, *, chunk, thinking_mode):  # type: ignore[no-untyped-def]
        captured_windows.append([(r[1], r[2]) for r in chunk])
        return {
            (prev, curr): {
                "prev_uuid": prev,
                "curr_uuid": curr,
                "prev_sentiment": None if prev is None else "neutral",
                "curr_sentiment": "neutral",
                "delta": None if prev is None else 0,
                "is_transition": False,
                "transition_kind": "none",
                "confidence": 0.85,
            }
            for prev, curr in [(r[1], r[2]) for r in chunk]
        }

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)
    monkeypatch.setattr(trajectory_worker, "_build_bedrock_client", lambda _settings: object())

    written = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    assert isinstance(written, int)
    assert written == 3, f"expected 3 windows, got {written}"

    df = read_all(tmp_settings.trajectory_parquet_path)
    assert df is not None
    assert df.height == 3
    # Exactly one row has prev_uuid IS NULL — the session-first window.
    null_prev = df.filter(pl.col("prev_uuid").is_null())
    assert null_prev.height == 1
    con.close()


def test_session_first_window_has_null_prev_uuid_and_neutral_synthetic(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthetic session-first window: prev_uuid=NULL, prev_sentiment=NULL, delta=NULL."""
    sid = "sess-first"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [
                    _user("u1", sid, "opener message clearing the 32-char floor", off=0),
                    _user("u2", sid, "second message clearing the 32-char floor", off=1),
                ],
            )
        ],
    )

    async def fake_chunk(client, settings, sem, *, chunk, thinking_mode):  # type: ignore[no-untyped-def]
        return {
            (prev, curr): {
                "prev_uuid": prev,
                "curr_uuid": curr,
                "prev_sentiment": None if prev is None else "neutral",
                "curr_sentiment": "neutral",
                "delta": None if prev is None else 0,
                "is_transition": False,
                "transition_kind": "none",
                "confidence": 0.85,
            }
            for prev, curr in [(r[1], r[2]) for r in chunk]
        }

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)
    monkeypatch.setattr(trajectory_worker, "_build_bedrock_client", lambda _settings: object())

    trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    df = read_all(tmp_settings.trajectory_parquet_path)
    assert df is not None
    first = df.filter(pl.col("prev_uuid").is_null()).to_dicts()[0]
    assert first["curr_uuid"] is not None
    assert first["prev_sentiment"] is None
    assert first["delta"] is None
    # curr_sentiment is the model output (neutral here).
    assert first["curr_sentiment"] == "neutral"
    con.close()


def test_session_with_more_than_16_text_messages_splits_into_chunks(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """20 turns → 2 chunks. Anchor-turn sharing: chunk N's last curr == chunk N+1's first prev."""
    sid = "sess-long"
    msgs = [
        _user(
            f"u{i:02d}",
            sid,
            f"turn {i:02d} body padded out so it clears the 32-char filter floor",
            off=i,
        )
        for i in range(20)
    ]
    con = _build_corpus(tmp_path, [(sid, msgs)])

    captured_chunks: list[list[tuple[str | None, str]]] = []

    async def fake_chunk(client, settings, sem, *, chunk, thinking_mode):  # type: ignore[no-untyped-def]
        keys = [(r[1], r[2]) for r in chunk]
        captured_chunks.append(keys)
        return {
            key: {
                "prev_uuid": key[0],
                "curr_uuid": key[1],
                "prev_sentiment": None if key[0] is None else "neutral",
                "curr_sentiment": "neutral",
                "delta": None if key[0] is None else 0,
                "is_transition": False,
                "transition_kind": "none",
                "confidence": 0.8,
            }
            for key in keys
        }

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)
    monkeypatch.setattr(trajectory_worker, "_build_bedrock_client", lambda _settings: object())

    written = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    assert written == 20
    # Two chunks: 16 + 4.
    assert len(captured_chunks) == 2
    assert len(captured_chunks[0]) == 16
    assert len(captured_chunks[1]) == 4
    # Anchor-turn invariant: chunk[0]'s last curr_uuid == chunk[1]'s first prev_uuid.
    assert captured_chunks[0][-1][1] == captured_chunks[1][0][0]
    con.close()


# ---------------------------------------------------------------------------
# Bounded-retry path
# ---------------------------------------------------------------------------


def test_array_response_with_missing_windows_triggers_retry(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock returns 14/16 windows on first call, all 16 on retry → 2 _classify_chunk calls."""
    sid = "sess-retry"
    msgs = [
        _user(
            f"u{i:02d}",
            sid,
            f"turn {i:02d} body padded out so it clears the 32-char filter floor",
            off=i,
        )
        for i in range(16)
    ]
    con = _build_corpus(tmp_path, [(sid, msgs)])

    call_count = {"n": 0}

    async def fake_chunk(client, settings, sem, *, chunk, thinking_mode):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        keys = [(r[1], r[2]) for r in chunk]
        # First call: drop the last 2 windows. Retry call (which sees only
        # the missing keys): return all of them.
        if call_count["n"] == 1:
            keys = keys[:-2]
        return {
            key: {
                "prev_uuid": key[0],
                "curr_uuid": key[1],
                "prev_sentiment": None if key[0] is None else "neutral",
                "curr_sentiment": "neutral",
                "delta": None if key[0] is None else 0,
                "is_transition": False,
                "transition_kind": "none",
                "confidence": 0.8,
            }
            for key in keys
        }

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)
    monkeypatch.setattr(trajectory_worker, "_build_bedrock_client", lambda _settings: object())

    written = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    assert written == 16
    # First chunk attempt + one retry of the missing pair = 2 calls.
    assert call_count["n"] == 2

    df = read_all(tmp_settings.trajectory_parquet_path)
    assert df is not None
    assert df.height == 16
    # All windows have non-zero confidence (no placeholders this time).
    assert (df["confidence"] > 0).all()
    con.close()


def test_array_response_with_persistent_misses_stamps_neutral_placeholders(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock returns 14/16 then 14/16 → 2 placeholder rows with confidence=0.0, transition_kind='none'."""
    sid = "sess-persistent-miss"
    msgs = [
        _user(
            f"u{i:02d}",
            sid,
            f"turn {i:02d} body padded out so it clears the 32-char filter floor",
            off=i,
        )
        for i in range(16)
    ]
    con = _build_corpus(tmp_path, [(sid, msgs)])

    async def fake_chunk(client, settings, sem, *, chunk, thinking_mode):  # type: ignore[no-untyped-def]
        # Always drop the last 2 keys regardless of which call we're on.
        keys = [(r[1], r[2]) for r in chunk][:-2] if len(chunk) >= 2 else []
        return {
            key: {
                "prev_uuid": key[0],
                "curr_uuid": key[1],
                "prev_sentiment": None if key[0] is None else "neutral",
                "curr_sentiment": "neutral",
                "delta": None if key[0] is None else 0,
                "is_transition": False,
                "transition_kind": "none",
                "confidence": 0.8,
            }
            for key in keys
        }

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)
    monkeypatch.setattr(trajectory_worker, "_build_bedrock_client", lambda _settings: object())

    written = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    assert written == 16

    df = read_all(tmp_settings.trajectory_parquet_path)
    assert df is not None
    placeholders = df.filter(pl.col("confidence") == 0.0)
    assert placeholders.height == 2, (
        f"expected 2 placeholder rows for the persistently missing pair, got {placeholders.height}"
    )
    # Placeholder rows have transition_kind='none' and curr_sentiment='neutral'.
    for row in placeholders.to_dicts():
        assert row["transition_kind"] == "none"
        assert row["curr_sentiment"] == "neutral"
    con.close()


# ---------------------------------------------------------------------------
# Old-shape cleanup
# ---------------------------------------------------------------------------


def test_old_per_message_parquet_shard_is_deleted_on_first_run(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale per-message shards are detected via parquet metadata + removed."""
    # Stamp a stale per-message-shape shard before the run.
    target = tmp_settings.trajectory_parquet_path
    target.mkdir(parents=True, exist_ok=True)
    stale_part = target / "part-9999999999999.parquet"
    pl.DataFrame(
        [
            {
                "uuid": "stale-1",
                "sentiment_delta": "neutral",
                "is_transition": False,
                "confidence": 0.5,
                "classified_at": datetime.now(UTC),
            }
        ],
        schema={
            "uuid": pl.Utf8,
            "sentiment_delta": pl.Utf8,
            "is_transition": pl.Boolean,
            "confidence": pl.Float32,
            "classified_at": pl.Datetime("us", "UTC"),
        },
    ).write_parquet(stale_part)
    assert stale_part.exists()

    sid = "sess-after-purge"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [
                    _user("u1", sid, "post-purge opener that clears the filter floor", off=0),
                    _user("u2", sid, "post-purge follower that clears the filter floor", off=1),
                ],
            )
        ],
    )

    async def fake_chunk(client, settings, sem, *, chunk, thinking_mode):  # type: ignore[no-untyped-def]
        return {
            (r[1], r[2]): {
                "prev_uuid": r[1],
                "curr_uuid": r[2],
                "prev_sentiment": None if r[1] is None else "neutral",
                "curr_sentiment": "neutral",
                "delta": None if r[1] is None else 0,
                "is_transition": False,
                "transition_kind": "none",
                "confidence": 0.85,
            }
            for r in chunk
        }

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)
    monkeypatch.setattr(trajectory_worker, "_build_bedrock_client", lambda _settings: object())

    trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    # The stale shard must be gone.
    assert not stale_part.exists(), "stale per-message shard was not purged"
    # And the new shape lives in fresh part files.
    new_parts = iter_part_files(target)
    assert new_parts, "no new part files written"
    df = read_all(target)
    assert df is not None
    # New schema column set.
    assert {"session_id", "prev_uuid", "curr_uuid", "transition_kind"}.issubset(set(df.columns))
    # Old columns are gone.
    assert "sentiment_delta" not in df.columns
    con.close()


# ---------------------------------------------------------------------------
# Cache-stat summary
# ---------------------------------------------------------------------------


def test_pipeline_cache_stats_emits_summary(
    tmp_path: Path,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pipeline_cache_stats('trajectory')`` emits one INFO line on exit."""
    sid = "sess-cache"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [
                    _user("u1", sid, "cache-stats opener with enough body to clear floor", off=0),
                    _user("u2", sid, "cache-stats follower with enough body to clear floor", off=1),
                ],
            )
        ],
    )

    async def fake_chunk(client, settings, sem, *, chunk, thinking_mode):  # type: ignore[no-untyped-def]
        return {
            (r[1], r[2]): {
                "prev_uuid": r[1],
                "curr_uuid": r[2],
                "prev_sentiment": None if r[1] is None else "neutral",
                "curr_sentiment": "neutral",
                "delta": None if r[1] is None else 0,
                "is_transition": False,
                "transition_kind": "none",
                "confidence": 0.85,
            }
            for r in chunk
        }

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)
    monkeypatch.setattr(trajectory_worker, "_build_bedrock_client", lambda _settings: object())

    captured: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is None:
            return
        if record["level"].name == "INFO":
            captured.append(record["message"])

    handler_id = logger.add(_sink, level="INFO")
    try:
        trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    finally:
        logger.remove(handler_id)

    cache_lines = [m for m in captured if "pipeline=trajectory" in m]
    assert cache_lines, f"expected a 'pipeline=trajectory' cache-stat line, got: {captured!r}"
    con.close()


# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------


def test_dry_run_returns_per_session_candidate_count(
    tmp_path: Path,
    tmp_settings: Settings,
) -> None:
    """``--dry-run`` reports per-session candidates (windowed pipeline shape)."""
    sid_a = "sess-a"
    sid_b = "sess-b"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid_a,
                [_user("ua1", sid_a, "session A opener over 32 chars filter floor", off=0)],
            ),
            (
                sid_b,
                [_user("ub1", sid_b, "session B opener over 32 chars filter floor", off=0)],
            ),
        ],
    )
    plan = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=True)
    assert isinstance(plan, dict)
    assert plan["pipeline"] == "trajectory"
    # Per-session candidate count (NOT per-message — the v1.0 windowed shape).
    assert plan["candidates"] == 2
    assert plan["dry_run"] is True
    assert plan["model"] == tmp_settings.sonnet_model_id
    con.close()
