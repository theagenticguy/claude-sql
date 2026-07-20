"""Coverage-lift tests for ``claude_sql.trajectory_worker``.

These tests target the pure helper functions and the dispatch-loop
branches that the semantically-organized ``test_trajectory_windowed.py``
suite doesn't exercise — corrupt-parquet probe, unlinkable shard,
``_load_windows`` / ``_chunk_windows`` / ``_format_chunk_xml`` edge
cases, ``_delta_value`` / ``_build_row`` fallbacks, ``_classify_chunk``
happy/refusal/exception/provider-unavailable paths, retry-queue drain logging,
0-window short-circuit, refusal + non-Bedrock exception branches in
``_process_session``, the outer ``except Exception`` that swallows
mid-loop failures, and the ``--dry-run`` ``since_days`` + ``limit`` cap.

The LLM path is fully mocked. ``_classify_chunk`` is monkeypatched per
test, or its ``LlmAnalyticsProvider`` seam is driven with a fake provider
(``_FakeProvider``) that returns a canned ``TrajectoryArrayResult`` or
raises a canned terminal error: no Bedrock / Strands / Luna call ever runs.
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

from claude_sql.application.use_cases import trajectory as trajectory_worker
from claude_sql.domain.errors import BedrockRefusalError
from claude_sql.infrastructure.bedrock import client as llm_shared
from claude_sql.infrastructure.duckdb_views import register_raw, register_views
from claude_sql.infrastructure.parquet_cache import read_all
from claude_sql.infrastructure.settings import Settings
from claude_sql.infrastructure.sqlite_state import retry_queue
from conftest import _seed_subagent_stub, make_user_msg, write_session_jsonl

# ---------------------------------------------------------------------------
# Plumbing — copied from test_trajectory_windowed.py
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client_cache() -> Iterator[None]:
    llm_shared._CLIENT_CACHE.clear()
    yield
    llm_shared._CLIENT_CACHE.clear()


@pytest.fixture(autouse=True)
def _noop_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_a, **_kw: None)


def _build_corpus(
    tmp_path: Path,
    sessions: list[tuple[str, list[dict[str, Any]]]],
) -> duckdb.DuckDBPyConnection:
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
    base = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
    ts = base.replace(microsecond=0).timestamp() + off
    iso = datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")
    return make_user_msg(uuid, sid, text, ts=iso)


# ---------------------------------------------------------------------------
# _stale_old_shape / _purge_old_shards exception paths (347, 350-352, 369-374)
# ---------------------------------------------------------------------------


def test_stale_old_shape_returns_false_on_corrupt_parquet(
    tmp_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """A 0-byte parquet makes pyarrow raise; probe warns + returns False (treat-as-fresh)."""
    target = tmp_settings.trajectory_parquet_path
    target.mkdir(parents=True, exist_ok=True)
    bad = target / "part-1.parquet"
    bad.write_bytes(b"")  # invalid parquet — pyarrow raises on open

    captured: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is None:
            return
        if record["level"].name == "WARNING":
            captured.append(record["message"])

    handler_id = logger.add(_sink, level="WARNING")
    try:
        result = trajectory_worker._stale_old_shape([bad])
    finally:
        logger.remove(handler_id)
    assert result is False
    assert any("metadata probe failed" in m for m in captured)


def test_stale_old_shape_empty_input_returns_false(tmp_path: Path) -> None:
    """Empty parts list → False (no probe needed)."""
    assert trajectory_worker._stale_old_shape([]) is False


def test_stale_old_shape_returns_false_when_curr_uuid_present(
    tmp_settings: Settings,
) -> None:
    """A new-shape parquet (has ``curr_uuid``) returns False (covers line 347)."""
    target = tmp_settings.trajectory_parquet_path
    target.mkdir(parents=True, exist_ok=True)
    new_part = target / "part-1234567890.parquet"
    pl.DataFrame(
        [
            {
                "session_id": "sid",
                "prev_uuid": None,
                "curr_uuid": "u1",
                "transition_kind": "none",
            }
        ],
        schema={
            "session_id": pl.Utf8,
            "prev_uuid": pl.Utf8,
            "curr_uuid": pl.Utf8,
            "transition_kind": pl.Utf8,
        },
    ).write_parquet(new_part)
    assert trajectory_worker._stale_old_shape([new_part]) is False


def test_purge_old_shards_logs_when_unlink_fails(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale-shape part whose ``Path.unlink`` raises emits a WARNING and is counted as 0 removed."""
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

    # Make every Path.unlink raise so we hit the warning branch.
    real_unlink = Path.unlink

    def boom(self: Path, *a: Any, **kw: Any) -> None:
        if self == stale_part:
            raise OSError("simulated permission error")
        real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", boom)

    captured: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is None:
            return
        if record["level"].name == "WARNING":
            captured.append(record["message"])

    handler_id = logger.add(_sink, level="WARNING")
    try:
        removed = trajectory_worker._purge_old_shards(target)
    finally:
        logger.remove(handler_id)
    assert removed == 0
    assert any("failed to delete legacy shard" in m for m in captured)
    # File still on disk because unlink failed.
    assert stale_part.exists()


# ---------------------------------------------------------------------------
# _load_windows / _chunk_windows / _format_chunk_xml edge cases
# ---------------------------------------------------------------------------


def test_load_windows_with_since_days_and_limit(tmp_path: Path, tmp_settings: Settings) -> None:
    """``since_days`` and ``limit`` clauses both render and bind correctly (covers 405, 425)."""
    sid = "sess-load"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [
                    _user("u1", sid, "first opener long enough to clear filtering floor", off=0),
                    _user("u2", sid, "second turn long enough to clear filtering floor", off=1),
                    _user("u3", sid, "third turn long enough to clear filtering floor", off=2),
                ],
            )
        ],
    )
    # since_days=3650 means "everything ever" (the corpus is dated 2026 and
    # the test runs after that, so a wide window covers it). limit=2 caps
    # the row count.
    rows = trajectory_worker._load_windows(con, active_sessions=None, since_days=3650, limit=2)
    assert len(rows) == 2
    con.close()


def test_load_windows_arrow_join_byte_identical_to_unnest_bind(
    tmp_path: Path, tmp_settings: Settings
) -> None:
    """The Arrow-relation JOIN filter is byte-identical to the old ``unnest(?)`` bind.

    PATTERN-H perf change: ``_load_windows`` filters ``active_sessions`` by
    JOINing a registered Arrow relation instead of binding the id list as a
    ``?`` parameter (kills DuckDB's ~2-probes-per-element pandas ``sys.path``
    storm on a cold-rebuild full-corpus run). The filter SEMANTICS must not
    move: run the production loader (Arrow JOIN) and the legacy bound-list SQL
    against the SAME connection and assert the row lists are identical, ordered
    and as a multiset.
    """
    sids = ["sess-a", "sess-b", "sess-c"]
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [
                    _user(f"{sid}-u1", sid, "first opener long enough to clear the floor", off=0),
                    _user(f"{sid}-u2", sid, "second turn long enough to clear the floor", off=1),
                    _user(f"{sid}-u3", sid, "third turn long enough to clear the floor", off=2),
                ],
            )
            for sid in sids
        ],
    )
    # Filter to a strict SUBSET so the predicate actually has to discriminate.
    active = {"sess-a", "sess-c"}

    # NEW shape: the production loader (registers the Arrow relation + JOINs).
    new_rows = trajectory_worker._load_windows(
        con, active_sessions=active, since_days=3650, limit=None
    )

    # OLD shape: the legacy bound-list SQL, run inline against the same con.
    old_sql = """
        SELECT CAST(tw.session_id AS VARCHAR) AS session_id,
               tw.prev_uuid, tw.curr_uuid, tw.prev_role, tw.curr_role,
               mt_prev.text_content AS prev_text,
               mt_curr.text_content AS curr_text,
               tw.window_idx
          FROM turn_window tw
          LEFT JOIN messages_text mt_prev ON CAST(mt_prev.uuid AS VARCHAR) = tw.prev_uuid
          LEFT JOIN messages_text mt_curr ON CAST(mt_curr.uuid AS VARCHAR) = tw.curr_uuid
         WHERE 1=1 AND CAST(tw.session_id AS VARCHAR) IN (SELECT unnest(?))
         ORDER BY tw.session_id, tw.window_idx
    """
    raw_old = con.execute(old_sql, [list(active)]).fetchall()
    # Apply the same null-curr drop + tuple projection the loader does.
    old_rows = [
        (sid, pu, cu, pr, cr, pt, ct)
        for sid, pu, cu, pr, cr, pt, ct, _idx in raw_old
        if cu is not None and ct is not None
    ]

    assert new_rows  # the subset is non-empty
    assert new_rows == old_rows  # ordered byte-identity
    assert sorted(map(repr, new_rows)) == sorted(map(repr, old_rows))  # multiset
    # Only the in-filter sessions appear; the excluded one is gone.
    assert {r[0] for r in new_rows} == active
    con.close()


def test_load_windows_skips_rows_with_null_curr(tmp_path: Path) -> None:
    """A row whose ``curr_uuid`` or ``curr_text`` is None is dropped (covers line 430).

    DuckDB connections have read-only attributes so we wrap the connection
    in a thin proxy that intercepts the turn_window query and returns
    a hand-crafted row set instead of round-tripping through SQL.
    """
    sid = "sess-null-curr"

    class FakeFetchResult:
        @staticmethod
        def fetchall() -> list[tuple[Any, ...]]:
            """Tuple shape: (sid, prev_uuid, curr_uuid, prev_role, curr_role, prev_text, curr_text, idx)."""
            return [
                (sid, None, None, None, "user", None, "body", 0),  # curr_uuid None → skipped
                (sid, None, "u1", None, "user", None, "body1", 1),  # survives
                (sid, "u1", "u2", "user", "user", "body1", None, 2),  # curr_text None → skipped
            ]

    class FakeCon:
        def execute(self, sql: str, params: Any = None) -> Any:
            return FakeFetchResult()

        def register(self, name: str, obj: Any) -> None:
            """No-op: the real loader registers the session-id Arrow relation
            before the turn_window query (perf: JOIN instead of a bound list)."""

        def unregister(self, name: str) -> None:
            """No-op twin of ``register`` — the loader unregisters in a finally."""

    rows = trajectory_worker._load_windows(
        FakeCon(),  # type: ignore[arg-type]
        active_sessions={sid},
        since_days=14,
        limit=10,
    )
    # Only the middle row survives.
    assert len(rows) == 1
    assert rows[0][2] == "u1"


def test_chunk_windows_empty_returns_empty_list() -> None:
    """``_chunk_windows([])`` returns ``[]`` (covers line 461)."""
    assert trajectory_worker._chunk_windows([]) == []


def test_format_chunk_xml_clips_long_bodies_and_handles_null_prev() -> None:
    """Long prev/curr bodies are clipped with the ``…(truncated)`` footer; None prev_text becomes empty body.

    Hits all of 478-500 (loop, prev/curr clipping, null-prev branch, the
    final XML composition + join).
    """
    long_prev = "a" * 2500
    long_curr = "b" * 2500
    chunk = [
        # Row with both prev and curr as long strings → exercises both clip branches.
        ("sid", "p1", "c1", "user", "user", long_prev, long_curr),
        # Row with prev_text=None → sets prev_body to "" (the `if prev_text is not None:` False branch).
        ("sid", None, "c2", None, "user", None, "short curr body"),
    ]
    xml = trajectory_worker._format_chunk_xml(chunk, max_text_chars=2000)
    # Both clip footers present.
    assert xml.count("…(truncated)") == 2
    # The null-prev row has an empty prev body.
    assert '<prev role="" uuid=""></prev>' in xml
    # idx=0 and idx=1 both rendered.
    assert "<window idx=0>" in xml
    assert "<window idx=1>" in xml


def test_xml_attr_and_xml_text_escape() -> None:
    """``_xml_attr`` and ``_xml_text`` escape the right characters (covers 505 and 512)."""
    assert trajectory_worker._xml_attr('a"b<c>d&e') == "a&quot;b&lt;c&gt;d&amp;e"
    # _xml_text does NOT escape quotes (they're harmless inside element bodies).
    assert trajectory_worker._xml_text("x<y>z&w") == "x&lt;y&gt;z&amp;w"


# ---------------------------------------------------------------------------
# _delta_value with None inputs (522-524)
# ---------------------------------------------------------------------------


def test_delta_value_none_inputs_return_none() -> None:
    """Either input None → result None (no arithmetic on missing labels)."""
    assert trajectory_worker._delta_value(None, "neutral") is None
    assert trajectory_worker._delta_value("neutral", None) is None
    assert trajectory_worker._delta_value(None, None) is None
    # Sanity: real labels still compute.
    assert trajectory_worker._delta_value("negative", "positive") == 2.0


# ---------------------------------------------------------------------------
# _build_row fallbacks (560, 572-573)
# ---------------------------------------------------------------------------


def test_build_row_unknown_transition_kind_falls_back_to_none() -> None:
    """A transition_kind outside the six-value enum is coerced to 'none'."""
    now = datetime.now(UTC)
    row = trajectory_worker._build_row(
        "sid",
        {
            "prev_uuid": "p",
            "curr_uuid": "c",
            "prev_sentiment": "neutral",
            "curr_sentiment": "neutral",
            "delta": 0,
            "is_transition": False,
            "transition_kind": "made_up_label",  # not in TRANSITION_KINDS
            "confidence": 0.7,
        },
        now,
    )
    assert row["transition_kind"] == "none"


def test_build_row_uncoercible_delta_recomputes_from_labels() -> None:
    """A non-numeric ``delta`` (e.g. an object) falls through to ``_delta_value``."""
    now = datetime.now(UTC)
    row = trajectory_worker._build_row(
        "sid",
        {
            "prev_uuid": "p",
            "curr_uuid": "c",
            "prev_sentiment": "negative",
            "curr_sentiment": "positive",
            "delta": object(),  # neither None nor float-coercible
            "is_transition": False,
            "transition_kind": "resolution",
            "confidence": 0.8,
        },
        now,
    )
    # Falls back to _delta_value("negative", "positive") = 2.0
    assert row["delta"] == 2.0


# ---------------------------------------------------------------------------
# _classify_chunk via the LlmAnalyticsProvider seam (Wave D)
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ``LlmAnalyticsProvider`` whose ``classify_structured`` returns a
    canned :class:`TrajectoryArrayResult` or raises a canned exception."""

    provider = "fake"

    def __init__(self, *, result: Any = None, exc: BaseException | None = None) -> None:
        self._result = result
        self._exc = exc

    async def classify_structured(self, *, system: str, prompt: str, schema: Any) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._result


def _drive_classify_chunk(provider: Any, chunk: list[tuple[Any, ...]]) -> Any:
    """Run ``_classify_chunk`` against the supplied provider and return its result."""
    import asyncio

    async def _run() -> Any:
        return await trajectory_worker._classify_chunk(provider, chunk=chunk)

    return asyncio.run(_run())


def test_classify_chunk_happy_path_returns_indexed_dict() -> None:
    """Happy path: provider returns a TrajectoryArrayResult → indexed by (prev,curr)."""
    from claude_sql.domain.models import TrajectoryArrayResult

    chunk = [
        ("sid", None, "c1", None, "user", None, "first body"),
        ("sid", "c1", "c2", "user", "user", "first body", "second body"),
    ]
    result_model = TrajectoryArrayResult.model_validate(
        {
            "windows": [
                {
                    "prev_uuid": None,
                    "curr_uuid": "c1",
                    "prev_sentiment": None,
                    "curr_sentiment": "neutral",
                    "delta": None,
                    "is_transition": False,
                    "transition_kind": "none",
                    "confidence": 0.85,
                },
                {
                    "prev_uuid": "c1",
                    "curr_uuid": "c2",
                    "prev_sentiment": "neutral",
                    "curr_sentiment": "neutral",
                    "delta": 0,
                    "is_transition": False,
                    "transition_kind": "none",
                    "confidence": 0.9,
                },
            ]
        }
    )
    result = _drive_classify_chunk(_FakeProvider(result=result_model), chunk)
    assert isinstance(result, dict)
    assert (None, "c1") in result
    assert ("c1", "c2") in result
    assert len(result) == 2


def test_classify_chunk_returns_refusal_error() -> None:
    """A BedrockRefusalError from the provider is returned as the value (not raised)."""
    chunk = [("sid", None, "c1", None, "user", None, "body")]
    result = _drive_classify_chunk(_FakeProvider(exc=BedrockRefusalError("policy refusal")), chunk)
    assert isinstance(result, BedrockRefusalError)


def test_classify_chunk_returns_generic_exception() -> None:
    """A non-refusal Exception from the provider is returned as the value."""
    chunk = [("sid", None, "c1", None, "user", None, "body")]
    result = _drive_classify_chunk(_FakeProvider(exc=RuntimeError("network blip")), chunk)
    assert isinstance(result, RuntimeError)
    assert "network blip" in str(result)


def test_classify_chunk_returns_provider_unavailable() -> None:
    """A Luna fail-open LlmAnalyticsUnavailable is returned as the value so the
    dispatch loop enqueues the session for a later run (not a crash)."""
    from claude_sql.domain.errors import LlmAnalyticsUnavailable

    chunk = [("sid", None, "c1", None, "user", None, "body")]
    result = _drive_classify_chunk(
        _FakeProvider(exc=LlmAnalyticsUnavailable("Luna transport down")), chunk
    )
    assert isinstance(result, LlmAnalyticsUnavailable)


# ---------------------------------------------------------------------------
# Dispatch loop: refusal branch (776-783) and generic-exception branch (785-802)
# ---------------------------------------------------------------------------


def test_chunk_refusal_stamps_neutral_placeholders(
    tmp_path: Path, tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BedrockRefusalError on a chunk → all rows in the chunk become neutral placeholders."""
    sid = "sess-refusal"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [
                    _user("u1", sid, "opener long enough to clear filtering floor", off=0),
                    _user("u2", sid, "follower long enough to clear filtering floor", off=1),
                ],
            )
        ],
    )

    async def fake_chunk(provider, *, chunk):  # type: ignore[no-untyped-def]
        return BedrockRefusalError("refused")

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)

    written = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    assert written == 2

    df = read_all(tmp_settings.trajectory_parquet_path)
    assert df is not None
    # Every row is a placeholder (confidence=0, transition_kind='none').
    assert (df["confidence"] == 0.0).all()
    assert (df["transition_kind"] == "none").all()
    con.close()


def test_chunk_generic_exception_enqueues_to_retry_queue(
    tmp_path: Path, tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-refusal Exception → session enqueued to retry_queue, no rows written, session NOT checkpointed."""
    sid = "sess-exc"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [
                    _user("u1", sid, "opener long enough to clear filtering floor", off=0),
                    _user("u2", sid, "follower long enough to clear filtering floor", off=1),
                ],
            )
        ],
    )

    async def fake_chunk(provider, *, chunk):  # type: ignore[no-untyped-def]
        return RuntimeError("transient bedrock failure")

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)

    written = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    assert written == 0

    # Session must be on the retry queue.
    pending = retry_queue.pending_count(tmp_settings.checkpoint_db_path, pipeline="trajectory")
    assert pending >= 1
    con.close()


# ---------------------------------------------------------------------------
# Outer except (862-877) + session_failed return (880)
# ---------------------------------------------------------------------------


def test_outer_exception_in_classify_chunk_path_enqueues_session(
    tmp_path: Path, tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected exception RAISED (not returned) from _classify_chunk hits the outer except.

    The dispatch loop catches it, enqueues the session, and returns
    cleanly so the task group keeps draining other sessions.
    """
    sid = "sess-outer"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [_user("u1", sid, "opener long enough to clear filtering floor", off=0)],
            )
        ],
    )

    async def fake_chunk(provider, *, chunk):  # type: ignore[no-untyped-def]
        # RAISE rather than return — the outer ``except Exception`` catches it.
        raise ValueError("schema invariant broken mid-loop")

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)

    written = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    assert written == 0
    pending = retry_queue.pending_count(tmp_settings.checkpoint_db_path, pipeline="trajectory")
    assert pending >= 1
    con.close()


# ---------------------------------------------------------------------------
# Empty-active and 0-windows short-circuits (706-707, 710, 713-714, 724-725)
# ---------------------------------------------------------------------------


def test_retry_queue_drain_logs_count(
    tmp_path: Path, tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-enqueueing a session causes the drain branch to log ``draining N retry-queue entries``."""
    sid = "sess-drain"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [
                    _user("u1", sid, "opener long enough to clear filtering floor", off=0),
                    _user("u2", sid, "follower long enough to clear filtering floor", off=1),
                ],
            )
        ],
    )

    # Force the session onto the retry queue with next_attempt_at in the past.
    retry_queue.enqueue(
        tmp_settings.checkpoint_db_path,
        pipeline="trajectory",
        unit_id=sid,
        error="prior failure",
        now=datetime(2020, 1, 1, tzinfo=UTC),
    )

    async def fake_chunk(provider, *, chunk):  # type: ignore[no-untyped-def]
        return {
            (r[1], r[2]): {
                "prev_uuid": r[1],
                "curr_uuid": r[2],
                "prev_sentiment": None if r[1] is None else "neutral",
                "curr_sentiment": "neutral",
                "delta": None if r[1] is None else 0,
                "is_transition": False,
                "transition_kind": "none",
                "confidence": 0.8,
            }
            for r in chunk
        }

    monkeypatch.setattr(trajectory_worker, "_classify_chunk", fake_chunk)

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
    assert any("draining" in m and "retry-queue" in m for m in captured)
    con.close()


def test_skipped_via_checkpoint_logs_count(
    tmp_path: Path, tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero ``skipped_sessions`` from filter_unchanged → ``skipped N sessions via checkpoint`` (covers 710).

    Patches ``session_bounds`` and ``checkpointer.filter_unchanged`` to
    deterministically return a ``(pending, skipped)`` tuple with
    ``skipped > 0`` — exercising the log line without depending on
    timestamp-comparison subtleties in the live checkpoint table.
    """
    sid_skipped = "sess-skipped"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid_skipped,
                [_user("u1", sid_skipped, "old message long enough to clear filter floor", off=0)],
            )
        ],
    )

    monkeypatch.setattr(
        trajectory_worker,
        "session_bounds",
        lambda *a, **kw: {sid_skipped: (None, None)},
    )
    monkeypatch.setattr(
        trajectory_worker.checkpointer,
        "filter_unchanged",
        lambda *a, **kw: ([], 1),
    )
    # The skipped-log fires before windows are loaded; stub loading to [] so
    # the run short-circuits hermetically (no provider / Bedrock call).
    monkeypatch.setattr(trajectory_worker, "_load_windows", lambda *a, **kw: [])

    captured: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is None:
            return
        if record["level"].name == "INFO":
            captured.append(record["message"])

    handler_id = logger.add(_sink, level="INFO")
    try:
        written = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    finally:
        logger.remove(handler_id)
    assert written == 0
    assert any("skipped" in m and "sessions via checkpoint" in m for m in captured)
    con.close()


def test_no_sessions_in_window_short_circuits(
    tmp_path: Path, tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus filtered to nothing by ``since_days`` → ``no sessions in window`` + 0 (covers 712-714).

    The fixture corpus is timestamped 2026-04-01; with ``since_days=1`` the
    session_bounds query yields ``{}`` and the active-set is empty too,
    triggering the early-return branch.
    """
    sid = "sess-old"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [_user("u1", sid, "old message long enough to clear filter floor", off=0)],
            )
        ],
    )

    captured: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is None:
            return
        if record["level"].name == "INFO":
            captured.append(record["message"])

    handler_id = logger.add(_sink, level="INFO")
    try:
        # since_days=1 filters out the 2026-04-01 corpus when "today" is 2026-05-13.
        written = trajectory_worker.trajectory_messages(
            con, tmp_settings, dry_run=False, since_days=1
        )
    finally:
        logger.remove(handler_id)
    assert written == 0
    assert any("no sessions in window" in m for m in captured)
    con.close()


def test_zero_windows_after_filtering_short_circuits(
    tmp_path: Path, tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If session_bounds yields a session but ``_load_windows`` returns 0 rows, log + 0 (covers 723-725)."""
    sid = "sess-empty-windows"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [_user("u1", sid, "single message long enough to clear filter floor", off=0)],
            )
        ],
    )

    # Force _load_windows to return [] regardless of the corpus.
    monkeypatch.setattr(trajectory_worker, "_load_windows", lambda *a, **kw: [])

    captured: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is None:
            return
        if record["level"].name == "INFO":
            captured.append(record["message"])

    handler_id = logger.add(_sink, level="INFO")
    try:
        written = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=False)
    finally:
        logger.remove(handler_id)
    assert written == 0
    assert any("0 windows pending" in m for m in captured)
    con.close()


# ---------------------------------------------------------------------------
# Dry-run since_days + limit cap (942, 952)
# ---------------------------------------------------------------------------


def test_dry_run_with_since_days_filters_count(tmp_path: Path, tmp_settings: Settings) -> None:
    """``--dry-run --since-days N`` injects the ts-window filter into the count SQL (covers 942)."""
    sid = "sess-dry-since"
    con = _build_corpus(
        tmp_path,
        [
            (
                sid,
                [_user("u1", sid, "single dry-run message long enough for filter", off=0)],
            )
        ],
    )
    plan = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=True, since_days=3650)
    assert isinstance(plan, dict)
    assert plan["since_days"] == 3650
    # since_days=3650 covers the corpus, so candidates >= 1.
    assert plan["candidates"] >= 1
    con.close()


def test_dry_run_with_limit_caps_session_count(tmp_path: Path, tmp_settings: Settings) -> None:
    """``--dry-run --limit 1`` caps ``candidates`` via ``min(sessions, limit)`` (covers 952)."""
    sid_a = "sess-cap-a"
    sid_b = "sess-cap-b"
    sid_c = "sess-cap-c"
    con = _build_corpus(
        tmp_path,
        [
            (sid_a, [_user("a1", sid_a, "session A opener clearing filter floor", off=0)]),
            (sid_b, [_user("b1", sid_b, "session B opener clearing filter floor", off=0)]),
            (sid_c, [_user("c1", sid_c, "session C opener clearing filter floor", off=0)]),
        ],
    )
    plan = trajectory_worker.trajectory_messages(con, tmp_settings, dry_run=True, limit=1)
    assert isinstance(plan, dict)
    assert plan["candidates"] == 1
    assert plan["limit"] == 1
    con.close()
