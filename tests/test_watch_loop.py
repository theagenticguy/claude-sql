"""The watch loop and its change-source adapters.

The loop is driven with a scripted :class:`FileWatcherPort` double and an
injected clock, so the debounce is exercised deterministically instead of by
sleeping. The refresh underneath is real: these run against a real DuckDB
connection over a real corpus, because the thing worth checking is that a flush
actually advances the snapshot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import duckdb
import pytest

from claude_sql.application.use_cases.watch import WatchConfig, run_watch
from claude_sql.infrastructure import duckdb_views, file_watcher
from claude_sql.infrastructure.file_watcher import (
    PollingWatcher,
    build_file_watcher,
    watchfiles_available,
)
from claude_sql.infrastructure.source_files import glob_root, glob_roots
from conftest import make_user_msg, write_session_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from claude_sql.infrastructure.settings import Settings

SECOND_NS = 1_000_000_000
SID_NEW = "44444444-4444-4444-4444-444444444444"


class ScriptedWatcher:
    """A :class:`FileWatcherPort` that replays a fixed list of batches."""

    def __init__(self, batches: Sequence[frozenset[str]]) -> None:
        self._batches = list(batches)

    def changed_batches(self) -> Iterator[frozenset[str]]:
        yield from self._batches


class FakeClock:
    """Monotonic-nanosecond clock that advances a fixed step per read."""

    def __init__(self, *, step_ns: int) -> None:
        self._now_ns = 0
        self._step_ns = step_ns

    def __call__(self) -> int:
        self._now_ns += self._step_ns
        return self._now_ns


def _settings_for(tmp_corpus: dict[str, Any], tmp_settings: Settings) -> Settings:
    """``tmp_settings`` re-pointed at the fixture corpus's globs."""
    return tmp_settings.model_copy(
        update={
            "default_glob": tmp_corpus["glob"],
            "subagent_glob": tmp_corpus["subagent_glob"],
            "subagent_meta_glob": tmp_corpus["subagent_meta_glob"],
        }
    )


def _registered(tmp_corpus: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    duckdb_views.register_raw(
        con,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
    )
    return con


def _write_new_session(root: Path) -> str:
    write_session_jsonl(
        root / f"{SID_NEW}.jsonl",
        messages=[
            make_user_msg(
                "w1",
                SID_NEW,
                "a session that landed while the watch loop was running",
                ts="2026-04-06T08:00:00.000Z",
            )
        ],
    )
    return str(root / f"{SID_NEW}.jsonl")


def _rows(con: duckdb.DuckDBPyConnection) -> int:
    row = con.execute("SELECT count(*) FROM v_raw_events").fetchone()
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_a_quiet_file_flushes_and_advances_the_snapshot(
    tmp_corpus: dict[str, Any],
    tmp_settings: Settings,
) -> None:
    con = _registered(tmp_corpus)
    try:
        before = _rows(con)
        path = _write_new_session(tmp_corpus["root"])
        # One batch announcing the write, then a heartbeat once the quiet period
        # has elapsed on the clock.
        watcher = ScriptedWatcher([frozenset({path}), frozenset()])

        totals = run_watch(
            con=con,
            settings=_settings_for(tmp_corpus, tmp_settings),
            watcher=watcher,
            config=WatchConfig(quiet_period_seconds=5.0, max_wait_seconds=60.0),
            monotonic_ns=FakeClock(step_ns=10 * SECOND_NS),
        )

        assert totals.flush_count == 1
        assert totals.rows_inserted_count == 1
        assert _rows(con) == before + 1
    finally:
        con.close()


def test_a_file_still_being_written_does_not_flush(
    tmp_corpus: dict[str, Any],
    tmp_settings: Settings,
) -> None:
    """Every batch arrives within the quiet period, so nothing is due."""
    con = _registered(tmp_corpus)
    try:
        before = _rows(con)
        path = _write_new_session(tmp_corpus["root"])
        watcher = ScriptedWatcher([frozenset({path})] * 4)

        totals = run_watch(
            con=con,
            settings=_settings_for(tmp_corpus, tmp_settings),
            watcher=watcher,
            config=WatchConfig(quiet_period_seconds=60.0, max_wait_seconds=0.0),
            monotonic_ns=FakeClock(step_ns=1 * SECOND_NS),
        )

        assert totals.flush_count == 0
        assert _rows(con) == before
    finally:
        con.close()


def test_a_continuously_written_file_still_flushes_at_the_max_wait_bound(
    tmp_corpus: dict[str, Any],
    tmp_settings: Settings,
) -> None:
    """Same schedule as the previous test, but with the starvation guard armed."""
    con = _registered(tmp_corpus)
    try:
        path = _write_new_session(tmp_corpus["root"])
        watcher = ScriptedWatcher([frozenset({path})] * 4)

        totals = run_watch(
            con=con,
            settings=_settings_for(tmp_corpus, tmp_settings),
            watcher=watcher,
            config=WatchConfig(quiet_period_seconds=60.0, max_wait_seconds=2.0),
            monotonic_ns=FakeClock(step_ns=1 * SECOND_NS),
        )

        assert totals.flush_count >= 1
    finally:
        con.close()


def test_the_loop_does_not_embed_unless_asked(
    tmp_corpus: dict[str, Any],
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embed_limit=0`` is the default and must mean zero provider calls."""
    con = _registered(tmp_corpus)
    calls: list[int] = []

    def _explode(*_args: object, **_kwargs: object) -> int:
        calls.append(1)
        raise AssertionError("watch must not embed when embed_limit is 0")

    try:
        monkeypatch.setattr("claude_sql.application.use_cases.watch._embed_new_messages", _explode)
        path = _write_new_session(tmp_corpus["root"])
        watcher = ScriptedWatcher([frozenset({path}), frozenset()])

        totals = run_watch(
            con=con,
            settings=_settings_for(tmp_corpus, tmp_settings),
            watcher=watcher,
            config=WatchConfig(quiet_period_seconds=5.0),
            monotonic_ns=FakeClock(step_ns=10 * SECOND_NS),
        )

        assert totals.flush_count == 1
        assert totals.embedded_count == 0
        assert calls == []
    finally:
        con.close()


def test_embedding_is_capped_per_flush_when_enabled(
    tmp_corpus: dict[str, Any],
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_limits: list[int] = []

    def _fake_embed(_con: object, _settings: object, *, limit: int) -> int:
        seen_limits.append(limit)
        return limit

    con = _registered(tmp_corpus)
    try:
        monkeypatch.setattr(
            "claude_sql.application.use_cases.watch._embed_new_messages", _fake_embed
        )
        path = _write_new_session(tmp_corpus["root"])
        watcher = ScriptedWatcher([frozenset({path}), frozenset()])

        totals = run_watch(
            con=con,
            settings=_settings_for(tmp_corpus, tmp_settings),
            watcher=watcher,
            config=WatchConfig(quiet_period_seconds=5.0, embed_limit=7),
            monotonic_ns=FakeClock(step_ns=10 * SECOND_NS),
        )

        assert seen_limits == [7]
        assert totals.embedded_count == 7
    finally:
        con.close()


def test_an_embedding_failure_does_not_stop_the_free_refresh(
    tmp_corpus: dict[str, Any],
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throttled provider must not take down the loop's primary job."""
    con = _registered(tmp_corpus)
    try:
        before = _rows(con)

        def _boom(*_args: object, **_kwargs: object) -> list[float]:
            raise RuntimeError("ThrottlingException")

        monkeypatch.setattr("claude_sql.application.use_cases.embed.run_backfill", _boom)
        path = _write_new_session(tmp_corpus["root"])
        watcher = ScriptedWatcher([frozenset({path}), frozenset()])

        totals = run_watch(
            con=con,
            settings=_settings_for(tmp_corpus, tmp_settings),
            watcher=watcher,
            config=WatchConfig(quiet_period_seconds=5.0, embed_limit=4),
            monotonic_ns=FakeClock(step_ns=10 * SECOND_NS),
        )

        assert totals.flush_count == 1
        assert totals.embedded_count == 0
        assert _rows(con) == before + 1
    finally:
        con.close()


def test_max_flushes_stops_the_loop(
    tmp_corpus: dict[str, Any],
    tmp_settings: Settings,
) -> None:
    con = _registered(tmp_corpus)
    try:
        path = _write_new_session(tmp_corpus["root"])
        watcher = ScriptedWatcher([frozenset({path}), frozenset(), frozenset(), frozenset()])

        totals = run_watch(
            con=con,
            settings=_settings_for(tmp_corpus, tmp_settings),
            watcher=watcher,
            config=WatchConfig(quiet_period_seconds=0.0, max_flushes=1),
            monotonic_ns=FakeClock(step_ns=SECOND_NS),
        )

        assert totals.flush_count == 1
    finally:
        con.close()


def test_flush_reports_stream_as_they_happen(
    tmp_corpus: dict[str, Any],
    tmp_settings: Settings,
) -> None:
    con = _registered(tmp_corpus)
    seen: list[dict[str, object]] = []
    try:
        path = _write_new_session(tmp_corpus["root"])
        watcher = ScriptedWatcher([frozenset({path}), frozenset()])

        run_watch(
            con=con,
            settings=_settings_for(tmp_corpus, tmp_settings),
            watcher=watcher,
            config=WatchConfig(quiet_period_seconds=5.0),
            on_flush=lambda report: seen.append(report.to_payload()),
            monotonic_ns=FakeClock(step_ns=10 * SECOND_NS),
        )

        assert len(seen) == 1
        assert seen[0]["flush_index"] == 1
        assert seen[0]["covers_through"] is not None
    finally:
        con.close()


def test_the_refresh_set_comes_from_the_watermark_not_the_event(
    tmp_corpus: dict[str, Any],
    tmp_settings: Settings,
) -> None:
    """A missed event must self-heal: the flush re-derives what moved.

    The scripted batch names a file that does NOT exist, while a different
    session lands on disk unannounced. A loop that refreshed only the announced
    paths would miss it permanently — dropped inotify events are real.
    """
    con = _registered(tmp_corpus)
    try:
        before = _rows(con)
        _write_new_session(tmp_corpus["root"])
        watcher = ScriptedWatcher([frozenset({"/nonexistent/never-written.jsonl"}), frozenset()])

        totals = run_watch(
            con=con,
            settings=_settings_for(tmp_corpus, tmp_settings),
            watcher=watcher,
            config=WatchConfig(quiet_period_seconds=5.0),
            monotonic_ns=FakeClock(step_ns=10 * SECOND_NS),
        )

        assert totals.flush_count == 1
        assert _rows(con) == before + 1
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def test_polling_watcher_baseline_scan_is_not_a_change(tmp_corpus: dict[str, Any]) -> None:
    """The first poll must not report the whole corpus as changed."""
    watcher = PollingWatcher(
        [tmp_corpus["glob"]], max_ticks=1, sleep=lambda _seconds: None, interval_seconds=0.0
    )

    batches = list(watcher.changed_batches())

    assert batches == [frozenset()]


def test_polling_watcher_reports_what_moved_after_the_baseline(
    tmp_corpus: dict[str, Any],
) -> None:
    watcher = PollingWatcher(
        [tmp_corpus["glob"]], max_ticks=2, sleep=lambda _seconds: None, interval_seconds=0.0
    )
    batches_iter = watcher.changed_batches()

    assert next(batches_iter) == frozenset()
    path = _write_new_session(tmp_corpus["root"])
    assert next(batches_iter) == frozenset({path})


def test_polling_watcher_stops_on_a_remote_glob() -> None:
    watcher = PollingWatcher(["s3://bucket/prefix/*/*.jsonl"], sleep=lambda _seconds: None)

    assert list(watcher.changed_batches()) == []


def test_glob_root_strips_wildcard_segments(tmp_path: Path) -> None:
    (tmp_path / "projects" / "proj").mkdir(parents=True)

    assert glob_root(str(tmp_path / "projects" / "*" / "*.jsonl")) == tmp_path / "projects"


def test_glob_root_is_none_for_a_remote_or_absent_root(tmp_path: Path) -> None:
    assert glob_root("s3://bucket/prefix/*.jsonl") is None
    assert glob_root(str(tmp_path / "absent" / "*.jsonl")) is None


def test_glob_roots_collapses_nested_roots(tmp_path: Path) -> None:
    """Watchers recurse, so a root under another root would double-deliver."""
    nested = tmp_path / "projects" / "a" / "subagents"
    nested.mkdir(parents=True)

    roots = glob_roots(
        [
            str(tmp_path / "projects" / "*" / "*.jsonl"),
            str(tmp_path / "projects" / "a" / "subagents" / "*.jsonl"),
        ]
    )

    assert roots == (tmp_path / "projects",)


def test_build_file_watcher_honors_force_polling(tmp_corpus: dict[str, Any]) -> None:
    watcher = build_file_watcher([tmp_corpus["glob"]], force_polling=True)

    assert isinstance(watcher, PollingWatcher)


def test_build_file_watcher_falls_back_when_the_extra_is_absent(
    tmp_corpus: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_watcher, "watchfiles_available", lambda: False)

    watcher = build_file_watcher([tmp_corpus["glob"]])

    assert isinstance(watcher, PollingWatcher)


@pytest.mark.skipif(not watchfiles_available(), reason="requires the 'watch' extra")
def test_watchfiles_watcher_resolves_its_roots(tmp_corpus: dict[str, Any]) -> None:
    from claude_sql.infrastructure.file_watcher import WatchfilesWatcher

    watcher = WatchfilesWatcher([tmp_corpus["glob"], tmp_corpus["subagent_glob"]])

    assert watcher.roots == (tmp_corpus["tmp_path"] / "projects",)
