"""The ``claude-sql watch`` command body and the adapter branches around it.

Covers the composition root: settings → watcher → registered connection → loop →
NDJSON stream → summary. The change source is swapped for a scripted double so
the test does not depend on OS event timing, but everything below it (registration,
``refresh_raw``, the debounce) is the real thing.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from claude_sql.infrastructure import file_watcher
from claude_sql.infrastructure.file_watcher import (
    PollingWatcher,
    WatchfilesWatcher,
    _is_transcript,
    build_file_watcher,
    watchfiles_available,
)
from claude_sql.interfaces.cli import app as cli
from claude_sql.interfaces.cli.app import Common
from claude_sql.interfaces.cli.output import OutputFormat
from conftest import make_user_msg, write_session_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

SID_WATCH = "55555555-5555-5555-5555-555555555555"


class ScriptedWatcher:
    """A :class:`FileWatcherPort` double replaying fixed batches."""

    def __init__(self, batches: Sequence[frozenset[str]]) -> None:
        self._batches = list(batches)

    def changed_batches(self) -> Iterator[frozenset[str]]:
        yield from self._batches


def _common(tmp_corpus: dict[str, Any], fmt: OutputFormat = OutputFormat.JSON) -> Common:
    os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = tmp_corpus["subagent_meta_glob"]
    return Common(
        verbose=False,
        quiet=True,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        format=fmt,
    )


@pytest.fixture(autouse=True)
def _purge_meta_glob_env() -> Iterator[None]:
    """Restore ``CLAUDE_SQL_SUBAGENT_META_GLOB`` so it does not bleed across modules."""
    prior = os.environ.get("CLAUDE_SQL_SUBAGENT_META_GLOB")
    yield
    if prior is None:
        os.environ.pop("CLAUDE_SQL_SUBAGENT_META_GLOB", None)
    else:
        os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = prior


def _write_new_session(root: Path) -> str:
    write_session_jsonl(
        root / f"{SID_WATCH}.jsonl",
        messages=[
            make_user_msg(
                "cw1",
                SID_WATCH,
                "a session that landed while the watch command was running",
                ts="2026-04-07T08:00:00.000Z",
            )
        ],
    )
    return str(root / f"{SID_WATCH}.jsonl")


# ---------------------------------------------------------------------------
# The command body
# ---------------------------------------------------------------------------


class WritingWatcher:
    """Writes a transcript on its first yield, then reports it.

    The write has to happen INSIDE the loop: ``watch`` registers the connection
    before consuming the source, so a file already on disk at startup is already
    in the snapshot and the refresh correctly finds nothing to do.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def changed_batches(self) -> Iterator[frozenset[str]]:
        path = _write_new_session(self._root)
        yield frozenset({path})
        yield frozenset[str]()


def test_watch_streams_a_flush_then_a_summary(
    tmp_corpus: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_resolve_settings",
        lambda common: cli.Settings(
            default_glob=tmp_corpus["glob"],
            subagent_glob=tmp_corpus["subagent_glob"],
            subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
        ),
    )
    monkeypatch.setattr(
        file_watcher,
        "build_file_watcher",
        lambda *_args, **_kwargs: WritingWatcher(tmp_corpus["root"]),
    )

    cli.watch(quiet_period=0.0, max_flushes=1, common=_common(tmp_corpus))

    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line.strip()]
    # One NDJSON flush line, then the pretty-printed summary object.
    flush = json.loads(lines[0])
    assert flush["flush_index"] == 1
    assert flush["rows_inserted_count"] >= 1
    assert flush["rebuilt"] is False
    summary = json.loads("\n".join(lines[1:]))
    assert summary["flush_count"] == 1


def test_watch_on_a_tty_format_does_not_stream_ndjson(
    tmp_corpus: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--format table`` is for a human; per-flush JSON would be noise."""
    path = _write_new_session(tmp_corpus["root"])
    monkeypatch.setattr(
        cli,
        "_resolve_settings",
        lambda common: cli.Settings(
            default_glob=tmp_corpus["glob"],
            subagent_glob=tmp_corpus["subagent_glob"],
            subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
        ),
    )
    monkeypatch.setattr(
        file_watcher,
        "build_file_watcher",
        lambda *_args, **_kwargs: ScriptedWatcher([frozenset({path}), frozenset()]),
    )

    cli.watch(
        quiet_period=0.0,
        max_flushes=1,
        common=_common(tmp_corpus, fmt=OutputFormat.TABLE),
    )

    out = capsys.readouterr().out
    assert "flush_index" not in out


def test_watch_exits_zero_on_keyboard_interrupt(
    tmp_corpus: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C is how a daemon is stopped, not a fault."""

    class InterruptingWatcher:
        def changed_batches(self) -> Iterator[frozenset[str]]:
            raise KeyboardInterrupt
            yield frozenset[str]()  # pragma: no cover - unreachable, satisfies the generator shape

    monkeypatch.setattr(
        cli,
        "_resolve_settings",
        lambda common: cli.Settings(
            default_glob=tmp_corpus["glob"],
            subagent_glob=tmp_corpus["subagent_glob"],
            subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
        ),
    )
    monkeypatch.setattr(
        file_watcher, "build_file_watcher", lambda *_args, **_kwargs: InterruptingWatcher()
    )

    # Returns normally — no SystemExit, no propagated KeyboardInterrupt.
    cli.watch(common=_common(tmp_corpus))


# ---------------------------------------------------------------------------
# Adapter branches
# ---------------------------------------------------------------------------


def test_only_jsonl_paths_reach_the_loop() -> None:
    """``*.meta.json`` binds as a live VIEW and DuckDB spill files are not corpus."""
    assert _is_transcript("/a/b/session.jsonl")
    assert not _is_transcript("/a/b/agent-x.meta.json")
    assert not _is_transcript("/a/b/duckdb_tmp/spill.tmp")


def test_watchfiles_watcher_stops_when_there_is_nothing_to_watch(tmp_path: Path) -> None:
    """No resolvable root means no subscription — say so instead of watching /."""
    watcher = WatchfilesWatcher([str(tmp_path / "absent" / "*" / "*.jsonl")])

    assert watcher.roots == ()
    assert list(watcher.changed_batches()) == []


@pytest.mark.skipif(not watchfiles_available(), reason="requires the 'watch' extra")
def test_build_file_watcher_prefers_os_events(tmp_corpus: dict[str, Any]) -> None:
    watcher = build_file_watcher([tmp_corpus["glob"]])

    assert isinstance(watcher, WatchfilesWatcher)


def test_glob_root_is_none_when_the_literal_prefix_is_a_file(tmp_path: Path) -> None:
    """A pattern rooted at a file has no watchable directory."""
    from claude_sql.infrastructure.source_files import glob_root

    afile = tmp_path / "afile.jsonl"
    afile.write_text("{}\n")

    assert glob_root(str(afile / "*.jsonl")) is None


def test_scan_skips_a_file_that_vanishes_mid_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-flight session can be rotated between the glob and the stat."""
    from claude_sql.infrastructure import source_files

    (tmp_path / "gone.jsonl").write_text("{}\n")
    (tmp_path / "stays.jsonl").write_text("{}\n")
    real_stat = os.stat

    def _stat_raising_for_gone(path: object, *args: object, **kwargs: object) -> object:
        if str(path).endswith("gone.jsonl"):
            raise FileNotFoundError(str(path))
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(source_files.os, "stat", _stat_raising_for_gone)

    scanned = source_files.scan_glob_mtimes(str(tmp_path / "*.jsonl"))

    assert set(scanned) == {str(tmp_path / "stays.jsonl")}


def test_polling_watcher_skips_a_vanished_file(tmp_corpus: dict[str, Any]) -> None:
    """A file removed between the glob expansion and the stat is not an error.

    An in-flight session can be rotated mid-scan; a scan is a best-effort
    observation, not a transaction.
    """
    watcher = PollingWatcher(
        [tmp_corpus["glob"]], max_ticks=2, sleep=lambda _seconds: None, interval_seconds=0.0
    )
    batches = watcher.changed_batches()
    assert next(batches) == frozenset()

    doomed = tmp_corpus["root"] / "66666666-6666-6666-6666-666666666666.jsonl"
    write_session_jsonl(
        doomed,
        messages=[
            make_user_msg("d1", "doomed", "written then removed", ts="2026-04-08T08:00:00.000Z")
        ],
    )
    doomed.unlink()

    # The removed file simply is not in the scan; no exception escapes.
    assert next(batches) == frozenset()
