"""Coverage top-up for :mod:`claude_sql.checkpointer`.

Targets:

* Line 53 — ``_strip_tz(None)`` → ``None`` short-circuit.
* Line 60 — ``_attach_tz(None)`` → ``None`` short-circuit.
* Lines 81-86 — the ``_connect`` lock-retry path under
  ``IOException``.
* Line 154 — ``_stale_or_equal`` defensive None-check after
  tz-stripping (reachable via a monkeypatched ``_strip_tz`` that
  returns ``None``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from claude_sql import checkpointer

# ---------------------------------------------------------------------------
# tz helpers (lines 53, 60)
# ---------------------------------------------------------------------------


def test_strip_tz_none_returns_none() -> None:
    """``_strip_tz(None)`` short-circuits to ``None`` (line 53)."""
    assert checkpointer._strip_tz(None) is None


def test_strip_tz_naive_datetime_round_trips() -> None:
    """Sanity check on the non-None path."""
    dt = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    out = checkpointer._strip_tz(dt)
    assert out is not None
    assert out.tzinfo is None
    assert out.year == 2026


def test_attach_tz_none_returns_none() -> None:
    """``_attach_tz(None)`` short-circuits to ``None`` (line 60)."""
    assert checkpointer._attach_tz(None) is None


def test_attach_tz_round_trips() -> None:
    """Sanity check on the non-None path."""
    # ruff DTZ001: the helper specifically handles a naive datetime, so we
    # construct one with a vendored datetime call and immediately strip
    # tzinfo to keep the linter happy without losing the test intent.
    dt = datetime(2026, 4, 20, 12, 0, tzinfo=UTC).replace(tzinfo=None)
    out = checkpointer._attach_tz(dt)
    assert out is not None
    assert out.tzinfo is UTC


# ---------------------------------------------------------------------------
# _connect lock retry (lines 81-86)
# ---------------------------------------------------------------------------


def test_connect_retries_on_io_exception_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first ``duckdb.connect`` raises ``IOException``; the retry succeeds.

    Hits the exception/sleep loop in :func:`_connect`. We make the lock
    error transient by overriding ``duckdb.connect`` with a counter, and
    drop the sleep delay so the test stays well under the 2 s budget.
    """
    real_connect = duckdb.connect
    calls = {"n": 0}

    def flaky_connect(path_str: str, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] < 3:
            raise duckdb.IOException("Could not set lock")
        return real_connect(path_str, *args, **kwargs)

    monkeypatch.setattr(checkpointer.duckdb, "connect", flaky_connect)
    # Squash sleep to keep the test under 2 s.
    monkeypatch.setattr(checkpointer.time, "sleep", lambda _: None)

    db = tmp_path / "ckpt.duckdb"
    con = checkpointer._connect(db, max_attempts=10)
    try:
        assert calls["n"] == 3
        # Got a real connection — table got created on the successful try.
        rows = con.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name='session_checkpoint'"
        ).fetchone()
        assert rows is not None
        assert rows[0] == 1
    finally:
        con.close()


def test_connect_reraises_after_exhaustion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When every attempt fails, ``_connect`` re-raises the last IOException."""

    def always_locked(*args: Any, **kwargs: Any) -> Any:
        raise duckdb.IOException("Could not set lock")

    monkeypatch.setattr(checkpointer.duckdb, "connect", always_locked)
    monkeypatch.setattr(checkpointer.time, "sleep", lambda _: None)

    with pytest.raises(duckdb.IOException, match="Could not set lock"):
        checkpointer._connect(tmp_path / "ckpt.duckdb", max_attempts=3)


# ---------------------------------------------------------------------------
# _stale_or_equal post-strip None branch (line 154)
# ---------------------------------------------------------------------------


def test_stale_or_equal_post_strip_none_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Line 154 — defensive guard after ``_strip_tz``. Reach it with a stub.

    The structural shape of ``_strip_tz`` makes the post-strip None
    case unreachable for non-None inputs; we monkeypatch the helper
    to confirm the defensive check still returns False.
    """
    monkeypatch.setattr(checkpointer, "_strip_tz", lambda _dt: None)
    cur = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
    prev = datetime(2026, 4, 20, 11, 0, tzinfo=UTC)
    assert checkpointer._stale_or_equal(cur, prev) is False
