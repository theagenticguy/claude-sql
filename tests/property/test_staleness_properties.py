"""Property tests for the durable-state staleness + backoff rules.

Both symbols moved under ``infrastructure/sqlite_state/`` in the hexagonal
cutover but stay pure functions (no DB handle needed):

* ``checkpointer._stale_or_equal(cur, prev)`` — True iff both timestamps are
  present and ``cur`` has NOT advanced past ``prev`` (i.e. skip). Invariants:
  idempotence (a second identical comparison agrees) and monotonicity
  (advancing ``cur`` past ``prev`` flips skip→pending and never the reverse).
* ``retry_queue._backoff_delta(attempts)`` — ``min(2**attempts, 60)`` minutes:
  monotone non-decreasing in attempts and capped at 60 minutes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import given, settings, strategies as st

from claude_sql.infrastructure.sqlite_state.checkpointer import _stale_or_equal
from claude_sql.infrastructure.sqlite_state.retry_queue import (
    _BACKOFF_CAP_MIN,
    _backoff_delta,
)

# tz-aware UTC datetimes — the shape the checkpointer boundary guarantees.
_AWARE_DT = st.datetimes(
    min_value=datetime(2000, 1, 1),  # noqa: DTZ001 — bound literal, tz added below
    max_value=datetime(2100, 1, 1),  # noqa: DTZ001 — bound literal, tz added below
    timezones=st.just(UTC),
)


# ---------------------------------------------------------------------------
# _stale_or_equal — idempotence
# ---------------------------------------------------------------------------


@given(cur=st.one_of(st.none(), _AWARE_DT), prev=st.one_of(st.none(), _AWARE_DT))
@settings(max_examples=150)
def test_stale_or_equal_idempotent(cur: datetime | None, prev: datetime | None) -> None:
    """The comparison is pure — evaluating it twice agrees."""
    assert _stale_or_equal(cur, prev) == _stale_or_equal(cur, prev)


@given(cur=_AWARE_DT, prev=_AWARE_DT)
@settings(max_examples=150)
def test_stale_or_equal_matches_comparison(cur: datetime, prev: datetime) -> None:
    """With both present, the result is exactly ``cur <= prev``."""
    assert _stale_or_equal(cur, prev) == (cur <= prev)


@given(dt=st.one_of(st.none(), _AWARE_DT))
@settings(max_examples=50)
def test_stale_or_equal_none_side_is_pending(dt: datetime | None) -> None:
    """A None on either side is treated as 'advance' (never skip)."""
    assert _stale_or_equal(None, dt) is False
    assert _stale_or_equal(dt, None) is False


# ---------------------------------------------------------------------------
# _stale_or_equal — monotonicity
# ---------------------------------------------------------------------------


@given(prev=_AWARE_DT, delta_secs=st.integers(min_value=1, max_value=10**8))
@settings(max_examples=150)
def test_stale_or_equal_advancing_cur_flips_skip_to_pending(
    prev: datetime, delta_secs: int
) -> None:
    """Advancing ``cur`` strictly past ``prev`` flips skip(True)→pending(False).

    At equality the rule skips (True); strictly after, it advances (False). The
    flip only ever goes skip→pending as ``cur`` grows, never the reverse.
    """
    at_equal = _stale_or_equal(prev, prev)
    after = _stale_or_equal(prev + timedelta(seconds=delta_secs), prev)
    before = _stale_or_equal(prev - timedelta(seconds=delta_secs), prev)
    assert at_equal is True  # equality is "stale or equal"
    assert after is False  # advanced past prev → pending
    assert before is True  # still behind prev → skip


@given(
    prev=_AWARE_DT,
    small=st.integers(min_value=1, max_value=1000),
    big=st.integers(min_value=1001, max_value=10**7),
)
@settings(max_examples=100)
def test_stale_or_equal_monotone_never_reverses(prev: datetime, small: int, big: int) -> None:
    """As ``cur`` increases, the skip flag is monotone non-increasing (True→False only)."""
    r_small = _stale_or_equal(prev + timedelta(seconds=small), prev)
    r_big = _stale_or_equal(prev + timedelta(seconds=big), prev)
    # skip==True is "1"; advancing cur can only lower or hold the flag.
    assert int(r_small) >= int(r_big)


# ---------------------------------------------------------------------------
# _backoff_delta — monotone + capped
# ---------------------------------------------------------------------------


@given(attempts=st.integers(min_value=0, max_value=64))
@settings(max_examples=100)
def test_backoff_delta_capped_at_60_min(attempts: int) -> None:
    """The backoff never exceeds the 60-minute cap and is always positive."""
    delta = _backoff_delta(attempts)
    assert timedelta(0) < delta <= timedelta(minutes=_BACKOFF_CAP_MIN)


@given(a=st.integers(min_value=0, max_value=64), b=st.integers(min_value=0, max_value=64))
@settings(max_examples=150)
def test_backoff_delta_monotone_non_decreasing(a: int, b: int) -> None:
    """More attempts never shortens the backoff."""
    lo, hi = sorted((a, b))
    assert _backoff_delta(lo) <= _backoff_delta(hi)
