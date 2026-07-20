"""Property tests for the pure dedup helpers (``domain/dedup.py``).

Invariant clusters:

* ``simhash64`` — determinism (same input → same signature) and the signed
  64-bit range contract DuckDB's BIGINT column depends on.
* ``hamming_distance_64`` — reflexivity (d(x,x)==0), symmetry (d(a,b)==d(b,a)),
  and the 0..64 bound.
* ``token_budget_bucket`` — monotone non-decreasing in the input token count and
  stable for the same input.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from claude_sql.domain.dedup import (
    hamming_distance_64,
    simhash64,
    token_budget_bucket,
)

# Full signed BIGINT range — the space simhash64 emits into.
_INT64 = st.integers(min_value=-(2**63), max_value=2**63 - 1)
_BUCKET_ORDER = {"xs": 0, "sm": 1, "md": 2, "lg": 3, "xl": 4}

_ANY_TEXT = st.text(max_size=300)


# ---------------------------------------------------------------------------
# simhash64
# ---------------------------------------------------------------------------


@given(text=_ANY_TEXT)
@settings(max_examples=100)
def test_simhash64_deterministic(text: str) -> None:
    """The same text always hashes to the same signature."""
    assert simhash64(text) == simhash64(text)


@given(text=_ANY_TEXT)
@settings(max_examples=100)
def test_simhash64_signed_bigint_range(text: str) -> None:
    """Every signature fits the signed 64-bit BIGINT range."""
    sig = simhash64(text)
    assert -(2**63) <= sig < 2**63


# ---------------------------------------------------------------------------
# hamming_distance_64
# ---------------------------------------------------------------------------


@given(x=_INT64)
@settings(max_examples=100)
def test_hamming_reflexive_zero(x: int) -> None:
    """Distance from a value to itself is zero."""
    assert hamming_distance_64(x, x) == 0


@given(a=_INT64, b=_INT64)
@settings(max_examples=150)
def test_hamming_symmetric(a: int, b: int) -> None:
    """Hamming distance is symmetric."""
    assert hamming_distance_64(a, b) == hamming_distance_64(b, a)


@given(a=_INT64, b=_INT64)
@settings(max_examples=150)
def test_hamming_bounded_0_to_64(a: int, b: int) -> None:
    """Distance over 64-bit values is in the closed range 0..64."""
    d = hamming_distance_64(a, b)
    assert 0 <= d <= 64


@given(a=_INT64, b=_INT64)
@settings(max_examples=100)
def test_hamming_matches_simhash_pairs(a: int, b: int) -> None:
    """Distance is consistent when both operands are real signatures too."""
    sa, sb = simhash64(str(a)), simhash64(str(b))
    assert 0 <= hamming_distance_64(sa, sb) <= 64
    assert hamming_distance_64(sa, sa) == 0


# ---------------------------------------------------------------------------
# token_budget_bucket
# ---------------------------------------------------------------------------


@given(tokens=st.integers(min_value=0, max_value=5_000_000))
@settings(max_examples=100)
def test_bucket_stable(tokens: int) -> None:
    """The bucket is a pure function of the input."""
    assert token_budget_bucket(tokens) == token_budget_bucket(tokens)


@given(
    a=st.integers(min_value=0, max_value=5_000_000),
    b=st.integers(min_value=0, max_value=5_000_000),
)
@settings(max_examples=200)
def test_bucket_monotone_non_decreasing(a: int, b: int) -> None:
    """More tokens never map to a smaller bucket."""
    lo, hi = sorted((a, b))
    assert _BUCKET_ORDER[token_budget_bucket(lo)] <= _BUCKET_ORDER[token_budget_bucket(hi)]
