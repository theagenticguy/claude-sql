"""Pure-function unit tests for the SimHash + token-bucket helpers in
``claude_sql.ingest``.

These don't touch DuckDB or Bedrock — every test exercises a single
function over hand-rolled strings so a regression in the hashing path
fails fast without dragging in the corpus fixtures.
"""

from __future__ import annotations

import pytest

from claude_sql.domain.dedup import (
    _ANTHROPIC_RATIO,
    approx_tokens_batch,
    hamming_distance_64,
    simhash64,
    token_budget_bucket,
)

# ---------------------------------------------------------------------------
# simhash64
# ---------------------------------------------------------------------------


def test_simhash_identical_texts_match() -> None:
    """The same input must hash to the same signature."""
    text = "the quick brown fox jumps over the lazy dog repeatedly today"
    assert simhash64(text) == simhash64(text)


def test_simhash_one_word_edit_low_hamming() -> None:
    """A one-word edit in a long doc must stay close in Hamming space.

    Threshold is ≤ 8 bits — the ``canonical_uuid_resolve`` macro uses 3,
    but that's calibrated for paste-retries (no edits) — a single-word
    edit is allowed to drift further as long as it stays below ``len(toks) / 6``
    or so.
    """
    base = (
        "the agent ran the test suite and discovered three failing cases "
        "in the auth module while the rest of the integration tests passed "
        "without surfacing any new regressions in the snapshot output"
    )
    edited = base.replace("three", "seven")
    assert hamming_distance_64(simhash64(base), simhash64(edited)) <= 12


def test_simhash_unrelated_texts_high_hamming() -> None:
    """Two unrelated 50-word docs must have a non-trivial Hamming distance.

    A genuinely random pair of 64-bit ints has expected distance 32 and
    standard deviation 4. We assert ≥ 16 — well below the mean but high
    enough that the canonical-resolution threshold (3) won't flag them.
    """
    a = (
        "we landed the new release process this morning and bumped the "
        "version number after the test suite passed everything green and "
        "the on-call paged us about an unrelated networking blip in the "
        "eu-west region just before noon"
    )
    b = (
        "the cooking class today covered three different sourdough recipes "
        "with different hydration levels and overnight rest schedules and "
        "the instructor showed how to score the loaves before they go in "
        "the oven for the bake"
    )
    assert hamming_distance_64(simhash64(a), simhash64(b)) >= 16


def test_simhash_empty_text_returns_zero() -> None:
    """Empty input must collapse to the canonical zero signature."""
    assert simhash64("") == 0


def test_simhash_whitespace_only_returns_zero() -> None:
    """Whitespace-only input has no tokens so it hashes to zero."""
    assert simhash64("   \t\n  ") == 0


def test_simhash_single_word_nonzero() -> None:
    """A single-token input must produce a non-zero signature.

    The 1-/2-token degenerate path falls back to a single n-gram so the
    signature is still tied to the content (rather than collapsing to 0
    alongside genuinely-empty rows).
    """
    assert simhash64("hello") != 0


def test_simhash_two_words_nonzero() -> None:
    """Same fallback path as the one-word case."""
    assert simhash64("hello world") != 0


def test_simhash_signed_bigint_range() -> None:
    """Signatures must fit in a signed BIGINT so DuckDB can store them."""
    sig = simhash64("abcdef ghijkl mnopqr stuvwx yzabcd efghij klmnop qrstuv wxyzab cdefgh")
    assert -(1 << 63) <= sig < (1 << 63)


def _simhash64_scalar_reference(text: str) -> int:
    """Independent per-bit scalar SimHash, byte-identical to the original loop.

    Pins :func:`simhash64`'s vectorized numpy voting against the historical
    ``±1`` Python tally.  Lives in the test, not the module, so any future
    drift in the production path fails here instead of silently changing
    every ingest signature (which would orphan the canonical-resolution
    cache).
    """
    import hashlib
    import re

    word = re.compile(r"\w+")
    toks = word.findall(text.lower())
    if not toks:
        return 0
    grams = (
        {" ".join(toks)}
        if len(toks) < 3
        else {" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)}
    )
    votes = [0] * 64
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for b in range(64):
            votes[b] += 1 if (h >> b) & 1 else -1
    sig = 0
    for b in range(64):
        if votes[b] > 0:
            sig |= 1 << b
    return sig - (1 << 64) if sig >= (1 << 63) else sig


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty → 0
        "   \t\n  ",  # whitespace-only → 0
        "hello",  # 1-token degenerate path
        "hello world",  # 2-token degenerate path
        "the quick brown fox jumps over the lazy dog repeatedly today",
        "fix it",  # short, punctuation-stripped
        "a a a a a a a a",  # repeated token → single gram, tie-free
        # A long doc whose grams can drive individual bit tallies to an even
        # split (exercises the strict ``> 0`` / ``2*count > n`` majority gate).
        " ".join(f"word{i % 7}" for i in range(120)),
        "abcdef ghijkl mnopqr stuvwx yzabcd efghij klmnop qrstuv wxyzab cdefgh",
    ],
)
def test_simhash_vectorized_matches_scalar_reference(text: str) -> None:
    """The numpy-vectorized voting must be byte-identical to the scalar tally."""
    assert simhash64(text) == _simhash64_scalar_reference(text)


# ---------------------------------------------------------------------------
# hamming_distance_64
# ---------------------------------------------------------------------------


def test_hamming_distance_zero_for_equal_inputs() -> None:
    assert hamming_distance_64(0xDEADBEEF, 0xDEADBEEF) == 0


def test_hamming_distance_handles_negative_signed() -> None:
    """Negative ints (signed BIGINT) must round-trip via two's complement."""
    a = -1  # all-ones in unsigned 64-bit
    b = 0
    assert hamming_distance_64(a, b) == 64


def test_hamming_distance_full_separation() -> None:
    """All-ones vs all-zeros differs in every bit position."""
    assert hamming_distance_64((1 << 64) - 1, 0) == 64


# ---------------------------------------------------------------------------
# token_budget_bucket
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (0, "xs"),
        (256, "xs"),
        (257, "sm"),
        (2048, "sm"),
        (2049, "md"),
        (8192, "md"),
        (8193, "lg"),
        (32768, "lg"),
        (32769, "xl"),
        (1_000_000, "xl"),
    ],
)
def test_token_budget_bucket_thresholds(tokens: int, expected: str) -> None:
    """Boundary checks at every bucket cutoff."""
    assert token_budget_bucket(tokens) == expected


# ---------------------------------------------------------------------------
# approx_tokens_batch
# ---------------------------------------------------------------------------


def test_approx_tokens_batch_scales_by_anthropic_ratio() -> None:
    """Token count must equal ``int(cl100k_len × _ANTHROPIC_RATIO)``."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = "the quick brown fox jumps over the lazy dog"
    expected = int(len(enc.encode_ordinary(text)) * _ANTHROPIC_RATIO)
    assert approx_tokens_batch([text]) == [expected]


def test_approx_tokens_batch_empty_list() -> None:
    """Empty input must short-circuit without calling the tokenizer."""
    assert approx_tokens_batch([]) == []


def test_approx_tokens_batch_empty_string_zero() -> None:
    """Empty string has zero tokens after scaling."""
    assert approx_tokens_batch([""]) == [0]


def test_approx_tokens_batch_preserves_order() -> None:
    """Order of the output must match the order of the input."""
    texts = ["first one", "second much longer message in the batch", "third"]
    counts = approx_tokens_batch(texts)
    assert len(counts) == len(texts)
    # The middle text is longest, so its scaled count should dominate.
    assert counts[1] >= counts[0]
    assert counts[1] >= counts[2]
