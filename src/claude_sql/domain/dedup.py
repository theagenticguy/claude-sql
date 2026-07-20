"""Pure dedup math: SimHash signatures, Hamming distance, token budgeting.

The dependency-free half of the ingest pipeline. Holds the hand-rolled SimHash
(``simhash64``), the 64-bit Hamming distance, the token-budget bucketer, and the
cl100k → Anthropic-scale token estimator. Nothing here touches DuckDB, polars,
or any adapter — the DuckDB stamp/resolve orchestration lives in the use-case
(``application.use_cases.ingest``), which imports these functions.

Compute-dep note
----------------
This module is "pure" in the domain sense (no adapters, no IO), but it *does*
lean on two compute libraries under the v2 structure relaxation:

* ``numpy`` — vectorizes the SimHash bit-vote (two reductions instead of a
  64-iteration Python loop); byte-identical to the scalar tally.
* ``tiktoken`` — the cl100k tokenizer used by ``approx_tokens_batch``. Kept in
  the domain (rather than pushed behind a port) because it is a deterministic
  local compute dep, not an external service.

Why hand-rolled simhash?
------------------------
The ``simhash`` PyPI package is stale, has no Python 3.13 wheels, and pulls in
a Cython build.  ``datasketch`` ships MinHash / HyperLogLog but no SimHash.
Hashing 3-grams with ``blake2b`` is ~30 LOC and stays inside the standard
library, so we avoid both pinpoints.

Tokens
------
``tiktoken`` ships ``cl100k_base`` (the OpenAI tokenizer used by
``gpt-3.5/4``).  Anthropic's actual billing tokenizer is closed source,
but the empirically-observed scaling factor between cl100k and the
Anthropic tokenizer for the prompts we send is ~0.78×.  See
``.erpaval/solutions/api-patterns/anthropic-prompt-cache-tokenizer-gap.md``
for the measurement methodology.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import tiktoken

#: cl100k_base encoding — same tokenizer OpenAI ships for ``gpt-3.5/4``.  We
#: don't have access to Anthropic's tokenizer, so this is the best public
#: stand-in.  Loaded once at module import; ``encode_ordinary_batch`` is
#: thread-safe.
_ENC = tiktoken.get_encoding("cl100k_base")

#: Word boundary used by the simhash 3-gram tokenizer.  Matches Python's
#: ``\w+`` (alphanumerics + underscore) so emoji / punctuation are dropped
#: before hashing — text like "fix it!" and "fix it" hash to the same set.
_WORD = re.compile(r"\w+")

#: Empirical scaling factor: cl100k tokens × this ≈ Anthropic billed tokens.
#: See ``.erpaval/solutions/api-patterns/anthropic-prompt-cache-tokenizer-gap.md``
#: for the measurement methodology.  Recomputed against fresh cache
#: receipts twice per minor release.
_ANTHROPIC_RATIO = 0.78

#: Mask of the low 64 bits — used to coerce a hashed integer into the
#: unsigned 64-bit range before splitting into per-bit votes.
_U64_MASK = (1 << 64) - 1

#: Per-bit position vector ``[0, 1, …, 63]`` and the matching powers of two,
#: precomputed once so :func:`simhash64` can vote across all 64 bits with two
#: vectorized numpy reductions instead of two Python ``for b in range(64)``
#: loops.  ``uint64`` throughout: the OR of distinct powers tops out at
#: ``2**64 - 1`` which is exactly representable, so the reduction is exact
#: (no float rounding, byte-identical to the scalar tally).
_BIT_POSITIONS = np.arange(64, dtype=np.uint64)
_BIT_POWERS = np.uint64(1) << _BIT_POSITIONS

#: Sentinel signed-BIGINT value for empty / hash-degenerate input.  Stored
#: alongside the real signatures so DuckDB doesn't need to special-case
#: NULL on join paths.
_EMPTY_SIMHASH = 0

#: Hamming-distance threshold for "the same message, copy-pasted".  Three
#: bit-flips out of 64 corresponds to ~5% surface edits on the source text;
#: empirically tuned on the live corpus to flag genuine retries (the user
#: re-pasting the same prompt) without lumping together independent
#: messages that happen to share a phrase.
NEAR_DUP_HAMMING_THRESHOLD = 3


def approx_tokens_batch(texts: list[str], num_threads: int | None = None) -> list[int]:
    """Return a per-text approximate token count (Anthropic-billing scale).

    Encodes every input with cl100k via ``encode_ordinary_batch`` (which
    runs in a Rust thread pool when ``num_threads > 1``) and multiplies
    each length by :data:`_ANTHROPIC_RATIO` so the integer matches
    Anthropic billing within ~5% on the prompts we send.

    Parameters
    ----------
    texts
        Inputs to tokenize.  Empty strings are tolerated (token count 0).
    num_threads
        Tokenizer thread count.  Defaults to ``os.cpu_count() - 1`` (leaves
        a core for the Python caller).  Pass 1 to disable parallelism, e.g.
        on a host where the GIL-free tokenizer threads compete with another
        worker.

    Returns
    -------
    list of int
        One count per input, in the same order.
    """
    if not texts:
        return []
    threads = num_threads if num_threads is not None else max(1, (os.cpu_count() or 1) - 1)
    encoded = _ENC.encode_ordinary_batch(texts, num_threads=threads)
    return [int(len(toks) * _ANTHROPIC_RATIO) for toks in encoded]


def simhash64(text: str) -> int:
    """Return a signed 64-bit SimHash signature over ``text``.

    Tokenization
    ------------
    Lower-case, split on ``\\w+``, take the set of word 3-grams (the unit
    that gives stable signatures across small edits while still discriminating
    between unrelated documents).  Single-token inputs collapse to a single
    1-gram so ``len(text) < 3 words`` doesn't return zero unconditionally.

    Hashing
    -------
    ``blake2b(digest_size=8)`` over each gram.  blake2b is a few times
    faster than SHA-256 on a modern x86, and 8 bytes is exactly the 64
    bits we vote over.

    Voting
    ------
    Standard SimHash: each bit position holds a +1 / -1 tally across grams,
    final sig has bit ``b`` set iff tally[b] > 0.

    Returns
    -------
    int
        Signed 64-bit signature in the range ``[-2**63, 2**63)``.  Empty /
        hash-degenerate inputs return :data:`_EMPTY_SIMHASH` (0).
    """
    toks = _WORD.findall(text.lower())
    if not toks:
        return _EMPTY_SIMHASH
    if len(toks) < 3:
        # Degenerate: 1- or 2-token input.  Fall back to a single n-gram
        # over whatever we have so the signature is still non-zero and
        # unique to the input rather than collapsing into the empty
        # bucket alongside genuinely empty rows.
        grams = {" ".join(toks)}
    else:
        grams = {" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)}
    # Vectorized SimHash voting.  blake2b each gram into a uint64, then vote
    # over all 64 bit positions at once: bit ``b`` of the signature is set iff
    # a majority of grams have bit ``b`` set, i.e. ``2 * set_count > n_grams``
    # (equivalent to the scalar ``±1`` tally being ``> 0``).  ``set_count`` is
    # exact in uint64 and the final signature is the OR of distinct powers of
    # two, so the result is byte-identical to the per-bit Python loop.
    digests = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
            for g in grams
        ),
        dtype=np.uint64,
        count=len(grams),
    )
    set_counts = ((digests[:, None] >> _BIT_POSITIONS) & np.uint64(1)).sum(axis=0, dtype=np.uint64)
    majority = (set_counts.astype(np.int64) * 2) > len(grams)
    sig_unsigned = int(_BIT_POWERS[majority].sum(dtype=np.uint64))
    # DuckDB BIGINT is signed; coerce so the Python int round-trips through
    # parquet → DuckDB without losing high-bit values.
    return sig_unsigned - (1 << 64) if sig_unsigned >= (1 << 63) else sig_unsigned


def hamming_distance_64(a: int, b: int) -> int:
    """Return the Hamming distance between two 64-bit integers.

    Treats both inputs as unsigned 64-bit values via :data:`_U64_MASK` so a
    Python ``int`` carrying a sign bit (the negative half of the signed
    BIGINT range that :func:`simhash64` emits) round-trips correctly.
    """
    a &= _U64_MASK
    b &= _U64_MASK
    return bin(a ^ b).count("1")


def token_budget_bucket(approx_tokens: int) -> str:
    """Bucket a token count into ``xs / sm / md / lg / xl``.

    Boundaries
    ----------
    ====   ================
    xs     ≤ 256 tokens
    sm     ≤ 2048 tokens
    md     ≤ 8192 tokens
    lg     ≤ 32768 tokens
    xl     > 32768 tokens
    ====   ================

    Used downstream to pick chunk sizes / cache strategies without
    re-deriving the histogram every time.
    """
    if approx_tokens <= 256:
        return "xs"
    if approx_tokens <= 2048:
        return "sm"
    if approx_tokens <= 8192:
        return "md"
    if approx_tokens <= 32768:
        return "lg"
    return "xl"


__all__ = [
    "NEAR_DUP_HAMMING_THRESHOLD",
    "approx_tokens_batch",
    "hamming_distance_64",
    "simhash64",
    "token_budget_bucket",
]
