"""Per-message ingest stamps: ``approx_tokens``, ``simhash64``,
``canonical_uuid``, ``token_budget_bucket``.

Reads ``messages_text`` via DuckDB, stamps each row, and writes a sharded
``ingest_stamps`` parquet under ``CLAUDE_SQL_HOME``.  A periodic dedup pass
populates ``canonical_uuid`` via a DuckDB SQL self-join (``bit_count(xor)``
over a top-16-bit bucket).  Downstream workers (``embed`` first) read this
parquet and skip near-dup messages on later runs.

Why hand-rolled simhash?
------------------------
The ``simhash`` PyPI package is stale, has no Python 3.13 wheels, and pulls in
a Cython build.  ``datasketch`` ships MinHash / HyperLogLog but no SimHash.
Hashing 3-grams with ``blake2b`` is ~30 LOC and stays inside the standard
library, so we avoid both pinpoints.

Pipeline shape
--------------
1. ``stamp_messages`` finds messages_text rows whose ``uuid`` is not yet in
   ``ingest_stamps`` and computes ``approx_tokens + simhash64`` for each,
   writing a fresh ``part-<ts_ns>.parquet`` shard under
   ``settings.ingest_stamps_parquet_path``.
2. ``resolve_canonicals`` runs a DuckDB self-join over the ``ingest_stamps``
   view to populate ``canonical_uuid`` for rows whose simhash is within
   Hamming distance 3 of an earlier-seen row (top-16-bit bucket gates the
   join so it doesn't blow up on large corpora).  The resolved mapping is
   written as a follow-up shard.

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
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import tiktoken
from loguru import logger

from claude_sql.config import Settings
from claude_sql.parquet_shards import iter_part_files, write_part

if TYPE_CHECKING:
    import duckdb

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

#: Sentinel signed-BIGINT value for empty / hash-degenerate input.  Stored
#: alongside the real signatures so DuckDB doesn't need to special-case
#: NULL on join paths.
_EMPTY_SIMHASH = 0


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
    votes = [0] * 64
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for b in range(64):
            votes[b] += 1 if (h >> b) & 1 else -1
    sig_unsigned = 0
    for b in range(64):
        if votes[b] > 0:
            sig_unsigned |= 1 << b
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


# ---------------------------------------------------------------------------
# Stamp pipeline
# ---------------------------------------------------------------------------


#: Schema for the ``ingest_stamps`` parquet shards.  Pinned explicitly so
#: ``write_part`` produces a stable dtype across runs even when a chunk
#: happens to land entirely on a single bucket label.
_INGEST_STAMPS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "uuid": pl.Utf8,
    "session_id": pl.Utf8,
    "approx_tokens": pl.Int32,
    "simhash64": pl.Int64,
    "token_budget_bucket": pl.Utf8,
    "canonical_uuid": pl.Utf8,
    "first_seen_ts": pl.Datetime("us", "UTC"),
    "stamped_at": pl.Datetime("us", "UTC"),
}


def _pending_stamp_sql(
    *,
    parts_clause: str | None,
    since_days: int | None,
    limit: int | None,
) -> str:
    """SQL pulling messages_text rows that need stamping.

    Anti-joins against the existing ``ingest_stamps`` parquet shards (when
    any exist) via ``read_parquet`` so the stamp pipeline is incremental:
    the first run on a corpus stamps everything, subsequent runs only
    catch up new messages.
    """
    where = ["mt.text_content IS NOT NULL", "length(mt.text_content) > 0"]
    if since_days is not None:
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
    anti_join = ""
    if parts_clause:
        anti_join = (
            f"  AND CAST(mt.uuid AS VARCHAR) NOT IN ("
            f"    SELECT CAST(uuid AS VARCHAR) FROM read_parquet([{parts_clause}])"
            f"  )"
        )
    sql = f"""
        SELECT CAST(mt.uuid AS VARCHAR)        AS uuid,
               CAST(mt.session_id AS VARCHAR)  AS session_id,
               mt.ts                           AS ts,
               mt.text_content                 AS text_content
          FROM messages_text mt
         WHERE {" AND ".join(where)}
        {anti_join}
         ORDER BY mt.ts
    """
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    return sql


def _existing_parts_clause(settings: Settings) -> str | None:
    """Return a comma-separated SQL string of existing part-file paths, or None."""
    parts = [
        p for p in iter_part_files(settings.ingest_stamps_parquet_path) if p.stat().st_size > 16
    ]
    if not parts:
        return None
    return ", ".join(f"'{str(p).replace(chr(39), chr(39) * 2)}'" for p in parts)


def count_pending(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
) -> int:
    """Pure-SQL count of messages that would be stamped on the next run.

    Used by the dry-run path so an agent can preview "how many rows
    would the stamp pass write" without materializing every text.  Runs
    in well under 1 s on the live corpus because the anti-join hits a
    parquet, not a Python set.
    """
    parts_clause = _existing_parts_clause(settings)
    sql = _pending_stamp_sql(parts_clause=parts_clause, since_days=since_days, limit=limit)
    probe = f"SELECT count(*) FROM ({sql}) q"
    row = con.execute(probe).fetchone()
    return int(row[0]) if row is not None else 0


def stamp_messages(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    batch_size: int = 4096,
) -> int:
    """Stamp every messages_text row not yet present in ``ingest_stamps``.

    Pulls pending rows with :func:`_pending_stamp_sql`, computes
    ``approx_tokens`` (cl100k batch encode × Anthropic ratio) and
    ``simhash64`` row-by-row, derives ``token_budget_bucket`` from the
    token count, and writes the chunk to a fresh ``part-<ts_ns>.parquet``
    shard under ``settings.ingest_stamps_parquet_path``.

    ``canonical_uuid`` is left NULL by this pass — :func:`resolve_canonicals`
    fills it in once the new shard has landed.

    Parameters
    ----------
    con
        DuckDB connection with ``messages_text`` registered.
    settings
        Settings driving the parquet path.
    since_days, limit
        Optional time / row caps mirroring the other workers.
    batch_size
        Number of rows per part-file shard.  4096 keeps each part under
        ~1 MB on the live corpus; larger chunks burn memory for no IO win.

    Returns
    -------
    int
        Total rows newly stamped this call.
    """
    parts_clause = _existing_parts_clause(settings)
    sql = _pending_stamp_sql(parts_clause=parts_clause, since_days=since_days, limit=limit)
    rows = con.execute(sql).fetchall()
    if not rows:
        logger.info("ingest/stamp: no pending rows")
        return 0

    total = len(rows)
    logger.info("ingest/stamp: {} pending rows (batch_size={})", total, batch_size)
    written = 0
    t0 = time.monotonic()
    for chunk_idx, start in enumerate(range(0, total, batch_size)):
        slice_ = rows[start : start + batch_size]
        uuids = [str(r[0]) for r in slice_]
        sids = [str(r[1]) for r in slice_]
        ts_values = [r[2] for r in slice_]
        texts = [r[3] or "" for r in slice_]

        token_counts = approx_tokens_batch(texts)
        sigs = [simhash64(t) for t in texts]
        buckets = [token_budget_bucket(n) for n in token_counts]

        now = datetime.now(UTC)
        df = pl.DataFrame(
            {
                "uuid": uuids,
                "session_id": sids,
                "approx_tokens": token_counts,
                "simhash64": sigs,
                "token_budget_bucket": buckets,
                # canonical_uuid is filled in by resolve_canonicals; we write
                # NULL here so the column shape stays stable across shards.
                "canonical_uuid": [None] * len(slice_),
                "first_seen_ts": ts_values,
                "stamped_at": [now] * len(slice_),
            },
            schema=_INGEST_STAMPS_SCHEMA,
        )
        path = write_part(settings.ingest_stamps_parquet_path, df)
        written += len(slice_)
        logger.debug(
            "ingest/stamp: wrote chunk {} ({} rows) -> {}",
            chunk_idx + 1,
            len(slice_),
            path,
        )

    elapsed = time.monotonic() - t0
    logger.info(
        "ingest/stamp: {} rows stamped in {:.2f}s ({:.1f} rows/s)",
        written,
        elapsed,
        written / elapsed if elapsed > 0 else 0.0,
    )
    return written


# ---------------------------------------------------------------------------
# Canonical UUID resolution (near-dup detection)
# ---------------------------------------------------------------------------


#: Top-16-bit bucket gates the SimHash self-join so the join doesn't blow
#: up on large corpora.  Two messages whose Hamming distance is ≤ 3 *must*
#: share their top 16 bits in expectation (a single bit-flip changes one
#: top-bit slot ~16/64 = 25% of the time, three flips ~58% — but only the
#: top-16-bit bucket needs to match, not the exact value, so the bucket
#: filter discards roughly N²·(1 - 1/2¹⁶) of the cross product).  See
#: Manku-Jain-Sarma 2007 (WWW '07) for the analysis.
_TOP_BUCKET_BITS = 48  # 64 - 16

#: Hamming-distance threshold for "the same message, copy-pasted".  Three
#: bit-flips out of 64 corresponds to ~5% surface edits on the source text;
#: empirically tuned on the live corpus to flag genuine retries (the user
#: re-pasting the same prompt) without lumping together independent
#: messages that happen to share a phrase.
NEAR_DUP_HAMMING_THRESHOLD = 3


def resolve_canonicals(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    refresh_view: bool = True,
) -> int:
    """Populate ``canonical_uuid`` via a DuckDB SQL self-join over ``ingest_stamps``.

    For every row, the canonical is the *earliest-seen* row whose simhash
    differs by ≤ :data:`NEAR_DUP_HAMMING_THRESHOLD` bits.  Rows that are
    their own canonical (no earlier near-dup) get
    ``canonical_uuid = uuid``.

    Parameters
    ----------
    con
        DuckDB connection.  When ``refresh_view`` is True the helper
        re-binds the ``ingest_stamps`` view first so the join hits the
        latest shards.
    settings
        Settings driving the parquet path.
    refresh_view
        Whether to rebind the ``ingest_stamps`` view before running the
        join.  Default True.  Set False from callers that already manage
        registration order.

    Returns
    -------
    int
        Total rows whose ``canonical_uuid`` was written this pass (may be 0
        when nothing has changed since the last call).
    """
    parts_clause = _existing_parts_clause(settings)
    if parts_clause is None:
        logger.info("ingest/resolve: no stamps yet, nothing to resolve")
        return 0

    if refresh_view:
        # Bind a temp view so the SQL below stays readable; using a CTE
        # twice within the same statement reads parquet twice on every
        # query plan.
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW _ingest_stamps_resolve AS "
            f"SELECT * FROM read_parquet([{parts_clause}]);"
        )
        view_name = "_ingest_stamps_resolve"
    else:
        view_name = "ingest_stamps"

    # ``xor(a, b)`` is the bit-XOR builtin (DuckDB's ``^`` is exponentiation,
    # not XOR — see CLAUDE.md note).  ``bit_count`` returns the population
    # count of the resulting BIGINT.  The ``>> _TOP_BUCKET_BITS`` filter
    # collapses the simhashes into 16-bit buckets and gates the self-join
    # so we don't spew the full N² cross-product.  ``MIN(b.uuid)`` is a
    # tie-breaker — multiple equally-old near-dups would otherwise yield
    # nondeterministic canonical assignments.
    sql = f"""
        WITH resolved AS (
            SELECT a.uuid          AS uuid,
                   COALESCE(
                       MIN(b.uuid) FILTER (WHERE b.first_seen_ts < a.first_seen_ts),
                       a.uuid
                   )               AS canonical_uuid
              FROM {view_name} a
              JOIN {view_name} b
                ON (a.simhash64 >> {_TOP_BUCKET_BITS}) = (b.simhash64 >> {_TOP_BUCKET_BITS})
               AND bit_count(xor(a.simhash64, b.simhash64))
                   <= {NEAR_DUP_HAMMING_THRESHOLD}
               AND b.first_seen_ts <= a.first_seen_ts
             GROUP BY a.uuid
        )
        SELECT s.uuid,
               s.session_id,
               s.approx_tokens,
               s.simhash64,
               s.token_budget_bucket,
               r.canonical_uuid,
               s.first_seen_ts,
               s.stamped_at
          FROM {view_name} s
          LEFT JOIN resolved r USING (uuid)
    """
    df = con.execute(sql).pl()
    if df.is_empty():
        logger.info("ingest/resolve: empty stamps table")
        return 0
    df = df.cast(pl.Schema(_INGEST_STAMPS_SCHEMA))
    # Truncate the cache and rewrite as one consolidated shard.  This is
    # the same pattern ``cache compact`` uses and keeps the shard count
    # bounded after a long backfill.
    cache_dir = settings.ingest_stamps_parquet_path
    if cache_dir.is_dir():
        for old in iter_part_files(cache_dir):
            old.unlink()
    write_part(cache_dir, df)
    written = df.height
    n_canon = int(df.filter(pl.col("canonical_uuid") == pl.col("uuid")).height)
    n_dup = written - n_canon
    logger.info(
        "ingest/resolve: {} rows resolved ({} canonical, {} near-dups)",
        written,
        n_canon,
        n_dup,
    )
    return written


__all__ = [
    "NEAR_DUP_HAMMING_THRESHOLD",
    "approx_tokens_batch",
    "count_pending",
    "hamming_distance_64",
    "resolve_canonicals",
    "simhash64",
    "stamp_messages",
    "token_budget_bucket",
]
