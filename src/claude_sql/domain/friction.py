"""Pure regex fast-path for user-friction classification.

MIGRATION Phase C (T-3-4): the friction pipeline's regex fast-path is a pure
function over a message string — stdlib ``re`` only, no DuckDB, no Bedrock, no
``Settings`` — so it belongs in the domain hexagon. It catches the unambiguous
friction shapes (status pings, hard interruption keywords, explicit
corrections) so those never pay Bedrock; everything ambiguous falls through to
the SQL-stamp layer and then the LLM (both of which stay in the
``application.use_cases.friction`` orchestration).

The label semantics are **frozen**: ``status_ping`` / ``interruption`` /
``correction`` at a flat 0.9 confidence. Ambiguous phrasings deliberately fall
through so a single mis-tuned pattern can't poison the corpus.

Consumed by ``application.use_cases.friction`` and imported directly by the
test suite as ``regex_fast_path`` / ``_REGEX_BANK`` (the pre-hexagonal
``analytics.friction_worker`` shim is gone).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regex fast-path
# ---------------------------------------------------------------------------
#
# These patterns catch the unambiguous cases so we don't pay Bedrock for them.
# Everything ambiguous falls through to the LLM, which is where the
# "screenshot?" / "tests?" class lives (those need semantic context to
# distinguish from genuine topic questions like "can you write tests?").


_REGEX_BANK: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Status pings — only multi-word phrasings that unambiguously ask about
    # progress.  Anything shorter or even slightly ambiguous ("status?",
    # "what's the status column called?") falls through to the LLM so it
    # can disambiguate via context.  The trailing-context guards below
    # require a question mark, end-of-string, or a progress-related word
    # to avoid matching "what's the status column called".
    (
        "status_ping",
        re.compile(
            r"""
            \bhow(?:'s|\s+is|\s+are|\s+it)?\s+(?:it|we|things|progress)\s+
                (?:going|coming|doing|looking|progressing|holding\s+up)\b
            | \bhow'?s?\s+progress\b(?=\s*[?.!]?\s*$)
            | \bany\s+update(?:s)?\b(?=\s*[?.!]?\s*$)
            | \bstatus\s+update\b
            | \bwhere\s+(?:are\s+we|we['\u2019]re)\s+(?:at|with)\b
            | \b(?:are\s+you\s+)?still\s+(?:working|going|running|on\s+it)\b
            | \bwhat'?s?\s+(?:the|your)\s+eta\b
            | \bhow\s+(?:much\s+)?long\s+(?:until|till|more|left)\b
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # Hard interruption keywords at the start of a message.
    (
        "interruption",
        re.compile(
            r"""
            ^\s*
            (?:
                wait(?:\s*[.,!]|\s+a\s+(?:sec|second|moment|minute)|[\s$])
              | stop(?:\s*[.,!]|\s+(?:right\s+)?there|[\s$])
              | hold\s+on
              | hold\s+up
              | hang\s+on
              | actually[,\s]
              | before\s+you\s+(?:do|go)
              | pause\b
              | nvm\b
              | never\s*mind\b
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # Explicit corrections.
    (
        "correction",
        re.compile(
            r"""
            ^\s*
            (?:
                no[,\s.!]
              | nope[,\s.!]?
              | nah[,\s.!]?
              | that'?s\s+(?:wrong|not\s+(?:right|it|what)|incorrect)
              | not\s+(?:that|what\s+i)
              | try\s+again
              | wrong\b
              | that'?s\s+not\s+
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
)


def regex_fast_path(text: str) -> tuple[str, float] | None:
    """Return ``(label, confidence)`` for a regex hit or ``None``.

    Confidence is a flat 0.9 for regex hits -- these are hand-picked,
    unambiguous phrasings.  Ambiguous shapes deliberately fall through to
    the LLM so a single misclassification in the bank does not poison the
    corpus.
    """
    if not text:
        return None
    probe = text[:512]
    for label, pat in _REGEX_BANK:
        if pat.search(probe):
            return label, 0.9
    return None


__all__ = ["_REGEX_BANK", "regex_fast_path"]
