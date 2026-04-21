"""Ungrounded-claim detector v0 — entity spotting over tool outputs.

For each assistant turn, extract *factual claims about internal systems*
(file paths, function names, config flags, CLI subcommands, table names,
env vars, Slack/work IDs) and check whether those entities appear in the
same session's tool-call outputs.  An assertion that names an entity
never seen in tool output is flagged as potentially ungrounded.

This is v0: fast, conservative, a regex+span-graph combo.  A later pass
can add Nova 2 Lite for claim-phrase extraction when the regex misses
semantic claims.

Schema written to parquet::

    session_id       STRING
    turn_idx         INT64
    claim_entity     STRING     # the specific entity name the agent asserted
    claim_kind       STRING     # path | function | flag | env_var | id | cli
    grounded         BOOLEAN    # True if seen in tool output
    tool_output_hits INT64      # count of exact matches in tool results
    freeze_sha       STRING
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Entity extractors
# ---------------------------------------------------------------------------

# Unix-ish paths with at least one '/' and a filename-shaped tail
_PATH_RE = re.compile(r"(?<![A-Za-z])(/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)")
# Python-style dotted function/attr references and bare ident(): handler.run, foo()
_FUNCTION_RE = re.compile(r"\b([a-z_][a-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\(")
# Flags: --flag-name, CLAUDE_SQL_CONCURRENCY, SOME_ENV_VAR (>=2 underscores OR upper + underscore)
_FLAG_RE = re.compile(r"(--[a-z][a-z0-9-]{2,})")
_ENV_VAR_RE = re.compile(r"\b([A-Z][A-Z0-9_]{4,}_[A-Z0-9_]+)\b")
# Work item / thread ts / session UUID patterns from blind_handover
_ID_RE = re.compile(
    r"\b(wi_[0-9a-f]{12}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b"
)
# Specific CLI subcommand style: `claude-sql <verb>`
_CLI_RE = re.compile(r"\bclaude-sql\s+([a-z][a-z0-9-]+)\b")


@dataclass(frozen=True)
class Claim:
    """One extracted claim: entity name + the category it belongs to."""

    entity: str
    kind: str  # "path" | "function" | "flag" | "env_var" | "id" | "cli"


def extract_claims(text: str) -> list[Claim]:
    """Pull factual-entity claims out of assistant text.

    Order matters: more specific patterns run first so a CLI subcommand
    isn't double-counted as a generic function call.
    """
    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()

    def push(entity: str, kind: str) -> None:
        key = (entity, kind)
        if key not in seen:
            seen.add(key)
            claims.append(Claim(entity=entity, kind=kind))

    for m in _CLI_RE.finditer(text):
        push(m.group(1), "cli")
    for m in _PATH_RE.finditer(text):
        push(m.group(1), "path")
    for m in _FUNCTION_RE.finditer(text):
        push(m.group(1), "function")
    for m in _FLAG_RE.finditer(text):
        push(m.group(1), "flag")
    for m in _ENV_VAR_RE.finditer(text):
        push(m.group(1), "env_var")
    for m in _ID_RE.finditer(text):
        push(m.group(1), "id")
    return claims


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------


def count_in(haystack: str, needle: str) -> int:
    """Count non-overlapping substring occurrences; safe for regex-unfriendly needles."""
    if not needle:
        return 0
    return haystack.count(needle)


def check_claims(claims: Iterable[Claim], tool_output_text: str) -> list[dict]:
    """For each claim, count hits in the tool-output text and decide grounded."""
    rows: list[dict] = []
    for claim in claims:
        hits = count_in(tool_output_text, claim.entity)
        rows.append(
            {
                "claim_entity": claim.entity,
                "claim_kind": claim.kind,
                "grounded": hits > 0,
                "tool_output_hits": hits,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Batch over sessions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One assistant turn paired with the tool output visible to it."""

    session_id: str
    turn_idx: int
    assistant_text: str
    tool_output_text: str  # concatenation of ToolResult content from this session


def detect(turns: list[Turn], freeze_sha: str) -> pl.DataFrame:
    """Run the detector over a batch of turns; return a parquet-shaped frame."""
    rows: list[dict] = []
    for t in turns:
        claims = extract_claims(t.assistant_text)
        checked = check_claims(claims, t.tool_output_text)
        for row in checked:
            rows.append(
                {
                    "session_id": t.session_id,
                    "turn_idx": t.turn_idx,
                    "claim_entity": row["claim_entity"],
                    "claim_kind": row["claim_kind"],
                    "grounded": row["grounded"],
                    "tool_output_hits": row["tool_output_hits"],
                    "freeze_sha": freeze_sha,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "session_id": pl.String,
                "turn_idx": pl.Int64,
                "claim_entity": pl.String,
                "claim_kind": pl.String,
                "grounded": pl.Boolean,
                "tool_output_hits": pl.Int64,
                "freeze_sha": pl.String,
            }
        )
    return pl.DataFrame(rows)


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    """Per-session rollup: ungrounded-claim count + rate."""
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "session_id": pl.String,
                "n_claims": pl.Int64,
                "n_ungrounded": pl.Int64,
                "ungrounded_rate": pl.Float64,
            }
        )
    return (
        df.group_by("session_id")
        .agg(
            pl.len().alias("n_claims"),
            (~pl.col("grounded")).sum().alias("n_ungrounded"),
            (1.0 - pl.col("grounded").cast(pl.Float64).mean()).alias("ungrounded_rate"),
        )
        .sort("ungrounded_rate", descending=True)
    )


def to_parquet(df: pl.DataFrame, path: Path) -> None:
    df.write_parquet(path)
