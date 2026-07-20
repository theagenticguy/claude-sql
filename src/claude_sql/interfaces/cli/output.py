"""Agent-friendly output formatting and error handling for claude-sql.

Every CLI subcommand emits results through :func:`emit_dataframe` (for tabular
output) or :func:`emit_json` (for structured non-tabular output).  The caller
picks a :class:`OutputFormat`; when it is :data:`OutputFormat.AUTO` the
formatter picks ``TABLE`` on a TTY and ``JSON`` otherwise so pipes and agent
subprocesses get machine-readable output without a flag.

Errors get classified and mapped to stable exit codes so agents can
distinguish *parse failed* from *unknown view* from *runtime error* without
pattern-matching tracebacks.  The taxonomy (:data:`EXIT_CODES`,
:class:`ClassifiedError`, :class:`InputValidationError`) lives in
``domain/errors.py``; the concrete ``duckdb.Error`` classifier lives in
``infrastructure/duckdb_errors.py``.  This module owns the rendering and the
``run_or_die`` bridge between them.
"""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from typing import Any

import duckdb
import polars as pl

from claude_sql.domain.errors import EXIT_CODES, ClassifiedError, InputValidationError
from claude_sql.infrastructure.duckdb_errors import classify_duckdb_error


class OutputFormat(StrEnum):
    """Supported output formats for tabular and structured CLI output.

    ``AUTO`` resolves to ``TABLE`` when stdout is a TTY and ``JSON`` otherwise.
    Keeping it a string Enum lets cyclopts parse ``--format json`` without any
    custom converter.

    Markdown rendering is intentionally absent: only ``review-sheet`` emits
    human prose, and it owns its own ``--render`` flag (see
    :class:`claude_sql.cli.RenderFormat`). Pulling markdown into this enum
    advertised the format on every subcommand even though no other command
    knows how to produce it.
    """

    AUTO = "auto"
    TABLE = "table"
    JSON = "json"
    NDJSON = "ndjson"
    CSV = "csv"


def resolve_format(fmt: OutputFormat | str) -> OutputFormat:
    """Resolve ``AUTO`` against the current stdout.  No-op for explicit formats."""
    resolved = OutputFormat(fmt) if isinstance(fmt, str) else fmt
    if resolved is not OutputFormat.AUTO:
        return resolved
    return OutputFormat.TABLE if sys.stdout.isatty() else OutputFormat.JSON


def emit_dataframe(
    df: pl.DataFrame,
    fmt: OutputFormat | str = OutputFormat.AUTO,
    *,
    table_rows: int = 100,
    table_str_len: int = 120,
) -> None:
    """Write a polars DataFrame to stdout in the requested format.

    Parameters
    ----------
    df
        The frame to emit.
    fmt
        One of :class:`OutputFormat`.  ``AUTO`` resolves per :func:`resolve_format`.
    table_rows
        Row cap for the pretty-printed table format only.  JSON / NDJSON / CSV
        always emit every row; rely on SQL ``LIMIT`` to cap upstream.
    table_str_len
        Column-cell string truncation for the table format only.
    """
    resolved = resolve_format(fmt)
    if resolved is OutputFormat.TABLE:
        with pl.Config(tbl_rows=table_rows, tbl_cols=20, fmt_str_lengths=table_str_len):
            print(df)
        return
    if resolved is OutputFormat.JSON:
        # ``write_json`` emits a JSON array of row objects -- the exact shape
        # agents expect for tabular results.
        sys.stdout.write(df.write_json())
        sys.stdout.write("\n")
        return
    if resolved is OutputFormat.NDJSON:
        df.write_ndjson(sys.stdout)
        return
    if resolved is OutputFormat.CSV:
        df.write_csv(sys.stdout)
        return
    # Defensive: unreachable while OutputFormat stays closed-set
    # (auto/table/json/ndjson/csv). Kept as a guard for future enum additions.
    raise ValueError(f"Unsupported format: {resolved}")  # pragma: no cover


def emit_json(payload: Any, fmt: OutputFormat | str = OutputFormat.AUTO) -> None:
    """Write a non-tabular payload as JSON (for schema, list-cache, errors).

    The ``TABLE`` path is handled by the caller (schema has a custom human
    layout); this helper is purely for machine-readable formats.
    """
    resolved = resolve_format(fmt)
    # JSON / NDJSON / CSV all reduce to a JSON document for non-tabular data.
    # CSV over a nested dict is meaningless, so fall through to JSON.
    sys.stdout.write(json.dumps(payload, indent=2, default=str))
    sys.stdout.write("\n")
    del resolved


def validate_glob(pattern: str | None, *, flag: str = "--glob") -> None:
    """Reject glob patterns DuckDB's ``read_json`` cannot accept.

    DuckDB raises ``IO Error: Cannot use multiple '**' in one path`` when a
    glob contains more than one recursive segment. We catch that up front so
    the failure surfaces with a useful hint instead of a raw traceback.

    Pass-through for ``None`` / empty strings: the caller will fall back to
    its default glob.
    """
    if not pattern:
        return
    if pattern.count("**") > 1:
        raise InputValidationError(
            f"{flag} pattern {pattern!r} contains more than one '**' segment; "
            "DuckDB's read_json rejects it.",
            hint=(
                "use at most one '**' recursive wildcard -- e.g. "
                "'/home/you/.claude/projects/**/*.jsonl' or "
                "'/home/you/.claude/projects/<project>/*.jsonl'"
            ),
        )


def emit_error(err: ClassifiedError, fmt: OutputFormat | str = OutputFormat.AUTO) -> None:
    """Write a classified error to stderr in the requested format.

    Agents running with ``--format json`` get the structured payload; humans on
    a TTY get a single readable line with the hint.  Either way the process
    exits with :attr:`ClassifiedError.exit_code`.
    """
    resolved = resolve_format(fmt)
    if resolved is OutputFormat.TABLE:
        prefix = f"[{err.kind}]"
        sys.stderr.write(f"{prefix} {err.message}\n")
        if err.hint:
            sys.stderr.write(f"hint: {err.hint}\n")
    else:
        sys.stderr.write(json.dumps(err.to_payload(), default=str))
        sys.stderr.write("\n")


def run_or_die(
    fn: Any,
    *args: Any,
    fmt: OutputFormat | str = OutputFormat.AUTO,
    **kwargs: Any,
) -> Any:
    """Invoke ``fn(*args, **kwargs)`` and translate DuckDB errors to exit codes.

    Keeps every subcommand's body clean of try/except bookkeeping.  On success
    returns ``fn``'s result; on :class:`duckdb.Error` writes a classified error
    and calls :func:`sys.exit` with the matching code.
    """
    try:
        return fn(*args, **kwargs)
    except InputValidationError as exc:
        err = ClassifiedError(
            kind="invalid_input",
            exit_code=EXIT_CODES["invalid_input"],
            message=str(exc),
            hint=exc.hint,
        )
        emit_error(err, fmt)
        sys.exit(err.exit_code)
    except duckdb.Error as exc:
        err = classify_duckdb_error(exc)
        emit_error(err, fmt)
        sys.exit(err.exit_code)


__all__ = [
    "OutputFormat",
    "emit_dataframe",
    "emit_error",
    "emit_json",
    "resolve_format",
    "run_or_die",
    "validate_glob",
]
