"""Agent-friendly output formatting and error handling for claude-sql.

Every CLI subcommand emits results through :func:`emit_dataframe` (for tabular
output) or :func:`emit_json` (for structured non-tabular output).  The caller
picks a :class:`OutputFormat`; when it is :data:`OutputFormat.AUTO` the
formatter picks ``TABLE`` on a TTY and ``JSON`` otherwise so pipes and agent
subprocesses get machine-readable output without a flag.

Errors from DuckDB get classified and mapped to stable exit codes so agents
can distinguish *parse failed* from *unknown view* from *runtime error*
without pattern-matching tracebacks.  See :data:`EXIT_CODES`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import duckdb
import polars as pl


class OutputFormat(StrEnum):
    """Supported output formats.

    ``AUTO`` resolves to ``TABLE`` when stdout is a TTY and ``JSON`` otherwise.
    Keeping it a string Enum lets cyclopts parse ``--format json`` without any
    custom converter.
    """

    AUTO = "auto"
    TABLE = "table"
    JSON = "json"
    NDJSON = "ndjson"
    CSV = "csv"


# Exit codes that agents can rely on.  Keep them stable -- wire protocols
# always rot fastest at the boundary.
EXIT_CODES: dict[str, int] = {
    "ok": 0,
    "no_embeddings": 2,
    "parse_error": 64,  # malformed SQL
    "catalog_error": 65,  # unknown view/macro/column
    "runtime_error": 70,  # everything else from duckdb.Error
    "duckdb_missing": 127,  # system `duckdb` binary not on PATH
}


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
    # unreachable if OutputFormat stays closed-set
    raise ValueError(f"Unsupported format: {resolved}")


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


@dataclass(frozen=True, slots=True)
class ClassifiedError:
    """The structured shape of a CLI error after classification."""

    kind: str  # "parse_error" | "catalog_error" | "runtime_error"
    exit_code: int
    message: str
    hint: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "kind": self.kind,
                "message": self.message,
                "hint": self.hint,
            }
        }


def classify_duckdb_error(exc: duckdb.Error) -> ClassifiedError:
    """Classify a ``duckdb.Error`` into one of our stable kinds + exit codes.

    DuckDB exposes :class:`duckdb.ParserException` and
    :class:`duckdb.CatalogException` at import time.  Everything else that
    inherits from :class:`duckdb.Error` is treated as a runtime error.
    """
    message = str(exc)
    if isinstance(exc, duckdb.ParserException):
        return ClassifiedError(
            kind="parse_error",
            exit_code=EXIT_CODES["parse_error"],
            message=message,
            hint="check SQL syntax; try `claude-sql schema --format json` for view/macro names",
        )
    if isinstance(exc, duckdb.CatalogException):
        return ClassifiedError(
            kind="catalog_error",
            exit_code=EXIT_CODES["catalog_error"],
            message=message,
            hint="unknown view or column; run `claude-sql schema --format json` for the catalog",
        )
    return ClassifiedError(
        kind="runtime_error",
        exit_code=EXIT_CODES["runtime_error"],
        message=message,
        hint=None,
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
    except duckdb.Error as exc:
        err = classify_duckdb_error(exc)
        emit_error(err, fmt)
        sys.exit(err.exit_code)
