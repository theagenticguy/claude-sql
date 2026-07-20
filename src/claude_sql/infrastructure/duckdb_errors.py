"""DuckDB error classification adapter.

The one piece of the exit-code taxonomy that needs ``import duckdb``: it maps a
concrete :class:`duckdb.Error` onto the pure :class:`ClassifiedError` domain
type + a stable exit code. Kept out of the domain (which stays duckdb-free) and
out of the interfaces layer (which stays free of the concrete driver's
exception hierarchy) so the wire contract lives at the adapter boundary where
the DuckDB dependency already sits.
"""

from __future__ import annotations

import duckdb

from claude_sql.domain.errors import EXIT_CODES, ClassifiedError


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


__all__ = ["classify_duckdb_error"]
