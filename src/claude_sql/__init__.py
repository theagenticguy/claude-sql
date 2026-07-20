"""claude-sql — query your Claude Code transcripts with SQL + semantic search.

The one importable convenience re-export is :class:`ClaudeSql`, the composition
facade. It is exposed via PEP 562 module ``__getattr__`` so a bare
``import claude_sql`` stays light — it does NOT import duckdb, polars, or any
adapter until you actually reach for ``claude_sql.ClaudeSql``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_sql.composition import ClaudeSql

__all__ = ["ClaudeSql"]


def __getattr__(name: str) -> Any:
    if name == "ClaudeSql":
        from claude_sql.composition import ClaudeSql

        return ClaudeSql
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
