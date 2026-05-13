"""Resolve the ``CLAUDE_SQL_HOME`` parent directory for derived caches.

Per RFC 0002 §5.1 and the matching backlog item, every analytics cache
written by claude-sql (LanceDB embeddings, parquet shards, the SQLite
checkpointer, the DuckDB spill dir, profiling JSONs) belongs under a
dedicated parent directory rather than mixed in with Claude Code's own
``~/.claude/`` state. This module owns the resolution rules so every
default-factory in :mod:`claude_sql.config` agrees on the answer.

Resolution order (first hit wins):

1. ``$CLAUDE_SQL_HOME`` if set — explicit override always wins.
2. ``${XDG_DATA_HOME}/claude-sql/`` on Linux when ``XDG_DATA_HOME`` is
   set (XDG Base Directory spec).
3. ``~/Library/Application Support/claude-sql/`` on macOS
   (``sys.platform == "darwin"``).
4. ``~/.claude-sql/`` as the universal fallback.

The resolved path is created with ``mkdir(parents=True, exist_ok=True)``
so callers never have to check existence before writing into it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Caches recognized as legitimate claude-sql output that lived under
#: ``~/.claude/`` prior to RFC 0002. The first-connect migration walks
#: this list and moves anything present into the new home. Order
#: doesn't matter — directories and individual files are both supported.
_LEGACY_CACHE_NAMES: tuple[str, ...] = (
    "embeddings_lance",
    "embeddings",
    "message_trajectory",
    "session_classifications",
    "session_conflicts",
    "user_friction",
    "clusters.parquet",
    "cluster_terms.parquet",
    "session_communities.parquet",
    "community_profile.parquet",
    "state.db",
    "duckdb_tmp",
    "profiling",
    "claude_sql.duckdb",
)


def claude_sql_home() -> Path:
    """Return the parent directory for every claude-sql derived cache.

    The directory is created on first call (``mkdir(parents=True,
    exist_ok=True)``); subsequent callers can rely on it existing.

    Resolution order is documented at module level. The function reads
    ``os.environ`` on every call so tests can flip env vars per-test
    via ``monkeypatch.setenv`` and observe the new value without
    needing module reloads.
    """
    explicit = os.environ.get("CLAUDE_SQL_HOME")
    if explicit:
        path = Path(explicit).expanduser()
    elif sys.platform == "darwin":
        path = Path("~/Library/Application Support/claude-sql").expanduser()
    elif xdg := os.environ.get("XDG_DATA_HOME"):
        path = Path(xdg).expanduser() / "claude-sql"
    else:
        path = Path("~/.claude-sql").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def recognized_legacy_caches(legacy_root: Path | None = None) -> dict[str, Path]:
    """Return ``{name: path}`` for every recognized legacy cache that exists.

    ``legacy_root`` defaults to ``~/.claude/`` (the historical claude-sql
    cache root). Pass an explicit path in tests to point at a tmp dir.

    Only entries that *actually exist* on disk are returned — the result
    is the migration manifest the auto-mover walks. Missing names are
    silently dropped so an empty dict means "nothing to migrate".
    """
    root = legacy_root if legacy_root is not None else Path("~/.claude").expanduser()
    if not root.exists() or not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for name in _LEGACY_CACHE_NAMES:
        candidate = root / name
        if candidate.exists():
            found[name] = candidate
    return found
