"""Resolve the ``CLAUDE_SQL_HOME`` parent directory for derived caches.

Per RFC 0002 §5.1 and the matching backlog item, every analytics cache
written by claude-sql (LanceDB embeddings, parquet shards, the SQLite
checkpointer, the DuckDB spill dir, profiling JSONs) belongs under a
dedicated parent directory rather than mixed in with Claude Code's own
``~/.claude/`` state. This module owns the resolution rules so every
default-factory in :mod:`claude_sql.infrastructure.settings` agrees on
the answer.

Resolution order (first hit wins):

1. ``$CLAUDE_SQL_HOME`` if set — explicit override always wins.
2. ``${XDG_DATA_HOME}/claude-sql/`` on Linux when ``XDG_DATA_HOME`` is
   set (XDG Base Directory spec).
3. ``~/Library/Application Support/claude-sql/`` on macOS
   (``sys.platform == "darwin"``).
4. ``~/.claude-sql/`` as the universal fallback.

The resolved path is created with ``mkdir(parents=True, exist_ok=True)``
so callers never have to check existence before writing into it.

Corpus scoping
--------------

Analytics caches are per-corpus, not process-wide: switching the INPUT
corpus (``CLAUDE_CONFIG_DIR`` or ``team_corpus_root``) must never read
another corpus's derived parquets. :func:`corpus_slug` maps one corpus
root to a stable, human-legible key; the ``Settings`` default factories
place every cache under ``claude_sql_home()/corpora/<corpus_key>/``.
The historical interactive corpus (``~/.claude``) maps to the reserved
key ``"default"`` so existing users' caches stay addressable after the
one-time move from the home root (see
``duckdb_connection.maybe_migrate_legacy_caches``).
"""

from __future__ import annotations

import hashlib
import os
import re
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

#: Reserved corpus key for the historical interactive corpus (``~/.claude``).
#: Pre-corpus-scoping caches at the home root are attributed to this key by
#: the one-time relocation in ``duckdb_connection.maybe_migrate_legacy_caches``
#: so existing users' caches remain addressable without a rebuild.
DEFAULT_CORPUS_KEY = "default"

#: Per-corpus caches that live under ``corpora/<corpus_key>/``. Everything
#: here is *derived from one transcript corpus* (analytics parquets, the
#: LanceDB store, the checkpointer + its WAL sidecars and one-time-migration
#: sentinels, the DuckDB spill dir). ``profiling/`` is deliberately absent:
#: query-profiling JSONs describe queries, not a corpus, and stay at the
#: home root.
_CORPUS_CACHE_NAMES: tuple[str, ...] = (
    "embeddings_lance",
    "embeddings",
    "message_trajectory",
    "session_classifications",
    "session_conflicts",
    "user_friction",
    "ingest_stamps",
    "clusters.parquet",
    "cluster_terms.parquet",
    "session_communities.parquet",
    "community_profile.parquet",
    "skills_catalog.parquet",
    "state.db",
    "state.db-wal",
    "state.db-shm",
    ".migrated_from_duckdb",
    "claude_sql.duckdb",
    "duckdb_tmp",
)

_SLUG_SANITIZE_RE = re.compile(r"[^a-z0-9]+")
_SLUG_MAX_NAME_LEN = 32


def corpus_slug(corpus_root: Path | str) -> str:
    """Map one corpus root to a stable, human-legible cache-directory key.

    Pure: no I/O beyond path resolution. The historical interactive corpus
    (``~/.claude``, however it is spelled — via ``CLAUDE_CONFIG_DIR`` or by
    default) maps to the reserved key :data:`DEFAULT_CORPUS_KEY` so existing
    users' caches stay addressable. Every other root maps to
    ``<sanitized-dirname>-<8-hex sha256 of the resolved path>`` — legible
    enough to eyeball in ``ls``, hashed enough that two roots sharing a
    dirname (``…/alice/.claude`` vs ``…/bob/.claude``) never collide.

    The root is ``expanduser().resolve()``-normalized first so symlinked
    spellings of the same corpus agree on one key.
    """
    resolved = Path(corpus_root).expanduser().resolve()
    if resolved == Path("~/.claude").expanduser().resolve():
        return DEFAULT_CORPUS_KEY
    name = _SLUG_SANITIZE_RE.sub("-", resolved.name.lower()).strip("-")[:_SLUG_MAX_NAME_LEN]
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{name}-{digest}" if name else digest


def corpus_caches_at_home_root(home: Path | None = None) -> dict[str, Path]:
    """Return ``{name: path}`` for per-corpus caches still at the home ROOT.

    These are caches written by pre-corpus-scoping versions, which pointed
    every corpus at one process-wide location directly under
    :func:`claude_sql_home`. They are the manifest for the one-time
    relocation into ``corpora/default/`` (see
    ``duckdb_connection.maybe_migrate_legacy_caches``). Mirrors
    :func:`recognized_legacy_caches`: only entries that exist are returned,
    so an empty dict means "nothing to relocate".
    """
    root = home if home is not None else claude_sql_home()
    if not root.exists() or not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for name in _CORPUS_CACHE_NAMES:
        candidate = root / name
        if candidate.exists():
            found[name] = candidate
    return found


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
