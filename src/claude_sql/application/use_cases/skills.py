"""Walk local skill + plugin directories and persist a catalog parquet.

Seeds ``~/.claude/skills_catalog.parquet`` from three filesystem sources plus
the built-in slash commands. The filesystem walkers live in the infrastructure
adapter (``infrastructure.skills_fs``); the pure parsing helpers and the
built-in constant live in the domain (``domain.skills``). This module owns the
orchestration: collect rows, de-dup, and write the parquet.

Multiple cached plugin versions collapse to the newest -- we sort the
version-dir names with :mod:`packaging.version` when the string is a
semver, and fall back to ``mtime`` when it isn't (e.g. the
``unknown/`` version directory Claude Code writes for some plugins).

Public API
----------
sync(settings, *, dry_run=False) -> dict[str, int]
    Walk filesystem, write parquet at ``settings.skills_catalog_parquet_path``.
    Returns ``{"rows": N, "skills": S, "commands": C, "builtins": B}``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from claude_sql.infrastructure.settings import Settings
from claude_sql.infrastructure.skills_fs import (
    _builtin_rows,
    _walk_plugins,
    _walk_user_skills,
)

_CATALOG_SCHEMA: dict[str, Any] = {
    "skill_id": pl.Utf8,
    "name": pl.Utf8,
    "plugin": pl.Utf8,
    "plugin_version": pl.Utf8,
    "source_kind": pl.Utf8,
    "description": pl.Utf8,
    "argument_hint": pl.Utf8,
    "source_path": pl.Utf8,
    "synced_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}


def _collect_rows(settings: Settings) -> list[dict[str, Any]]:
    """Produce the full, de-duplicated row set for :func:`sync`."""
    now = datetime.now(tz=UTC)
    rows: list[dict[str, Any]] = []
    rows.extend(_walk_user_skills(settings.user_skills_dir, now))
    rows.extend(_walk_plugins(settings.plugins_cache_dir, now))
    rows.extend(_builtin_rows(now))

    # De-dup on (skill_id, source_kind) keeping the first occurrence.
    # user-skill wins over plugin-skill for the same bare name because we
    # yield user skills first; plugin-skill wins over plugin-command.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (row["skill_id"], row["source_kind"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def sync(settings: Settings, *, dry_run: bool = False) -> dict[str, int]:
    """Walk local skills + plugins and write the catalog parquet.

    Parameters
    ----------
    settings
        Configured :class:`claude_sql.config.Settings`; reads
        :attr:`Settings.user_skills_dir`,
        :attr:`Settings.plugins_cache_dir`, and
        :attr:`Settings.skills_catalog_parquet_path`.
    dry_run
        When ``True`` the walk runs and the row counts are returned, but
        nothing is written to disk.

    Returns
    -------
    dict
        ``{"rows": total, "skills": user+plugin skill rows, "commands":
        plugin-command rows, "builtins": builtin rows}``.

    Notes
    -----
    The parquet is written atomically -- to a sibling ``.tmp`` path then
    renamed -- so a crashed sync never leaves the catalog view staring at
    a truncated file.
    """
    rows = _collect_rows(settings)
    stats = {
        "rows": len(rows),
        "skills": sum(1 for r in rows if r["source_kind"] in ("user-skill", "plugin-skill")),
        "commands": sum(1 for r in rows if r["source_kind"] == "plugin-command"),
        "builtins": sum(1 for r in rows if r["source_kind"] == "builtin"),
    }

    if dry_run:
        logger.info(
            "skills_catalog.sync(dry_run=True): {} rows ({} skills, {} commands, {} builtins)",
            stats["rows"],
            stats["skills"],
            stats["commands"],
            stats["builtins"],
        )
        return stats

    out_path: Path = settings.skills_catalog_parquet_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows, schema=_CATALOG_SCHEMA)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(out_path)
    logger.info(
        "skills_catalog.sync: wrote {} rows to {} ({} skills, {} commands, {} builtins)",
        stats["rows"],
        out_path,
        stats["skills"],
        stats["commands"],
        stats["builtins"],
    )
    return stats


__all__ = ["sync"]
