"""Walk local skill + plugin directories and persist a catalog parquet.

Seeds ``~/.claude/skills_catalog.parquet`` from three filesystem sources:

1. ``~/.claude/skills/<name>/SKILL.md`` -- user-installed skills, keyed by
   bare name only.
2. ``~/.claude/plugins/cache/<owner>/<plugin>/<version>/skills/<name>/SKILL.md``
   -- plugin-provided skills, emitted as *both* the bare ``<name>`` and the
   namespaced ``<plugin>:<name>`` (both invocation shapes show up in
   transcripts, so both need to join through ``skill_usage``).
3. ``~/.claude/plugins/cache/<owner>/<plugin>/<version>/commands/<name>.md``
   -- plugin-provided slash commands, same bare + namespaced treatment.

Plus a constant :data:`BUILTIN_SLASH_COMMANDS` written verbatim so the
view can tag ``/clear``, ``/compact``, ``/plugin`` etc. without SQL
hardcoding.

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

import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from loguru import logger
from packaging.version import InvalidVersion
from packaging.version import Version as _Version

from claude_sql.config import Settings

# Built-in Claude Code slash commands.  These never map to a SKILL.md on
# disk but show up as ``<command-name>/clear</command-name>`` in the
# transcripts; tagging them keeps the ``skill_usage`` view's
# ``is_builtin`` column honest.
BUILTIN_SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("clear", "Clear the current conversation context."),
    ("compact", "Compact the current conversation to free up context."),
    ("cost", "Show token and cost usage for the current session."),
    ("help", "Show the Claude Code help menu."),
    ("ide", "Connect to an IDE extension."),
    ("init", "Initialize a new CLAUDE.md for the working directory."),
    ("mcp", "Manage MCP servers attached to this session."),
    ("memory", "Manage persistent conversation memory."),
    ("model", "Switch the active Claude model."),
    ("plugin", "Manage installed Claude Code plugins."),
    ("reload-plugins", "Reload plugin definitions without restarting."),
    ("resume", "Resume a prior session."),
    ("review", "Request a code review of the current changes."),
    ("security-review", "Run a security review over pending changes."),
    ("status", "Show current session status."),
)


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(path: Path) -> dict[str, Any]:
    """Return the YAML frontmatter at the top of ``path`` as a dict.

    Silent-fallback on missing / unparseable frontmatter so a malformed
    SKILL.md file doesn't abort the whole sync -- we just get a row with
    ``description=None`` and a warning.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("skills_catalog: cannot read {}: {}", path, exc)
        return {}
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning("skills_catalog: yaml parse error in {}: {}", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _coerce_str(value: Any) -> str | None:
    """Flatten multiline YAML scalars (literal ``>``, ``|``) to a single line."""
    if value is None:
        return None
    text = str(value).strip()
    # Multiline YAML can leave newlines; flatten so the parquet stays narrow.
    text = re.sub(r"\s+", " ", text)
    return text or None


def _read_plugin_manifest(version_dir: Path) -> dict[str, Any]:
    """Return the parsed ``.claude-plugin/plugin.json`` or an empty dict."""
    manifest = version_dir / ".claude-plugin" / "plugin.json"
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("skills_catalog: cannot read {}: {}", manifest, exc)
        return {}


def _version_sort_key(version_dir: Path) -> tuple[int, Any, float]:
    """Sort plugin-version directories newest-first.

    Returns a tuple ``(tier, version_or_none, mtime)``:

    * ``tier=0`` when the directory name parses as a PEP 440 / semver
      :class:`packaging.version.Version` -- sort those on ``Version``.
    * ``tier=1`` for anything else (``unknown/`` etc.); those fall back
      to the directory's ``mtime``.

    The outer caller reverses the sort so newest wins.
    """
    name = version_dir.name
    try:
        ver = _Version(name)
        return (0, ver, version_dir.stat().st_mtime)
    except InvalidVersion:
        try:
            return (1, None, version_dir.stat().st_mtime)
        except OSError:
            return (1, None, 0.0)


def _walk_user_skills(root: Path, now: datetime) -> Iterable[dict[str, Any]]:
    """Yield catalog rows for each ``~/.claude/skills/<name>/SKILL.md``."""
    if not root.exists():
        return
    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        meta = _parse_frontmatter(skill_md)
        # ``name`` in frontmatter wins over the directory name when both are
        # present; we keep them aligned in the canonical ``name`` field.
        name = _coerce_str(meta.get("name")) or skill_dir.name
        yield {
            "skill_id": name,
            "name": name,
            "plugin": None,
            "plugin_version": None,
            "source_kind": "user-skill",
            "description": _coerce_str(meta.get("description")),
            "argument_hint": _coerce_str(meta.get("argument-hint"))
            or _coerce_str(meta.get("argument_hint")),
            "source_path": str(skill_md),
            "synced_at": now,
        }


def _walk_plugins(root: Path, now: datetime) -> Iterable[dict[str, Any]]:
    """Yield catalog rows for every skill + command under the plugins cache.

    Plugin layout:

    .. code-block:: text

        <root>/<owner>/<plugin>/<version>/.claude-plugin/plugin.json
        <root>/<owner>/<plugin>/<version>/skills/<name>/SKILL.md
        <root>/<owner>/<plugin>/<version>/commands/<name>.md

    Only the newest ``<version>`` per ``(owner, plugin)`` is emitted --
    older cached generations are ignored.  Each skill + command emits
    two rows: one bare (``skill_id=<name>``) and one namespaced
    (``skill_id=<plugin>:<name>``) -- both invocation shapes show up in
    real transcripts, so the catalog must carry both keys.
    """
    if not root.exists():
        return

    for owner_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for plugin_dir in sorted(p for p in owner_dir.iterdir() if p.is_dir()):
            version_dirs = [p for p in plugin_dir.iterdir() if p.is_dir()]
            if not version_dirs:
                continue
            # Newest version wins.  Tier-0 (semver) beats tier-1 (unknown);
            # within a tier the higher Version / later mtime wins.
            version_dirs.sort(key=_version_sort_key, reverse=True)
            chosen = version_dirs[0]
            manifest = _read_plugin_manifest(chosen)
            plugin_name = _coerce_str(manifest.get("name")) or plugin_dir.name
            plugin_version = _coerce_str(manifest.get("version")) or chosen.name

            skills_root = chosen / "skills"
            if skills_root.is_dir():
                for skill_dir in sorted(skills_root.iterdir()):
                    skill_md = skill_dir / "SKILL.md"
                    if not skill_md.exists():
                        continue
                    meta = _parse_frontmatter(skill_md)
                    name = _coerce_str(meta.get("name")) or skill_dir.name
                    description = _coerce_str(meta.get("description"))
                    base_row = {
                        "name": name,
                        "plugin": plugin_name,
                        "plugin_version": plugin_version,
                        "source_kind": "plugin-skill",
                        "description": description,
                        "argument_hint": _coerce_str(meta.get("argument-hint"))
                        or _coerce_str(meta.get("argument_hint")),
                        "source_path": str(skill_md),
                        "synced_at": now,
                    }
                    # Bare form AND namespaced form -- both show up in the
                    # transcripts and we want either key to join cleanly.
                    yield {**base_row, "skill_id": name}
                    yield {**base_row, "skill_id": f"{plugin_name}:{name}"}

            commands_root = chosen / "commands"
            if commands_root.is_dir():
                for command_file in sorted(commands_root.glob("*.md")):
                    meta = _parse_frontmatter(command_file)
                    name = command_file.stem
                    description = _coerce_str(meta.get("description"))
                    base_row = {
                        "name": name,
                        "plugin": plugin_name,
                        "plugin_version": plugin_version,
                        "source_kind": "plugin-command",
                        "description": description,
                        "argument_hint": _coerce_str(meta.get("argument-hint"))
                        or _coerce_str(meta.get("argument_hint")),
                        "source_path": str(command_file),
                        "synced_at": now,
                    }
                    yield {**base_row, "skill_id": name}
                    yield {**base_row, "skill_id": f"{plugin_name}:{name}"}


def _builtin_rows(now: datetime) -> Iterable[dict[str, Any]]:
    """Yield one row per :data:`BUILTIN_SLASH_COMMANDS` entry."""
    for name, blurb in BUILTIN_SLASH_COMMANDS:
        yield {
            "skill_id": name,
            "name": name,
            "plugin": None,
            "plugin_version": None,
            "source_kind": "builtin",
            "description": blurb,
            "argument_hint": None,
            "source_path": None,
            "synced_at": now,
        }


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
