"""Filesystem walkers for the skills catalog.

Reads the three on-disk skill sources and yields catalog rows:

1. ``~/.claude/skills/<name>/SKILL.md`` -- user-installed skills, keyed by
   bare name only.
2. ``~/.claude/plugins/cache/<owner>/<plugin>/<version>/skills/<name>/SKILL.md``
   -- plugin-provided skills, emitted as *both* the bare ``<name>`` and the
   namespaced ``<plugin>:<name>``.
3. ``~/.claude/plugins/cache/<owner>/<plugin>/<version>/commands/<name>.md``
   -- plugin-provided slash commands, same bare + namespaced treatment.

Plus :func:`_builtin_rows` for the :data:`BUILTIN_SLASH_COMMANDS` constant.

Pure parsing helpers (``_coerce_str``, ``_version_sort_key``,
``BUILTIN_SLASH_COMMANDS``, ``_FRONTMATTER_RE``) live in
:mod:`claude_sql.domain.skills`; this adapter owns the IO (reading files,
parsing YAML frontmatter / plugin manifests, walking directory trees).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from claude_sql.domain.skills import (
    _FRONTMATTER_RE,
    BUILTIN_SLASH_COMMANDS,
    _coerce_str,
    _version_sort_key,
)


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


def _read_plugin_manifest(version_dir: Path) -> dict[str, Any]:
    """Return the parsed ``.claude-plugin/plugin.json`` or an empty dict."""
    manifest = version_dir / ".claude-plugin" / "plugin.json"
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("skills_catalog: cannot read {}: {}", manifest, exc)
        return {}


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


__all__ = [
    "_builtin_rows",
    "_parse_frontmatter",
    "_read_plugin_manifest",
    "_walk_plugins",
    "_walk_user_skills",
]
