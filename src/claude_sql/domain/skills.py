"""Pure skills-catalog helpers: frontmatter regex, scalar coercion, version sort.

The dependency-free half of the skills catalog. Holds the built-in slash-command
constant, the frontmatter-block regex, the multiline-scalar flattener, and the
plugin-version sort key. The filesystem walkers that consume these live in the
infrastructure adapter (``infrastructure.skills_fs``); the sync orchestration
lives in the use-case (``application.use_cases.skills``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version as _Version

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


def _coerce_str(value: Any) -> str | None:
    """Flatten multiline YAML scalars (literal ``>``, ``|``) to a single line."""
    if value is None:
        return None
    text = str(value).strip()
    # Multiline YAML can leave newlines; flatten so the parquet stays narrow.
    text = re.sub(r"\s+", " ", text)
    return text or None


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


__all__ = [
    "BUILTIN_SLASH_COMMANDS",
    "_FRONTMATTER_RE",
    "_coerce_str",
    "_version_sort_key",
]
