"""Discover where ``claude-sql`` was installed from.

Four install shapes reach a user's ``PATH``: the published wheel
(``uv tool install claude-sql``), a local checkout
(``uv tool install --from . claude-sql``), a git URL, and this project's own
venv. ``uv`` records the source of every tool install in
``$UV_TOOL_DIR/<tool>/uv-receipt.toml``, so ``claude-sql --version`` (and the
``version`` subcommand) reads that receipt and names which one it is.

**A registry install is identified by the ABSENCE of a local-source key**, not
by a positive marker. Probed 2026-08-20 against a real
``uv tool install claude-sql==2.1.0``: the requirement entry is
``{name = "claude-sql", specifier = "==2.1.0"}`` — no ``directory``, ``url``, or
``git``. Treating "no local-source key" as unknown reports the placeholder
``source: ?`` for the most common install of all, so :data:`_REGISTRY` names it.

The receipt schema is not a public contract — uv has changed it between
releases. Every read is wrapped in ``try/except`` so a future schema change
degrades to "source unknown" instead of crashing the CLI.
"""

from __future__ import annotations

import os
import tomllib
from importlib.metadata import version as _version
from pathlib import Path

__version__ = _version("claude-sql")

#: Receipt keys that mark a *local* install source. A requirement entry carrying
#: none of these came from a package registry.
_LOCAL_SOURCE_KEYS: tuple[str, ...] = ("directory", "url", "git")

#: ``source_kind`` for a registry install. Not a uv receipt key — this module
#: mints it, because uv records a registry install as the absence of the keys
#: above rather than as a value.
_REGISTRY: str = "registry"


def _tool_dir() -> Path:
    """Return the uv tool-install root, respecting ``$UV_TOOL_DIR`` / XDG."""
    if override := os.environ.get("UV_TOOL_DIR"):
        return Path(override)
    if xdg := os.environ.get("XDG_DATA_HOME"):
        return Path(xdg) / "uv" / "tools"
    return Path.home() / ".local" / "share" / "uv" / "tools"


def read_install_source(tool: str = "claude-sql") -> dict[str, str] | None:
    """Parse ``uv-receipt.toml`` for an installed uv tool.

    Returns a dict with keys ``source_kind`` (``"directory"`` / ``"url"`` /
    ``"git"`` / ``"registry"``), ``source`` (the value — for a registry install,
    the requirement string such as ``claude-sql==2.1.0``), and optionally
    ``install_path`` (the resolved entrypoint). Returns ``None`` when the receipt
    is missing, the TOML is unreadable, or it names no requirement for ``tool``.
    """
    receipt = _tool_dir() / tool / "uv-receipt.toml"
    try:
        data = tomllib.loads(receipt.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    info: dict[str, str] = {}
    for req in (data.get("tool") or {}).get("requirements") or []:
        if not isinstance(req, dict) or req.get("name") != tool:
            continue
        for key in _LOCAL_SOURCE_KEYS:
            if val := req.get(key):
                info["source_kind"] = key
                info["source"] = str(val)
                break
        else:
            info["source_kind"] = _REGISTRY
            info["source"] = f"{tool}{req.get('specifier') or ''}"
        break
    for ep in (data.get("tool") or {}).get("entrypoints") or []:
        if not isinstance(ep, dict) or ep.get("name") != tool:
            continue
        if path := ep.get("install-path"):
            info["install_path"] = str(path)
        break
    return info or None


def format_version() -> str:
    """Return ``"claude-sql X.Y.Z"`` plus an install-source line when known."""
    lines = [f"claude-sql {__version__}"]
    src = read_install_source()
    if src is None:
        lines.append("installed from: project venv (not via `uv tool install`)")
        return "\n".join(lines)
    kind = src.get("source_kind", "source")
    where = src.get("source", "?")
    lines.append(f"installed from {kind}: {where}")
    if ip := src.get("install_path"):
        lines.append(f"entrypoint: {ip}")
    return "\n".join(lines)
