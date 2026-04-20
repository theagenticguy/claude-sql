"""Discover where ``claude-sql`` was installed from.

The tool is not published to PyPI — users install from a local checkout via
``uv tool install --from . claude-sql``. ``uv`` records the source of every
tool install in ``$UV_TOOL_DIR/<tool>/uv-receipt.toml``. This module reads
that receipt so ``claude-sql --version`` (and the ``version`` subcommand) can
tell the user whether the binary on their ``PATH`` came from a directory
checkout, a git URL, or (fallback) this project's own venv.

The receipt schema is not a public contract — uv has changed it between
releases. Every read is wrapped in ``try/except`` so a future schema change
degrades to "source unknown" instead of crashing the CLI.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from claude_sql import __version__


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
    ``"git"``), ``source`` (the value), and optionally ``install_path`` (the
    resolved entrypoint). Returns ``None`` when the receipt is missing or the
    TOML is unreadable.
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
        for key in ("directory", "url", "git"):
            if val := req.get(key):
                info["source_kind"] = key
                info["source"] = str(val)
                break
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
