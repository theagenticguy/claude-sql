"""Tests for :mod:`claude_sql.install_source`."""

from __future__ import annotations

from pathlib import Path

import pytest

from claude_sql.install_source import _tool_dir, format_version, read_install_source


def _write_receipt(root: Path, text: str, tool: str = "claude-sql") -> Path:
    tool_dir = root / tool
    tool_dir.mkdir(parents=True, exist_ok=True)
    receipt = tool_dir / "uv-receipt.toml"
    receipt.write_text(text.lstrip())
    return receipt


def test_tool_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", "/custom/tools")
    assert _tool_dir() == Path("/custom/tools")


def test_tool_dir_xdg_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert _tool_dir() == tmp_path / "uv" / "tools"


def test_tool_dir_home_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert _tool_dir() == Path.home() / ".local" / "share" / "uv" / "tools"


def test_read_install_source_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    assert read_install_source("claude-sql") is None


def test_read_install_source_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    _write_receipt(
        tmp_path,
        """
[tool]
requirements = [{ name = "claude-sql", directory = "/repo/claude-sql" }]
entrypoints = [
  { name = "claude-sql", install-path = "/home/u/.local/bin/claude-sql", from = "claude-sql" },
]
""",
    )
    info = read_install_source("claude-sql")
    assert info == {
        "source_kind": "directory",
        "source": "/repo/claude-sql",
        "install_path": "/home/u/.local/bin/claude-sql",
    }


def test_read_install_source_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    _write_receipt(
        tmp_path,
        """
[tool]
requirements = [{ name = "claude-sql", git = "https://example/x.git" }]
""",
    )
    info = read_install_source("claude-sql")
    assert info == {
        "source_kind": "git",
        "source": "https://example/x.git",
    }


def test_read_install_source_malformed_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    _write_receipt(tmp_path, "this is [[not valid toml")
    # Malformed TOML must degrade to None, not crash.
    assert read_install_source("claude-sql") is None


def test_read_install_source_skips_other_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    # A receipt with only other tools returns None.
    _write_receipt(
        tmp_path,
        """
[tool]
requirements = [{ name = "ruff", directory = "/repo/ruff" }]
""",
    )
    assert read_install_source("claude-sql") is None


def test_format_version_with_directory_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    _write_receipt(
        tmp_path,
        '[tool]\nrequirements = [{ name = "claude-sql", directory = "/repo" }]\n',
    )
    out = format_version()
    assert out.splitlines()[0].startswith("claude-sql ")
    assert "installed from directory: /repo" in out


def test_format_version_no_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    out = format_version()
    assert "project venv" in out
