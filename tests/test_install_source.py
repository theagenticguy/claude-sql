"""Tests for :mod:`claude_sql.install_source`."""

from __future__ import annotations

from pathlib import Path

import pytest

from claude_sql.interfaces.cli.install_source import _tool_dir, format_version, read_install_source


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


def test_read_install_source_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Verbatim receipt from a real `uv tool install claude-sql==2.1.0`
    # (probed 2026-08-20): a registry install carries a `specifier` and NONE of
    # directory / url / git, so the source kind has to be inferred from absence.
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    _write_receipt(
        tmp_path,
        """
[tool]
requirements = [{ name = "claude-sql", specifier = "==2.1.0" }]
entrypoints = [
    { name = "claude-sql", install-path = "/home/u/.local/bin/claude-sql", from = "claude-sql" },
]
""",
    )
    info = read_install_source("claude-sql")
    assert info == {
        "source_kind": "registry",
        "source": "claude-sql==2.1.0",
        "install_path": "/home/u/.local/bin/claude-sql",
    }


def test_read_install_source_registry_without_specifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `uv tool install claude-sql` pins no version, so the entry is name-only.
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    _write_receipt(tmp_path, '[tool]\nrequirements = [{ name = "claude-sql" }]\n')
    assert read_install_source("claude-sql") == {
        "source_kind": "registry",
        "source": "claude-sql",
    }


def test_format_version_names_a_registry_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The regression this locks: the published wheel is the most common install
    # of all, and reporting it as the unknown-source placeholder `source: ?`
    # makes `--version` useless for the users most likely to run it.
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path))
    _write_receipt(
        tmp_path,
        '[tool]\nrequirements = [{ name = "claude-sql", specifier = "==2.1.0" }]\n',
    )
    out = format_version()
    assert "installed from registry: claude-sql==2.1.0" in out
    assert "source: ?" not in out
    assert "project venv" not in out


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
