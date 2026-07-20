"""Tests for :mod:`claude_sql.home` and the ``CLAUDE_SQL_HOME`` plumbing.

Covers RFC 0002 §5.1: the parent-directory resolution rules, the legacy
cache discovery helper, and the per-field cascade through
:class:`claude_sql.config.Settings`. The Settings tests confirm that
``CLAUDE_SQL_HOME`` re-roots every default-derived path while
user-pinned ``CLAUDE_SQL_*_PATH`` env vars still beat the cascade.

The ``_purge_home_env`` autouse fixture snapshots and restores every
env var this module touches per test, so we don't leak ``CLAUDE_SQL_*``
state into other test modules. See
``.erpaval/solutions/best-practices/os-environ-leak-across-test-modules.md``
for the rationale.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from claude_sql.infrastructure.home import claude_sql_home, recognized_legacy_caches
from claude_sql.infrastructure.settings import Settings

_ENV_VARS_TO_PURGE: tuple[str, ...] = (
    "CLAUDE_SQL_HOME",
    "XDG_DATA_HOME",
    "CLAUDE_SQL_TRAJECTORY_PARQUET_PATH",
    "CLAUDE_SQL_CLUSTERS_PARQUET_PATH",
)


@pytest.fixture(autouse=True)
def _purge_home_env() -> Iterator[None]:
    """Snapshot/restore every env var these tests mutate.

    Catches anyone who reaches for ``os.environ[...] = ...`` instead of
    ``monkeypatch.setenv``, and shields the rest of the test suite from
    leaked ``CLAUDE_SQL_HOME`` state when something goes sideways.
    """
    prior: dict[str, str | None] = {var: os.environ.get(var) for var in _ENV_VARS_TO_PURGE}
    yield
    for var, value in prior.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


def test_explicit_env_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``CLAUDE_SQL_HOME`` overrides every other resolution rule."""
    monkeypatch.setenv("CLAUDE_SQL_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "should-be-ignored"))
    assert claude_sql_home() == tmp_path
    assert tmp_path.is_dir()


def test_xdg_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """On Linux, ``XDG_DATA_HOME`` resolves to ``$XDG_DATA_HOME/claude-sql``."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("CLAUDE_SQL_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    expected = tmp_path / "claude-sql"
    assert claude_sql_home() == expected
    assert expected.is_dir()


def test_macos_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """On macOS without env, returns ``~/Library/Application Support/claude-sql``."""
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.delenv("CLAUDE_SQL_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    expected = fake_home / "Library" / "Application Support" / "claude-sql"
    assert claude_sql_home() == expected
    assert expected.is_dir()


def test_default_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No envs, non-darwin → ``~/.claude-sql``."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("CLAUDE_SQL_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    expected = fake_home / ".claude-sql"
    assert claude_sql_home() == expected
    assert expected.is_dir()


def test_recognized_legacy_caches_lists_known_paths(tmp_path: Path) -> None:
    """The legacy walker picks up exactly the recognized cache names."""
    legacy_root = tmp_path / "legacy-claude"
    legacy_root.mkdir()
    # A mix of files and directories; the walker should pick all of these up.
    (legacy_root / "embeddings_lance").mkdir()
    (legacy_root / "message_trajectory").mkdir()
    (legacy_root / "clusters.parquet").write_bytes(b"")
    (legacy_root / "state.db").write_bytes(b"")
    # Unrelated paths should be ignored.
    (legacy_root / "projects").mkdir()
    (legacy_root / "random_file.txt").write_bytes(b"")

    found = recognized_legacy_caches(legacy_root)
    assert "embeddings_lance" in found
    assert "message_trajectory" in found
    assert "clusters.parquet" in found
    assert "state.db" in found
    assert "projects" not in found
    assert "random_file.txt" not in found


def test_recognized_legacy_caches_missing_root_returns_empty(tmp_path: Path) -> None:
    """A nonexistent legacy root yields no entries (no crash)."""
    nonexistent = tmp_path / "does-not-exist"
    assert recognized_legacy_caches(nonexistent) == {}


def test_settings_paths_default_under_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Setting ``CLAUDE_SQL_HOME`` re-roots every default-derived cache path."""
    monkeypatch.setenv("CLAUDE_SQL_HOME", str(tmp_path))
    settings = Settings()
    home = tmp_path
    # Every path field that should live under CLAUDE_SQL_HOME by default.
    cache_paths = [
        settings.embeddings_parquet_path,
        settings.lance_uri,
        settings.classifications_parquet_path,
        settings.trajectory_parquet_path,
        settings.conflicts_parquet_path,
        settings.clusters_parquet_path,
        settings.cluster_terms_parquet_path,
        settings.communities_parquet_path,
        settings.community_profile_parquet_path,
        settings.user_friction_parquet_path,
        settings.skills_catalog_parquet_path,
        settings.checkpoint_db_path,
        settings.duckdb_temp_dir,
        settings.ingest_stamps_parquet_path,
    ]
    for p in cache_paths:
        assert home in p.parents, f"{p} is not under CLAUDE_SQL_HOME={home}"


def test_settings_user_pinned_path_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit ``CLAUDE_SQL_*_PATH`` env beats the ``CLAUDE_SQL_HOME`` cascade."""
    home = tmp_path / "home"
    pinned = tmp_path / "elsewhere" / "trajectory"
    monkeypatch.setenv("CLAUDE_SQL_HOME", str(home))
    monkeypatch.setenv("CLAUDE_SQL_TRAJECTORY_PARQUET_PATH", str(pinned))
    settings = Settings()
    # The pinned field is preserved exactly.
    assert settings.trajectory_parquet_path == pinned
    # Other fields still cascade under CLAUDE_SQL_HOME.
    assert home in settings.clusters_parquet_path.parents


def test_settings_ingest_stamps_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The new ``ingest_stamps_parquet_path`` field defaults under home."""
    monkeypatch.setenv("CLAUDE_SQL_HOME", str(tmp_path))
    settings = Settings()
    assert settings.ingest_stamps_parquet_path == tmp_path / "ingest_stamps"
