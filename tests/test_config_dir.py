"""Tests for ``CLAUDE_CONFIG_DIR``-aware transcript-root resolution.

Claude Code relocates its own config dir (transcripts, skills, plugin cache)
to ``$CLAUDE_CONFIG_DIR`` when set — a fleet run service points it under
``~/agent-fleet/config-dirs/``. These tests pin the resolution precedence:

    explicit ``CLAUDE_SQL_*`` glob > ``team_corpus_root`` > ``CLAUDE_CONFIG_DIR`` > ``~/.claude``

and confirm the skills / plugin-cache dirs follow the same root. The helper
reads ``os.environ`` at call time (matching the ``home.claude_sql_home``
precedent), so ``monkeypatch.setenv`` is observed without a module reload.

The ``_purge_config_env`` autouse fixture snapshots/restores every env var
this module touches so ``CLAUDE_CONFIG_DIR`` state never leaks across modules
(see ``.erpaval/solutions/best-practices/os-environ-leak-across-test-modules.md``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from claude_sql.infrastructure.duckdb_views import register_all
from claude_sql.infrastructure.settings import (
    Settings,
    _claude_config_root,
    _default_glob,
    _default_plugins_cache_dir,
    _default_subagent_glob,
    _default_subagent_meta_glob,
    _default_user_skills_dir,
)

_ENV_VARS_TO_PURGE: tuple[str, ...] = (
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_SQL_DEFAULT_GLOB",
    "CLAUDE_SQL_SUBAGENT_GLOB",
    "CLAUDE_SQL_SUBAGENT_META_GLOB",
    "CLAUDE_SQL_TEAM_CORPUS_ROOT",
    "CLAUDE_SQL_USER_SKILLS_DIR",
    "CLAUDE_SQL_PLUGINS_CACHE_DIR",
)


@pytest.fixture(autouse=True)
def _purge_config_env() -> Iterator[None]:
    """Snapshot/restore every env var these tests mutate."""
    prior: dict[str, str | None] = {var: os.environ.get(var) for var in _ENV_VARS_TO_PURGE}
    yield
    for var, value in prior.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


# ---------------------------------------------------------------------------
# Helper resolution
# ---------------------------------------------------------------------------


def test_config_root_defaults_to_dot_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset ``CLAUDE_CONFIG_DIR`` falls back to the historical ``~/.claude``."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert _claude_config_root() == os.path.expanduser("~/.claude")


def test_config_root_honors_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``CLAUDE_CONFIG_DIR`` re-roots the helper, read at call time."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert _claude_config_root() == str(tmp_path)


def test_config_root_expands_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``~`` in ``CLAUDE_CONFIG_DIR`` is expanded."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "~/agent-fleet/config-dirs/agent-7")
    assert _claude_config_root() == os.path.expanduser("~/agent-fleet/config-dirs/agent-7")


# ---------------------------------------------------------------------------
# Glob factories derive from the config root
# ---------------------------------------------------------------------------


def test_globs_derive_from_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """All three transcript globs are rooted under ``CLAUDE_CONFIG_DIR``."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert _default_glob() == f"{tmp_path}/projects/*/*.jsonl"
    assert _default_subagent_glob() == f"{tmp_path}/projects/*/*/subagents/agent-*.jsonl"
    assert _default_subagent_meta_glob() == f"{tmp_path}/projects/*/*/subagents/agent-*.meta.json"


def test_skills_and_plugins_dirs_derive_from_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The read-only skills / plugin-cache roots follow ``CLAUDE_CONFIG_DIR`` too."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert _default_user_skills_dir() == tmp_path / "skills"
    assert _default_plugins_cache_dir() == tmp_path / "plugins" / "cache"


def test_settings_pick_up_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A fresh ``Settings`` binds every config-dir-derived field to the new root."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    settings = Settings()
    assert settings.default_glob == f"{tmp_path}/projects/*/*.jsonl"
    assert settings.subagent_glob == f"{tmp_path}/projects/*/*/subagents/agent-*.jsonl"
    assert settings.subagent_meta_glob == f"{tmp_path}/projects/*/*/subagents/agent-*.meta.json"
    assert settings.user_skills_dir == tmp_path / "skills"
    assert settings.plugins_cache_dir == tmp_path / "plugins" / "cache"


# ---------------------------------------------------------------------------
# Precedence: explicit glob > team_corpus_root > CLAUDE_CONFIG_DIR > ~/.claude
# ---------------------------------------------------------------------------


def test_explicit_glob_beats_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A user-pinned ``default_glob`` wins over ``CLAUDE_CONFIG_DIR``."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    custom = "/some/custom/path/*.jsonl"
    settings = Settings(default_glob=custom)
    assert settings.default_glob == custom


def test_team_corpus_root_beats_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``team_corpus_root`` wins over ``CLAUDE_CONFIG_DIR`` for the derived globs.

    ``CLAUDE_CONFIG_DIR`` still governs what the factory defaults look like, so
    the team-corpus validator's user-pin check (which compares against those
    factory defaults) does not misfire; the resulting globs are rooted at the
    team corpus, not the config dir.
    """
    config_dir = tmp_path / "config"
    team_root = tmp_path / "team"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    settings = Settings(team_corpus_root=team_root)
    resolved = team_root.expanduser().resolve()
    assert settings.default_glob == f"{resolved}/*/projects/*/*.jsonl"
    assert str(config_dir) not in settings.default_glob


def test_explicit_glob_beats_team_corpus_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The full precedence chain: an explicit glob tops both other sources."""
    config_dir = tmp_path / "config"
    team_root = tmp_path / "team"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    custom = "/pinned/*.jsonl"
    settings = Settings(team_corpus_root=team_root, default_glob=custom)
    assert settings.default_glob == custom
    # The other two stay at the CLAUDE_CONFIG_DIR-rooted factory defaults
    # (we never partially rewrite for team corpus once a pin is detected).
    assert settings.subagent_glob == _default_subagent_glob()
    assert settings.subagent_meta_glob == _default_subagent_meta_glob()


# ---------------------------------------------------------------------------
# End-to-end: register_all binds against a CLAUDE_CONFIG_DIR-rooted corpus
# ---------------------------------------------------------------------------


def _write_min_session(path: Path, session_id: str) -> None:
    """Write a one-message JSONL in the Claude Code transcript shape."""
    record = {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "uuid": f"u-{session_id}",
        "timestamp": "2026-04-01T10:00:00.000Z",
        "sessionId": session_id,
        "version": "2.0.0",
        "gitBranch": "main",
        "cwd": "/home/agent/proj",
        "userType": "external",
        "message": {
            "id": f"m-{session_id}",
            "type": "message",
            "role": "user",
            "model": None,
            "content": [{"type": "text", "text": "hi"}],
        },
    }
    with path.open("w") as fh:
        fh.write(json.dumps(record) + "\n")


def test_register_all_binds_relocated_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A relocated config dir surfaces its sessions through the standard views."""
    config_dir = tmp_path / "config-dirs" / "agent-3" / ".claude"
    proj = config_dir / "projects" / "-home-agent-proj"
    proj.mkdir(parents=True)
    _write_min_session(proj / "sess-1.jsonl", "11111111-1111-1111-1111-111111111111")
    _write_min_session(proj / "sess-2.jsonl", "22222222-2222-2222-2222-222222222222")
    # A subagent stub so register_raw's subagent read_json glob matches at least
    # one file (DuckDB raises IOException on a no-match glob). It is otherwise
    # inert — the local layout does not satisfy the parent-session UUID regex.
    sub_dir = proj / "sess-1" / "subagents"
    sub_dir.mkdir(parents=True)
    (sub_dir / "agent-aaaa1111aaaa1111.meta.json").write_text(
        json.dumps({"agentType": "general-purpose", "description": "stub"})
    )
    _write_min_session(sub_dir / "agent-aaaa1111aaaa1111.jsonl", "sub-aaaa1111aaaa1111")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    cache = tmp_path / "cache"
    cache.mkdir()
    settings = Settings(
        embeddings_parquet_path=cache / "embeddings",
        classifications_parquet_path=cache / "session_classifications",
        trajectory_parquet_path=cache / "message_trajectory",
        conflicts_parquet_path=cache / "session_conflicts",
        clusters_parquet_path=cache / "clusters.parquet",
        cluster_terms_parquet_path=cache / "cluster_terms.parquet",
        communities_parquet_path=cache / "session_communities.parquet",
        user_friction_parquet_path=cache / "user_friction",
        skills_catalog_parquet_path=cache / "skills_catalog.parquet",
        user_skills_dir=cache / "skills",
        plugins_cache_dir=cache / "plugins",
        checkpoint_db_path=cache / "state.db",
        duckdb_temp_dir=cache / "duckdb_tmp",
    )
    con = duckdb.connect(":memory:")
    register_all(con, settings=settings, include_analytics=False)
    n = con.execute("SELECT count(DISTINCT session_id) FROM sessions").fetchone()
    assert n is not None
    assert n[0] == 2
