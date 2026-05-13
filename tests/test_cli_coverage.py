"""Patch-coverage tests for the v1.0 windowed-pipelines additions to ``cli.py``.

Targets the load-bearing PR-diff lines that the existing test suite missed:

* :func:`claude_sql.cli._maybe_migrate_legacy_caches` — the one-time
  ``~/.claude/`` → ``CLAUDE_SQL_HOME`` migrator (RFC 0002 §5.1).
* :func:`claude_sql.cli._describe_lance_entry` — the LanceDB row in
  ``list-cache`` (rglob over the dataset directory + count_rows).
* :func:`claude_sql.cli.ingest` — both the dry-run and real-run paths
  for the new ``ingest`` subcommand (zero-cost SimHash stamping +
  canonical_uuid resolution).
* :func:`claude_sql.cli.shell` — DuckDB binary missing exit path.

All tests redirect ``CLAUDE_SQL_HOME`` and ``HOME`` under ``tmp_path`` so
neither the user's real ``~/.claude/`` nor the new home dir are touched.
Bedrock and LanceDB writes are mocked out — the ingest tests still build
a small JSONL fixture corpus and run real DuckDB / parquet writes
because the worker is pure CPU.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from claude_sql import cli
from claude_sql.cli import Common
from claude_sql.output import OutputFormat

# ---------------------------------------------------------------------------
# Cache redirection — same shape as test_cli.py so settings paths are sane.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Re-root every cache path + ``CLAUDE_SQL_HOME`` under ``tmp_path``.

    Critical: every test in this module mutates ``CLAUDE_SQL_HOME``;
    ``monkeypatch.setenv`` scrubs each variable on teardown so leaks
    can't bleed into other modules (``test_home`` enforces this for
    itself; we just need the same hygiene here).
    """
    cache = tmp_path / "claude_home"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "duckdb_tmp").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("CLAUDE_SQL_HOME", str(cache))
    monkeypatch.setenv("CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH", str(cache / "embeddings"))
    monkeypatch.setenv("CLAUDE_SQL_LANCE_URI", str(cache / "embeddings_lance"))
    monkeypatch.setenv("CLAUDE_SQL_CLASSIFICATIONS_PARQUET_PATH", str(cache / "classifications"))
    monkeypatch.setenv("CLAUDE_SQL_TRAJECTORY_PARQUET_PATH", str(cache / "trajectory"))
    monkeypatch.setenv("CLAUDE_SQL_CONFLICTS_PARQUET_PATH", str(cache / "conflicts"))
    monkeypatch.setenv("CLAUDE_SQL_USER_FRICTION_PARQUET_PATH", str(cache / "user_friction"))
    monkeypatch.setenv("CLAUDE_SQL_CLUSTERS_PARQUET_PATH", str(cache / "clusters.parquet"))
    monkeypatch.setenv(
        "CLAUDE_SQL_CLUSTER_TERMS_PARQUET_PATH", str(cache / "cluster_terms.parquet")
    )
    monkeypatch.setenv("CLAUDE_SQL_COMMUNITIES_PARQUET_PATH", str(cache / "communities.parquet"))
    monkeypatch.setenv("CLAUDE_SQL_INGEST_STAMPS_PARQUET_PATH", str(cache / "ingest_stamps"))
    monkeypatch.setenv("CLAUDE_SQL_CHECKPOINT_DB_PATH", str(cache / "claude_sql.duckdb"))
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_TEMP_DIR", str(cache / "duckdb_tmp"))
    monkeypatch.setenv("CLAUDE_SQL_EMBED_CONCURRENCY", "2")
    monkeypatch.setenv("CLAUDE_SQL_LLM_CONCURRENCY", "2")
    monkeypatch.setenv("CLAUDE_SQL_BATCH_SIZE", "4")
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_THREADS", "2")
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_MEMORY_LIMIT", "1GB")
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    (tmp_path / "fake_home" / ".claude").mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def _purge_meta_glob_env() -> Iterator[None]:
    prior = os.environ.get("CLAUDE_SQL_SUBAGENT_META_GLOB")
    yield
    if prior is None:
        os.environ.pop("CLAUDE_SQL_SUBAGENT_META_GLOB", None)
    else:
        os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = prior


# ---------------------------------------------------------------------------
# JSONL fixture builder (small subset of conftest.py — kept local so this
# module reads independently and we don't import private helpers).
# ---------------------------------------------------------------------------


def _msg_record(*, uuid: str, sid: str, ts: str, text: str, role: str = "user") -> dict[str, Any]:
    return {
        "parentUuid": None,
        "isSidechain": False,
        "type": role,
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": sid,
        "version": "2.0.0",
        "gitBranch": "main",
        "cwd": "/home/u/proj",
        "userType": "external",
        "entrypoint": "cli",
        "permissionMode": "acceptEdits",
        "promptId": f"p-{uuid}",
        "message": {
            "id": f"m-{uuid}",
            "type": "message",
            "role": role,
            "model": "claude-sonnet-4-6" if role == "assistant" else None,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def _build_corpus(tmp_path: Path) -> dict[str, str]:
    """Tiny on-disk JSONL corpus the ``ingest`` tests can use.

    Two sessions, three messages per session, all long enough to clear
    the 32-char filter the ``messages_text`` view enforces.
    """
    proj = tmp_path / "projects" / "proj-x"
    sid = "11111111-1111-1111-1111-111111111111"
    proj.mkdir(parents=True, exist_ok=True)
    records = [
        _msg_record(
            uuid="u1",
            sid=sid,
            ts="2026-04-01T10:00:00.000Z",
            text="please help me debug a python function that runs slowly under load",
            role="user",
        ),
        _msg_record(
            uuid="a1",
            sid=sid,
            ts="2026-04-01T10:00:10.000Z",
            text="sure thing — let me look at the function and check the obvious culprits",
            role="assistant",
        ),
        _msg_record(
            uuid="u2",
            sid=sid,
            ts="2026-04-01T10:00:30.000Z",
            text="thanks; the bottleneck looked like a python loop so I rewrote it in numpy",
            role="user",
        ),
    ]
    with (proj / f"{sid}.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec))
            fh.write("\n")
    # Subagent stub so register_raw doesn't crash on empty subagent globs.
    sa_dir = tmp_path / "projects" / "proj-x" / "00000000-0000-0000-0000-000000000000" / "subagents"
    sa_dir.mkdir(parents=True, exist_ok=True)
    sa_record = _msg_record(
        uuid="sa-stub",
        sid="placeholder",
        ts="2026-01-01T00:00:00.000Z",
        text="subagent stub record so duckdb read_json infers a schema",
        role="user",
    )
    with (sa_dir / "agent-stub.jsonl").open("w") as fh:
        fh.write(json.dumps(sa_record))
        fh.write("\n")
    (sa_dir / "agent-stub.meta.json").write_text(
        json.dumps({"agentType": "stub", "description": "stub"})
    )
    sa_glob = str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl")
    sa_meta_glob = str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.meta.json")
    return {
        "glob": str(tmp_path / "projects" / "*" / "*.jsonl"),
        "subagent_glob": sa_glob,
        "subagent_meta_glob": sa_meta_glob,
    }


def _common_for(corpus: dict[str, str], fmt: OutputFormat = OutputFormat.JSON) -> Common:
    os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = corpus["subagent_meta_glob"]
    return Common(
        verbose=False,
        quiet=True,
        glob=corpus["glob"],
        subagent_glob=corpus["subagent_glob"],
        format=fmt,
    )


# ---------------------------------------------------------------------------
# _maybe_migrate_legacy_caches — RFC 0002 §5.1 one-time migration
# ---------------------------------------------------------------------------


def test_migrate_legacy_caches_moves_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Recognized ``~/.claude/`` caches are moved into ``CLAUDE_SQL_HOME``.

    Pin the happy path: a legacy ``embeddings_lance/`` dir under the
    fake ``$HOME/.claude/`` and a ``state.db`` legacy file get moved to
    the new home, and the ``.migration_complete`` sentinel is stamped.
    A second call is a no-op (sentinel exists).
    """
    fake_home = tmp_path / "user_home"
    legacy = fake_home / ".claude"
    legacy.mkdir(parents=True)
    # Two legacy caches: one directory + one file.
    (legacy / "embeddings_lance").mkdir()
    (legacy / "embeddings_lance" / "data.lance").write_bytes(b"\x00" * 16)
    (legacy / "state.db").write_bytes(b"\x00" * 8)
    monkeypatch.setenv("HOME", str(fake_home))

    new_home = tmp_path / "new_home"
    monkeypatch.setenv("CLAUDE_SQL_HOME", str(new_home))

    cli._maybe_migrate_legacy_caches()

    assert (new_home / "embeddings_lance" / "data.lance").exists()
    assert (new_home / "state.db").exists()
    assert (new_home / ".migration_complete").exists()
    assert not (legacy / "embeddings_lance").exists()
    assert not (legacy / "state.db").exists()

    # Idempotency: second call short-circuits on the sentinel.
    cli._maybe_migrate_legacy_caches()
    # Marker still there — function returned early without crashing.
    assert (new_home / ".migration_complete").exists()


def test_migrate_legacy_caches_skips_when_destination_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pre-populated destination is preserved; legacy stays in place.

    The migrator is conservative: if the new home already has a cache
    with the same name (e.g. a fresh install on a box that also has a
    legacy layout), don't overwrite — let the user reconcile manually.
    """
    fake_home = tmp_path / "user_home"
    legacy = fake_home / ".claude"
    legacy.mkdir(parents=True)
    (legacy / "embeddings_lance").mkdir()
    (legacy / "embeddings_lance" / "data.lance").write_bytes(b"legacy")
    monkeypatch.setenv("HOME", str(fake_home))

    new_home = tmp_path / "new_home"
    new_home.mkdir()
    (new_home / "embeddings_lance").mkdir()
    (new_home / "embeddings_lance" / "existing.lance").write_bytes(b"already-here")
    monkeypatch.setenv("CLAUDE_SQL_HOME", str(new_home))

    cli._maybe_migrate_legacy_caches()

    # Legacy preserved (not moved over the populated destination).
    assert (legacy / "embeddings_lance" / "data.lance").exists()
    # Destination's prior contents intact.
    assert (new_home / "embeddings_lance" / "existing.lance").exists()
    assert (new_home / ".migration_complete").exists()


def test_migrate_legacy_caches_no_legacy_stamps_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty legacy root still stamps the marker so we don't re-probe.

    A box that never had ``~/.claude/`` (or had it but no recognized
    caches) should not keep walking the legacy tree on every CLI call.
    """
    fake_home = tmp_path / "user_home"
    fake_home.mkdir()
    # Note: NO ``.claude`` dir at all under fake_home.
    monkeypatch.setenv("HOME", str(fake_home))

    new_home = tmp_path / "new_home"
    monkeypatch.setenv("CLAUDE_SQL_HOME", str(new_home))

    cli._maybe_migrate_legacy_caches()

    assert (new_home / ".migration_complete").exists()


def test_migrate_legacy_caches_oserror_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An ``OSError`` during the move is logged, not raised.

    The migrator must not crash startup on a hostile filesystem
    (read-only mount, EACCES on a single subtree). Patches
    ``shutil.move`` to raise and asserts the function returns normally.
    """
    fake_home = tmp_path / "user_home"
    legacy = fake_home / ".claude"
    legacy.mkdir(parents=True)
    (legacy / "state.db").write_bytes(b"\x00")
    monkeypatch.setenv("HOME", str(fake_home))

    new_home = tmp_path / "new_home"
    monkeypatch.setenv("CLAUDE_SQL_HOME", str(new_home))

    def _raise(*_a: object, **_kw: object) -> None:
        raise OSError("simulated read-only mount")

    monkeypatch.setattr(cli.shutil, "move", _raise)

    # Must not raise — the loop swallows OSError and warns.
    cli._maybe_migrate_legacy_caches()


def test_migrate_legacy_caches_marker_touch_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing ``marker.touch()`` on the empty-legacy path is logged, not raised.

    Covers the inner ``except OSError`` branch around the empty-legacy
    marker stamp. Patches ``Path.touch`` to raise so we exercise the
    warning-and-return path without crashing startup.
    """
    fake_home = tmp_path / "no_legacy_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    new_home = tmp_path / "new_home_locked"
    monkeypatch.setenv("CLAUDE_SQL_HOME", str(new_home))

    real_touch = Path.touch

    def _selective_touch(self: Path, *a: object, **kw: object) -> None:
        # Only fail when stamping the migration marker; let mkdir / other
        # path operations succeed normally.
        if self.name == ".migration_complete":
            raise OSError("simulated EACCES on marker stamp")
        real_touch(self, *a, **kw)

    monkeypatch.setattr(Path, "touch", _selective_touch)

    # Must NOT raise — the function logs and returns.
    cli._maybe_migrate_legacy_caches()


# ---------------------------------------------------------------------------
# _describe_lance_entry — list-cache LanceDB row
# ---------------------------------------------------------------------------


def test_describe_lance_entry_for_existing_dataset(tmp_path: Path) -> None:
    """A populated lance dataset reports nonzero ``bytes`` and an mtime."""
    lance_dir = tmp_path / "embeddings_lance"
    lance_dir.mkdir()
    # Drop a couple of fake fragment files so the rglob loop has work.
    (lance_dir / "fragment-1.lance").write_bytes(b"\x00" * 64)
    sub = lance_dir / "_versions"
    sub.mkdir()
    (sub / "1.manifest").write_bytes(b"\x00" * 32)

    entry = cli._describe_lance_entry(lance_dir)
    assert entry["name"] == "embeddings_lance"
    assert entry["exists"] is True
    bytes_val = entry["bytes"]
    assert isinstance(bytes_val, int)
    assert bytes_val > 0
    assert "mtime" in entry
    # rows is None because there's no real Lance table here — count_rows
    # raises and the helper traps OSError/ValueError/RuntimeError → None.
    assert "rows" in entry


def test_describe_lance_entry_for_missing_path(tmp_path: Path) -> None:
    """A nonexistent path yields an entry with ``exists=False`` only."""
    entry = cli._describe_lance_entry(tmp_path / "no-such-dataset")
    assert entry["exists"] is False
    assert "bytes" not in entry
    assert "mtime" not in entry


# ---------------------------------------------------------------------------
# ingest subcommand — zero-cost SimHash + canonical_uuid stamping
# ---------------------------------------------------------------------------


def test_ingest_dry_run_emits_candidate_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dry-run path emits ``{pipeline:ingest, candidates:N, dry_run:true}``.

    Pure CPU — no Bedrock. Builds a small JSONL fixture and verifies
    the ``candidates`` count reflects rows in ``messages_text``.
    """
    corpus = _build_corpus(tmp_path)
    cli.ingest(dry_run=True, common=_common_for(corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline"] == "ingest"
    assert payload["dry_run"] is True
    assert isinstance(payload["candidates"], int)
    assert payload["candidates"] >= 1


def test_ingest_real_run_stamps_and_resolves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real-run path: stamps + resolves canonicals, emits rows_processed.

    Exercises the full ingest pipeline body (lines 1833-1839 in cli.py)
    end-to-end against a real DuckDB connection + parquet writes.
    """
    corpus = _build_corpus(tmp_path)
    cli.ingest(dry_run=False, common=_common_for(corpus))
    payload = json.loads(capsys.readouterr().out)
    # _emit_worker_result wraps the int return into a JSON envelope.
    assert payload["pipeline"] == "ingest"
    assert payload["dry_run"] is False
    assert isinstance(payload["rows_processed"], int)
    assert payload["rows_processed"] >= 1


def test_ingest_dry_run_with_filters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Dry-run honors ``--since-days`` and ``--limit`` and reflects them in JSON."""
    corpus = _build_corpus(tmp_path)
    cli.ingest(since_days=365, limit=2, dry_run=True, common=_common_for(corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload["since_days"] == 365
    assert payload["limit"] == 2


# ---------------------------------------------------------------------------
# shell — duckdb binary missing path
# ---------------------------------------------------------------------------


def _patch_mkstemp_for_duckdb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Patch ``tempfile.mkstemp`` to return a fresh, never-created path.

    DuckDB rejects a zero-byte file at the path with ``IOException``.
    The real ``shell`` command works because ``mkstemp`` writes nothing
    to the file (just creates+closes the fd) but DuckDB's ``connect``
    treats ANY existing zero-byte file as corrupt. The fix: return a
    path that doesn't yet exist so DuckDB initializes a fresh database.
    """
    db_path = tmp_path / "shell_test.duckdb"
    # Don't create the file — let duckdb.connect create a fresh DB.
    fd_holder = os.open(os.devnull, os.O_RDONLY)

    def _fake_mkstemp(*_a: object, **_kw: object) -> tuple[int, str]:
        return fd_holder, str(db_path)

    monkeypatch.setattr(cli.tempfile, "mkstemp", _fake_mkstemp)
    return db_path


def test_shell_missing_duckdb_binary_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shell`` exits non-zero when the ``duckdb`` binary is not on PATH.

    Patches ``subprocess.run`` to raise ``FileNotFoundError`` (the OS's
    way of saying "binary missing"). The CLI must exit with the
    ``duckdb_missing`` exit code rather than letting the exception
    propagate as an uncaught traceback.
    """
    corpus = _build_corpus(tmp_path)
    _patch_mkstemp_for_duckdb(monkeypatch, tmp_path)

    def _raise(*_a: object, **_kw: object) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError("duckdb")

    monkeypatch.setattr(cli.subprocess, "run", _raise)

    with pytest.raises(SystemExit) as exc:
        cli.shell(common=_common_for(corpus))
    # Whatever the configured exit code is — assert it's non-zero rather
    # than hard-coding the value in case EXIT_CODES gets renumbered.
    assert exc.value.code != 0


def test_shell_invokes_duckdb_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shell`` happy path: subprocess.run is called with the duckdb binary."""
    corpus = _build_corpus(tmp_path)
    _patch_mkstemp_for_duckdb(monkeypatch, tmp_path)
    captured: list[list[str]] = []

    def _capture(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
        captured.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", _capture)

    cli.shell(common=_common_for(corpus))

    assert captured, "expected subprocess.run to be called"
    assert captured[0][0] == "duckdb"
    # The DB path argument exists on disk (duckdb.connect created it).
    assert Path(captured[0][1]).exists()


# ---------------------------------------------------------------------------
# list_cache surfaces the legacy:* breadcrumb when caches predate migration
# ---------------------------------------------------------------------------


def test_list_cache_surfaces_legacy_breadcrumb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``list-cache`` flags any legacy cache that wasn't auto-migrated.

    Stamps the migration marker first so the auto-migrator short-circuits
    on the next CLI call, then plants a legacy ``state.db`` under the
    fake ``~/.claude/``. The breadcrumb iterator in ``list_cache`` should
    surface a ``legacy:state.db`` entry.
    """
    fake_home = tmp_path / "fake_home_for_legacy"
    legacy = fake_home / ".claude"
    legacy.mkdir(parents=True)
    (legacy / "state.db").write_bytes(b"\x00")
    monkeypatch.setenv("HOME", str(fake_home))

    # Stamp marker so the auto-migrator on _open_connection_full doesn't
    # move the legacy cache before list_cache can see it.
    new_home = tmp_path / "claude_home"
    new_home.mkdir(exist_ok=True)
    (new_home / ".migration_complete").touch()

    corpus = _build_corpus(tmp_path)
    cli.list_cache(common=_common_for(corpus, OutputFormat.JSON))
    payload = json.loads(capsys.readouterr().out)
    names = {entry["name"] for entry in payload}
    assert any(name.startswith("legacy:") for name in names)


# ---------------------------------------------------------------------------
# ingest_stamps cache shows up in list-cache (covers the new entry binding)
# ---------------------------------------------------------------------------


def test_list_cache_includes_ingest_stamps_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ingest_stamps`` cache row appears in ``list-cache`` output."""
    corpus = _build_corpus(tmp_path)
    cli.list_cache(common=_common_for(corpus))
    payload = json.loads(capsys.readouterr().out)
    names = {entry["name"] for entry in payload}
    assert "ingest_stamps" in names
