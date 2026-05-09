"""Tests for ``claude_sql.binding`` — the RFC 0001 reference implementation.

Covers the ten test cases enumerated in the plan:

1. ``test_compute_digest_stable`` — identical bytes → identical hex.
2. ``test_compute_digest_changes_on_byte_change`` — newline append → new hex.
3. ``test_projectify_canonical`` — known cwd → known dash-form mappings.
4. ``test_find_active_transcript_picks_latest_mtime`` — three fakes, latest wins.
5. ``test_write_trailer_roundtrip`` — write trailer, parse with ``git
   interpret-trailers --parse``, assert keys + values.
6. ``test_write_note_roundtrip`` — ``git init`` repo, commit, write note,
   read note, assert JSON.
7. ``test_resolve_precedence_trailer_wins`` — both surfaces present and
   identical → returns merged binding with trailer wire fields.
8. ``test_resolve_mismatch_loud`` — different digest in trailer vs note
   → ``BindingMismatchError``.
9. ``test_resolve_no_binding_exit_2`` — fresh commit → ``LookupError``.
10. ``test_bind_cli_dry_run`` — invoke ``claude-sql bind --dry-run``
    against a tmp git repo, parse stdout, assert plan shape.

All tests use ``tmp_path`` and a real ``git`` subprocess — no Bedrock,
no mocks. Per the project's existing test style (test_freeze.py:50)
we monkeypatch ``HOME`` so JSONL discovery and notes don't touch the
user's real ``~/.claude``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from claude_sql import binding

# ---------------------------------------------------------------------------
# Skip-if-git-too-old gate
# ---------------------------------------------------------------------------


def _git_version_tuple() -> tuple[int, int, int] | None:
    """Parse ``git --version`` into (major, minor, patch); ``None`` if missing."""
    if shutil.which("git") is None:
        return None
    cp = subprocess.run(["git", "--version"], capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        return None
    # Output looks like ``git version 2.40.1`` (or ``2.40.1.windows.1``).
    parts = cp.stdout.strip().split()
    if len(parts) < 3:
        return None
    raw = parts[2].split(".")
    try:
        return int(raw[0]), int(raw[1]), int(raw[2]) if len(raw) > 2 else 0
    except (ValueError, IndexError):
        return None


_GIT_VERSION = _git_version_tuple()
_REQUIRES_GIT = pytest.mark.skipif(
    _GIT_VERSION is None or _GIT_VERSION < (2, 30, 0),
    reason="binding round-trip tests require git >=2.30 (interpret-trailers --parse)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``$HOME`` at a tmp dir so transcript discovery doesn't escape."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("CLAUDE_AGENT_RUNTIME", raising=False)
    monkeypatch.delenv("CLAUDE_SQL_BIND_COMMIT_MSG", raising=False)
    return fake_home


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Initialize a tmp_path git repo with one commit; return repo root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    cmds = [
        ["git", "init", "-q", "-b", "main", str(repo)],
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        ["git", "-C", str(repo), "config", "commit.gpgsign", "false"],
    ]
    for cmd in cmds:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert cp.returncode == 0, f"git setup failed: {cmd}: {cp.stderr}"
    seed = repo / "README.md"
    seed.write_text("seed\n", encoding="utf-8")
    cp = subprocess.run(
        ["git", "-C", str(repo), "add", "README.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    cp = subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "chore: seed"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    return repo


def _head_sha(repo: Path) -> str:
    cp = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    return cp.stdout.strip()


def _make_jsonl(path: Path, *, session_id: str = "01HSESSION") -> Path:
    """Write a tiny but realistic JSONL transcript for digesting/discovery."""
    lines = [
        {"type": "user", "sessionId": session_id, "uuid": "u1", "content": "hi"},
        {"type": "assistant", "sessionId": session_id, "uuid": "a1", "content": "hi back"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. compute_digest stability
# ---------------------------------------------------------------------------


def test_compute_digest_stable(tmp_path: Path) -> None:
    """Same bytes hash to the same hex twice in a row."""
    p = _make_jsonl(tmp_path / "session.jsonl")
    d1 = binding.compute_digest(p)
    d2 = binding.compute_digest(p)
    assert d1 == d2
    assert d1.startswith(binding.DIGEST_PREFIX)
    # ``sha256:`` + 64 hex chars
    assert len(d1) == len(binding.DIGEST_PREFIX) + 64


# ---------------------------------------------------------------------------
# 2. compute_digest sensitivity
# ---------------------------------------------------------------------------


def test_compute_digest_changes_on_byte_change(tmp_path: Path) -> None:
    """A single trailing newline append produces a different digest."""
    p = _make_jsonl(tmp_path / "session.jsonl")
    before = binding.compute_digest(p)
    with p.open("ab") as fh:
        fh.write(b"\n")
    after = binding.compute_digest(p)
    assert before != after


# ---------------------------------------------------------------------------
# 3. projectify canonical mapping
# ---------------------------------------------------------------------------


def test_projectify_canonical() -> None:
    """``/foo/bar`` → ``-foo-bar`` and a few other anchors."""
    assert binding.projectify(Path("/foo/bar")) == "-foo-bar"
    assert binding.projectify(Path("/efs/laith/workplace/claude-sql")) == (
        "-efs-laith-workplace-claude-sql"
    )
    assert binding.projectify(Path("/")) == "-"
    assert binding.projectify(Path("/a")) == "-a"


# ---------------------------------------------------------------------------
# 4. find_active_transcript picks latest mtime
# ---------------------------------------------------------------------------


def test_find_active_transcript_picks_latest_mtime(isolated_home: Path, tmp_path: Path) -> None:
    """Three transcripts, the newest mtime wins."""
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    projects_dir = isolated_home / ".claude" / "projects" / binding.projectify(cwd)
    projects_dir.mkdir(parents=True)
    a = _make_jsonl(projects_dir / "a.jsonl", session_id="A")
    b = _make_jsonl(projects_dir / "b.jsonl", session_id="B")
    c = _make_jsonl(projects_dir / "c.jsonl", session_id="C")
    # Force mtimes apart deterministically; ``time.time()`` resolution
    # is fine on Linux but we set explicit mtimes via ``os.utime`` to
    # avoid flake on fast filesystems.
    import os as _os

    base = time.time()
    _os.utime(a, (base, base))
    _os.utime(b, (base + 1, base + 1))
    _os.utime(c, (base + 2, base + 2))
    found = binding.find_active_transcript(cwd)
    assert found == c
    # Sanity: empty projects dir → None.
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    assert binding.find_active_transcript(empty_cwd) is None


# ---------------------------------------------------------------------------
# 5. write_trailer roundtrip
# ---------------------------------------------------------------------------


@_REQUIRES_GIT
def test_write_trailer_roundtrip(tmp_path: Path) -> None:
    """Write the three trailers, parse with git interpret-trailers --parse."""
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text("feat: example commit\n\nbody paragraph.\n", encoding="utf-8")
    bnd = binding.TranscriptBinding(
        digest="sha256:" + "a" * 64,
        uri="file:///tmp/session.jsonl",
        agent_runtime="claude-code/0.42.1",
        transcript_id="01HSESSION",
        captured_at="2026-05-09T03:51:08+00:00",
    )
    binding.write_trailer(msg, bnd)
    cp = subprocess.run(
        ["git", "interpret-trailers", "--parse"],
        input=msg.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    pairs = binding._parse_trailer_block(cp.stdout)
    assert pairs[binding.TRAILER_DIGEST] == bnd.digest
    assert pairs[binding.TRAILER_URI] == bnd.uri
    assert pairs[binding.TRAILER_RUNTIME] == bnd.agent_runtime


# ---------------------------------------------------------------------------
# 6. write_note + read_note roundtrip
# ---------------------------------------------------------------------------


@_REQUIRES_GIT
def test_write_note_roundtrip(git_repo: Path) -> None:
    """Round-trip a binding through refs/notes/transcripts."""
    bnd = binding.TranscriptBinding(
        digest="sha256:" + "b" * 64,
        uri="file:///tmp/session.jsonl",
        agent_runtime="claude-code/0.42.1",
        transcript_id="01HSESSION",
        captured_at="2026-05-09T03:51:08+00:00",
    )
    sha = _head_sha(git_repo)
    binding.write_note(git_repo, sha, bnd)
    # raw git read
    cp = subprocess.run(
        ["git", "-C", str(git_repo), "notes", "--ref=transcripts", "show", sha],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    payload = json.loads(cp.stdout.strip())
    assert payload["digest"] == bnd.digest
    assert payload["uri"] == bnd.uri
    assert payload["agent_runtime"] == bnd.agent_runtime
    assert payload["transcript_id"] == bnd.transcript_id
    assert payload["captured_at"] == bnd.captured_at
    # binding-side reader
    parsed = binding.read_note(sha, repo=git_repo)
    assert parsed is not None
    assert parsed == bnd


# ---------------------------------------------------------------------------
# 7. resolve precedence — trailer wins, note supplements forensic fields
# ---------------------------------------------------------------------------


@_REQUIRES_GIT
def test_resolve_precedence_trailer_wins(git_repo: Path) -> None:
    """Both surfaces present and digest-aligned: trailer wire wins; note
    contributes transcript_id + captured_at to the merged binding."""
    bnd = binding.TranscriptBinding(
        digest="sha256:" + "c" * 64,
        uri="file:///tmp/session.jsonl",
        agent_runtime="claude-code/0.42.1",
        transcript_id="01HSESSION",
        captured_at="2026-05-09T03:51:08+00:00",
    )
    # Amend HEAD to carry the trailers, then write note.
    msg_path = git_repo / ".git" / "COMMIT_EDITMSG"
    msg_path.write_text("chore: seed\n", encoding="utf-8")
    binding.write_trailer(msg_path, bnd)
    cp = subprocess.run(
        [
            "git",
            "-C",
            str(git_repo),
            "commit",
            "--amend",
            "-q",
            "-F",
            str(msg_path),
            "--no-verify",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    sha = _head_sha(git_repo)
    binding.write_note(git_repo, sha, bnd)
    resolved = binding.resolve_commit_to_transcript(sha, repo=git_repo)
    assert resolved.digest == bnd.digest
    assert resolved.uri == bnd.uri
    assert resolved.agent_runtime == bnd.agent_runtime
    assert resolved.transcript_id == bnd.transcript_id
    assert resolved.captured_at == bnd.captured_at


# ---------------------------------------------------------------------------
# 8. resolve mismatch — loud failure on disagreement
# ---------------------------------------------------------------------------


@_REQUIRES_GIT
def test_resolve_mismatch_loud(git_repo: Path) -> None:
    """Trailer and note carry different digests → BindingMismatchError."""
    trailer_bnd = binding.TranscriptBinding(
        digest="sha256:" + "1" * 64,
        uri="file:///tmp/trailer.jsonl",
        agent_runtime="claude-code/0.42.1",
        transcript_id="",
        captured_at="",
    )
    note_bnd = binding.TranscriptBinding(
        digest="sha256:" + "2" * 64,
        uri="file:///tmp/note.jsonl",
        agent_runtime="claude-code/0.42.1",
        transcript_id="01HSESSION",
        captured_at="2026-05-09T03:51:08+00:00",
    )
    msg_path = git_repo / ".git" / "COMMIT_EDITMSG"
    msg_path.write_text("chore: seed\n", encoding="utf-8")
    binding.write_trailer(msg_path, trailer_bnd)
    cp = subprocess.run(
        [
            "git",
            "-C",
            str(git_repo),
            "commit",
            "--amend",
            "-q",
            "-F",
            str(msg_path),
            "--no-verify",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    sha = _head_sha(git_repo)
    binding.write_note(git_repo, sha, note_bnd)
    with pytest.raises(binding.BindingMismatchError) as exc_info:
        binding.resolve_commit_to_transcript(sha, repo=git_repo)
    err = exc_info.value
    assert err.trailer is not None
    assert err.note is not None
    assert err.trailer.digest != err.note.digest


# ---------------------------------------------------------------------------
# 9. resolve no-binding — LookupError → CLI exits 2
# ---------------------------------------------------------------------------


@_REQUIRES_GIT
def test_resolve_no_binding_lookup_error(git_repo: Path) -> None:
    """Fresh commit with no trailer + no note → LookupError."""
    sha = _head_sha(git_repo)
    with pytest.raises(LookupError):
        binding.resolve_commit_to_transcript(sha, repo=git_repo)


# ---------------------------------------------------------------------------
# 10. bind CLI dry-run
# ---------------------------------------------------------------------------


@_REQUIRES_GIT
def test_bind_cli_dry_run(
    isolated_home: Path,
    git_repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``claude-sql bind --dry-run`` against a tmp repo prints the plan."""
    # Plant a transcript in the projectified location for git_repo.
    proj_dir = isolated_home / ".claude" / "projects" / binding.projectify(git_repo)
    proj_dir.mkdir(parents=True)
    _make_jsonl(proj_dir / "session.jsonl")
    monkeypatch.setenv("CLAUDE_AGENT_RUNTIME", "claude-code/test-1.0.0")
    monkeypatch.chdir(git_repo)

    from claude_sql.cli import bind_cmd

    bind_cmd(repo=git_repo, dry_run=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["bound"] is False
    assert payload["dry_run"] is True
    assert "binding" in payload
    assert payload["binding"]["digest"].startswith("sha256:")
    assert payload["binding"]["uri"].startswith("file://")
    assert payload["binding"]["agent_runtime"] == "claude-code/test-1.0.0"
    assert payload["repo"] == str(git_repo)
    # No real writes happened: the .git/refs/notes ref doesn't exist.
    notes_ref = git_repo / ".git" / "refs" / "notes" / "transcripts"
    assert not notes_ref.exists()


# ---------------------------------------------------------------------------
# Bonus coverage — failure modes the public API has to handle
# ---------------------------------------------------------------------------


def test_compute_digest_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        binding.compute_digest(tmp_path / "nonexistent.jsonl")


def test_resolve_all_sources_returns_typed_dict(git_repo: Path) -> None:
    """``resolve_all_sources`` never raises on absence; returns both as None."""
    sha = _head_sha(git_repo)
    sources = binding.resolve_all_sources(sha, repo=git_repo)
    assert sources["trailer"] is None
    assert sources["note"] is None


def test_resolve_commit_to_transcript_all_sources_true_rejected(
    git_repo: Path,
) -> None:
    """The split API rejects ``all_sources=True`` so callers reach for the
    statically-typed ``resolve_all_sources`` helper instead."""
    sha = _head_sha(git_repo)
    with pytest.raises(TypeError, match="resolve_all_sources"):
        binding.resolve_commit_to_transcript(sha, repo=git_repo, all_sources=True)


def test_detect_agent_runtime_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default runtime is the documented ``claude-code/unknown`` sentinel."""
    monkeypatch.delenv("CLAUDE_AGENT_RUNTIME", raising=False)
    assert binding.detect_agent_runtime() == "claude-code/unknown"
    monkeypatch.setenv("CLAUDE_AGENT_RUNTIME", "cursor/0.42")
    assert binding.detect_agent_runtime() == "cursor/0.42"


# Belt & suspenders for the import path so the CLI module isn't lazy-broken
# (this is the kind of regression where importing `binding` from `cli`
# silently breaks because of a circular import).
def test_binding_module_imports_via_cli() -> None:
    from claude_sql.cli import bind_cmd, resolve_cmd

    assert callable(bind_cmd)
    assert callable(resolve_cmd)
    assert "claude_sql.cli" in sys.modules
