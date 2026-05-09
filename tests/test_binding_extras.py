"""Coverage top-up for :mod:`claude_sql.binding`.

Targets the failure-mode branches the existing test_binding.py doesn't
already hit:

* :class:`GitInvocationError` constructor (lines 125-131).
* :func:`_read_transcript_id` empty / non-dict / parse-error fallbacks
  (lines 302, 305, 310-314).
* :func:`_resolve_repo` git failure (lines 385-393).
* :func:`write_trailer` and :func:`write_note` git failures (lines
  434, 469).
* :func:`_parse_trailer_block` skip-empty / skip-malformed / dedupe
  paths (lines 500, 503, 506).
* :func:`read_trailer` git failures on either subprocess (lines 544,
  551).
* :func:`_from_note_payload` non-string typing guards (line 584).
* :func:`read_note` non-absence git failure / empty body / non-JSON /
  non-dict (lines 632, 635, 638-639, 641).
* :func:`resolve_commit_to_transcript` trailer-only and note-only
  branches (lines 735-740).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from claude_sql import binding

# ---------------------------------------------------------------------------
# Skip-if-git-too-old gate (mirrors test_binding.py)
# ---------------------------------------------------------------------------


def _git_version_tuple() -> tuple[int, int, int] | None:
    if shutil.which("git") is None:
        return None
    cp = subprocess.run(["git", "--version"], capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        return None
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
    reason="git >=2.30 required for interpret-trailers --parse",
)


# ---------------------------------------------------------------------------
# GitInvocationError __init__ (lines 125-131)
# ---------------------------------------------------------------------------


def test_git_invocation_error_carries_argv_and_exit_metadata() -> None:
    """The constructor surfaces argv + returncode + stdout + stderr."""
    exc = binding.GitInvocationError(
        argv=["git", "log"],
        returncode=128,
        stdout="some stdout",
        stderr="fatal: bad sha\n",
    )
    assert exc.argv == ["git", "log"]
    assert exc.returncode == 128
    assert exc.stdout == "some stdout"
    assert exc.stderr == "fatal: bad sha\n"
    msg = str(exc)
    assert "exit 128" in msg
    assert "git log" in msg
    assert "fatal: bad sha" in msg


# ---------------------------------------------------------------------------
# _read_transcript_id fallbacks (lines 302, 305, 310-314)
# ---------------------------------------------------------------------------


def test_read_transcript_id_empty_first_line_falls_back_to_stem(tmp_path: Path) -> None:
    """An empty first line falls through to the file stem (line 302)."""
    p = tmp_path / "empty-first.jsonl"
    p.write_text('\n{"sessionId":"actual"}\n', encoding="utf-8")
    assert binding._read_transcript_id(p) == "empty-first"


def test_read_transcript_id_non_dict_record_falls_back_to_stem(tmp_path: Path) -> None:
    """A non-dict first line falls through to the file stem (line 305)."""
    p = tmp_path / "list-line.jsonl"
    p.write_text("[1, 2, 3]\n", encoding="utf-8")
    assert binding._read_transcript_id(p) == "list-line"


def test_read_transcript_id_parse_error_falls_back_to_stem(tmp_path: Path) -> None:
    """An undecodable first line falls back via the JSONDecodeError path."""
    p = tmp_path / "bad.jsonl"
    p.write_text("not-valid-json\n", encoding="utf-8")
    assert binding._read_transcript_id(p) == "bad"


def test_read_transcript_id_picks_alternate_id_keys(tmp_path: Path) -> None:
    """``session_id`` and ``uuid`` keys are also recognized (line 306-309)."""
    p1 = tmp_path / "snake.jsonl"
    p1.write_text('{"session_id":"snake-id"}\n', encoding="utf-8")
    assert binding._read_transcript_id(p1) == "snake-id"
    p2 = tmp_path / "uuid.jsonl"
    p2.write_text('{"uuid":"uuid-id"}\n', encoding="utf-8")
    assert binding._read_transcript_id(p2) == "uuid-id"


# ---------------------------------------------------------------------------
# _resolve_repo failure (lines 385-393)
# ---------------------------------------------------------------------------


def test_resolve_repo_raises_outside_a_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_resolve_repo(None)`` outside a repo raises ``GitInvocationError``."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(binding.GitInvocationError):
        binding._resolve_repo(None)


def test_resolve_repo_passes_through_explicit_path(tmp_path: Path) -> None:
    """An explicit Path is returned resolved without subprocessing."""
    out = binding._resolve_repo(tmp_path)
    assert out == tmp_path.resolve()


# ---------------------------------------------------------------------------
# write_trailer / write_note git failures (lines 434, 469)
# ---------------------------------------------------------------------------


def test_write_trailer_raises_on_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``git interpret-trailers`` exits non-zero, raise ``GitInvocationError``."""
    msg = tmp_path / "MSG"
    msg.write_text("subject\n", encoding="utf-8")

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=argv, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(binding, "_run_git", fake_run)
    bnd = binding.TranscriptBinding(
        digest="sha256:" + "a" * 64,
        uri="file:///tmp/x.jsonl",
        agent_runtime="claude-code/0.1.0",
        transcript_id="",
        captured_at="",
    )
    with pytest.raises(binding.GitInvocationError):
        binding.write_trailer(msg, bnd)


def test_write_note_raises_on_git_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero exit on ``git notes add`` raises ``GitInvocationError``."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv, returncode=1, stdout="", stderr="cannot add note"
        )

    monkeypatch.setattr(binding, "_run_git", fake_run)
    bnd = binding.TranscriptBinding(
        digest="sha256:" + "b" * 64,
        uri="file:///tmp/y.jsonl",
        agent_runtime="claude-code/0.1.0",
        transcript_id="",
        captured_at="",
    )
    with pytest.raises(binding.GitInvocationError):
        binding.write_note(tmp_path, "deadbeef", bnd)


# ---------------------------------------------------------------------------
# _parse_trailer_block edge cases (lines 500, 503, 506)
# ---------------------------------------------------------------------------


def test_parse_trailer_block_skips_blank_and_malformed_lines() -> None:
    """Blank lines and lines that don't match the trailer regex are dropped."""
    raw = (
        "Foo: bar\n"
        "\n"  # blank line — line 500 skip
        "this-is-not-a-trailer-line\n"  # no colon — line 503 regex miss
        "Baz: qux\n"
    )
    out = binding._parse_trailer_block(raw)
    assert out == {"Foo": "bar", "Baz": "qux"}


def test_parse_trailer_block_dedupes_by_lowered_key() -> None:
    """First occurrence of a case-folded key wins (line 506)."""
    raw = "Foo: first\nFOO: second\n"
    out = binding._parse_trailer_block(raw)
    assert out == {"Foo": "first"}


# ---------------------------------------------------------------------------
# read_trailer git failures (lines 544, 551)
# ---------------------------------------------------------------------------


def test_read_trailer_raises_on_git_log_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``git log`` itself fails, ``read_trailer`` re-raises."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=argv, returncode=128, stdout="", stderr="bad sha")

    monkeypatch.setattr(binding, "_run_git", fake_run)
    with pytest.raises(binding.GitInvocationError):
        binding.read_trailer("nope", repo=tmp_path)


def test_read_trailer_raises_on_interpret_trailers_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the trailer-parse subprocess fails, ``read_trailer`` re-raises."""
    calls = {"n": 0}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        if calls["n"] == 1:
            # git log succeeds, returning a body.
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="subject\n\nbody\n", stderr=""
            )
        # interpret-trailers fails on the second call.
        return subprocess.CompletedProcess(
            args=argv, returncode=2, stdout="", stderr="parse failed"
        )

    monkeypatch.setattr(binding, "_run_git", fake_run)
    with pytest.raises(binding.GitInvocationError):
        binding.read_trailer("anything", repo=tmp_path)


# ---------------------------------------------------------------------------
# _from_note_payload non-string fields (line 584)
# ---------------------------------------------------------------------------


def test_from_note_payload_rejects_non_string_required_fields() -> None:
    """Required wire-format fields must be strings or the parser returns None."""
    out = binding._from_note_payload({"digest": 123, "uri": "x", "agent_runtime": "y"})
    assert out is None
    out = binding._from_note_payload({"digest": "x", "uri": None, "agent_runtime": "y"})
    assert out is None


def test_from_note_payload_coerces_optional_fields_to_empty_when_non_string() -> None:
    """``transcript_id`` / ``captured_at`` non-strings coerce to empty (line 591-592)."""
    out = binding._from_note_payload(
        {
            "digest": "sha256:abc",
            "uri": "file:///x.jsonl",
            "agent_runtime": "claude/0",
            "transcript_id": 42,
            "captured_at": ["bad"],
        }
    )
    assert out is not None
    assert out.transcript_id == ""
    assert out.captured_at == ""


# ---------------------------------------------------------------------------
# read_note non-absence failures + body shapes (lines 632, 635, 638-639, 641)
# ---------------------------------------------------------------------------


def test_read_note_raises_on_unrelated_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-"no note" stderr re-raises (line 632)."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv,
            returncode=128,
            stdout="",
            stderr="fatal: ambiguous argument 'deadbeef'",
        )

    monkeypatch.setattr(binding, "_run_git", fake_run)
    with pytest.raises(binding.GitInvocationError):
        binding.read_note("deadbeef", repo=tmp_path)


def test_read_note_returns_none_when_no_note_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "no note found" stderr is the absence signal."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv, returncode=1, stdout="", stderr="error: no note found for object x"
        )

    monkeypatch.setattr(binding, "_run_git", fake_run)
    assert binding.read_note("anything", repo=tmp_path) is None


def test_read_note_returns_none_on_empty_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful but empty note body returns None (line 635)."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="\n", stderr="")

    monkeypatch.setattr(binding, "_run_git", fake_run)
    assert binding.read_note("anything", repo=tmp_path) is None


def test_read_note_returns_none_on_non_json_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A note that isn't JSON returns None (lines 638-639)."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="not-json", stderr="")

    monkeypatch.setattr(binding, "_run_git", fake_run)
    assert binding.read_note("anything", repo=tmp_path) is None


def test_read_note_returns_none_when_payload_not_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON list (not dict) at the note body returns None (line 641)."""

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="[1,2,3]", stderr="")

    monkeypatch.setattr(binding, "_run_git", fake_run)
    assert binding.read_note("anything", repo=tmp_path) is None


# ---------------------------------------------------------------------------
# resolve_commit_to_transcript trailer-only and note-only branches (735-740)
# ---------------------------------------------------------------------------


def test_resolve_returns_trailer_when_only_trailer_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trailer-only path returns the trailer binding directly (line 735-736)."""
    trailer = binding.TranscriptBinding(
        digest="sha256:" + "a" * 64,
        uri="file:///tmp/t.jsonl",
        agent_runtime="claude-code/0.1",
        transcript_id="",
        captured_at="",
    )
    monkeypatch.setattr(binding, "_resolve_repo", lambda _r: tmp_path)
    monkeypatch.setattr(binding, "read_trailer", lambda sha, repo: trailer)
    monkeypatch.setattr(binding, "read_note", lambda sha, repo: None)
    out = binding.resolve_commit_to_transcript("deadbeef", repo=tmp_path)
    assert out is trailer


def test_resolve_returns_note_when_only_note_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Note-only path falls through to the note binding (line 738-740)."""
    note = binding.TranscriptBinding(
        digest="sha256:" + "b" * 64,
        uri="file:///tmp/n.jsonl",
        agent_runtime="claude-code/0.1",
        transcript_id="01HID",
        captured_at="2026-05-09T00:00:00+00:00",
    )
    monkeypatch.setattr(binding, "_resolve_repo", lambda _r: tmp_path)
    monkeypatch.setattr(binding, "read_trailer", lambda sha, repo: None)
    monkeypatch.setattr(binding, "read_note", lambda sha, repo: note)
    out = binding.resolve_commit_to_transcript("deadbeef", repo=tmp_path)
    assert out is note


# ---------------------------------------------------------------------------
# Smoke: GitInvocationError surfaces when reading the trailer of a missing sha
# ---------------------------------------------------------------------------


@_REQUIRES_GIT
def test_read_trailer_on_missing_sha_raises_git_error(tmp_path: Path) -> None:
    """A real git repo + a bogus SHA reaches the line-544 raise."""
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
        assert cp.returncode == 0, cp.stderr
    seed = repo / "f.txt"
    seed.write_text("seed\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "f.txt"], capture_output=True, text=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "chore: seed"],
        capture_output=True,
        text=True,
        check=True,
    )
    with pytest.raises(binding.GitInvocationError):
        binding.read_trailer("0" * 40, repo=repo)
