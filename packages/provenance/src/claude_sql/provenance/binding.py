"""Transcript-to-PR binding via commit trailers + git notes.

Implements RFC 0001 (`docs/rfc/0001-transcript-pr-binding.md`). Pure-stdlib
helpers for writing and reading the three-trailer + JSON-note convention
that points a merged commit at the AI-agent transcript that produced it.

Design boundaries:

* No new dependencies. ``hashlib``, ``subprocess``, ``pathlib``,
  ``dataclasses``, ``json``, ``os``, ``re`` — all stdlib.
* Subprocess to ``git`` only via ``subprocess.run([...], check=False,
  capture_output=True, text=True)``; we inspect ``returncode`` and
  ``stderr`` explicitly. No ``shell=True``; no ``check=True``. The
  caller's branch is responsible for raising; this keeps the helpers
  composable.
* All public functions carry full type hints and pass ``ty`` strict
  mode. The dataclass is ``frozen=True`` so a ``TranscriptBinding`` is
  hashable and safe to share across threads.
* Every function is independently unit-testable against a real
  ``git init`` repository under ``tmp_path``. No Bedrock, no live
  filesystem outside the JSONL discovery helpers.

The wire format (three commit trailers + a JSON-shaped git note under
``refs/notes/transcripts``) is host-agnostic: it survives squash-merge,
GitHub outages, and the eventual death of any one transcript host. See
RFC 0001 §Specification for the full contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, TypedDict


class BindingSources(TypedDict):
    """Diagnostic dict shape returned by :func:`resolve_all_sources`.

    Defined as a ``TypedDict`` rather than a ``dataclass`` so the
    return is ``json``-serializable as-is and so callers can
    structurally narrow on ``trailer`` / ``note`` keys without
    importing the type. The forward references resolve under
    ``from __future__ import annotations``; ``TranscriptBinding`` is
    declared further down in this module.
    """

    trailer: TranscriptBinding | None
    note: TranscriptBinding | None


# ---------------------------------------------------------------------------
# Public constants — wire format. RFC 0001 §Specification.
# ---------------------------------------------------------------------------

DIGEST_PREFIX: str = "sha256:"
"""Prefix on every ``Claude-Transcript-Digest:`` value. The rest is
a 64-character hex digest from ``hashlib.sha256``."""

NOTES_REF: str = "transcripts"
"""Short ref name passed to ``git notes --ref=...``. Resolves to
``refs/notes/transcripts`` per ``git-notes(1)``."""

TRAILER_DIGEST: str = "Claude-Transcript-Digest"
"""Trailer key carrying the SHA-256 digest of the JSONL transcript."""

TRAILER_URI: str = "Claude-Transcript-URI"
"""Trailer key carrying the URI where the transcript can be retrieved.

One of ``file://<path>``, ``s3://<bucket>/<key>``, or
``git-notes://<refname>``. The reference implementation in v0 only
emits ``file://``; the other two are spec-only entry points for future
emitters and readers per RFC 0001 §Specification.URI scheme.
"""

TRAILER_RUNTIME: str = "Claude-Agent-Runtime"
"""Trailer key carrying the agent runtime identifier (``vendor/version``)."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BindingMismatchError(RuntimeError):
    """Raised when trailer and note disagree about the bound transcript.

    Carries both surfaces so callers can render a useful error message
    (CLI surfaces this as exit 70 with a structured payload). The
    ``trailer`` and ``note`` attributes are the parsed bindings; either
    may be ``None`` if only one side was present and the other was
    malformed beyond rescue.
    """

    def __init__(
        self,
        message: str,
        *,
        trailer: TranscriptBinding | None,
        note: TranscriptBinding | None,
    ) -> None:
        super().__init__(message)
        self.trailer = trailer
        self.note = note


class GitInvocationError(RuntimeError):
    """Raised when a ``git`` subprocess returns a non-zero exit code.

    Wraps ``returncode``, ``stdout``, and ``stderr`` so the caller can
    classify the failure (e.g., commit not found vs. notes ref empty
    vs. dirty working tree) without re-running ``git``.
    """

    def __init__(
        self,
        argv: list[str],
        *,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        super().__init__(
            f"git command failed (exit {returncode}): {' '.join(argv)}\n{stderr.strip()}"
        )
        self.argv = argv
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptBinding:
    """One commit's pointer to the transcript that authored it.

    All fields carry the wire-format strings from RFC 0001
    §Specification — no parsing or validation beyond what the writer
    already enforced. Read-side callers wanting structured access
    (e.g., parsing the URI scheme) operate on the string fields
    directly.
    """

    digest: str
    """``sha256:<64-hex-chars>`` — same value as the
    ``Claude-Transcript-Digest:`` trailer."""

    uri: str
    """``file://...`` | ``s3://...`` | ``git-notes://...`` — same value
    as the ``Claude-Transcript-URI:`` trailer."""

    agent_runtime: str
    """``vendor/version`` — same value as the
    ``Claude-Agent-Runtime:`` trailer (e.g., ``claude-code/0.42.1``)."""

    transcript_id: str
    """Opaque session identifier from inside the JSONL artifact (e.g.,
    Claude Code's session-id field). Lives in the note only; trailers
    don't carry it because it's forensic metadata, not human-readable
    provenance."""

    captured_at: str
    """ISO-8601 UTC timestamp for when the binding was written. Note-only."""

    # Class-level reserved JSON keys — mirrored in ``to_note_payload``
    # and ``_from_note_payload``. Anything unknown is ignored on read,
    # forward-compatibility per RFC 0001 §Compatibility.
    _NOTE_KEYS: ClassVar[tuple[str, ...]] = (
        "digest",
        "uri",
        "agent_runtime",
        "transcript_id",
        "captured_at",
    )

    def to_note_payload(self) -> dict[str, str]:
        """Serialize to the JSON shape stored in ``refs/notes/transcripts``."""
        return {
            "uri": self.uri,
            "digest": self.digest,
            "agent_runtime": self.agent_runtime,
            "transcript_id": self.transcript_id,
            "captured_at": self.captured_at,
        }

    def to_dict(self) -> dict[str, str]:
        """Plain dict view for ``emit_json`` / structured CLI output."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure helpers (no subprocess, no I/O beyond reading a path)
# ---------------------------------------------------------------------------


def projectify(cwd: Path) -> str:
    """Mirror Claude Code's projectified-cwd convention.

    Claude Code stores transcripts under
    ``~/.claude/projects/<projectified>/`` where ``<projectified>`` is
    the absolute path of the working directory with leading ``/``
    stripped, every remaining ``/`` replaced with ``-``, and a leading
    ``-`` re-prepended.

    Examples:
        ``/foo/bar`` → ``-foo-bar``
        ``/efs/laith/workplace/claude-sql`` →
            ``-efs-laith-workplace-claude-sql``
        ``/`` → ``-``

    The result is a relative path component, never absolute. Callers
    join it with ``~/.claude/projects/`` to form the discovery root.
    """
    text = str(cwd)
    # ``Path("/").as_posix()`` returns "/"; treat root specially so we
    # don't end up with the empty string after stripping.
    if text == "/":
        return "-"
    stripped = text.lstrip("/")
    return "-" + stripped.replace("/", "-")


def compute_digest(jsonl_path: Path) -> str:
    """SHA-256 hex digest of the JSONL artifact's raw bytes.

    Returns the digest with the ``sha256:`` algorithm prefix per RFC
    0001. Raw-bytes hashing — *not* canonical-JSON-per-line —
    deliberately so the digest is recomputable in one line of any
    language and matches the artifact reviewers will inspect. See RFC
    0001 §Security.Digest determinism for the trade-off.

    Raises ``FileNotFoundError`` if ``jsonl_path`` doesn't exist; the
    caller (typically ``write_trailer``-side code) handles this — there
    is no recovery, the binding can't be written without a transcript.
    """
    digest = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    return f"{DIGEST_PREFIX}{digest}"


def detect_agent_runtime() -> str:
    """Identify the agent runtime that emitted the transcript.

    Reads ``CLAUDE_AGENT_RUNTIME`` from the environment if set (the
    first emitter convention is for the runtime itself to set this
    before invoking the bind hook). Falls back to ``claude-code/unknown``
    when unset; any future runtime gets a different default by setting
    its own env var name (or, more usefully, by setting
    ``CLAUDE_AGENT_RUNTIME=cursor/0.42`` from inside its hook wiring).
    """
    explicit = os.environ.get("CLAUDE_AGENT_RUNTIME")
    if explicit:
        return explicit.strip()
    return "claude-code/unknown"


def find_active_transcript(cwd: Path) -> Path | None:
    """Resolve the active transcript JSONL for ``cwd``, if any.

    Lists ``~/.claude/projects/<projectified-cwd>/*.jsonl`` and returns
    the most recently-modified file. Returns ``None`` when the
    projects directory doesn't exist or contains no JSONL — for
    example, the user is committing from a directory that hasn't been
    used inside Claude Code yet, or has explicitly opted out of
    transcript persistence. The caller (``claude-sql bind``) treats
    ``None`` as "no transcript to bind, exit cleanly with no trailer."
    """
    home = Path(os.path.expanduser("~"))
    projects_root = home / ".claude" / "projects" / projectify(cwd)
    if not projects_root.is_dir():
        return None
    candidates = sorted(
        projects_root.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_transcript_id(jsonl_path: Path) -> str:
    """Best-effort transcript-id extraction from the first JSON line.

    Claude Code's JSONL emits a ``sessionId`` (or ``session_id``) field
    on each line. We read just the first line — same convention as the
    on-disk JSONL invariant — and parse it to find the id. Falls back
    to the file's basename (stem) if no id is present, so the field
    is always populated for the note.

    Raises nothing: a malformed first line falls through to the
    basename fallback. Forensic data; we don't fail the binding over
    a parse error here.
    """
    try:
        with jsonl_path.open(encoding="utf-8") as fh:
            first_line = fh.readline()
        if not first_line.strip():
            return jsonl_path.stem
        record = json.loads(first_line)
        if not isinstance(record, dict):
            return jsonl_path.stem
        for key in ("sessionId", "session_id", "uuid"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    except (OSError, json.JSONDecodeError):
        # Fallthrough: forensic metadata, we'd rather have a stable
        # filename-derived id than a ``KeyError`` on bind.
        pass
    return jsonl_path.stem


def build_binding(
    *,
    transcript_path: Path,
    runtime: str | None = None,
    captured_at: str | None = None,
) -> TranscriptBinding:
    """Compose a fully-populated ``TranscriptBinding`` for ``transcript_path``.

    Convenience constructor for callers that have a JSONL path and
    want every field filled. ``runtime`` defaults to
    :func:`detect_agent_runtime`; ``captured_at`` defaults to
    ``datetime.now(UTC).isoformat()``. Both are accepted as
    overrides primarily for tests that need deterministic output.
    """
    digest = compute_digest(transcript_path)
    uri = transcript_path.resolve().as_uri()
    transcript_id = _read_transcript_id(transcript_path)
    if runtime is None:
        runtime = detect_agent_runtime()
    if captured_at is None:
        captured_at = datetime.now(UTC).isoformat()
    return TranscriptBinding(
        digest=digest,
        uri=uri,
        agent_runtime=runtime,
        transcript_id=transcript_id,
        captured_at=captured_at,
    )


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run_git(
    argv: list[str],
    *,
    cwd: Path | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a ``git`` subprocess with our standard knobs.

    All git invocations in this module funnel through here so the
    options are uniform: ``check=False`` (we inspect ``returncode``
    explicitly), ``capture_output=True`` (we always want stdout +
    stderr captured), ``text=True`` (we treat git output as UTF-8
    text). Never ``shell=True``.
    """
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        input=stdin,
    )


def _resolve_repo(repo: Path | None) -> Path:
    """Resolve the repo root.

    ``None`` means ``git rev-parse --show-toplevel`` from the current
    process cwd. An explicit ``Path`` is passed through verbatim. The
    resolved path is always absolute.
    """
    if repo is not None:
        return repo.resolve()
    cp = _run_git(["git", "rev-parse", "--show-toplevel"])
    if cp.returncode != 0:
        raise GitInvocationError(
            ["git", "rev-parse", "--show-toplevel"],
            returncode=cp.returncode,
            stdout=cp.stdout,
            stderr=cp.stderr,
        )
    return Path(cp.stdout.strip()).resolve()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_trailer(commit_msg_path: Path, binding: TranscriptBinding) -> None:
    """Append the three trailers to ``commit_msg_path`` in place.

    Three calls to ``git interpret-trailers --in-place``, each writing
    one ``Key: Value`` pair. We pass ``--if-exists replace`` so the
    operation is idempotent under ``git commit --amend`` and repeated
    ``prepare-commit-msg`` invocations — re-running ``bind`` on an
    amended commit replaces the existing trailer rather than
    appending a duplicate.

    Order of invocation matches the order in RFC 0001's wire format:
    digest, URI, runtime. ``git interpret-trailers`` ensures a single
    blank line precedes the trailer block, even if the message had
    none.
    """
    pairs: tuple[tuple[str, str], ...] = (
        (TRAILER_DIGEST, binding.digest),
        (TRAILER_URI, binding.uri),
        (TRAILER_RUNTIME, binding.agent_runtime),
    )
    for key, value in pairs:
        argv = [
            "git",
            "interpret-trailers",
            "--in-place",
            "--if-exists",
            "replace",
            "--trailer",
            f"{key}: {value}",
            str(commit_msg_path),
        ]
        cp = _run_git(argv)
        if cp.returncode != 0:
            raise GitInvocationError(
                argv,
                returncode=cp.returncode,
                stdout=cp.stdout,
                stderr=cp.stderr,
            )


def write_note(repo: Path, commit_sha: str, binding: TranscriptBinding) -> None:
    """Write ``binding`` as a JSON note under ``refs/notes/transcripts``.

    Uses ``git notes --ref=transcripts add -f -m '<json>' <sha>``. The
    ``-f`` (force) flag overwrites an existing note for the same
    commit, which is the right semantics for re-runs (e.g., amend);
    re-running the bind step with a different transcript replaces
    the entry instead of failing with "note already exists."

    The JSON is single-line (no ``indent``) so the note body stays
    one line — matches `git`'s preference for compact note formats.
    """
    payload = json.dumps(binding.to_note_payload(), ensure_ascii=False, separators=(",", ":"))
    argv = [
        "git",
        "-C",
        str(repo),
        "notes",
        f"--ref={NOTES_REF}",
        "add",
        "-f",
        "-m",
        payload,
        commit_sha,
    ]
    cp = _run_git(argv)
    if cp.returncode != 0:
        raise GitInvocationError(
            argv,
            returncode=cp.returncode,
            stdout=cp.stdout,
            stderr=cp.stderr,
        )


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


_TRAILER_LINE_RE: re.Pattern[str] = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):\s*(.*)$")


def _parse_trailer_block(text: str) -> dict[str, str]:
    """Parse the ``key: value`` pairs that ``git interpret-trailers --parse`` emits.

    The output format is one trailer per line, ``Key: Value``. We
    case-fold keys for the lookup map but preserve the original
    capitalization in the returned dict so callers see exactly what
    the writer emitted. When a key appears multiple times, the *first*
    occurrence wins — matches RFC 0001 §Compatibility.Trailer survival
    (rebase / fixup squash duplication tolerance).
    """
    parsed: dict[str, str] = {}
    seen_lower: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TRAILER_LINE_RE.match(line)
        if match is None:
            continue
        key, value = match.group(1), match.group(2).strip()
        if key.lower() in seen_lower:
            continue
        parsed[key] = value
        seen_lower.add(key.lower())
    return parsed


def read_trailer(
    commit_sha: str,
    *,
    repo: Path | None = None,
) -> TranscriptBinding | None:
    """Read the binding trailer set off a commit's message, if present.

    Subprocess sequence:

    1. ``git -C <repo> log --format=%B -1 <sha>`` — the commit message body.
    2. ``git interpret-trailers --parse`` (with the message piped on stdin)
       — emits one line per trailer.

    Returns ``None`` when:

    * The commit doesn't exist (``git log`` fails — surfaced as
      ``GitInvocationError`` and re-raised; callers map to exit 65).
    * The message has no trailers at all.
    * The trailer block is present but missing one or more of the
      three required keys.

    Returns the populated :class:`TranscriptBinding` when all three
    trailers are present. ``transcript_id`` and ``captured_at`` are
    note-only fields — they default to empty strings on the
    trailer-only path, and resolution callers wanting those should
    fall back to the note. This matches RFC 0001 §Specification.Resolution
    precedence: trailer first, note fallback.
    """
    repo_path = _resolve_repo(repo)
    log_argv = ["git", "-C", str(repo_path), "log", "--format=%B", "-1", commit_sha]
    cp = _run_git(log_argv)
    if cp.returncode != 0:
        raise GitInvocationError(
            log_argv, returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr
        )
    message = cp.stdout
    parse_argv = ["git", "interpret-trailers", "--parse"]
    parsed_cp = _run_git(parse_argv, stdin=message)
    if parsed_cp.returncode != 0:
        raise GitInvocationError(
            parse_argv,
            returncode=parsed_cp.returncode,
            stdout=parsed_cp.stdout,
            stderr=parsed_cp.stderr,
        )
    pairs = _parse_trailer_block(parsed_cp.stdout)
    digest = pairs.get(TRAILER_DIGEST)
    uri = pairs.get(TRAILER_URI)
    runtime = pairs.get(TRAILER_RUNTIME)
    if not (digest and uri and runtime):
        return None
    return TranscriptBinding(
        digest=digest,
        uri=uri,
        agent_runtime=runtime,
        transcript_id="",
        captured_at="",
    )


def _from_note_payload(payload: dict[str, Any]) -> TranscriptBinding | None:
    """Build a binding from a parsed JSON note dict.

    Returns ``None`` when any of the three wire-format fields
    (``digest``, ``uri``, ``agent_runtime``) is missing or non-string.
    Unknown keys are ignored — forward-compatibility per RFC 0001
    §Compatibility.Forward compatibility.
    """
    digest = payload.get("digest")
    uri = payload.get("uri")
    runtime = payload.get("agent_runtime")
    if not isinstance(digest, str) or not isinstance(uri, str) or not isinstance(runtime, str):
        return None
    transcript_id = payload.get("transcript_id", "")
    captured_at = payload.get("captured_at", "")
    return TranscriptBinding(
        digest=digest,
        uri=uri,
        agent_runtime=runtime,
        transcript_id=transcript_id if isinstance(transcript_id, str) else "",
        captured_at=captured_at if isinstance(captured_at, str) else "",
    )


def read_note(commit_sha: str, *, repo: Path) -> TranscriptBinding | None:
    """Read the binding's JSON note under ``refs/notes/transcripts``.

    Subprocess: ``git -C <repo> notes --ref=transcripts show <sha>``.

    Returns ``None`` when:

    * The commit has no note (``git notes show`` exits non-zero with
      "no note found for object" — we treat this as the absence
      signal, not an error).
    * The note exists but isn't valid JSON.
    * The note is JSON but is missing one of the three wire-format
      fields.

    Raises :class:`GitInvocationError` on any other ``git`` failure
    (e.g., the commit SHA itself doesn't exist — git emits a
    different stderr in that case, which the caller may want to
    distinguish from "note absent").
    """
    argv = [
        "git",
        "-C",
        str(repo),
        "notes",
        f"--ref={NOTES_REF}",
        "show",
        commit_sha,
    ]
    cp = _run_git(argv)
    if cp.returncode != 0:
        # ``git notes show`` returns 1 with "error: no note found for
        # object <sha>" when the object exists but has no note. Treat
        # that as the absence signal; everything else bubbles.
        stderr = cp.stderr.lower()
        if "no note found" in stderr or "no note for object" in stderr:
            return None
        raise GitInvocationError(argv, returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr)
    body = cp.stdout.strip()
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _from_note_payload(payload)


# ---------------------------------------------------------------------------
# Resolution — the public read-side entry point
# ---------------------------------------------------------------------------


def resolve_all_sources(
    commit_sha: str,
    *,
    repo: Path | None = None,
) -> BindingSources:
    """Diagnostic resolver: return both surfaces without merging.

    Never raises on trailer/note disagreement — that's what the
    diagnostic flow exists to investigate. Returns a typed dict with
    ``trailer`` and ``note`` keys, each either a populated
    :class:`TranscriptBinding` or ``None``.

    Underlying ``git`` failures (e.g., the commit SHA itself doesn't
    exist) still raise :class:`GitInvocationError`.
    """
    repo_path = _resolve_repo(repo)
    trailer = read_trailer(commit_sha, repo=repo_path)
    note = read_note(commit_sha, repo=repo_path)
    return {"trailer": trailer, "note": note}


def resolve_commit_to_transcript(
    commit_sha: str,
    *,
    repo: Path | None = None,
    all_sources: bool = False,
) -> TranscriptBinding:
    """Resolve a commit SHA to its bound transcript.

    Implements RFC 0001 §Specification.Resolution precedence:

    1. Trailer first.
    2. Note fallback.
    3. Loud failure on disagreement.

    Returns the merged :class:`TranscriptBinding`: trailer wins on
    wire-format fields (``digest``, ``uri``, ``agent_runtime``);
    note supplements with forensic fields (``transcript_id``,
    ``captured_at``) when both surfaces are present.

    Raises:

    * :class:`BindingMismatchError` when both surfaces are present
      and the digest disagrees.
    * :class:`LookupError` when neither surface is present (CLI maps
      to exit 2).

    The ``all_sources`` parameter is preserved for API compatibility
    with the plan signature; setting it ``True`` short-circuits to
    :func:`resolve_all_sources` and raises ``TypeError`` (callers
    wanting the dict shape should call :func:`resolve_all_sources`
    directly so the return type is statically narrowable).
    """
    if all_sources:
        raise TypeError(
            "resolve_commit_to_transcript(all_sources=True) is not supported; "
            "call resolve_all_sources() directly for the diagnostic dict shape"
        )
    repo_path = _resolve_repo(repo)
    trailer = read_trailer(commit_sha, repo=repo_path)
    note = read_note(commit_sha, repo=repo_path)

    if trailer is None and note is None:
        raise LookupError(f"no transcript binding for commit {commit_sha}")

    if trailer is not None and note is not None:
        if trailer.digest != note.digest:
            raise BindingMismatchError(
                (
                    f"trailer/note disagreement on commit {commit_sha}: "
                    f"trailer digest={trailer.digest!r}, note digest={note.digest!r}"
                ),
                trailer=trailer,
                note=note,
            )
        # Both agree on wire-format. Take trailer's wire values (per
        # RFC 0001 precedence) and supplement with note's forensic fields.
        return TranscriptBinding(
            digest=trailer.digest,
            uri=trailer.uri,
            agent_runtime=trailer.agent_runtime,
            transcript_id=note.transcript_id,
            captured_at=note.captured_at,
        )

    if trailer is not None:
        return trailer
    # ``note is not None`` here by exhaustion: lines above returned/raised on
    # (both None) and (both non-None); the only remaining state is note-only.
    assert note is not None  # noqa: S101  type-narrow for the type checker
    return note
