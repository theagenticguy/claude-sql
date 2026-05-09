---
title: PR↔transcript binding via commit-trailers + git notes
track: knowledge
category: api-patterns
module: src/claude_sql/binding.py
component: git plumbing (subprocess)
severity: info
tags: [git, trailer, notes, binding, transcript, prepare-commit-msg]
applies_when:
  - "Need to attach metadata to a commit that survives rebase, squash, and cross-tool review"
  - "Want a hook-callable CLI that decorates the commit message and writes a side-channel note"
pattern: |
  Use `git interpret-trailers --in-place --trailer "Key: Value" <commit-msg-path>` for the
  decorator, and `git notes --ref=<custom-ref>` for the side-channel JSON. Run the binder from
  `prepare-commit-msg` (NOT `commit-msg` or `pre-commit`):
    - prepare-commit-msg gets the commit-msg path as $1
    - prepare-commit-msg runs BEFORE the editor, so the user sees the trailer in their session
    - prepare-commit-msg cannot be bypassed with `git commit --no-verify`

  Resolution precedence on the read path:
    1. Trailer first (it's the canonical claim, survives rebase by default)
    2. Note as fallback (carries supplemental fields like `transcript_id`, `captured_at`)
    3. Loud failure on mismatch — exit 70, emit_error JSON, do NOT silently prefer one source

  git-notes gotchas to document at adoption:
    - `git push origin refs/notes/transcripts` is non-default — push policy must be configured
    - `git notes show` returns nothing (exit 1) when no entry exists — distinguish from a hard error
    - GitHub's PR UI does not surface notes by default; CLI / API / GitHub App needed for visibility

  Subprocess discipline:
    - List-form `subprocess.run([...])` always; no shell=True
    - `check=False, capture_output=True, text=True` — handle returncode explicitly
    - Wrap subprocess errors in a domain exception (e.g., `GitInvocationError`) for clean exit-code mapping
example_files:
  - docs/rfc/0001-transcript-pr-binding.md
  - src/claude_sql/binding.py
  - tests/test_binding.py
---

# Why this matters

Binding a transcript to a PR via something other than commit-trailers + git notes (e.g.,
out-of-band index, GitHub annotation, vendor-locked sidebar) couples your binding to a
single host and a single tool. Trailers and notes are commodity git infrastructure shipped
since git 1.6 — they survive rebase, squash, push, fork, and pull-merge. Linux kernel
`Signed-off-by:` and Gerrit `Change-Id` are the canonical precedents.

The hook-stage choice matters more than it looks. `commit-msg` runs after the user's editor
and is bypassable with `--no-verify`. `pre-commit` doesn't have access to the message file.
`prepare-commit-msg` is the only one that decorates the message before the editor, can't be
trivially bypassed, and gets `$1` as the path to the message file.

# Example

```python
# Trailer write — three keys, one subprocess each. Git's parser handles ordering.
import subprocess
TRAILER_DIGEST = "Claude-Transcript-Digest"
TRAILER_URI = "Claude-Transcript-URI"
TRAILER_RUNTIME = "Claude-Agent-Runtime"

def write_trailer(commit_msg_path: Path, binding: TranscriptBinding) -> None:
    for key, value in (
        (TRAILER_DIGEST, binding.digest),
        (TRAILER_URI, binding.uri),
        (TRAILER_RUNTIME, binding.agent_runtime),
    ):
        subprocess.run(
            ["git", "interpret-trailers", "--in-place", "--trailer", f"{key}: {value}", str(commit_msg_path)],
            check=False, capture_output=True, text=True,
        )

# Note write — JSON under refs/notes/transcripts
def write_note(repo: Path, commit_sha: str, binding: TranscriptBinding) -> None:
    payload = json.dumps({
        "uri": binding.uri,
        "digest": binding.digest,
        "agent_runtime": binding.agent_runtime,
        "transcript_id": binding.transcript_id,
        "captured_at": binding.captured_at,
    })
    subprocess.run(
        ["git", "-C", str(repo), "notes", "--ref=transcripts", "add", "-f", "-m", payload, commit_sha],
        check=False, capture_output=True, text=True,
    )
```
