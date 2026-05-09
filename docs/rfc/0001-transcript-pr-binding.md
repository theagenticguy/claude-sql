# RFC 0001 — Transcript-to-PR Binding

| Field      | Value                                                   |
| ---------- | ------------------------------------------------------- |
| RFC        | 0001                                                    |
| Title      | Transcript-to-PR Binding via Commit Trailers + Git Notes |
| Status     | Draft                                                   |
| Authors    | Laith Al-Saadoon (`@lalsaado`)                          |
| Created    | 2026-05-09                                              |
| Updated    | 2026-05-09                                              |
| Track      | Standards (convention; reference impl in `claude-sql`)  |
| Implements | `.erpaval/strategy/transcripts-as-compiler/strategy-memo.md` §Coherent Actions #1 |
| License    | Apache-2.0 (reference implementation) / CC-BY-4.0 (this document) |

## Abstract

This RFC defines a **stable, host-neutral convention** for binding a merged
git commit to the AI-agent transcript that produced it. The binding is
encoded in two complementary surfaces shipped with every modern git
install: (i) three commit-message **trailers** parsed by
`git-interpret-trailers(1)`, and (ii) a JSON entry under
`refs/notes/transcripts` written and read with `git-notes(1)`. Both
surfaces are commodity primitives — trailers have been used since git
1.6 (2008) and are the load-bearing mechanism behind the Linux kernel's
`Signed-off-by:` and Gerrit's `Change-Id:` workflows; notes have shipped
since git 1.6.6 (2010). The convention requires no new git infrastructure,
no central registry, no vendor account, and no schema server. A commit
that carries the trailers is one `git log --format=%B HEAD` away from
naming the transcript that authored it; a reviewer that wants the
prompt→tool-call→correction trail behind a diff resolves it through the
URI in the trailer or through `git notes --ref=transcripts show <sha>`.

A reference implementation ships as part of `claude-sql` v0.3+ under two
subcommands: `claude-sql bind` (writes the binding from a
`prepare-commit-msg` hook) and `claude-sql resolve <sha>` (reads it from
either surface, with a loud failure on trailer/note disagreement). The
implementation is pure stdlib Python — `hashlib`, `subprocess`,
`pathlib`, `dataclasses`, `json` — and has no dependency on any
particular agent runtime; `claude-code` is the first emitter, but
`Claude-Agent-Runtime:` is an open-set extension point so `cursor/…`,
`amp/…`, `windsurf/…`, and any future runtime can publish bindings
through the same convention.

## Status

Draft. This document tracks the design behind branch
`feat/transcript-binding` (PR forthcoming) and is the first RFC in the
repository's `docs/rfc/` series. It will be promoted to "Accepted" when
(a) the reference implementation lands on `main`, (b) one external
agent runtime (claude-code is internal; the second mover is open) ships
a compatible writer, and (c) the convention has produced at least one
post-merge transcript-grounded review on a real production PR. Until
then the trailer names, the URI scheme, and the JSON schema are
**stable but reservation-only**: external implementations may write
them with the expectation that the reference reader treats them as
authoritative, but the RFC explicitly reserves the right to add fields
in a backward-compatible way (additional trailers, additional JSON
keys) before the "Accepted" promotion.

## Motivation

The strategy memo's diagnosis (`strategy-memo.md` §Diagnosis) frames
the problem in four symptoms and one root cause. This section
restates those grounded in the operational reality the RFC has to
solve.

**Symptom 1: AI-authored PRs get skimmed, not reviewed.** GitHub
Copilot Code Review, Graphite Diamond, Greptile, and Cursor Bugbot
all ship LLM-generated review comments derived from the diff alone.
None of them have access to the transcript that produced the diff,
because the transcript is on the agent operator's laptop in
`~/.claude/projects/**/*.jsonl` (or the equivalent path under
`~/.cursor/`, `~/.amp/`, etc.) and gets discarded at `git commit`
time. Reviewers read the merged diff without the
prompt→decision→tool-call path that produced it; they cannot tell
"the agent considered three approaches and the human picked this
one" from "the agent shipped the first thing that compiled."

**Symptom 2: Incident retros and software archaeology dead-end at
the squash-merge boundary.** When `git blame` lands on an
AI-authored line, the answer to "why is this line here?" is the
transcript, not the diff. Today the answer is unrecoverable —
transcripts live under a user's home directory, are not pushed, are
not addressable from a commit SHA, and are typically lost when the
agent operator changes machines. For a discipline (code review,
incident response, software archaeology) that has spent thirty years
treating the diff as the source of truth, AI-authored code is the
first widely-deployed source of compiled output committed *as if it
were source*. The `.o` shipped without the `.c` is the operative
metaphor (`strategy-memo.md` §Diagnosis): technically runnable, but
every downstream activity that depends on causal reasoning degrades
to "inspect the object code and guess." Operationally, the metaphor
is cashed out by the binding defined in this RFC — the pointer from
`.o` back to `.c` that every agent runtime currently discards at
`git commit` time.

**Symptom 3: Team knowledge leaks when agent operators leave.**
Pattern discovery across a team — recurring corrections, ineffective
prompt patterns, subagents that consistently fail, friction points in
specific codebases — is blocked not by analytics (`claude-sql`
already ships clusters, communities, friction classification, stance
conflict detection over the personal corpus) but by the absence of a
cross-author binding from PR to transcript. Without that, "patterns
across the team's AI-authored code" reduces to "patterns within
whoever was on shift."

**Symptom 4: Vendor-locked race-to-the-binding.** Third-party Cursor
session-replay extensions (SpecStory, cursor-replay, vibe-replay)
prove there is real demand for "see what the agent actually did";
none of them attach the transcript to the merged commit. The
incumbents' incentive — agent vendors win by locking transcripts
into their cloud, code-host vendors win by locking reviews into
their UI — is to *not* ship a cross-vendor binding. The window
before one of them ships a *vendor-locked* binding is the 12–18
months named in the strategy memo (`strategy-memo.md` §Risks #1).
The remediation is to ship a neutral binding *first*, under
permissive licenses, so any future vendor convention has to either
adopt this one or explain in writing why theirs is different and
better.

The diagnosis converges on a single missing primitive: **a stable,
content-addressed pointer from a merged commit to the transcript that
produced it**. The primitive must be (a) writable by any agent
runtime without that runtime needing the cooperation of any code
host, (b) readable by any reviewer or analytics tool without that
tool needing the cooperation of any agent vendor, and (c) shipped
on infrastructure already universally deployed — i.e., git itself.
Trailers + notes is the smallest convention that meets all three.

## Specification

### Trailers

A commit message that carries a transcript binding **MUST** include
the following three trailers, in any order, in the trailer block at
the end of the commit message (the block separated from the
preceding paragraph by exactly one blank line per
`git-interpret-trailers(1)`).

```
Claude-Transcript-Digest: sha256:<64-hex-chars>
Claude-Transcript-URI: <URI>
Claude-Agent-Runtime: <runtime-id>
```

Field semantics:

* **`Claude-Transcript-Digest`** is the SHA-256 hex digest of the
  transcript artifact's raw bytes, prefixed with the algorithm
  identifier `sha256:`. The digest is computed over the file as it
  existed on disk at the moment of binding — no re-canonicalization,
  no JSON re-serialization. This is a deliberate choice; see
  §"Digest determinism" below for the rationale and the alternative
  considered.

* **`Claude-Transcript-URI`** identifies where the bound transcript
  can be retrieved. The value is a URI in one of three schemes:
  * `file://<absolute-path>` — local filesystem, for solo and
    devbox use. The path is the absolute path to the JSONL file at
    the moment of binding; it may not be reachable from a different
    machine, and consumers should fall back to the
    `Claude-Transcript-Digest` for content verification.
  * `s3://<bucket>/<key>` — object storage, for team and
    organization use. The reference implementation ships only the
    `file://` writer in v0; the `s3://` reader is also out of scope
    for v0 and reserved for a future RFC update.
  * `git-notes://<refname>` — the binding's own `git notes` entry,
    used as a fallback URI when the trailer is the only available
    surface. Consumers reading this scheme **MUST** consult the
    note for the actual URI.

* **`Claude-Agent-Runtime`** identifies the agent runtime that
  emitted the transcript. The value is a free-form
  `<vendor>/<version>` string, e.g. `claude-code/0.42.1`,
  `cursor/0.42`, `amp/2025.4.1`. The reference implementation
  defaults to `claude-code/unknown` when the version is not
  detectable. Consumers **MUST NOT** parse the version for behavior
  decisions — the field is for provenance and analytics, not for
  feature gating.

The "Claude-" prefix is the v0 emitter convention, not a vendor
lock. The strategy memo (`§Coherent Actions #1`) explicitly anchored
the trailer names to the first emitter for grep-friendliness; future
RFC revisions may introduce a vendor-neutral alias (e.g. `Agent-
Transcript-Digest`) once a second runtime ships and the alias has a
named user. Until then, third-party runtimes are encouraged to use
the same trailer names — `Claude-` is read here as the *first
emitter convention*, not as a vendor namespace.

#### Idempotency

`bind` operations **MUST** be idempotent under `git commit --amend`
and any `prepare-commit-msg` re-run. The reference implementation
achieves this by passing `--if-exists replace` to
`git interpret-trailers`, which replaces an existing trailer of the
same key rather than appending a duplicate. This matches the
behavior of `Signed-off-by:` under repeated `git commit -s`
invocations on the kernel.

#### Survival across rebase, squash, cherry-pick

Trailers are part of the commit message. Standard git operations
preserve them:

* **`git rebase`** replays each commit's full message, trailers
  included. Trailers survive without intervention.
* **`git rebase -i` with `squash`/`fixup`** concatenates messages;
  trailers from both messages end up in the squashed commit's
  message, where `git interpret-trailers --parse` will return all
  of them. The reference resolver tolerates duplicates by taking
  the first-seen value per key and emitting a warning when keys
  conflict.
* **`git merge --squash`** drops the source commits' messages
  entirely and produces an empty commit message; if the resulting
  squash commit is not re-bound (e.g. by re-running the
  `prepare-commit-msg` hook on the squasher's machine), the
  binding is lost. This is a known gotcha; see §Compatibility.
* **`git cherry-pick`** replays the original message; trailers
  survive. The cherry-picked commit refers to the original
  transcript, which is the desired behavior.

### Git notes

A commit that carries a transcript binding **SHOULD** also write a
JSON entry under the `refs/notes/transcripts` ref. The note's body
is a single-line JSON object:

```json
{
  "uri": "file:///home/laith/.claude/projects/-efs-laith-workplace-claude-sql/2026-05-09T03-48-12.jsonl",
  "digest": "sha256:9d4c0…",
  "agent_runtime": "claude-code/0.42.1",
  "transcript_id": "01HXYZ-…-session-id",
  "captured_at": "2026-05-09T03:51:08+00:00"
}
```

Field semantics:

* `uri` — same value as `Claude-Transcript-URI` trailer.
* `digest` — same value as `Claude-Transcript-Digest` trailer.
* `agent_runtime` — same value as `Claude-Agent-Runtime` trailer.
* `transcript_id` — opaque identifier from the transcript artifact
  itself (e.g. the session-id field inside Claude Code JSONL).
  Never derived from the commit; always carried over from the
  transcript so a single transcript binds to the same id even when
  shipped through different commits (cherry-pick, rebase).
* `captured_at` — ISO-8601 UTC timestamp at the moment the
  binding was written. Used for forensics and stale-binding
  detection; not authoritative for ordering.

The note **complements** the trailers — it carries the same `uri`
and `digest` for redundancy, plus the two fields (`transcript_id`
and `captured_at`) that don't belong in a commit message because
they're forensic metadata, not human-readable provenance.

The choice of JSON over the kernel-style `key=value` note format is
deliberate: this repo's machine-readable bias (parquet caches, JSON
schema-validated structured outputs, CLI `--format json`) makes JSON
the lowest-friction format for downstream tooling. `git notes` does
not constrain the body format; conventions are per-ref.

#### Push policy

`git notes` are **not pushed by default**. To make a binding
visible on the remote — which the reference integration assumes —
both the writer's and the reader's side must opt in:

```
# Writer side (per-clone or in `git config --global`):
git config --add remote.origin.push 'refs/notes/transcripts:refs/notes/transcripts'

# Reader side:
git config --add remote.origin.fetch '+refs/notes/transcripts:refs/notes/transcripts'
```

This is a documented gotcha — a freshly-cloned repository will not
have the notes ref unless the cloner adds the fetch line. The
reference integration's documentation (and the post-merge GitHub
App, Action 4 in the strategy memo) will surface this as a one-time
setup step. A future RFC update may consider promoting the notes
ref to the default-pushed set if `git` itself ships a hook for that.

### URI scheme

Three URI schemes are permitted in `Claude-Transcript-URI` and the
note's `uri` field. The reference implementation in v0 ships the
`file://` writer only; the `s3://` and `git-notes://` schemes are
specified here so that future emitters can adopt them without an
RFC churn.

* `file://<absolute-path>` — local filesystem path. The reader
  resolves the path verbatim. If the path is not reachable from
  the reader's machine, the reader **MUST** fall back to the
  digest for content verification (i.e., request the file out of
  band, hash it, compare).
* `s3://<bucket>/<key>` — object storage. Authentication is the
  reader's responsibility (typically `AWS_PROFILE` or instance
  metadata). Future RFC updates may extend this to other
  object-storage URIs (`gs://`, `azure://`).
* `git-notes://<refname>` — sentinel scheme indicating the URI is
  carried by the note rather than the trailer. Useful when the
  emitter wants to keep the trailer block compact and push the
  full URI into the note. Readers seeing this scheme look up the
  note's `uri` field.

A reader that encounters a URI scheme it does not understand
**MUST** treat the binding as advisory only and continue resolving
through the digest; it **MUST NOT** raise an error.

### Resolution precedence

Given a commit SHA, a reader resolves the binding in three steps:

1. **Trailer first.** Run `git log --format=%B -1 <sha>` and parse
   trailers via `git interpret-trailers --parse`. If the three
   `Claude-Transcript-*` trailers are present, take the values.
2. **Note fallback.** If any trailer is missing, look up
   `git notes --ref=transcripts show <sha>` and parse the JSON.
   If the note is present, take the values.
3. **Disagreement is loud.** If both surfaces are present and
   `digest` disagrees, the reader **MUST** raise an error with
   exit code 70 (runtime error) and emit a structured error
   payload identifying both values. Disagreement is a forensic
   signal — typically an amended commit whose trailer was updated
   but whose note was not, or vice versa.

The reference resolver (`resolve_commit_to_transcript`) accepts an
`all_sources=True` flag that returns both surfaces as a dict for
diagnostic purposes; this is the only path that does not raise on
disagreement.

### Hook stage

The reference writer (`claude-sql bind`) is designed for the
`prepare-commit-msg` hook, **not** `pre-commit` or `commit-msg`.
The hook receives the commit-message file as `$1` and runs *before*
the editor opens, so the user sees the trailer in their editor and
can review or remove it. This is the same pattern git ships for
`git commit -s` (the `Signed-off-by:` injector), which is also a
`prepare-commit-msg` hook in spirit.

`prepare-commit-msg` cannot be bypassed with `--no-verify`, which is
the right behavior for a provenance pointer — bypassing should be
explicit (delete the trailer in the editor) rather than implicit
(silent skip).

The hook script is a one-liner:

```bash
# .git/hooks/prepare-commit-msg
exec claude-sql bind --commit-msg "$1" --source "$2"
```

Or as a `lefthook.yml` entry:

```yaml
prepare-commit-msg:
  jobs:
    - name: claude-sql-bind
      run: 'uv run claude-sql bind --commit-msg "{0}" --source "{1}" || true'
```

The trailing `|| true` is a deliberate choice for the lefthook
wiring: a binding failure (e.g., the user is committing from a
working tree that isn't tracked under `~/.claude/projects/...`)
should not block the commit. The strategy is "bind when possible,
silent on failure"; the reviewer side checks for the trailer
explicitly and treats its absence as "not AI-authored or runtime
not yet emitting bindings" rather than as an error.

## Reference implementation

The reference implementation lives in `claude-sql` and is delivered
through two CLI subcommands and a public Python module.

### `src/claude_sql/binding.py`

Pure stdlib Python. No new dependencies on top of what `claude-sql`
already ships. The module exports:

```
TRAILER_DIGEST  = "Claude-Transcript-Digest"
TRAILER_URI     = "Claude-Transcript-URI"
TRAILER_RUNTIME = "Claude-Agent-Runtime"
DIGEST_PREFIX   = "sha256:"
NOTES_REF       = "transcripts"

@dataclass(frozen=True)
class TranscriptBinding:
    digest: str           # "sha256:<hex>"
    uri: str              # "file://..." | "s3://..." | "git-notes://..."
    agent_runtime: str    # "claude-code/<version>" | "cursor/<version>" | ...
    transcript_id: str
    captured_at: str      # ISO-8601 UTC

def projectify(cwd: Path) -> str: ...
def find_active_transcript(cwd: Path) -> Path | None: ...
def compute_digest(jsonl_path: Path) -> str: ...
def detect_agent_runtime() -> str: ...
def write_trailer(commit_msg_path: Path, binding: TranscriptBinding) -> None: ...
def write_note(repo: Path, commit_sha: str, binding: TranscriptBinding) -> None: ...
def read_trailer(commit_sha: str, *, repo: Path | None = None) -> TranscriptBinding | None: ...
def read_note(commit_sha: str, *, repo: Path) -> TranscriptBinding | None: ...
def resolve_commit_to_transcript(
    commit_sha: str,
    *,
    repo: Path | None = None,
    all_sources: bool = False,
) -> TranscriptBinding | dict: ...
```

### `claude-sql bind`

Pre-commit-hook entry point. Reads the active transcript from
`~/.claude/projects/<projectified-cwd>/*.jsonl` (latest by mtime),
computes the digest, builds the URI and runtime fields, and writes
both trailer and note. Defaults to `--dry-run` when called outside
a hook (no `$1` argument and no fallback `.git/COMMIT_EDITMSG`),
which prints the would-be binding for inspection.

### `claude-sql resolve <commit-sha>`

Reader entry point. Implements the resolution precedence:
trailer → note → loud failure on disagreement. Exits 0 on success
with the binding emitted as JSON; exits 2 when no binding is
present (matches `claude-sql`'s "no embeddings" exit code for
absent-but-not-broken state); exits 70 on trailer/note disagreement
(matches the `runtime_error` band).

`--all-sources` returns both surfaces in a `{trailer: ..., note: ...}`
dict for diagnostic flows. Useful when investigating a stale or
amended binding.

### Test coverage

Ten tests in `tests/test_binding.py` cover:

1. Digest stability under repeat hashing of the same bytes.
2. Digest sensitivity to a single-byte change (newline append).
3. `projectify` round-trip for known cwd → claude-projects path
   mappings (`/foo/bar` → `-foo-bar`).
4. `find_active_transcript` picks the latest mtime.
5. `write_trailer` round-trips through
   `git interpret-trailers --parse`.
6. `write_note` round-trips through `git notes --ref=transcripts
   show` against a real `git init`'d tmp_path repo.
7. Resolve precedence: when both surfaces agree, trailer wins.
8. Resolve mismatch: when trailer and note digests disagree,
   resolution exits 70.
9. Resolve no-binding: when neither surface is present, exits 2.
10. `claude-sql bind --dry-run` against a tmp git repo emits a
    valid JSON plan and writes nothing to the repo state.

## Compatibility

### Trailer survival

The single biggest compatibility concern is the
**`git merge --squash`** workflow used by some teams to
collapse a feature branch into one commit on `main`. Because
`merge --squash` produces an empty commit message and the squasher
re-types it, the trailers from the source branch are not carried
forward by default. Recommended mitigations:

* The reference `prepare-commit-msg` hook re-runs on the squashed
  commit (since the squasher is the one running `git commit`), so
  `claude-sql bind` re-binds against the squasher's active
  transcript — which is the right answer if the squasher made a
  non-trivial edit, and the wrong answer if the squasher is a
  release engineer who didn't author the change.
* Teams that want to preserve the source branch's binding through
  squash should use `git merge --squash --log` (which dumps the
  source commits' messages into the squash commit's message;
  trailers come along), or use `--ff-only` / merge commits instead
  of squash.

`git rebase`, `git rebase -i`, and `git cherry-pick` all preserve
trailers by default; no special configuration needed. Multiple
copies of the same trailer (e.g. from a `fixup!` commit collapsed
during interactive rebase) are tolerated by the reader, which
emits a warning and takes the first occurrence per key.

### Git notes push policy

As noted in §Specification, `git notes` are not pushed by default.
Teams enabling the binding need a one-time configuration on each
clone (writer-side `push` config, reader-side `fetch` config). The
reference integration documents this as a setup step and the
GitHub App (strategy memo Action 4) carries the responsibility of
fetching the notes ref before resolving on the server side.

### Non-default push gotcha

A specific gotcha worth calling out: a contributor who runs
`claude-sql bind` locally but does not configure the
`refs/notes/transcripts` push will see the trailer on the remote
(it ships with the commit) but not the note. The trailer alone is
sufficient for resolution; the note is supplementary. Readers
that find a trailer but no note should not warn — that's the
expected steady state for any clone without the fetch config.

### Backward compatibility

Commits without trailers and without notes are unbound. The
reference resolver returns "no binding" (exit 2) for them, which is
distinct from "binding broken" (exit 70). Tools that consume the
binding (review-sheet generation, forensic analytics, the GitHub
App) should treat unbound commits as out-of-scope rather than as
errors.

### Forward compatibility

Adding new trailers or new JSON fields is forward-compatible by
construction. Readers ignore unknown trailers and unknown JSON
keys. The RFC explicitly reserves the right to add fields; a
future revision adding e.g. `Claude-Transcript-Compression:` or a
new `model_versions` JSON key will not break existing readers.

Removing or renaming an existing field is a **breaking change**
and would require an RFC supersession (this RFC marked
"Superseded" with a successor) plus a transition period during
which both names are accepted.

## Security

### Digest collision space

SHA-256 has a 256-bit output. Birthday-bound collision resistance
is approximately 2^128 — sufficient for any forseeable corpus
size. The digest is a content-integrity check, not a
cryptographic commitment in the protocol-design sense; we do not
rely on its non-malleability for any security claim, only for
"the bytes I bound and the bytes you fetched are the same bytes."
Pre-image and second-pre-image resistance are also at SHA-256's
documented strength; an adversary who could forge a transcript
matching a fixed digest already has sufficient power to do worse
things to the rest of the codebase.

### Digest determinism

The reference implementation hashes the **raw bytes** of the JSONL
file as written by Claude Code (`hashlib.sha256(p.read_bytes())
.hexdigest()`). This was a deliberate choice over the alternative
of canonicalizing each line through `json.loads` + `json.dumps(
sort_keys=True, separators=(',', ':'))` before hashing.

Reasoning:

* **Reviewers can recompute.** The trailer's digest must be
  reproducible by anyone with the file in hand; raw-bytes hashing
  is a one-line operation in any language. Canonical-JSON
  hashing requires the reviewer to know the canonicalization
  recipe and apply it identically.
* **The artifact is the file as written.** Claude Code emits one
  JSON object per line; mutation of that file *is* a new
  artifact. We bind to the artifact, not to a derived semantic
  view of it.
* **SRI and npm precedent.** W3C Subresource Integrity hashes raw
  resource bytes, and npm's `package-lock.json` does the same.
  Both are content-integrity systems for content-addressed
  artifacts; both pick raw-bytes hashing for the same reasons.

The trade-off is fragility: any whitespace or key-order
difference between writers — for example, if the agent runtime is
updated and starts emitting sorted keys — produces a different
digest, even when the semantic transcript is identical.
Acknowledged. The mitigation is the `Claude-Agent-Runtime:`
trailer, which carries the runtime version explicitly so a
digest-mismatch investigation can rule out cross-version
incompatibility quickly.

### Scrub-secrets-before-bind

Transcripts can contain secrets that the user pasted into prompts
("here's our staging API key, debug this 502"), tool output that
the agent saw mid-execution (database credentials in a connection
string), or third-party IP. The strategy memo's risk register
(`§Risks #2`) names this as a partial mitigation: a regex-based
secrets scrub is in scope for the v0 binding, a policy-driven
redaction service is not.

The reference implementation **does not** ship a redaction step
in v0. The expected pattern for v1 is a pre-bind transformation
that runs each line through a permissive but high-recall regex
filter before the digest is computed:

```python
SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"AKIA[0-9A-Z]{16}"),                    # AWS access key
    re.compile(r"(ghp|ghu|gho|ghs|ghr)_[A-Za-z0-9]{36}"),  # GitHub PAT
    re.compile(r"sk-[A-Za-z0-9]{48}"),                  # OpenAI API key
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # ... extension point
]
```

A redacted line is replaced with the literal string
`"[REDACTED:<pattern-name>]"` before hashing. The redacted
artifact becomes the bound artifact; reviewers who fetch the
transcript see redactions, not secrets. This is the v1 plan; v0
ships **without** redaction and explicitly documents the gap.

Operators concerned about secret exposure today have three
options:

1. Don't bind. The trailer is opt-in.
2. Run a pre-bind grep over the transcript and abort on hit.
3. Wait for v1.

### IP / PII deferral

Beyond regex secrets, transcripts contain customer data,
proprietary code, and content from third parties (other vendors'
documentation pasted in by the user, snippets from libraries the
agent searched). The strategy memo (`§Risks #2`) explicitly
defers a policy-driven redaction service to post-90-day; the v0
binding is "safe-by-default on secrets," not "audit-grade on IP."

This RFC inherits that scope cut. Teams with an IP / PII story
beyond regex secret scrubbing should not enable the binding on
sensitive repositories until a future RFC adds a redaction step
they can configure. Naming the gap here is the gap's mitigation;
hiding it would be worse.

### Authentication

The convention does not specify an authentication model. A reader
fetching `s3://...` URIs uses whatever credentials the local
environment provides (the reference implementation uses boto3's
standard chain, gated by `AWS_PROFILE`). A reader resolving
trailers and notes uses git's standard auth (SSH keys, HTTPS
tokens, etc.).

The post-merge GitHub App / Action (strategy memo Action 4) will
introduce a service-side authentication model when it ships;
it is out of scope for this RFC.

## Open questions

1. **`Claude-` prefix — when, if ever, do we promote to a vendor-
   neutral alias?** First emitter convention is reasonable for v0
   (matches `Signed-off-by:` history — it's "DCO" in the
   `Documentation/process/submitting-patches.rst` text but the
   trailer is `Signed-off-by:`, not `DCO-Signed-off-by:`). The
   tracking question is whether a v1 RFC introduces an
   `Agent-Transcript-Digest:` alias once a second runtime ships.

2. **`Claude-Agent-Runtime:` value taxonomy.** Free-form
   `vendor/version` is the v0 spec. A v1 RFC may consider adding
   a controlled list of vendor strings to prevent typos
   (`claude-code` vs `claude_code` vs `Claude Code`). The
   trade-off is registry overhead vs. typo fragility; deferred.

3. **Notes ref naming — `transcripts` vs `claude-sql/transcripts`?**
   The current spec uses `refs/notes/transcripts` — short, easy to
   type. Risk is collision with another tool that wants the same
   ref. Mitigation is the JSON shape: any reader sees the
   `agent_runtime` field and decides whether to honor the entry.
   Alternative is namespacing the ref. Deferred; revisit when a
   second tool wants the same ref.

4. **`captured_at` clock skew.** The note's `captured_at` is the
   writer's local clock at bind time. Forensic comparison across
   machines that disagree on UTC time will produce ordering
   anomalies. Acceptable for v0; a v1 RFC may add a server-side
   timestamp via the GitHub App.

5. **Squash-merge re-binding semantics.** The current spec has the
   `prepare-commit-msg` hook re-bind on the squashed commit's
   commit phase, which captures the squasher's transcript rather
   than the source branch's. For solo workflows where the
   squasher is the original author, this is correct. For PR
   workflows where the squasher is a release engineer, it is
   wrong. v1 may add a `--inherit-trailers-from-source-branch`
   mode to the bind command. Deferred until a real complaint
   lands.

6. **Multi-transcript binding.** A complex feature may span
   multiple Claude Code sessions, each producing its own JSONL.
   The current spec is one-trailer-per-key, which forces "one
   commit = one transcript." A v1 RFC may add comma-separated
   digest lists or repeated trailer entries to bind multiple
   transcripts to one commit. Deferred until a real use case
   surfaces; the strategy memo's Action 1 explicitly named "no
   multi-transcript binding" as out of scope.

## Citations

* `git-interpret-trailers(1)` — https://git-scm.com/docs/git-interpret-trailers
* `git-notes(1)` — https://git-scm.com/docs/git-notes
* `githooks(5)` — https://git-scm.com/docs/githooks
* Linux kernel `Signed-off-by:` precedent —
  https://www.kernel.org/doc/html/latest/process/submitting-patches.html#sign-your-work-the-developer-s-certificate-of-origin
* Gerrit `Change-Id:` trailer convention —
  https://gerrit-review.googlesource.com/Documentation/user-changeid.html
* W3C Subresource Integrity (raw-bytes hashing precedent) —
  https://www.w3.org/TR/SRI/
* GitHub Copilot Code Review (LLM-over-diff incumbent without
  transcript provenance) —
  https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review
* `claude-sql` strategy memo (this RFC's parent) —
  `.erpaval/strategy/transcripts-as-compiler/strategy-memo.md`
  §Coherent Actions #1
* `claude-sql` Plan (operationalization of this RFC) —
  `.erpaval/sessions/session-2293a5/plan.md` §Action 1
