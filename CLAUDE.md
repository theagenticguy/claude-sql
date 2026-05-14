# claude-sql — project instructions

This file is read by Claude Code (and any agent running in this repo). It tells
you how to work on and with `claude-sql` without re-deriving the design each
time.

## What this project is

A standalone Python CLI that makes the user's own Claude Code transcripts —
everything under `~/.claude/projects/**/*.jsonl` plus subagent sidecar files —
queryable in place. Four analytics layers stack on top of those JSONLs:

1. **SQL** — DuckDB reads the JSONLs with zero copy; 18 views + 14 macros
   normalize messages, tool calls, todos, subagents, costs.
2. **Semantic search** — Cohere Embed v4 (Bedrock, global CRIS) embeds every
   message; DuckDB VSS HNSW cosine index serves top-k in milliseconds.
3. **LLM analytics** — Sonnet 4.6 (global CRIS, `output_config.format`
   structured output) classifies sessions (autonomy tier, work category,
   success, goal), scores per-message sentiment, detects stance conflicts,
   and classifies short user messages for friction signals (status pings,
   unmet expectations, confusion, interruption, correction, frustration).
4. **Structure** — UMAP + HDBSCAN cluster message embeddings, c-TF-IDF names
   the clusters, Leiden + CPM groups sessions into communities over a
   mutual-kNN cosine graph of session centroids. The `community` subcommand
   is agent-first: it emits *signals* (medoid sessions, coherence,
   resolution-profile sidecar, `--neighbors-of` lookup) so a calling LLM
   can do progressive disclosure into the corpus before reading specific
   transcripts.

Every expensive output (embeddings, classifications, clusters, communities,
friction) is cached in parquet under `~/.claude/`. Views register only the
parquets that exist — missing ones warn and no-op, never crash.

## How to work on it

- **Package manager:** `uv` only. Never hand-edit `[dependencies]` in
  `pyproject.toml` — use `uv add` / `uv remove` so `uv.lock` stays in sync.
- **Task runner:** `mise`. Every developer command is a mise task. Run
  `mise tasks` to see the full list. Do not invent bare `uv run` / `ruff`
  invocations for things that already have a task.
- **Quality gate:** `mise run check` must pass before any commit —
  that's `lint + fmt + typecheck + test` in parallel. Treat every
  non-zero exit as a blocker, not a "pre-existing" issue to skip past.
- **Security gate:** `mise run security` runs all four SAST/SCA/secrets
  scanners in parallel — `bandit + semgrep + osv + leaks` — and writes
  SARIF under `.sarif/` (gitignored). Mirrors what CI uploads to GitHub
  code scanning. Each scanner uses `--exit-zero` / `--exit-code=0` per
  `.erpaval/solutions/best-practices/sarif-scanner-report-vs-gate.md`;
  gating happens in code scanning UI, not the scanner exit code. Run
  before opening a PR if you've touched anything load-bearing
  (binding/ingestion/CLI surface).
- **Python floor:** `3.13` (`.python-version` + `pyproject.toml`
  `requires-python` + `mise.toml [tools].python` all agree). 3.14 is
  deferred pending `hdbscan` cp314 wheels — see
  `docs/adr/0015-stack-modernization.md`.
- **Formatting:** `mise run fmt:write` applies ruff formatting. Line length
  is 100. Ruff runs a 32-family strict selector set (E, W, F, I, N, UP, B,
  SIM, ANN, ASYNC, BLE, C4, DTZ, ERA, FBT, G, ICN, ISC, LOG, PERF, PIE, PL,
  PT, PTH, RET, RSE, S, T20, TID, TRY, PGH, RUF) with principled ignores
  documented inline in `pyproject.toml`. See the ADR for the rationale.
- **Type checker:** `ty` in strict mode (`[tool.ty.rules] all = "error"`)
  over `src/ tests/`. Promotes every warn-default diagnostic so new ty
  rules fail CI instead of drifting into noise. Tests carry a narrow
  `[[tool.ty.overrides]]` for the DuckDB-`Optional`-subscript false-
  positive class. Editor Pyright warnings about unresolved imports or
  `Optional[...]` subscript are false positives from a global install
  that can't see the project's `.venv`; trust `ty`.
- **Reproducibility:** `[tool.uv]` pins `required-version >= 0.11.7`,
  `python-preference = only-managed`, `compile-bytecode = true`,
  `link-mode = clone`. `mise run lock:check` is the CI freshness gate.
- **Tests:** `pytest` under `tests/`. `mise run test`. Tests must not hit
  Bedrock — mock the client. The live corpus is the one under
  `~/.claude/projects/`, so integration tests should use a small fixture
  directory, not the live one.

## CodeQL hygiene (the rules ruff misses)

GitHub Advanced Security runs CodeQL on every PR. Ruff's 32-family
selector catches most issues, but four CodeQL rules fire on patterns
ruff lets through. Each has a one-line fix; each is easy to forget.

- **`py/empty-except` — every `try/except: pass` block needs a comment
  explaining why.** Ruff's `S110` only fires on broad excepts
  (`except Exception:`). A *narrow* except (`except (ImportError,
  AttributeError):`) with a bare `pass` body is invisible to ruff,
  but CodeQL flags it without an inline explanation. Always pair the
  `except: pass` with a one-line comment naming what the exception
  represents and why ignoring it is correct. Example shape:

  ```python
  try:
      import optional_dep
  except ImportError:
      # Optional dep — no-op when not installed; the caller's fallback handles it.
      pass
  ```

- **`py/unused-local-variable` — classes / functions defined inside a
  test function must be referenced.** Ruff's `F841` only flags assigned
  names, not nested `class Foo: …` declarations. Don't define test
  scaffolding (mock classes, fake exceptions, helper functions) you
  don't end up using. If a class was kept "just in case" or for a
  branch that never materialized, delete it before committing.

- **`py/catch-base-exception` — never `except BaseException` in an
  anyio task body or async dispatcher.** `BaseException` swallows
  `KeyboardInterrupt`, `SystemExit`, and crucially
  `asyncio.CancelledError`. In an anyio task group, swallowing
  `CancelledError` deadlocks shutdown — the parent's `aclose()` waits
  forever for child tasks that observed cancel but never re-raised it.
  The fix is one character: `BaseException` → `Exception`. Recoverable
  errors (network blip, parquet schema mismatch, refused refusal) are
  all `Exception` subclasses; cancellation cleanly cascades. Verified
  on PR #42 (`trajectory_worker.py:647/:862`). The only legitimate
  `except BaseException` lives in CLI top-level `try/except` blocks
  that re-raise after logging — never in worker code.

- **`py/file-not-closed` — when monkeypatching `tempfile.mkstemp` to
  return a `(fd, path)` tuple, open the fd inside the closure per-call,
  not at module level.** Production code conventionally calls
  `os.close(fd)` because mkstemp's contract gives the caller fd
  ownership. A module-level `fd = os.open(...)` reused across calls
  trips CodeQL's data-flow tracker (open-site and close-site are
  decoupled) AND a per-test `addFinalizer(lambda: os.close(fd))`
  produces `Bad file descriptor` on a double-close because the
  consumer ALSO closes. Open inside the closure each call so producer
  and consumer pair statically. See
  `.erpaval/solutions/best-practices/codeql-py-file-not-closed-in-test-mkstemp-fakes.md`
  for the full pattern.

All four checks run automatically — there's no `mise run codeql` task.
Treat the comment / unused-name / no-BaseException / per-call-mkstemp
discipline as part of writing the test in the first place. The rules
below in `pyproject.toml` should be kept maximally aggressive so the
*next* class of CodeQL findings has a ruff-side analog wherever one
exists.

## Logging: loguru only, no stdlib `logging`

Every module in `claude_sql` logs through `from loguru import logger`.
Stdlib `logging` is **banned** — `[tool.ruff.lint.flake8-tidy-imports.banned-api]`
in `pyproject.toml` rejects `import logging` / `from logging import …`
with a `TID251` error. Mixing the two leaves messages stuck in the
stdlib root logger that the user's loguru sink never sees.

The historic exception was tenacity's `before_sleep_log(stdlib_logger,
level)` callback, which required a `logging.LoggerProtocol`. That has
been replaced by `claude_sql.logging_setup.loguru_before_sleep("LEVEL")`,
which emits the same retry-state line shape via loguru. Use it on every
new tenacity `@retry` decorator:

```python
from claude_sql.logging_setup import loguru_before_sleep
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    before_sleep=loguru_before_sleep("WARNING"),
    reraise=True,
)
def call_bedrock(...): ...
```

If you find yourself reaching for `logging` because some third-party API
demands a stdlib logger, write a small loguru-backed adapter under
`logging_setup.py` rather than punching a hole in the ban.

## Git hooks (lefthook) and commit conventions (commitizen)

First-time setup after cloning:

```bash
mise run install          # also runs hooks:install as a dependency
```

That installs `pre-commit`, `commit-msg`, and `pre-push` git hooks via
lefthook. See `lefthook.yml` for the exact wiring. TL;DR:

- **pre-commit** — runs `ruff check --fix` + `ruff format` on staged Python
  files (auto-stages fixes) and `ty check src/` across the whole source
  tree. Fast in parallel.
- **commit-msg** — `cz check --allow-abort --commit-msg-file {1}` validates
  the message against the conventional-commits schema.
- **pre-push** — full `pytest` run. The belt + suspenders for the belt.

Every commit message must follow
[Conventional Commits](https://www.conventionalcommits.org/). The
commitizen types supported out of the box are `feat`, `fix`, `docs`,
`style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
Scope is optional but preferred (`feat(friction): ...`).

Use `mise run commit` for the interactive wizard if you want prompts, or
write the message yourself — either way the `commit-msg` hook validates it.
Bypass with `git commit --no-verify` only when you're landing a WIP and
intend to amend it into a conventional message before pushing.

## Publishing to PyPI

Trusted Publishing (OIDC) wires `pypi.org` and `test.pypi.org` to this
repo via `.github/workflows/publish.yml`. There are no API tokens; the
workflow assumes a short-lived OIDC token at job runtime. Two paths:

- **PyPI on release published.** `gh release create vX.Y.Z` →
  `release: types: [published]` fires → the `publish-pypi` job builds
  + uploads to https://pypi.org/p/claude-sql. Same trigger as
  `sbom.yml`, so the release flow already produces both the SBOM and
  the wheel in one motion.
- **TestPyPI on workflow_dispatch.** `gh workflow run publish.yml -f
  target=testpypi` (or click "Run workflow" in the Actions UI) builds
  + uploads to https://test.pypi.org/p/claude-sql for dry-run
  verification of metadata + the upload shape. Use this first when a
  contributor PR changes anything load-bearing about the package
  metadata.

Trusted Publisher entries on PyPI / TestPyPI must list:

- Project name: `claude-sql`
- Owner / repo: `theagenticguy / claude-sql`
- Workflow filename: `publish.yml`
- Environment name: `pypi` (production) or `testpypi` (dry-run)

The workflow declares `environment: name: pypi` / `environment: name:
testpypi` per job — that's the load-bearing scope check. Don't drop
the environment block; without it any compromised PR would inherit the
ability to publish.

If a wheel ever lands on PyPI and you need to redact it: `pip install
--upgrade pkginfo` then `twine` can't unpublish but you can mark it
`yanked` via the project page, then publish a corrected version.
Trusted Publishing also supports per-environment `pending` publishers
that auto-clear once the first release lands.

## Version bumping & changelog

**`CHANGELOG.md` is write-only via `cz bump`.** Never hand-edit it. Never
run `mise run changelog` between releases — its output drifts against
`cz`'s post-squash-merge projection (the `## Unreleased` placeholder
+ `(#NNN)` suffix injection) and any drift gate built on top will
fight you. CI no longer enforces any drift check; the file mutates
exactly twice per release: once when `cz bump` writes the new
`## vX.Y.Z` block on the release branch, and once when squash-merge
appends `(#NNN)` to the merged commit's bullet.

Between releases, query commit history directly:
- `mise run bump:dry-run` — preview next version + the bullets `cz bump`
  would write.
- `git log --oneline v0.7.0..HEAD` — raw conventional-commit list.

Version management is driven by `cz bump`, which reads commit history,
picks MAJOR/MINOR/PATCH from the conventional-commits types, updates
`pyproject.toml` + `uv.lock` (via `version_provider = "uv"`), writes
`CHANGELOG.md`, and creates an annotated tag.

- `mise run bump:dry-run` — preview the next version + tag without writing.
- `mise run bump` — bump, update changelog, commit as
  `chore(release): X → Y`, create annotated `vY` tag.

Commitizen config lives in `[tool.commitizen]` in `pyproject.toml`:
`tag_format = "v$version"`, `version_provider = "uv"`,
`major_version_zero = true`, `update_changelog_on_bump = true`,
`annotated_tag = true`.

### Release flow under branch-protection (the branch-PR-tag dance)

`origin/main` is branch-protected: direct pushes are rejected with
`GH013: Repository rule violations found`. The release commit MUST go
through a PR. Don't `mise run bump` on `main` and try to push — the
local commit + tag will land in limbo. Use this sequence:

1. **Branch the bump.**
   ```bash
   git checkout main && git fetch origin && git reset --hard origin/main
   git checkout -b chore/release-X.Y.Z
   mise run bump:dry-run    # confirm version + increment
   mise run bump            # creates the commit AND the local tag
   ```
2. **Push branch + open PR.**
   ```bash
   git push -u origin chore/release-X.Y.Z
   gh pr create --title "chore(release): A → B" --body "$(cat <<'EOF'
   ...notes from CHANGELOG.md...
   EOF
   )"
   ```
3. **Wait for CI green**, then squash-merge:
   ```bash
   gh pr checks <pr-num> --watch
   gh pr merge <pr-num> --squash --delete-branch
   ```
4. **Re-tag the merge SHA.** Squash-merge rewrites the commit, so the
   local tag from step 1 points at a SHA that doesn't exist on main.
   ```bash
   git checkout main && git fetch origin && git reset --hard origin/main
   git tag -d vY                                    # drop pre-merge tag
   git tag -a vY -m "vY" $(git rev-parse HEAD)      # tag the merge SHA
   git push origin vY                               # tags bypass branch-protection
   ```
5. **Cut the GitHub release** so `release: types: [published]`
   workflows (`sbom.yml` etc.) fire:
   ```bash
   awk '/^## vY/{flag=1; next} /^## v/{flag=0} flag' CHANGELOG.md > /tmp/notes.md
   gh release create vY --title vY --notes-file /tmp/notes.md --verify-tag
   ```

**If a release-triggered workflow at the tag is broken** (e.g. the v0.3.0
SBOM run hit a bad cyclonedx-py flag shape), don't `gh workflow run --ref vY`
to retry — that re-runs the *tagged* (broken) workflow body. Fix the
workflow on a follow-up PR for the next release, and recover the missed
artifact locally:
```bash
uvx --from cyclonedx-bom cyclonedx-py environment .venv \
  --output-format JSON --output-file /tmp/SBOM.cdx.json
gh release upload vY /tmp/SBOM.cdx.json --clobber
```

See `.erpaval/solutions/best-practices/cz-bump-via-pr-with-branch-protection.md`
for the lesson capturing this; the cyclonedx-py flag-shape fix is in
`.erpaval/solutions/api-patterns/cyclonedx-python-uv-environment.md`.

## How to run it

- **As a uv tool** (preferred end-user path): `mise run tool:install` →
  `claude-sql --version`. Reinstall after pulling with
  `mise run tool:upgrade` (same command — `uv tool upgrade claude-sql`
  does NOT work because the package is not on PyPI). Remove with
  `mise run tool:uninstall`.
- **In the project venv** (development): `mise run install` →
  `mise run cli -- <subcommand>`.
- **REPL:** `claude-sql shell` drops into DuckDB with every view + macro
  already registered.

## Bedrock

- **Region:** `us-east-1`. Embeds + Sonnet classification go through the
  **global** CRIS profiles — `global.cohere.embed-v4:0` and
  `global.anthropic.claude-sonnet-4-6`. Direct model IDs and US-only CRIS
  throttle under load; global is the only path that sustains throughput.
- **IAM:** `bedrock:InvokeModel` on both inference profiles above.
- **Credentials:** `AWS_PROFILE=<your-profile>` in the environment. The CLI
  reads it via boto3's standard chain.
- **Cost guard:** every command that spends real money defaults to
  `--dry-run`. `analyze` chains embed → cluster → classify → trajectory →
  conflicts → friction and respects `--dry-run` on every step. The dry-run
  path uses a pure-SQL count, not a full materialization, so it stays fast
  on large corpora.

## DuckDB

- **Version floor: ≥1.5.1** — the `lance` core extension (used by the
  embeddings store) became core in 1.5.1. Below that, the extension has
  to be installed from the community repo and pin churn breaks the
  embeddings cache. Pinned `>=1.5.2,<2` in `pyproject.toml`; upper bound
  keeps us inside the 1.x train through any 2.0 surprise.
- **Non-blocking checkpointing** — 1.5.x replaces the old stop-the-world
  checkpoint with a concurrent path: reads, writes, inserts, and deletes
  proceed during checkpoint flushes. Matters less now that the embeddings
  store moved to LanceDB (no DuckDB writer-lock storms), but still
  benefits the analytics views that read JSONL transcripts in place.
- **Throughput** — 1.5.2 is ~17% faster on TPC-H than 1.4.x in upstream
  benchmarks. Our analytics views (mostly aggregate-heavy SQL over the
  JSONL corpus) sit in the same shape, so the improvement carries over.
- **`date_trunc(DATE)` returns TIMESTAMP in 1.5.0+** (was DATE). Audit
  result: only one usage in `sql_views.py` (`autonomy_trend` macro) and
  it operates on `classified_at` which is TIMESTAMP-typed in the parquet
  schema, so no breaking change for us. Future `date_trunc` calls on
  DATE-typed columns that feed DATE-expecting downstream consumers must
  add an explicit `::DATE` cast.

## Structured output (Sonnet classification)

- Uses Bedrock's GA `output_config.format` field (not tool_use/tool_choice).
- Schemas are pydantic v2 models in `schemas.py`; `model_json_schema()` is
  then flattened: inline `$ref`, inject `additionalProperties: false`,
  strip the numeric/string constraints the validator rejects from Draft
  2020-12 subset.
- Adaptive thinking stays on. `citations` is the only feature incompatible
  with structured output — do not add it.

## Trajectory classifier (`trajectory_worker.py`) — windowed v1.0

The v1.0 trajectory pipeline (RFC 0002 §3.4 / §4.1) is **per-session
windowed**, not per-message. Each text turn produces one row keyed on
`(prev_uuid, curr_uuid)`; the session-first turn gets a synthetic pair
with `prev_uuid IS NULL`. Sonnet 4.6 sees up to 16 windows per chunk in
one structured-output call (`TrajectoryArrayResult` schema in
`schemas.py`) — sessions longer than 16 text turns split into
anchor-sharing chunks where chunk N's last `curr_uuid` equals chunk N+1's
first `prev_uuid`. The host pipeline echoes `(prev_uuid, curr_uuid)`
tuples back from the response and runs ONE bounded retry of just the
missing windows; persistent misses become neutral placeholder rows so a
single refusing chunk never wedges the pipeline. The output parquet
schema is `(session_id, prev_uuid, curr_uuid, prev_sentiment,
curr_sentiment, delta, is_transition, transition_kind, confidence,
classified_at)`. `transition_kind` is the new categorical signal — a
six-value enum: `frustration_spike`, `resolution`, `reset`, `drift`,
`clarification`, `none`. Stale per-message shards (the pre-v1.0 schema
with `uuid`/`sentiment_delta` columns) are detected via parquet metadata
and deleted on first run. The whole run is wrapped in
`pipeline_cache_stats("trajectory")` so the system-prompt 1h cache write
+ subsequent reads emit one summary line at INFO. The system prompt is
padded past Sonnet 4.6's 2048-input-token cache floor — verified by
`test_system_prompt_clears_cache_floor`. The `sentiment_arc(sid)` macro
joins `messages.uuid` to `message_trajectory.curr_uuid` and surfaces
`(ts, role, curr_sentiment, delta, transition_kind, is_transition,
confidence)`. Old `sentiment_delta` callers must rebase to `delta` and
`curr_sentiment`.

## Conflicts classifier (`conflicts_worker.py`) — pair-keyed v1.0

The v1.0 conflicts pipeline (RFC 0002 §3.4) is **pair-keyed** on
`(turn_a_uuid, turn_b_uuid)`. Sessions with no conflicts produce **zero
rows** — the legacy `empty=True` sentinel is gone, so any caller that
wants every session in the result set must `LEFT JOIN sessions` and
coalesce missing counts to 0. Two enums supplement the pair: `conflict_kind`
∈ {`disagreement`, `correction`, `reversal`, `impasse`} and `severity` ∈
{`low`, `medium`, `high`}, plus `agent_position` / `user_position` /
`confidence`. The output parquet shape is `(session_id, turn_a_uuid,
turn_b_uuid, conflict_kind, severity, agent_position, user_position,
confidence, detected_at)`. Stale shards from the pre-v1.0 schema (anything
carrying `conflict_idx` or `empty` columns) are detected via parquet
metadata and the entire cache directory is deleted on first run before
new shards land. The whole run is wrapped in
`pipeline_cache_stats("conflicts")` so the 1h system-prompt cache write
+ reads surface one summary line at INFO. The new `conflicts_summary`
view is a simple `count(*) GROUP BY session_id` over `session_conflicts`
— sessions with no conflict rows do not appear there. The pair-scanner
(RFC §4.2) that would replace the whole-session prompt with one row per
adjacent turn pair is v1.1 work and explicitly out of scope for v1.0; the
existing whole-session prompt still runs but now elicits the new
pair-keyed shape.

## Friction classifier (`friction_worker.py`)

Detects user-friction signals in short user-role messages (≤300 chars by
default). Seven labels: `status_ping`, `unmet_expectation`, `confusion`,
`interruption`, `correction`, `frustration`, `none`. Pipeline:

1. Pull user-role messages ≤`friction_max_chars` via `messages_text`.
2. Regex fast-path (`regex_fast_path`) catches unambiguous cases —
   `status_ping`, `interruption`, `correction`. Confidence 0.9.
3. Everything else goes to Sonnet 4.6 with `USER_FRICTION_SCHEMA`.
4. Session-level checkpointer + per-uuid anti-join so reruns on untouched
   sessions are free.
5. Output: `~/.claude/user_friction.parquet` with `source ∈ {regex, llm,
   refused}`.

**Why `unmet_expectation` lives in the LLM path, not regex:** a message
like `screenshot?` needs session context to disambiguate from a genuine
topic question. The LLM does this right; regex can't.

## Analytics pipeline determinism

`CLAUDE_SQL_SEED=42` (default) seeds UMAP, HDBSCAN, and Leiden so cluster
IDs and community IDs are stable across reruns. Don't reach for different
seeds without a reason. The Leiden seed flows into both
`leidenalg.find_partition(seed=...)` and `Optimiser.set_rng_seed(...)` for
the resolution-profile bisection, so same-seed reruns produce byte-equal
parquets.

## Leiden+CPM note

We use `leidenalg.find_partition(... CPMVertexPartition ...)` over an
`igraph` mutual-kNN graph (k=15, edge floor 0.3 by default) of session
centroids. **Do not reintroduce networkx Louvain** or the abandoned
`python-louvain` / `community` package. CPM is chosen over modularity
because for cosine-similarity edges γ has a closed-form interpretation
(Traag-Van Dooren-Nesterov 2011): communities have internal density ≥ γ
and external density ≤ γ, both expressed in cosine units. Modularity's
resolution parameter has no such semantics on weighted graphs and is
prone to the Fortunato-Barthélemy resolution limit; CPM is
resolution-limit-free for the dense-centroid scale claude-sql operates at.

Auto-γ runs `Optimiser.resolution_profile` over `(0.05, 0.95)` and picks
the longest-plateau γ — this is a free byproduct that produces a
sidecar parquet (`community_profile.parquet`) so an agent can ask "what γ
would give 50 communities?" without rerunning Leiden. Override the auto
pick with `community --gamma <float>` (skips the profile + sidecar) or
with `--resolution {coarse, medium, fine}` (picks alternate plateaus
from the same profile — no extra Leiden runs).

Connectivity post-check is **warn-only**. Park et al. (2024) reported up
to 16% of Leiden communities disconnected on biomedical citation graphs;
on symmetric mutual-kNN over normalized cosine centroids the rate is
materially lower. We log a warning if any community's induced subgraph
has multiple weakly-connected components but do NOT split. If the
warning ever fires on the live corpus regularly, file an issue and we'll
add the Park et al. Connectivity Modifier as a focused follow-up — but
don't pre-emptively land 30+ LOC of split + relabel logic that might
never run.

Top terms per community come from the existing `community_top_topics(cid, n)`
macro at `sql_views.py:898`, which composes from the message-level
`cluster_terms` parquet on demand. We do **not** snapshot per-community
top terms into `session_communities.parquet` — the macro is live and the
parquet column would freeze a derivation that should update whenever
clusters re-run.

The edge attribute on the igraph graph MUST be named exactly `"weight"`;
`leidenalg` looks it up by string when `find_partition(weights="weight",
...)` is called. Versions are pinned `leidenalg>=0.11.0,<0.12` and
`igraph>=1.0.0,<2.0` — both are reference-grade and the pin protects us
from determinism drift across patch versions.

## c-TF-IDF note

We use `sklearn.feature_extraction.text.CountVectorizer` + our own c-TF-IDF
math (per-class TF, IDF, L1 norm, ngram (1,2), min_df=2). Do not pull in
`bertopic` — we want the weighting logic visible and patchable.

## Resilience patterns to preserve

- `read_json(..., filename=true, union_by_name=true, sample_size=-1,
  ignore_errors=true, maximum_object_size=67108864)` — the corpus always
  has a few truncated or growing files; those flags keep them from
  aborting the query.
- `session_text.build_session_text` wraps its DuckDB query in
  `try/except duckdb.IOException` and skips stale JSONLs with a warning.
- `llm_worker._count_pending_sessions` uses a pure SQL
  `COUNT(DISTINCT session_id)` for dry-run — do not replace with a full
  materialization.
- Tenacity retries catch `SSLError`, `ConnectionError`, and
  `ThrottlingException` with exponential backoff. Botocore's own retry is
  disabled so tenacity owns the policy.
- Embeddings parquet: write with explicit `pl.Array(pl.Float32,
  output_dimension)` schema, otherwise polars infers `Object` and the
  roundtrip breaks.
- Model-ID cost lookup: strip the dated suffix via
  `re.sub(r'-\d{8}$', '', model_id)` before looking it up, so
  `claude-sonnet-4-6-20260315` matches the same entry as
  `claude-sonnet-4-6`.
- `BedrockRefusalError` is terminal: the LLM pipelines stamp a neutral
  placeholder row and clear the retry queue so refused messages don't
  cycle forever.
- **LanceDB embeddings store (`~/.claude/embeddings_lance/`).** Replaces
  the prior `embeddings/part-*.parquet` + `~/.claude/hnsw.duckdb` combo.
  Lance holds the FLOAT[1024] vectors AND the IVF_HNSW_SQ index in one
  versioned directory; DuckDB reads it back via the `lance` core
  extension (`INSTALL lance; LOAD lance; ATTACH '<dir>' AS lance_store
  (TYPE LANCE)`). `register_vss` probes the LanceDB namespace via
  `lance_store._has_table`; if the embeddings table is absent (fresh
  install), it creates an empty `message_embeddings` DuckDB table so
  the `semantic_search` macro still binds. Recovery for the user:
  `rm -rf ~/.claude/embeddings_lance/` then `claude-sql embed`. The
  legacy `embeddings/part-*.parquet` directory is left in place for
  rollback; one-time migration is automatic on first connect (idempotent
  via row-count check).
- **Empty-namespace gate.** DuckDB's `ATTACH (TYPE LANCE)` succeeds on
  any path — even a directory without a Lance dataset. The catalog
  error fires later at SELECT/CREATE-VIEW time. Always probe via
  `lance_store._has_table(db, TABLE_NAME)` before binding the view, NOT
  via filesystem heuristics like `any(dir.iterdir())`. This is
  pinned by `tests/test_lance_store.py`.
- **Sharded parquet caches.** Workers (`embed`, `classify`, `trajectory`,
  `conflicts`, `friction`) write each chunk as a fresh
  `<cache>/part-<ts_ns>.parquet` instead of read-concat-rewrite. Readers
  glob the directory via `claude_sql.parquet_shards.iter_part_files`.
  The legacy single-file path still works (Settings field is unchanged
  shape) for back-compat; new installs default to a directory. Recovery
  / consolidation is `claude-sql cache compact`. One-time migration of
  an old single-file cache is `claude-sql cache migrate` (defaults to
  `--dry-run`).
- **DuckDB tuning PRAGMAs.** Set inside `cli._open_connection` from
  `Settings.duckdb_threads / duckdb_memory_limit / duckdb_temp_dir`.
  `memory_limit` accepts percentage strings (`'70%'`); we resolve to
  MiB at apply time because DuckDB rejects `%` as a unit.
  `temp_directory` is `~/.claude/duckdb_tmp` instead of `/tmp` —
  Amazon devboxes ship `/tmp` as a 4 GB tmpfs and a clustering spill
  there thrashes the host.
- **Cluster mtime sidecar.** `cluster_worker.run_clustering` skips the
  ~40 s UMAP+HDBSCAN refit when the embeddings haven't moved by
  comparing the latest part-file mtime against
  `clusters.parquet.embeddings_mtime`. Older installs without a sidecar
  get one stamped on the next call. `force=True` always rebuilds.

## Agent-friendly CLI surface (load-bearing — do not regress)

- `--format {auto,table,json,ndjson,csv}` on every subcommand. `auto`
  resolves to `table` on TTY, `json` on pipe — agents calling via
  subprocess get JSON for free.
- Structured DuckDB errors: parse → exit 64, catalog → exit 65, runtime
  → exit 70. Non-TTY stderr carries `{"error": {"kind","message","hint"}}`.
  Keep `claude_sql.output.classify_duckdb_error` in sync if new DuckDB
  exception subclasses land.
- `list-cache` reports every parquet's `{exists, bytes, mtime, rows}`.
  When adding a new analytics parquet, extend `_describe_cache_entry`
  calls in `cli.py` so it shows up.
- `explain` default is static (`EXPLAIN`, no execution). `--analyze`
  opts into `EXPLAIN ANALYZE`. Do NOT flip this back — the prior default
  silently executed slow queries when agents probed plans.
- `--quiet` is honored by every subcommand via `_configure`. View
  registration logs are at DEBUG on purpose so the default stderr stays
  empty for read-only flows. Warnings are reserved for genuinely
  actionable state (e.g., missing embeddings parquet).

## Environment variables

All prefixed `CLAUDE_SQL_`. Defaults are in `config.py`; see the README
for the full table. The common overrides:

- `CLAUDE_SQL_DEFAULT_GLOB` — override the main transcript glob
- `CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH` — move the embeddings cache
  (path now points at a directory by default; legacy single-file path
  still works)
- `CLAUDE_SQL_USER_FRICTION_PARQUET_PATH` — move the friction cache
- `CLAUDE_SQL_FRICTION_MAX_CHARS` — short-message cutoff (default 300)
- `CLAUDE_SQL_EMBED_CONCURRENCY` — parallel Cohere Embed v4 calls
  (default 8 — sustained on global CRIS without throttling)
- `CLAUDE_SQL_LLM_CONCURRENCY` — parallel Sonnet 4.6 calls (default 2)
- `CLAUDE_SQL_DUCKDB_THREADS` — override worker threads (default
  `os.cpu_count()`)
- `CLAUDE_SQL_DUCKDB_MEMORY_LIMIT` — `'70%'` of host RAM by default;
  also accepts absolute units like `'4GB'`
- `CLAUDE_SQL_DUCKDB_TEMP_DIR` — spill directory; default
  `~/.claude/duckdb_tmp`
- `CLAUDE_SQL_LANCE_URI` — LanceDB embeddings store; default
  `~/.claude/embeddings_lance`

## Operational rollback paths

- LanceDB corruption / version mismatch — `rm -rf ~/.claude/embeddings_lance/`,
  then `claude-sql embed --since-days 14 --no-dry-run` (or just `analyze`).
  The legacy parquet shards remain at `~/.claude/embeddings/` and re-migrate
  automatically on the next connect.
- Sharded parquet directory grows large — `claude-sql cache compact
  --no-dry-run` consolidates `<cache>/part-*.parquet` into a single
  consolidated file.
- Memory pressure on a shared host — set
  `CLAUDE_SQL_DUCKDB_MEMORY_LIMIT='4GB'` (or any absolute size) to
  bound DuckDB.
- Bedrock throttling on a new model — drop
  `CLAUDE_SQL_EMBED_CONCURRENCY` or `CLAUDE_SQL_LLM_CONCURRENCY` to 2.
  Tenacity already absorbs short bursts.
- Slow query investigation — `claude-sql query "..." --profile-json`
  drops a JSON timing tree under `~/.claude/profiling/`.

## Deferred decisions (named so they don't get re-derived)

- **Snapshot tier / `CacheNode` DAG.** Considered and deferred — the
  2 GB / 15K-JSONL corpus doesn't justify the abstraction yet. Reopen
  when `EXPLAIN ANALYZE` shows JSONL re-scan dominating wall clock, or
  the corpus crosses ~10 GB.
- **`CreateModelInvocationJob` batch path / vector quantization.**
  Async-on-asyncio at concurrency 8 saturates global CRIS without
  throttling; VSS doesn't natively support FLOAT16/INT8. Reopen when
  embeddings parquet crosses ~10 GB.

## Commits & pushes

- Keep commits small and focused — one logical change per commit, per the
  project's established rhythm.
- **All commits must be conventional** — the `commit-msg` hook enforces
  this; `git commit --no-verify` is an escape hatch, not a policy.
- `mise run check` must pass before every commit (pre-commit runs
  ruff + ty automatically; `mise run check` additionally runs pytest).
- **Direct push to `origin/main` is rejected** by branch-protection
  (GH013). Open a PR for every change, including the `cz bump` release
  commit. `pre-push` runs pytest as a final safety net before the push
  to your feature branch.
- Use `mise run bump` (not hand-rolled tags) for version releases. It
  updates `CHANGELOG.md` + `pyproject.toml` + `uv.lock` atomically. See
  the *Release flow under branch-protection* section above for the full
  branch → PR → squash-merge → re-tag → push-tag → cut-release sequence.
