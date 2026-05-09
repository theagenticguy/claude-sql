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
   the clusters, Louvain groups sessions into communities over cosine-
   similarity session centroids.

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
selector catches most issues, but two CodeQL rules fire on patterns
ruff lets through. Both have a one-line fix; both are easy to forget.

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

Both checks run automatically — there's no `mise run codeql` task.
Treat the comment / unused-name discipline as part of writing the test
in the first place. The rules below in `pyproject.toml` should be kept
maximally aggressive so the *next* class of CodeQL findings has a
ruff-side analog wherever one exists.

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

## Version bumping & changelog

Version management is driven by `cz bump`, which reads commit history,
picks MAJOR/MINOR/PATCH from the conventional-commits types, updates
`pyproject.toml` + `uv.lock` (via `version_provider = "uv"`), writes
`CHANGELOG.md`, and creates an annotated tag.

- `mise run bump:dry-run` — preview the next version + tag without writing.
- `mise run bump` — bump, update changelog, commit as
  `chore(release): X → Y`, create annotated `vY` tag.
- `mise run changelog` — regenerate `CHANGELOG.md` from history without bumping.

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

## Structured output (Sonnet classification)

- Uses Bedrock's GA `output_config.format` field (not tool_use/tool_choice).
- Schemas are pydantic v2 models in `schemas.py`; `model_json_schema()` is
  then flattened: inline `$ref`, inject `additionalProperties: false`,
  strip the numeric/string constraints the validator rejects from Draft
  2020-12 subset.
- Adaptive thinking stays on. `citations` is the only feature incompatible
  with structured output — do not add it.

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

`CLAUDE_SQL_SEED=42` (default) seeds UMAP, HDBSCAN, and Louvain so cluster
IDs and community IDs are stable across reruns. Don't reach for different
seeds without a reason.

## Louvain note

We use `networkx.community.louvain_communities` (built into networkx ≥3.4).
Do not reintroduce the abandoned `python-louvain` / `community` package.

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
- **Persistent HNSW (`~/.claude/hnsw.duckdb`).** Built lazily on first
  command and reused across runs via DuckDB's
  `hnsw_enable_experimental_persistence` flag. `register_vss` ATTACHes
  it as `hnsw_store`, mtime-checks against the embeddings parquet, and
  rebuilds when the parquet is newer or the catalog comes back empty.
  ATTACH on a corrupt file is caught and the store is unlinked +
  rebuilt from parquet automatically. Recovery for the user:
  `rm ~/.claude/hnsw.duckdb`. Do NOT couple this with
  `claude_sql.duckdb` — separate file per state surface.
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
- `CLAUDE_SQL_CONCURRENCY` — DEPRECATED: aliases onto both pipelines
  with a `DeprecationWarning`. Removed in the next release.
- `CLAUDE_SQL_DUCKDB_THREADS` — override worker threads (default
  `os.cpu_count()`)
- `CLAUDE_SQL_DUCKDB_MEMORY_LIMIT` — `'70%'` of host RAM by default;
  also accepts absolute units like `'4GB'`
- `CLAUDE_SQL_DUCKDB_TEMP_DIR` — spill directory; default
  `~/.claude/duckdb_tmp`
- `CLAUDE_SQL_HNSW_DB_PATH` — persistent HNSW store; default
  `~/.claude/hnsw.duckdb`

## Operational rollback paths

- HNSW corruption / version mismatch — `rm ~/.claude/hnsw.duckdb`. Next
  command rebuilds the index from `embeddings/`.
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
