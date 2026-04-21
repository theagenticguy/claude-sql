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
- **Formatting:** `mise run fmt:write` applies ruff formatting. Line length
  is 100. Ruff lint selectors are `E, F, I, N, UP, B, SIM`.
- **Type checker:** `ty` (not `mypy` or `pyright`). `mise run typecheck`.
  Editor Pyright warnings about unresolved imports are false positives from
  a global install that can't see the project's `.venv`; trust `ty`.
- **Tests:** `pytest` under `tests/`. `mise run test`. Tests must not hit
  Bedrock — mock the client. The live corpus is the one under
  `~/.claude/projects/`, so integration tests should use a small fixture
  directory, not the live one.

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
- `CLAUDE_SQL_USER_FRICTION_PARQUET_PATH` — move the friction cache
- `CLAUDE_SQL_FRICTION_MAX_CHARS` — short-message cutoff (default 300)
- `CLAUDE_SQL_CONCURRENCY` — bump parallel Bedrock calls (default 2)

## Commits & pushes

- Keep commits small and focused — one logical change per commit, per the
  project's established rhythm.
- **All commits must be conventional** — the `commit-msg` hook enforces
  this; `git commit --no-verify` is an escape hatch, not a policy.
- `mise run check` must pass before every commit (pre-commit runs
  ruff + ty automatically; `mise run check` additionally runs pytest).
- Push to `origin/main` only after the local check is green. `pre-push`
  runs pytest as a final safety net.
- Use `mise run bump` (not hand-rolled tags) for version releases. It
  updates `CHANGELOG.md` + `pyproject.toml` + `uv.lock` atomically.
