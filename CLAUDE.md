# claude-sql — project instructions

This file is read by Claude Code (and any agent running in this repo). It tells
you how to work on and with `claude-sql` without re-deriving the design each
time.

## What this project is

A standalone Python CLI that makes the user's own Claude Code transcripts —
everything under `~/.claude/projects/**/*.jsonl` plus subagent sidecar files —
queryable in place. Three layers stack on top of those JSONLs:

1. **SQL** — DuckDB reads the JSONLs with zero copy; 18 views + 12 macros
   normalize messages, tool calls, todos, subagents, costs.
2. **Semantic search** — Cohere Embed v4 (Bedrock, global CRIS) embeds every
   message; DuckDB VSS HNSW cosine index serves top-k in milliseconds.
3. **LLM analytics** — Sonnet 4.6 (global CRIS, `output_config.format`
   structured output) classifies sessions (autonomy tier, work category,
   success, goal), scores per-message sentiment, and detects stance conflicts.
   UMAP + HDBSCAN cluster message embeddings; c-TF-IDF names the clusters;
   Louvain groups sessions into communities.

Every expensive output (embeddings, classifications, clusters, communities)
is cached in parquet under `~/.claude/`. Views register only the parquets
that exist — missing ones warn and no-op, never crash.

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
- **Type checker:** `ty` (not `mypy`). `mise run typecheck`.
- **Tests:** `pytest` under `tests/`. `mise run test`. Tests must not hit
  Bedrock — mock the client. The live corpus is the one under
  `~/.claude/projects/`, so integration tests should use a small fixture
  directory, not the live one.

## How to run it

- **As a uv tool** (preferred end-user path): `mise run tool:install` →
  `claude-sql --version`. Upgrade with `uv tool upgrade claude-sql`, remove
  with `mise run tool:uninstall`.
- **In the project venv** (development): `mise run install` → `mise run cli -- <subcommand>`.
- **REPL:** `claude-sql shell` drops into DuckDB with every view + macro
  already registered.

## Bedrock

- **Region:** `us-east-1`. Embeds + Sonnet classification go through the
  **global** CRIS profiles — `global.cohere.embed-v4:0` and
  `global.anthropic.claude-sonnet-4-6`. Direct model IDs and US-only CRIS
  throttled under load; global is the only path that sustains throughput.
- **IAM:** `bedrock:InvokeModel` on both inference profiles above.
- **Credentials:** `AWS_PROFILE=<your-profile>` in the environment. The CLI
  reads it via boto3's standard chain.
- **Cost guard:** every command that spends real money defaults to
  `--dry-run`. `analyze` chains embed → cluster → classify → trajectory →
  conflicts and respects `--dry-run` on every step. The dry-run path uses
  a pure-SQL count, not a full materialization, so it stays fast on large
  corpora.

## Structured output (Sonnet classification)

- Uses Bedrock's GA `output_config.format` field (not tool_use/tool_choice).
- Schemas are pydantic v2 models in `schemas.py`; `model_json_schema()` is
  then flattened: inline `$ref`, inject `additionalProperties: false`,
  strip the numeric/string constraints the validator rejects from Draft
  2020-12 subset.
- Adaptive thinking stays on. `citations` is the only feature incompatible
  with structured output — do not add it.

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
- Model-ID cost lookup: strip the dated suffix via `re.sub(r'-\d{8}

, '', model_id)`
  before looking it up, so `claude-sonnet-4-6-20260315` matches the same
  entry as `claude-sonnet-4-6`.

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
- `CLAUDE_SQL_CONCURRENCY` — bump parallel Bedrock calls (default 2)

## Commits & pushes

- Keep commits small and focused — one logical change per commit, per the
  project's established rhythm.
- `mise run check` must pass before every commit.
- Push to `origin/main` only after the local check is green.
