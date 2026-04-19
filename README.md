# claude-sql

Zero-copy SQL over `~/.claude/` JSONL transcripts with Cohere Embed v4
semantic search (via DuckDB VSS / HNSW).

## What it does

No ETL, no parquet materialization for the transcripts themselves. DuckDB
reads `~/.claude/projects/**/*.jsonl` directly via `read_json(...,
filename=true, union_by_name=true, sample_size=-1)`, and 11 business-level
views are registered on top at connection open. Task tracking (`todo_events`,
`todo_state_current`) and subagents (`subagent_sessions`, `subagent_messages`)
are first-class citizens alongside `sessions` and `messages`. Semantic search
runs on an HNSW cosine index over Cohere Embed v4 vectors stored in a single
parquet file that gets rebuilt into memory on every `claude-sql` invocation.

## Install

```bash
uv sync
```

Optional: `mise install` if you want the `mise run check` / `mise run test`
tasks. The `duckdb` binary on PATH is needed for `claude-sql shell`; Bedrock
access (AWS creds in the environment) is needed for `claude-sql embed` and
`claude-sql search`. Everything else is purely local.

## Quick tour

```bash
# Inspect everything that's registered (11 views, 6 macros).
uv run claude-sql schema

# Run a query — this is the work-item acceptance prompt.
uv run claude-sql query "
  SELECT session_id, model_used(session_id) AS model,
         cost_estimate(session_id) AS usd
  FROM sessions
  WHERE started_at >= current_timestamp - INTERVAL 30 DAY
    AND model_used(session_id) LIKE '%opus%'
    AND cost_estimate(session_id) > 5.0
  ORDER BY usd DESC
"

# EXPLAIN ANALYZE plan with pushdown markers highlighted in green.
uv run claude-sql explain "SELECT * FROM messages WHERE session_id = '11111111-...' LIMIT 1"

# Drop into the DuckDB REPL with everything pre-registered.
uv run claude-sql shell

# Backfill embeddings (Cohere Embed v4 on Bedrock).
AWS_PROFILE=lalsaado-handson uv run claude-sql embed --since-days 30

# Semantic search.
uv run claude-sql search "temporal workflow determinism" --k 10
```

See [docs/cookbook.md](docs/cookbook.md) for more recipes with real outputs
pasted from the dev-host corpus.

## v2 analytics

Three new capabilities sit on top of the v1 substrate:

1. **Session classification** — Sonnet 4.6 with Bedrock structured output
   tags every session with `autonomy_tier`, `work_category`, `success`,
   `goal`, and `confidence`.
2. **Message clustering** — UMAP 50d -> UMAP 2d -> HDBSCAN over the Cohere
   Embed v4 vectors; in-house c-TF-IDF labels each cluster with its top 10
   1-2gram terms.
3. **Session community detection** — Louvain (`networkx` >= 3.4) over a
   cosine-similarity graph of session centroids, yielding coarse topic
   neighborhoods per session.

End-to-end dry-run:

```bash
uv run claude-sql analyze --dry-run --since-days 30
```

Prints pending counts and cost estimates for every LLM stage; pass
`--no-dry-run` to spend real money. `--skip-<stage>` drops an individual
stage; `--force-cluster` / `--force-community` rebuild those (non-LLM)
parquets.

**Six new subcommands.** All LLM-touching commands default to `--dry-run`.

| Subcommand | One-line |
|---|---|
| `classify` | Sonnet 4.6 session-level autonomy + work + success + goal |
| `trajectory` | Per-message sentiment delta + is_transition (regex prefilter -> Sonnet) |
| `conflicts` | Per-session stance-conflict detection via Sonnet |
| `cluster` | UMAP + HDBSCAN over `message_embeddings`; writes `clusters.parquet` |
| `community` | Louvain community detection over session centroids |
| `analyze` | Orchestrator: embed -> cluster + community -> classify -> trajectory -> conflicts |

(`terms` also exists as a standalone step; `analyze` runs it automatically
after `cluster`.)

**Six new analytics macros.**

| Macro | Signature |
|---|---|
| `autonomy_trend` | `(window_days) -> TABLE(week, autonomy_tier, n)` |
| `work_mix` | `(since_days) -> TABLE(work_category, n)` |
| `success_rate_by_work` | `(since_days) -> TABLE(work_category, sessions, success_rate, failure_rate, partial_rate)` |
| `cluster_top_terms` | `(cluster_id, n) -> TABLE(term, weight, rank)` |
| `community_top_topics` | `(community_id, n) -> TABLE(cluster_id, n_msgs, top_terms)` |
| `sentiment_arc` | `(session_id) -> TABLE(ts, role, sentiment_delta, is_transition, text)` |

**Seven new parquet-backed views** register at connection open if their
parquet exists (missing ones warn and no-op): `session_classifications`,
`session_goals`, `message_trajectory`, `session_conflicts`,
`message_clusters`, `cluster_terms`, `session_communities`.

`claude-sql classify --dry-run` is the recommended pre-flight — it prints
a cost estimate ($/MTok * pending-session count) before you commit to the
real Bedrock call.

See [docs/analytics_cookbook.md](docs/analytics_cookbook.md) for runnable
recipes per capability.

## Architecture

Four modules under `src/claude_sql/`:

- `config.py` — pydantic v2 `Settings` with env prefix `CLAUDE_SQL_`.
- `sql_views.py` — DuckDB views + macros + VSS/HNSW setup.
- `embed_worker.py` — async Bedrock embedding worker with tenacity retry
  and `asyncio.gather` over a semaphore.
- `cli.py` — cyclopts CLI.

Data flow:

```
~/.claude/projects/*/*.jsonl ──┐
                               │
~/.claude/projects/*/*/        │  read_json(filename=true,
  subagents/agent-*.jsonl ─────┼──> union_by_name=true,      ──> v_raw_*
                               │  sample_size=-1)                views
~/.claude/projects/*/*/        │
  subagents/agent-*.meta.json ─┘
                                     │
                                     ▼
                               11 business-level views
                               (sessions, messages, content_blocks,
                                messages_text, tool_calls, tool_results,
                                todo_events, todo_state_current, task_spawns,
                                subagent_sessions, subagent_messages)
                                     │
        ┌────────────────────────────┼──────────────────────────┐
        │                            │                          │
        ▼                            ▼                          ▼
   claude-sql query /          claude-sql embed          claude-sql search
   explain / schema            (writes parquet)           (HNSW + macro)
                                     │                          ▲
                                     ▼                          │
                         ~/.claude/embeddings.parquet ──────────┘
                         (rebuilt into message_embeddings
                          + HNSW idx on every connection open)
```

## Views

| View | One-line description | Key column(s) |
|---|---|---|
| `sessions` | One row per top-level transcript file. | `session_id`, `started_at` |
| `messages` | User + assistant events with usage counters. | `uuid`, `session_id`, `model` |
| `content_blocks` | Flattened `message.content[]`. | `block_type`, `tool_name` |
| `messages_text` | Text-only substrate for embeddings. | `uuid`, `text_content` |
| `tool_calls` | `content_blocks` where `block_type='tool_use'`. | `tool_name`, `tool_use_id` |
| `tool_results` | `content_blocks` where `block_type='tool_result'`. | `tool_use_id`, `content` |
| `todo_events` | One row per todo per `TodoWrite` snapshot. | `subject`, `status`, `snapshot_ix` |
| `todo_state_current` | Latest status per `(session, subject)`. | `status`, `written_at` |
| `task_spawns` | `Task` / `Agent` / `TaskCreate` launch sites. | `subagent_type`, `prompt` |
| `subagent_sessions` | One row per subagent run. | `parent_session_id`, `agent_hex`, `agent_type` |
| `subagent_messages` | user+assistant events from subagent transcripts. | `parent_session_id`, `agent_hex` |

Full column listings in [docs/jsonl_schema_v1.sql](docs/jsonl_schema_v1.sql).

## Macros

| Macro | Signature | Description |
|---|---|---|
| `model_used` | `(sid) -> VARCHAR` | Latest `model` observed in a session. |
| `cost_estimate` | `(sid) -> DOUBLE` | USD via `config.DEFAULT_PRICING` (in + cache-write at in_rate, out at out_rate). |
| `tool_rank` | `(last_n_days) -> TABLE(tool_name, n)` | Tool-use leaderboard. |
| `todo_velocity` | `(sid) -> DOUBLE` | `completed / distinct_subjects` in `todo_state_current`. |
| `subagent_fanout` | `(sid) -> BIGINT` | Count of `subagent_sessions` with `parent_session_id = sid`. |
| `semantic_search` | `(query_vec, k) -> TABLE(uuid, sim, distance)` | HNSW cosine top-k over `message_embeddings`. |

## Env vars

All settings are overridable via env vars prefixed `CLAUDE_SQL_` (or via a
`.env` file in the working directory). Fields live on
`claude_sql.config.Settings`.

| Env var | Default | Purpose |
|---|---|---|
| `CLAUDE_SQL_DEFAULT_GLOB` | `~/.claude/projects/*/*.jsonl` | Top-level session transcripts. |
| `CLAUDE_SQL_SUBAGENT_GLOB` | `~/.claude/projects/*/*/subagents/agent-*.jsonl` | Subagent transcripts. |
| `CLAUDE_SQL_SUBAGENT_META_GLOB` | `~/.claude/projects/*/*/subagents/agent-*.meta.json` | Sibling meta for subagents. |
| `CLAUDE_SQL_REGION` | `us-east-1` | Bedrock region. |
| `CLAUDE_SQL_MODEL_ID` | `global.cohere.embed-v4:0` | Cohere Embed v4 global CRIS profile (sustained ~220 vec/s with zero throttling). |
| `CLAUDE_SQL_OUTPUT_DIMENSION` | `1024` | Embedding dim (256 / 512 / 1024 / 1536). |
| `CLAUDE_SQL_EMBEDDING_TYPE` | `int8` | Type Cohere returns (cast to `FLOAT[]` at load). |
| `CLAUDE_SQL_CONCURRENCY` | `8` | Embed-worker concurrency. |
| `CLAUDE_SQL_BATCH_SIZE` | `96` | Max texts per Cohere request. |
| `CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH` | `~/.claude/embeddings.parquet` | Where the embed worker writes. |
| `CLAUDE_SQL_HNSW_METRIC` | `cosine` | HNSW distance metric. |
| `CLAUDE_SQL_HNSW_EF_CONSTRUCTION` | `128` | HNSW build param. |
| `CLAUDE_SQL_HNSW_EF_SEARCH` | `64` | HNSW query-time param. |
| `CLAUDE_SQL_HNSW_M` | `16` | HNSW `M`. |
| `CLAUDE_SQL_HNSW_M0` | `32` | HNSW `M0`. |

Why `global.cohere.embed-v4:0`: the direct `cohere.embed-v4:0` profile and
the US CRIS pool both throttle aggressively on modest token rates (8 × 96
batches blew through the TPM bucket in seconds in real-corpus testing).
The `global.*` CRIS profile routes to the worldwide Bedrock capacity pool
and sustained 223 vec/s at concurrency=8 with zero `ThrottlingException`s.
Kept as a single default; no routing knob.

## Development

```bash
mise run check   # ruff lint + ruff format --check + ty check src/ + pytest (15 tests)
mise run test
mise run lint
mise run typecheck
```

## Links

- [Research report](../claude-sql-zero-copy-engine-research.md) — 22 sources,
  design rationale, pricing comparisons.
- [docs/cookbook.md](docs/cookbook.md) — runnable recipes with real outputs.
- [docs/research_notes.md](docs/research_notes.md) — schema stability + HNSW
  + embedding design notes.
- [docs/jsonl_schema_v1.sql](docs/jsonl_schema_v1.sql) — captured DESCRIBE
  for every registered view and the three raw read_json sources.
- [docs/queries/thread_walk.sql](docs/queries/thread_walk.sql) — recursive
  CTE that walks `parent_uuid -> uuid` to reconstruct conversation trees.
