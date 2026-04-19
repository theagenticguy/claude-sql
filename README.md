# claude-sql

Zero-copy SQL over `~/.claude/projects/**/*.jsonl` transcripts, with
Cohere Embed v4 semantic search via DuckDB VSS. The corpus is queried in
place through `read_json` — no ETL, no parquet ingest (except for
embeddings, which are append-only). Views and macros are registered into
a fresh in-memory DuckDB connection on every command.

## Install

```bash
uv sync
```

Requires Python 3.12+ and the `duckdb` binary on PATH if you want to use
`claude-sql shell`. Bedrock access is needed for `claude-sql embed` /
`claude-sql search`; everything else is local.

## Quick start

```bash
# List every registered view with columns, plus the macro inventory.
uv run claude-sql schema

# Run a SQL query (results print as a polars table).
uv run claude-sql query "SELECT count(*) FROM sessions"

# Show the EXPLAIN ANALYZE plan with pushdown markers highlighted.
uv run claude-sql explain "SELECT uuid, role FROM messages WHERE session_id = '<uuid>'"

# Backfill embeddings via Cohere Embed v4 on Bedrock.
uv run claude-sql embed --since-days 30

# Semantic search (requires embeddings).
uv run claude-sql search "temporal determinism" --k 5

# Or drop into the DuckDB REPL with everything pre-registered.
uv run claude-sql shell
```

See `docs/cookbook.md` for ready-to-run recipes against the real corpus.

## Views

- **sessions** — one row per top-level transcript, rolled up from raw
  JSONL grouped by the filename-derived session id.
- **messages** — user + assistant events with token usage and
  `content_json` kept as JSON for lazy flattening.
- **content_blocks** — one row per element in `message.content[]`, typed
  by `block_type` (text, tool_use, tool_result, thinking).
- **messages_text** — text blocks only; substrate for embeddings.
- **tool_calls** — `content_blocks` where `block_type = 'tool_use'`.
- **tool_results** — `content_blocks` where `block_type = 'tool_result'`.
- **todo_events** — one row per todo per TodoWrite snapshot, ordered by
  `snapshot_ix`.
- **todo_state_current** — latest status per `(session_id, subject)`.
- **task_spawns** — `tool_calls` filtered to Task / Agent / TaskCreate
  launch sites.
- **subagent_sessions** — rolled-up subagent runs joined with the sibling
  `agent-*.meta.json` files for `agent_type` + `description`.
- **subagent_messages** — user + assistant events from subagent
  transcripts.

## Macros

- **model_used(sid)** — latest model used in a session.
- **cost_estimate(sid)** — USD via `config.DEFAULT_PRICING`.
- **tool_rank(last_n_days)** — table macro; tool-use leaderboard.
- **todo_velocity(sid)** — completed / distinct todos.
- **subagent_fanout(sid)** — count of subagent runs for a session.
- **semantic_search(query_vec, k)** — table macro issuing HNSW-backed
  top-k over `message_embeddings`.

## Config

All settings are overridable via env vars prefixed `CLAUDE_SQL_` (or via
`.env` in the working directory). Fields live on
`claude_sql.config.Settings`.

| Env var | Default | Purpose |
|---|---|---|
| `CLAUDE_SQL_DEFAULT_GLOB` | `~/.claude/projects/*/*.jsonl` | Top-level session transcripts. |
| `CLAUDE_SQL_SUBAGENT_GLOB` | `~/.claude/projects/*/*/subagents/agent-*.jsonl` | Subagent transcripts. |
| `CLAUDE_SQL_SUBAGENT_META_GLOB` | `~/.claude/projects/*/*/subagents/agent-*.meta.json` | Sibling meta for subagents. |
| `CLAUDE_SQL_REGION` | `us-east-1` | Bedrock region. |
| `CLAUDE_SQL_MODEL_ID` | `cohere.embed-v4:0` | Direct on-demand model id. |
| `CLAUDE_SQL_CRIS_MODEL_ID` | `us.cohere.embed-v4:0` | CRIS failover profile. |
| `CLAUDE_SQL_USE_CRIS` | `false` | Send requests to the CRIS profile instead. |
| `CLAUDE_SQL_OUTPUT_DIMENSION` | `1024` | Embedding dim (256 / 512 / 1024 / 1536). |
| `CLAUDE_SQL_EMBEDDING_TYPE` | `int8` | Type Cohere returns (stored as `FLOAT[]` in DuckDB). |
| `CLAUDE_SQL_CONCURRENCY` | `8` | Embed-worker concurrency. |
| `CLAUDE_SQL_BATCH_SIZE` | `96` | Max texts per Cohere request. |
| `CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH` | `~/.claude/embeddings.parquet` | Where the embed worker writes. |
| `CLAUDE_SQL_HNSW_METRIC` | `cosine` | HNSW distance metric. |
| `CLAUDE_SQL_HNSW_EF_CONSTRUCTION` | `128` | HNSW build param. |
| `CLAUDE_SQL_HNSW_EF_SEARCH` | `64` | HNSW query-time param. |
| `CLAUDE_SQL_HNSW_M` | `16` | HNSW `M`. |
| `CLAUDE_SQL_HNSW_M0` | `32` | HNSW `M0`. |

`model_id` vs `cris_model_id`: `model_id` is the direct on-demand profile
(fastest path in-region). `cris_model_id` is the cross-region inference
profile — flip `CLAUDE_SQL_USE_CRIS=true` when the direct profile throttles
or when running outside us-east-1.

## Related docs

- [docs/cookbook.md](docs/cookbook.md) — recipes with real-corpus output.
- [docs/research_notes.md](docs/research_notes.md) — design rationale and
  what we verified live.
- [docs/jsonl_schema_v1.sql](docs/jsonl_schema_v1.sql) — captured DESCRIBE
  for every registered view.
- [docs/queries/thread_walk.sql](docs/queries/thread_walk.sql) — recursive
  CTE that walks `parent_uuid -> uuid` to reconstruct conversation trees.
