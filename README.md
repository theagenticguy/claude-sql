# claude-sql

> **Ask your Claude Code transcripts anything.**
> Your sessions are already on disk. This turns them into a searchable,
> explorable, self-improving record of your work.

## What you get out of it

**Remember what you worked on.**

- "What was that thing I did last Tuesday with DuckDB and HNSW?"
- "Show me every conversation I've had about temporal workflows,
  ranked by relevance."
- "Which week did I finally figure out the memory RCA?"

**See where your time and money actually go.**

- "Which sessions cost me more than $5 on Opus this month — and what
  was I trying to do?"
- "Which tools am I leaning on most? Which ones fail the most?"
- "Where am I spending hours on prose vs. on tool calls?"

**Notice patterns in how you work.**

- "When do I hand-hold the agent step-by-step vs. let it run on its own?
  Has that shifted?"
- "What kinds of work am I doing most — coding, strategy, admin, writing?"
- "Which session types actually finish successfully vs. trail off?"
- "Which todos do I create and never close out?"

**Surface themes across hundreds of conversations.**

- "Group my sessions by what they're *about* and tell me what moved
  this month."
- "Show me the 10 biggest themes in my work and what's trending."
- "When I've wrestled with the same problem across multiple sessions,
  group them together so I can see the arc."

**Catch yourself disagreeing with yourself.**

- "Find sessions where I took two opposing positions on the same
  decision — and flag which ones got resolved vs. abandoned."

`claude-sql` turns every one of those into a SQL query that runs in
under a second on the live JSONL corpus — no export, no pipeline.

## How it works

```mermaid
flowchart LR
    J["~/.claude/projects/<br/>*.jsonl"] -->|read_json| R[raw views]
    J2["subagents/<br/>agent-*.jsonl"] -->|read_json| R
    R --> V[18 business views]
    V --> Q[["claude-sql query<br/>claude-sql explain<br/>claude-sql schema"]]
    V --> E["claude-sql embed<br/>(Cohere v4 on Bedrock)"]
    E --> P["embeddings.parquet"]
    P --> H["HNSW index<br/>(DuckDB VSS)"]
    H --> S[["claude-sql search"]]
    V --> L["claude-sql classify / trajectory /<br/>conflicts (Sonnet 4.6 +<br/>output_config.format)"]
    L --> PA["classifications / trajectory /<br/>conflicts parquets"]
    P --> C["claude-sql cluster<br/>(UMAP + HDBSCAN)"]
    C --> PC["clusters + cluster_terms<br/>(c-TF-IDF)"]
    P --> CM["claude-sql community<br/>(Louvain over centroids)"]
    CM --> PM["session_communities<br/>parquet"]
    PA --> AV[analytics views + macros]
    PC --> AV
    PM --> AV
    AV --> Q
```

Every parquet is cached and rebuilt only on explicit re-run; views register
over whichever parquets exist at connection open (missing ones warn and
no-op).

## Install

### As a uv tool (recommended)

`claude-sql` is **not published to PyPI**. You install it from a local
checkout of this repo. `mise run tool:install` wraps `uv tool install
--from . claude-sql --force --reinstall` so the binary on your `PATH`
lands in an isolated uv-managed venv.

```bash
git clone https://github.com/theagenticguy/claude-sql.git
cd claude-sql
mise run tool:install     # → uv tool install --from . claude-sql --force --reinstall
claude-sql --version      # prints version + "installed from directory: /path/to/checkout"
```

To upgrade after pulling new commits, re-run the same task (or the
alias `tool:upgrade` — identical command, nicer grep):

```bash
git pull
mise run tool:upgrade     # same as tool:install, just clearer intent
```

> :warning: `uv tool upgrade claude-sql` does **not** work — it resolves
> against the PyPI registry, which has no `claude-sql` package. Always
> reinstall from your checkout.

Remove:

```bash
mise run tool:uninstall   # → uv tool uninstall claude-sql
```

### Project install (for development)

```bash
git clone https://github.com/theagenticguy/claude-sql.git
cd claude-sql
mise install              # fetch pinned Python 3.12 + uv
mise run install          # uv sync --all-extras
mise run check            # ruff + fmt + ty + pytest
```

`mise` auto-activates `.venv` on `cd`. Every command below is also
available as a mise task: `mise tasks` prints the full list.

### AWS creds

Semantic search + Sonnet classification require Bedrock access.

```bash
export AWS_PROFILE=your-profile
```

The IAM policy needs `bedrock:InvokeModel` on
`inference-profile/global.cohere.embed-v4:0` and
`inference-profile/global.anthropic.claude-sonnet-4-6`.

## Quick tour

```bash
# Inspect every registered view + macro
claude-sql schema

# Answer the work-item acceptance prompt
claude-sql query "
  SELECT session_id, model_used(session_id) AS model,
         cost_estimate(session_id) AS usd
  FROM sessions
  WHERE started_at >= current_timestamp - INTERVAL 30 DAY
    AND model_used(session_id) LIKE '%opus%'
    AND cost_estimate(session_id) > 5.0
  ORDER BY usd DESC
"

# See the EXPLAIN ANALYZE plan with pushdown markers highlighted
claude-sql explain "SELECT * FROM messages WHERE session_id = '...' LIMIT 1"

# Drop into the DuckDB REPL with everything pre-registered
claude-sql shell

# Backfill embeddings (Cohere Embed v4 global CRIS)
AWS_PROFILE=... claude-sql embed --since-days 30

# Semantic search
claude-sql search "temporal workflow determinism" --k 10

# Classify every recent session (dry-run prints a cost estimate first)
claude-sql classify --dry-run --since-days 30
AWS_PROFILE=... claude-sql classify --no-dry-run --since-days 30

# Full analytics pipeline (embed → cluster → classify → trajectory → conflicts)
AWS_PROFILE=... claude-sql analyze --since-days 30 --no-dry-run
```

More recipes in [docs/analytics_cookbook.md](docs/analytics_cookbook.md).

## CLI surface

All 13 subcommands share top-level flags: `--verbose` / `--quiet`,
`--glob`, `--subagent-glob`, and `--format {auto,table,json,ndjson,csv}`.
Commands that spend real Bedrock money default to `--dry-run`.

| Command | Purpose |
|---|---|
| `schema` | List every view + its columns, plus every macro |
| `query <sql>` | Run a query, emit result as table / JSON / NDJSON / CSV |
| `explain <sql>` | Static `EXPLAIN` by default; `--analyze` for `EXPLAIN ANALYZE` |
| `shell` | Launch the `duckdb` REPL with everything pre-registered |
| `list-cache` | Report freshness + row counts for every parquet cache |
| `embed` | Backfill embeddings via Cohere Embed v4 on Bedrock |
| `search <text>` | HNSW cosine semantic search over embeddings |
| `classify` | Sonnet 4.6 → session autonomy + work category + success + goal |
| `trajectory` | Per-message sentiment + is_transition |
| `conflicts` | Per-session stance-conflict detection |
| `cluster` | UMAP → HDBSCAN → c-TF-IDF over message embeddings |
| `community` | Louvain over session centroids |
| `analyze` | Run the whole pipeline in dependency order |

### Agent-friendly defaults

- **`--format auto`** emits a human table on a TTY and JSON when stdout is
  piped, so agents don't have to set a flag. `json`, `ndjson`, and `csv` are
  always available explicitly.
- **Classified exit codes** for DuckDB errors — `64` for parse errors, `65`
  for unknown view / column / macro, `70` for other runtime errors, `2` when
  `search` is called before `embed` has run. On non-TTY stdout the error
  comes back as `{"error": {"kind", "message", "hint"}}` on stderr so agents
  don't have to scrape tracebacks.
- **`list-cache`** reports every parquet (embeddings, classifications,
  trajectory, conflicts, clusters, cluster_terms, communities) with its
  `{exists, bytes, mtime, rows}`, so an agent can decide whether to run a
  prerequisite stage before issuing a `search` or `query`.
- **`explain`** is a static plan by default (no query execution); pass
  `--analyze` for `EXPLAIN ANALYZE` when you actually want timings.
- **`--quiet`** suppresses all INFO / WARNING logs to ERROR-only; view
  registration happens at DEBUG level, so the default `query` stderr is
  already empty unless something actually warrants attention.

## Views

| View | Grain | Key columns |
|---|---|---|
| `sessions` | one per transcript file | `session_id`, `started_at`, `ended_at` |
| `messages` | one per chat message | `uuid`, `session_id`, `role`, `model`, token usage |
| `content_blocks` | flattened `message.content[]` | `block_type`, `tool_name` |
| `messages_text` | text blocks aggregated per message | `uuid`, `text_content` |
| `tool_calls` | `content_blocks` where `type='tool_use'` | `tool_name`, `tool_use_id` |
| `tool_results` | `content_blocks` where `type='tool_result'` | `tool_use_id`, `content` |
| `todo_events` | one row per todo per TodoWrite snapshot | `subject`, `status`, `snapshot_ix` |
| `todo_state_current` | latest status per `(session, subject)` | `status`, `written_at` |
| `task_spawns` | `Task`/`Agent`/`TaskCreate` launch sites | `subagent_type`, `prompt` |
| `subagent_sessions` | rolled-up subagent runs | `parent_session_id`, `agent_hex`, `agent_type`, `description`, `started_at`, `ended_at`, `message_count`, `transcript_path` |
| `subagent_messages` | user+assistant events from subagent transcripts | `uuid`, `parent_session_id` |
| `session_classifications` | one row per classified session | `autonomy_tier`, `work_category`, `success`, `goal` |
| `session_goals` | projection over classifications | `session_id`, `goal` |
| `message_trajectory` | per-message sentiment + is_transition | `sentiment_delta` (`positive`/`neutral`/`negative`), `is_transition` |
| `session_conflicts` | per-session stance conflicts | `stance_a`, `stance_b`, `resolution` |
| `message_clusters` | cluster id + 2d viz coords | `cluster_id`, `x`, `y`, `is_noise` |
| `cluster_terms` | c-TF-IDF top terms per cluster | `cluster_id`, `term`, `weight`, `rank` |
| `session_communities` | Louvain community per session | `community_id`, `size` |

## Macros

| Macro | Signature | What it does |
|---|---|---|
| `model_used(sid)` | scalar → VARCHAR | Latest `model` observed in the session |
| `cost_estimate(sid)` | scalar → DOUBLE | USD spend (dated model IDs prefix-matched) |
| `tool_rank(last_n_days)` | table | Tool-use leaderboard over a window |
| `todo_velocity(sid)` | scalar → DOUBLE | Completed / distinct todos ratio |
| `subagent_fanout(sid)` | scalar → INT | Subagent runs for a session |
| `semantic_search(query_vec, k)` | table | HNSW top-k over embeddings |
| `autonomy_trend(window_days)` | table | Weekly autonomy-tier mix |
| `work_mix(since_days)` | table | Work-category distribution |
| `success_rate_by_work(since_days)` | table | success / failure / partial rates per category |
| `cluster_top_terms(cid, n)` | table | Top-N terms for a cluster |
| `community_top_topics(cid, n)` | table | Dominant clusters within a community |
| `sentiment_arc(sid)` | table | Per-message sentiment timeline for one session |

## Environment variables

All configurable via `CLAUDE_SQL_*`:

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_SQL_DEFAULT_GLOB` | `~/.claude/projects/*/*.jsonl` | Main transcript glob |
| `CLAUDE_SQL_SUBAGENT_GLOB` | `~/.claude/projects/*/*/subagents/agent-*.jsonl` | Subagent transcripts |
| `CLAUDE_SQL_REGION` | `us-east-1` | Bedrock region |
| `CLAUDE_SQL_MODEL_ID` | `global.cohere.embed-v4:0` | Embedding model |
| `CLAUDE_SQL_SONNET_MODEL_ID` | `global.anthropic.claude-sonnet-4-6` | Classification model |
| `CLAUDE_SQL_OUTPUT_DIMENSION` | `1024` | Matryoshka embedding dim |
| `CLAUDE_SQL_CONCURRENCY` | `2` | Parallel Bedrock calls |
| `CLAUDE_SQL_BATCH_SIZE` | `96` | Cohere batch size |
| `CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH` | `~/.claude/embeddings.parquet` | Embeddings cache |
| `CLAUDE_SQL_SEED` | `42` | UMAP/HDBSCAN/Louvain determinism |

## Development

```bash
mise run check           # lint + fmt-check + typecheck + 40 tests
mise run fmt:write       # auto-apply ruff formatting
mise run upgrade         # uv lock --upgrade && uv sync
mise run build           # uv build → dist/*.whl + *.tar.gz
mise run tool:install    # install claude-sql as a uv tool (global)
mise run cli -- schema   # run the CLI in the project venv
mise tasks               # list every mise task
```

## Design notes

- Zero-copy reads: `read_json(..., filename=true, union_by_name=true,
  sample_size=-1, ignore_errors=true)` so the corpus is queried in place.
- Nested `message.content[]` kept as JSON and flattened via `UNNEST +
  json_extract_string`, not eagerly shredded — resilient to new block
  types.
- Cohere Embed v4 via the `global.cohere.embed-v4:0` CRIS profile sustains
  the highest throughput with no throttling in tests; direct and US CRIS
  both saturate at low TPM.
- HNSW cosine index (`DuckDB VSS`) rebuilt from the parquet on every
  connection open; no experimental persistence.
- Sonnet 4.6 structured output uses Bedrock's GA `output_config.format`
  (not `tool_use` / `tool_choice`) with adaptive thinking on, satisfying
  JSON Schema Draft 2020-12 subset via a pydantic → flattener pipeline
  that inlines `$ref` and strips numeric/string constraints the validator
  rejects.
- UMAP → HDBSCAN runs with `random_state=42` so cluster IDs are stable
  across runs.
- Louvain community detection uses `networkx.community.louvain_communities`
  (built into `networkx>=3.4`), not the abandoned `python-louvain`.

## Links

- Research report: [claude-sql-zero-copy-engine-research.md](../claude-sql-zero-copy-engine-research.md)
  (22 sources)
- Cookbook: [docs/cookbook.md](docs/cookbook.md)
- Analytics cookbook: [docs/analytics_cookbook.md](docs/analytics_cookbook.md)
- Research notes: [docs/research_notes.md](docs/research_notes.md)

## License

Apache 2.0. See [LICENSE](LICENSE).
