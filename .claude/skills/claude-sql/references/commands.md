# claude-sql commands reference

Every subcommand, its flags, and the shape of its output. All commands
share the top-level flags `--verbose/-v`, `--quiet/-q`, `--glob`,
`--subagent-glob`. Paths default to `~/.claude/projects/*/*.jsonl` and
`~/.claude/projects/*/*/subagents/agent-*.jsonl`.

## `schema`

List every registered view with its columns, then every macro with its
signature. Run this first whenever you're unsure what's queryable.

```bash
claude-sql schema
```

No flags beyond the shared set. Output is a polars DataFrame grouped
by view.

## `query <sql>`

Run an arbitrary SQL statement and print the result as a polars table.

```bash
claude-sql query "SELECT count(*) FROM sessions"
claude-sql query "
  SELECT session_id,
         model_used(session_id) AS model,
         cost_estimate(session_id) AS usd
  FROM sessions
  WHERE started_at >= current_timestamp - INTERVAL 7 DAY
  ORDER BY usd DESC
  LIMIT 20
"
```

Flags:
- `--limit N` (default 100) — cap rows printed. Use `--limit 0` for no cap.
- `--format <table|json|csv|parquet>` (default `table`).

## `explain <sql>`

`EXPLAIN ANALYZE` with the plan printed and any predicate-pushdown
markers highlighted in the output. Use this when a query is slower
than expected.

```bash
claude-sql explain "SELECT * FROM messages WHERE session_id = '...' LIMIT 1"
```

## `shell`

Drop into the `duckdb` REPL with every view, macro, and the HNSW index
pre-registered. Exit with `.quit`.

```bash
claude-sql shell
```

## `embed`

Backfill Cohere Embed v4 embeddings for every message that doesn't yet
have one. Writes to `~/.claude/embeddings.parquet` in chunked
checkpoints so interruptions don't lose work.

```bash
# Dry-run (default): counts pending messages and estimates cost
claude-sql embed --since-days 30

# Actually run
AWS_PROFILE=... claude-sql embed --since-days 30 --no-dry-run
```

Flags:
- `--since-days N` — only embed messages newer than N days
- `--dry-run / --no-dry-run` — default `--dry-run`
- `--concurrency N` (default 2) — parallel Bedrock calls
- `--batch-size N` (default 96) — Cohere batch size per call
- `--output-dimension D` (default 1024) — Matryoshka truncation
- `--model-id ID` (default `global.cohere.embed-v4:0`)

## `search <text>`

HNSW cosine top-k semantic search over embeddings. Prints session_id,
uuid, role, and a text snippet per hit. Requires that `embed` has run.

```bash
claude-sql search "temporal workflow determinism" --k 10
claude-sql search "the part where I debugged the Louvain community detection"
```

Flags:
- `--k N` (default 10) — number of hits

## `classify`

Sonnet 4.6 classifies each session along four axes: autonomy_tier
(1/2/3), work_category (coding/strategy/admin/writing/research/other),
success (success/partial/failure), and a short free-text goal. Writes
to `~/.claude/classifications.parquet`.

```bash
claude-sql classify --since-days 30
AWS_PROFILE=... claude-sql classify --since-days 30 --no-dry-run
```

Flags:
- `--since-days N` — only classify sessions newer than N days
- `--dry-run / --no-dry-run` — default `--dry-run`
- `--concurrency N` (default 2)
- `--model-id ID` (default `global.anthropic.claude-sonnet-4-6`)

The dry-run path uses a pure SQL count of pending sessions, so it's
fast even on a huge corpus.

## `trajectory`

Per-message sentiment delta (-1.0 to +1.0) and `is_transition` flag.
Useful for plotting the arc of a single session or spotting where a
conversation turned around. Writes to `~/.claude/trajectory.parquet`.

```bash
claude-sql trajectory --session-id <uuid> --no-dry-run
claude-sql trajectory --since-days 7 --no-dry-run
```

## `conflicts`

Per-session stance conflict detection. Outputs `stance_a`, `stance_b`,
and `resolution` (resolved/abandoned/unresolved). Writes to
`~/.claude/conflicts.parquet`.

```bash
claude-sql conflicts --since-days 30 --no-dry-run
```

## `cluster`

UMAP (cosine, 50d + 2d viz) → HDBSCAN (min_cluster_size=20) → c-TF-IDF
(ngram (1,2), min_df=2). Deterministic with `CLAUDE_SQL_SEED=42`.
Writes `~/.claude/clusters.parquet` and
`~/.claude/cluster_terms.parquet`.

```bash
claude-sql cluster --min-cluster-size 20 --n-neighbors 15
```

No Bedrock cost — CPU only.

## `community`

Louvain community detection over session centroids (mean of every
message embedding in the session). Builds a k-NN graph in-memory and
runs `networkx.community.louvain_communities`. Writes to
`~/.claude/session_communities.parquet`.

```bash
claude-sql community --k 10
```

No Bedrock cost.

## `analyze`

Chains the whole analytics pipeline in dependency order:
embed → cluster → classify → trajectory → conflicts → community.
Every step respects `--dry-run` individually. Use this when you want
a full refresh after a long break.

```bash
claude-sql analyze --since-days 30               # dry-run, prints cost
AWS_PROFILE=... claude-sql analyze --since-days 30 --no-dry-run
```

## Environment

All configurable via `CLAUDE_SQL_*` env vars (see README). Common ones:

- `CLAUDE_SQL_DEFAULT_GLOB` — main transcript glob
- `CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH` — embeddings cache path
- `CLAUDE_SQL_CONCURRENCY` — parallel Bedrock calls
- `CLAUDE_SQL_SEED` — determinism for UMAP/HDBSCAN/Louvain
