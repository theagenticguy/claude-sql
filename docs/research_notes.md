# claude-sql Research Notes

> Zero-copy DuckDB engine over `~/.claude/projects/**/*.jsonl` transcripts,
> with Cohere Embed v4 semantic search via DuckDB VSS, plus a v2 analytics
> layer (UMAP + HDBSCAN clusters, Louvain communities, Sonnet 4.6
> classifications / trajectory / conflicts / friction).

This file captures the design decisions that survived into the shipped
engine and the tuning knobs worth revisiting. It is a companion to the
code under `src/claude_sql/`, not a user guide — see
[`../README.md`](../README.md), [`cookbook.md`](cookbook.md), and
[`analytics_cookbook.md`](analytics_cookbook.md) for those.

## 1. The zero-copy read (DuckDB `read_json`)

The canonical scan (see `register_raw` in
[`src/claude_sql/sql_views.py`](../src/claude_sql/sql_views.py)):

```sql
CREATE OR REPLACE VIEW v_raw_events AS
SELECT *,
       filename AS source_file,
       regexp_extract(filename, '([^/]+)\.jsonl$', 1) AS session_id_file
FROM read_json(
    '~/.claude/projects/*/*.jsonl',
    format              = 'newline_delimited',
    union_by_name       = true,
    filename            = true,
    ignore_errors       = true,
    sample_size         = -1,
    maximum_object_size = 67108864
);
```

**Glob, learned the hard way.** The conversation-transcript glob is
`~/.claude/projects/*/*.jsonl` (two segments, not `**`). The recursive
`**` pattern pulls in subagent side-files under
`projects/<project>/<session>/subagents/agent-*.jsonl`, which breaks the
`sessions` `GROUP BY` because the same `session_id` appears in both the
top-level transcript and every subagent run. Subagent files have their
own glob (`SUBAGENT_GLOB`) and their own views.

Overrides from `read_json` defaults:

| Param | Default | Our value | Why |
|---|---|---|---|
| `sample_size` | 20480 | `-1` | Schema inference must visit every file; content types evolve across months. |
| `maximum_object_size` | 16 MB | 64 MB | Large `tool_result` payloads (search output, repo snapshots) regularly exceed 16 MB. |
| `union_by_name` | `false` | `true` | Transcripts span months of Claude Code schema drift; union-by-name tolerates added / dropped fields. |
| `ignore_errors` | `false` | `true` | A small fraction of JSONL rows are truncated by crashes — tolerate, don't abort. |

## 2. Nested `content[]` — lazy type pattern

`message.content` stays as JSON on `messages.content_json`. Block-type
dispatch happens in `content_blocks`:

```sql
CREATE OR REPLACE VIEW content_blocks AS
SELECT
    m.session_id,
    m.uuid                                      AS message_uuid,
    m.ts, m.role,
    json_extract_string(block, '$.type')        AS block_type,
    json_extract_string(block, '$.text')        AS text,
    json_extract_string(block, '$.name')        AS tool_name,
    json_extract(block, '$.input')              AS tool_input,
    json_extract_string(block, '$.tool_use_id') AS tool_use_id,
    json_extract(block, '$.content')            AS tool_result_content,
    json_extract_string(block, '$.thinking')    AS thinking
FROM messages m,
     UNNEST(json_extract(m.content_json, '$[*]')) AS t(block);
```

Why lazy over eager `STRUCT` shredding: the block-type vocabulary evolves
(e.g. `thinking` blocks and various MCP shapes arrived late). A typed
`STRUCT` would force a schema migration every time Anthropic ships a new
content type. The `json_extract_string` / `UNNEST` path has a measurable
per-row cost, but the tradeoff is that `content_blocks` tolerates schema
drift without touching ingest. For queries that hammer
`block_type = 'text'` specifically (semantic-search embed pipeline), we
materialize a thin `messages_text` view on top.

## 3. Views + macros

Registration chain (`register_all` in
[`src/claude_sql/sql_views.py`](../src/claude_sql/sql_views.py)):

```text
register_all(con, settings)
├── register_raw(con, glob=..., subagent_glob=..., subagent_meta_glob=...)
│   ├── v_raw_events            ← read_json(projects/*/*.jsonl)
│   ├── v_raw_subagents         ← read_json(projects/*/*/subagents/agent-*.jsonl)
│   └── v_raw_subagent_meta     ← read_json(projects/*/*/subagents/agent-*.meta.json)
├── register_views(con)
├── register_vss(con, embeddings_parquet=..., dim=1024, metric='cosine', ...)
│   ├── INSTALL + LOAD vss
│   ├── CREATE OR REPLACE TABLE message_embeddings (from parquet if present)
│   └── CREATE INDEX idx_msg_hnsw USING HNSW (embedding)
├── register_analytics(con, settings)
│   ├── session_classifications, session_goals
│   ├── message_trajectory
│   ├── session_conflicts
│   ├── message_clusters, cluster_terms
│   ├── session_communities
│   └── user_friction
└── register_macros(con, settings)
```

Order matters: `register_vss` must run before `register_macros` because
the `semantic_search` macro body references `message_embeddings`, and
DuckDB resolves macro bodies at creation time. Analytics views register
against whichever parquets exist — missing ones warn and no-op, never
crash.

See [`jsonl_schema_v1.sql`](jsonl_schema_v1.sql) for the full column
listings of the v1 views (`sessions`, `messages`, `content_blocks`,
`messages_text`, `tool_calls`, `tool_results`, `todo_events`,
`todo_state_current`, `task_spawns`, `subagent_sessions`,
`subagent_messages`). The README lists v2 views and every macro.

## 4. Semantic search (Cohere Embed v4 + DuckDB VSS)

Model config (from `config.Settings`):

- `model_id = "global.cohere.embed-v4:0"` — Cohere Embed v4 global CRIS
  inference profile. Hardcoded as the single default; see the routing
  callout below.
- `output_dimension = 1024` — Matryoshka-truncatable later without
  re-embedding; 256 is the smallest supported by Embed v4.
- `embedding_type = "int8"` — requested from Cohere for compactness on
  the wire.

> **Routing decision.** The direct `cohere.embed-v4:0` and the US CRIS
> `us.cohere.embed-v4:0` both throttled aggressively under real load
> (`8 × 96`-batch fan-out saturated the TPM bucket in seconds). The
> global CRIS profile `global.cohere.embed-v4:0` sustained high
> throughput at `concurrency=8` with zero `ThrottlingException`s. Since
> the global pool is always ≥ the direct / US pools, we hardcode it — no
> `use_cris` / `routing` knobs. IAM requirement: `bedrock:InvokeModel`
> on `arn:aws:bedrock:*:*:inference-profile/global.cohere.embed-v4:0`.

**Storage shape.** Cohere returns `int8`, DuckDB VSS only supports
`FLOAT` element type. `register_vss` converts on insert:

```sql
CREATE OR REPLACE TABLE message_embeddings AS
SELECT uuid, model, dim,
       CAST(embedding AS FLOAT[1024]) AS embedding
FROM read_parquet(?);
```

We lose the 4× wire savings at rest. Acceptable for a single-user
corpus; revisit if the parquet grows past roughly 5 GB.

### HNSW index — wired in v1

`register_vss` unconditionally builds the index whenever the embeddings
parquet exists:

```sql
INSTALL vss;
LOAD vss;

CREATE INDEX IF NOT EXISTS idx_msg_hnsw
ON message_embeddings
USING HNSW (embedding)
WITH (
    metric          = 'cosine',
    ef_construction = 128,
    ef_search       = 64,
    M               = 16,
    M0              = 32
);
```

**Persistence clarification.** The HNSW index is **rebuilt at connection
open** from `~/.claude/embeddings.parquet`;
`hnsw_enable_experimental_persistence` is **not** used. DuckDB still
flags HNSW persistence as experimental (WAL recovery isn't fully
implemented for custom indexes), so we treat the parquet as the source
of truth and the in-memory HNSW as a rebuildable cache. Rebuild is
minutes-scale for modest corpora; acceptable for an interactive CLI
because `register_all` runs only once per command.

`semantic_search` triggers the index rewrite by issuing
`ORDER BY array_distance(embedding, query_vec) LIMIT k` — the VSS
planner hooks that shape and emits `HNSW_INDEX_SCAN`.

## 5. Pricing alternatives for embeddings

| Model | Price per 1M tokens | Dims | Verdict |
|---|---|---|---|
| Cohere Embed v4 (Bedrock) | $0.12 | 256–1536 MRL | Primary |
| Titan Text Embeddings V2 | $0.02 | 256 / 512 / 1024 | Budget fallback |
| OpenAI `text-embedding-3-large` | $0.13 | 256–3072 MRL | Skip (off-Bedrock) |
| Voyage 3.5 | $0.12 | 256–2048 MRL | Skip (off-Bedrock) |

Order-of-magnitude check: one million messages × 500 tokens average is
500M tokens, so a full-corpus Cohere embed is roughly $60 one-time
on-demand. Bedrock batch inference
(`CreateModelInvocationJob`) roughly halves that; we have not wired a
batch path in `embed_worker` yet.

## 6. v2 analytics design

v2 layers session-level analytics on top of the zero-copy substrate.
Every expensive output is cached in a parquet under `~/.claude/`; views
register only the parquets that exist.

### Sonnet 4.6 structured output

- **Model ID**: `global.anthropic.claude-sonnet-4-6`. CRIS-only (no
  direct on-demand `claude-sonnet-4-6` ARN), 1M-context native, no beta
  header required. Pricing `$3 / MTok` input, `$15 / MTok` output.
- **Bedrock `output_config.format` is GA** and replaces the earlier
  `tool_use` / `tool_choice` plan. Supported on Sonnet 4.5 / 4.6, Opus
  4.5 / 4.6, Haiku 4.5. Request shape is the standard Converse body plus
  `output_config = {"format": {"json": {"schema": <JSON Schema>}}}`. The
  model is required to emit a parseable JSON object conforming to the
  schema.
- **Schema rules.** JSON Schema Draft 2020-12 subset: `$ref`, `$defs`,
  `allOf` union, and several other features are disallowed. Pydantic v2
  `model_json_schema()` output must be flattened (inline every `$ref`)
  with `additionalProperties: false` injected on every object.
  `claude_sql.schemas` exposes a `_flatten_schema` helper that does this.
- **Incompatibilities.** `citations` is the only documented feature
  incompatible with structured output. `thinking` is not — we keep
  `thinking: {"type": "adaptive"}` on by default for classify /
  trajectory / conflicts / friction, with a `--no-thinking` CLI escape
  hatch on each command for when the dry-run estimate is uncomfortable.

### Louvain — `networkx` beats `python-louvain`

`networkx.algorithms.community.louvain_communities` has been built in
since `networkx >= 3.4` and is the current maintained implementation.
The old `python-louvain` package is stuck at its 2018 release — no bug
fixes, no parallel modularity tweaks, and depends on the deprecated
`community` shim. Switching saved one dependency and delivered a
measurable speedup on the real graph.

Pipeline: build a cosine-similarity graph from session centroids, keep
edges with `sim >= threshold` (default `0.75`), pass to
`louvain_communities(..., resolution=1.0)`. Seeded by
`CLAUDE_SQL_SEED=42` so community IDs are stable across runs.

### UMAP → HDBSCAN → c-TF-IDF

- **UMAP 50d**: `n_neighbors=30, min_dist=0.0, metric='cosine'`, seeded
  by `random_state=42`.
- **UMAP 2d**: re-fit on the 50d output for visualization coords.
- **HDBSCAN**: `min_cluster_size=20, min_samples=5, metric='euclidean'`
  on the 50d coords.

Textual-embedding HDBSCAN literature puts a healthy noise band at
25–45%; lowering `min_cluster_size` to 10 reduces noise but splits
several coherent clusters.

**c-TF-IDF**: BERTopic pulls in sentence-transformers + UMAP + HDBSCAN +
scikit-learn with pinned versions. We already have UMAP / HDBSCAN, so
c-TF-IDF is implemented directly over `CountVectorizer`:

- `CountVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))`.
- Pseudo-document per cluster via DuckDB `string_agg(text_content, ' ')`.
- Weight formula: normalized per-cluster term frequency ×
  `log(1 + avg / col_sum)` where `avg` is the corpus-average column sum
  and `col_sum` is the per-term column sum across clusters (BERTopic's
  original formulation).

Output: top 10 terms per cluster in
`~/.claude/cluster_terms.parquet`, exposed by `cluster_top_terms(cid,
n)` and `community_top_topics(cid, n)` (which joins
`session_communities → session_clusters → cluster_terms` to label a
community by the terms of the clusters its sessions' messages land in).

### Friction classifier

`friction_worker.py` detects user-friction signals in short user-role
messages (≤ `CLAUDE_SQL_FRICTION_MAX_CHARS`, default 300). Seven labels:
`status_ping`, `unmet_expectation`, `confusion`, `interruption`,
`correction`, `frustration`, `none`. Hybrid pipeline:

1. Pull user-role messages under the length cutoff via `messages_text`.
2. `regex_fast_path` catches unambiguous cases —
   `status_ping` / `interruption` / `correction` — at confidence `0.9`.
3. Everything else goes to Sonnet 4.6 with `USER_FRICTION_SCHEMA`.
4. Session-level checkpointer + per-uuid anti-join keep reruns on
   untouched sessions free.
5. Output: `~/.claude/user_friction.parquet` with
   `source ∈ {regex, llm, refused}`.

`unmet_expectation` deliberately stays in the LLM path: a message like
`screenshot?` needs session context to disambiguate from a genuine topic
question.

## 7. Resilience patterns to preserve

- **Stale-JSONL skip.** `session_text.build_session_text` wraps its
  DuckDB query in `try/except duckdb.IOException`. Over a corpus's
  lifetime, some JSONL files get deleted (worktree cleanup, failed
  rm-rf replays) but the session UUID lingers inside other files that
  reference it. Without the catch, the classify pipeline aborts
  mid-run. With it, the session is skipped with a `logger.warning(...)`
  and the pipeline continues; if the JSONL is restored on the next
  run, it gets classified then.
- **Pure-SQL dry-run.** `llm_worker._count_pending_sessions` uses
  `COUNT(DISTINCT session_id)` instead of materializing the pending
  set. Keeps `--dry-run` fast on large corpora.
- **Tenacity retries** catch `SSLError`, `ConnectionError`, and
  `ThrottlingException` with exponential backoff. Botocore's own retry
  is disabled so tenacity owns the policy end-to-end.
- **Explicit parquet schema** on embedding write:
  `pl.Array(pl.Float32, output_dimension)`. Otherwise polars infers
  `Object` and the roundtrip breaks.
- **Dated model-ID normalization.** `cost_estimate(sid)` joins
  `regexp_replace(m.model, '-\d{8}$', '')` against the pricing table so
  dated snapshots (e.g. `claude-haiku-4-5-20251001`) resolve to the
  base rate without requiring a `DEFAULT_PRICING` update.
- **Refusal is terminal.** `BedrockRefusalError` stamps a neutral
  placeholder row and clears the retry queue so refused messages don't
  cycle forever.

## 8. Open questions

- [ ] Benchmark HNSW build time at 1M × 1024d on modest hardware.
- [ ] 50-query manual retrieval eval to validate Cohere Embed v4
      against Titan Text Embeddings V2.
- [ ] Nightly batch vs. incremental-on-demand threshold for embed
      backfill (break-even is probably around 1000 new messages / day,
      unmeasured).
- [ ] Subagent transcripts contain `sessionId` too — currently we key
      `subagent_sessions` by `parent_session_id + agent_hex` derived
      from the filesystem path. Revisit once workloads where agents
      cross-reference each other appear.

## 9. What we verified live

- **Predicate pushdown.** `claude-sql explain "SELECT uuid, role FROM
  messages WHERE session_id = '<uuid>'"` prints a green-highlighted
  `READ_JSON` node with a narrow `Projections:` list (4 of 15 inferred
  columns) and the `Filters:` clause attached to the scan. Filename
  predicates (`AND source_file LIKE '%<session>.jsonl'`) prune to a
  single file.
- **Corpus-specific observation.** `isSidechain = true` rows do not
  appear in the modern corpus — subagents live exclusively in the
  `subagents/` side-files, not mixed into the parent transcript. We
  keep the `is_sidechain` column in `messages` for historical
  compatibility, but the dedicated `subagent_*` views carry the weight.
- **Glob correctness.** `~/.claude/projects/*/*.jsonl` returns the
  top-level session transcripts; the `**` form leaks subagent JSONL
  into the `sessions` rollup and inflates `record_count` by the
  subagent message events.
