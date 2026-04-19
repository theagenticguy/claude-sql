# claude-sql Research Notes (wi_1aca3fd44d8d)

> Zero-copy DuckDB engine over `~/.claude/projects/**/*.jsonl` transcripts,
> with Cohere Embed v4 semantic search via DuckDB VSS.
> HNSW wired in v1, not deferred.
> Last verified against corpus: 2026-04-19.

Corpus snapshot at capture time: 9,503 top-level sessions, 3,145 subagent
sessions, 10,077 TodoWrite events. `mise run check` green (ruff lint + ruff
format + `ty check src/` + pytest, 15 tests).

Companion to the long-form research report at
`../../claude-sql-zero-copy-engine-research.md` (22 sources). This file
captures only the design decisions that survived into v1 and the tuning
knobs worth revisiting; the original report covers alternatives,
benchmarks, and citations in depth.

## 1. The Zero-Copy Read (DuckDB `read_json`)

The canonical scan (see `register_raw` in `src/claude_sql/sql_views.py`):

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

Glob fix, learned the hard way: the conversation-transcript glob is
`~/.claude/projects/*/*.jsonl` (two segments, not `**`). The recursive
`**` pattern also pulls in subagent side-files under
`projects/<project>/<session>/subagents/agent-*.jsonl`, which breaks the
`sessions` GROUP BY because the same `session_id` appears in both the
top-level transcript and every subagent run. Subagent files have their
own glob (`SUBAGENT_GLOB`) and their own views.

Overrides from `read_json` defaults:

| Param | Default | Our value | Why |
|---|---|---|---|
| `sample_size` | 20480 | `-1` | Schema inference must visit every file at 9,505 files with evolving content types. |
| `maximum_object_size` | 16 MB | 64 MB | Large tool_result payloads (search output, repo snapshots) regularly exceed 16 MB. |
| `union_by_name` | false | true | Transcripts span months of Claude Code schema drift; union by name tolerates added/dropped fields. |
| `ignore_errors` | false | true | A small fraction of JSONL rows are truncated by crashes — tolerate, don't abort. |

Proof of predicate-pushdown is in section 6.

## 2. Nested `content[]` — Lazy Type Pattern

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

Why lazy over eager `STRUCT` shredding: block-type vocabulary evolves
(`thinking` and various MCP shapes arrived late). A typed `STRUCT` would
force a schema migration every time Anthropic ships a new content type.
The `json_extract_string` / `UNNEST` path has a measurable per-row cost
but the tradeoff is that `content_blocks` tolerates schema drift without
touching ingest. For queries that hammer `block_type = 'text'`
specifically (semantic-search embed pipeline), we materialize a thin
`messages_text` view on top.

## 3. Eleven Views + Six Macros

Registration chain (`register_all` in `src/claude_sql/sql_views.py`):

```
register_all(con, settings)
├── register_raw(con, glob=..., subagent_glob=..., subagent_meta_glob=...)
│   ├── v_raw_events            <- read_json(projects/**/*.jsonl)
│   ├── v_raw_subagents         <- read_json(projects/*/*/subagents/agent-*.jsonl)
│   └── v_raw_subagent_meta     <- read_json(projects/*/*/subagents/agent-*.meta.json)
├── register_views(con)
│   └── 11 views (see below)
├── register_vss(con, embeddings_parquet=..., dim=1024, metric='cosine', ...)
│   ├── INSTALL + LOAD vss
│   ├── CREATE OR REPLACE TABLE message_embeddings (from parquet if present)
│   └── CREATE INDEX idx_msg_hnsw USING HNSW (embedding)
└── register_macros(con, settings)
    └── 6 macros: model_used, cost_estimate, tool_rank, todo_velocity,
        subagent_fanout, semantic_search
```

Order matters: `register_vss` must run before `register_macros` because the
`semantic_search` macro body references `message_embeddings`, and DuckDB
resolves macro bodies at creation time.

Views (11 total, all registered by `register_views` in order):

- `sessions` — one row per transcript, rolled up from `v_raw_events`.
- `messages` — user + assistant events with token usage + `content_json`.
- `content_blocks` — one row per element in `message.content[]`, typed by
  `block_type`.
- `messages_text` — text blocks only; substrate for embeddings.
- `tool_calls` — `content_blocks` where `block_type = 'tool_use'`.
- `tool_results` — `content_blocks` where `block_type = 'tool_result'`.
- `todo_events` — one row per todo per TodoWrite snapshot (with
  `snapshot_ix` for ordering).
- `todo_state_current` — latest status per `(session_id, subject)`.
- `task_spawns` — `tool_calls` filtered to `Task` / `Agent` / `TaskCreate`
  / `mcp__tasks__task_create`.
- `subagent_sessions` — rolled-up subagent runs from
  `v_raw_subagents` + `v_raw_subagent_meta`.
- `subagent_messages` — user + assistant events from subagent transcripts.

Macros (six, all registered by `register_macros`):

- `model_used(sid)` — latest `model` observed in the session.
- `cost_estimate(sid)` — USD using `config.DEFAULT_PRICING` (per-1M rates).
- `tool_rank(last_n_days)` — table macro; tool-use leaderboard.
- `todo_velocity(sid)` — completed / distinct todos.
- `subagent_fanout(sid)` — count of subagent runs for a session.
- `semantic_search(query_vec, k)` — table macro issuing HNSW-backed top-k.

## 4. Semantic Search (Cohere Embed v4 + DuckDB VSS)

Model config (from `config.Settings`):

- `model_id = "global.cohere.embed-v4:0"` — Cohere Embed v4 global CRIS
  inference profile. Hardcoded as the single default; see the routing
  callout below.
- `output_dimension = 1024` — Matryoshka-truncatable later without
  re-embedding; 256 is the smallest supported by Embed v4.
- `embedding_type = "int8"` — requested from Cohere for compactness on
  the wire.

> **Routing decision.** The direct `cohere.embed-v4:0` and the US CRIS
> `us.cohere.embed-v4:0` both throttled aggressively on the real corpus
> (8×96 batch fan-out saturated the TPM bucket in seconds). The global
> CRIS profile `global.cohere.embed-v4:0` sustained 223 vec/s at
> concurrency=8 with zero `ThrottlingException`s. Since the global pool
> always ≥ the direct/us pools, we hardcode it. No `use_cris`/`routing`
> knobs. IAM requirement: `bedrock:InvokeModel` on
> `arn:aws:bedrock:*:*:inference-profile/global.cohere.embed-v4:0` —
> already present on `lalsaado-handson`.

Storage shape: Cohere returns int8, DuckDB VSS only supports `FLOAT`
element type. `register_vss` converts on insert:

```sql
CREATE OR REPLACE TABLE message_embeddings AS
SELECT uuid, model, dim,
       CAST(embedding AS FLOAT[1024]) AS embedding
FROM read_parquet(?);
```

We lose the 4× wire savings at rest. Acceptable for a single-user corpus;
revisit if the parquet grows past ~5 GB.

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

**Persistence clarification.** HNSW index is **rebuilt at connection open
from `~/.claude/embeddings.parquet`** — `hnsw_enable_experimental_persistence`
is **NOT** used. DuckDB flags HNSW persistence as experimental in 2026
(WAL recovery isn't fully implemented for custom indexes), so we treat the
parquet as the source of truth and the in-memory HNSW as a rebuildable
cache. Rebuild cost is a one-time minutes-scale build per `claude-sql`
invocation; acceptable for an interactive CLI because `register_all` only
runs once per command.

`semantic_search` triggers the index rewrite by issuing
`ORDER BY array_distance(embedding, query_vec) LIMIT k` — the VSS planner
hooks that shape and emits `HNSW_INDEX_SCAN`.

## 5. Pricing & Alternatives

| Model | Price / 1M tokens | Dims | Verdict |
|---|---|---|---|
| Cohere Embed v4 (Bedrock) | $0.12 | 256–1536 MRL | Primary |
| Titan Text Embeddings V2 | $0.02 | 256 / 512 / 1024 | Budget fallback |
| OpenAI text-embedding-3-large | $0.13 | 256–3072 MRL | Skip (off-Bedrock) |
| Voyage 3.5 | $0.12 | 256–2048 MRL | Skip (off-Bedrock) |

At a rough 1M messages × 500 tokens avg = 500M tokens, a full-corpus Cohere
embed is ~$60 one-time on-demand. Batch inference
(`CreateModelInvocationJob`) roughly halves that; we have not yet wired a
batch path in `embed_worker`.

## 6. Open Questions / Next Steps

- [ ] Benchmark HNSW build time at 1M × 1024d on dev hardware. First
      measurement pending.
- [ ] 50-query manual retrieval eval on the real corpus to validate Cohere
      v4 choice against Titan V2.
- [ ] Decide nightly batch vs incremental-on-demand threshold for embed
      backfill (break-even roughly >1000 new messages/day).
- [ ] Subagent transcripts contain `sessionId` too — currently we key
      `subagent_sessions` by `parent_session_id + agent_hex` derived from
      the filesystem path. Revisit once we see workloads where agents
      cross-reference each other.
- [ ] Normalize dated model IDs (`claude-haiku-4-5-20251001`) against the
      pricing keys in `config.DEFAULT_PRICING` so `cost_estimate` doesn't
      silently NULL them out.

### What we verified live

- **Predicate pushdown**: `uv run claude-sql explain "SELECT uuid, role
  FROM messages WHERE session_id = '<uuid>'"` prints a green-highlighted
  `READ_JSON` node with `Projections: type, sessionId, message.role,
  uuid` (4 of 15 inferred columns) and `Total Files Read: 9505`. Wall
  time on the CLI's `EXPLAIN ANALYZE` for that single-session filter is
  ~32s for the first run / ~38s when cold, dominated by the full-corpus
  JSON scan that `sessions` triggers at connection time. Filename-based
  predicates (`AND source_file LIKE '%<session>.jsonl'`) prune the scan
  to a single file.
- **Corpus-specific observations**: `isSidechain = true` rows do not
  appear in the modern corpus — subagents live exclusively in the
  `subagents/` side-files, not mixed into the parent transcript. We keep
  the `is_sidechain` column in `messages` for historical compatibility
  but the dedicated `subagent_*` views carry the weight.
- **Glob correctness**: `~/.claude/projects/*/*.jsonl` returns the 9,503
  session transcripts; the `**` form leaks subagent JSONL into the
  `sessions` rollup and inflates `record_count` by the subagent message
  events.
- **Quality gates green**: `mise run check` passes ruff lint, ruff format
  check, `ty check src/`, and the 15-test pytest suite.

## v2 research

v2 layers session-level analytics on top of the zero-copy substrate:
clusters, communities, LLM-judged classifications/trajectory/conflicts. The
research below captures design decisions that survived into v2 and the
measurements that drove them. Verified against the corpus on 2026-04-19.

### Corpus snapshot at v2 capture

- 26,158 Cohere Embed v4 embeddings at `~/.claude/embeddings.parquet`.
- 26,158 HDBSCAN cluster assignments at `~/.claude/clusters.parquet` —
  299 signal clusters + 9,859 noise points (37.7%).
- 2,990 c-TF-IDF term rows at `~/.claude/cluster_terms.parquet`
  (299 clusters × 10 top 1-2gram terms).
- 6,329 session-centroid embeddings grouped into 1,512 Louvain communities
  at `~/.claude/session_communities.parquet`. Top-5 community sizes:
  759, 711, 686, 450, 309.
- `session_classifications.parquet`, `message_trajectory.parquet`, and
  `session_conflicts.parquet` are pending. Full-corpus Sonnet 4.6 run is
  roughly $455; estimate via `claude-sql classify --dry-run` before spending.

### Sonnet 4.6 model and Bedrock structured output

- **Model ID**: `global.anthropic.claude-sonnet-4-6`. CRIS-only (no direct
  on-demand `claude-sonnet-4-6` ARN), 1M-context native, **no** beta header
  required. Pricing $3/MTok input, $15/MTok output.
- **Bedrock `output_config.format` is GA and replaces the earlier
  tool_use/tool_choice plan.** Supported on Sonnet 4.5 / 4.6, Opus 4.5 / 4.6,
  and Haiku 4.5. The request shape is `Converse`'s standard body plus
  `output_config = {"format": {"json": {"schema": <JSON Schema>}}}` — the
  model is *required* to emit a parseable JSON object conforming to the
  schema.
- **Schema rules.** JSON Schema Draft 2020-12 subset: `$ref`,
  `$defs`, `allOf` union, and a few other features are disallowed. The
  pydantic v2 `model_json_schema()` output must be flattened (inline every
  `$ref`) and every object needs `additionalProperties: false` injected.
  `claude_sql.schemas` has a `_flatten_schema` helper that does exactly this.
- **`citations` is the *only* documented incompatibility** — `thinking`
  is not mentioned as incompatible with structured output. We keep
  `thinking: {"type": "adaptive"}` on by default for classify/trajectory/
  conflicts, and expose a `--no-thinking` CLI escape hatch on each command
  for when a run is hitting budget ceilings. (Adaptive thinking adds a few
  hundred output tokens on average; disable if the dry-run estimate is
  uncomfortable.)

### Louvain community detection — networkx beats python-louvain

`networkx.algorithms.community.louvain_communities` has been built in
since networkx 3.4 and is the current maintained implementation. The old
`python-louvain` package is stuck at its 2018 release — no bug fixes, no
parallel modularity tweaks, and depends on the deprecated `community` shim.
Switching saved one dependency and got us a measurable speedup on the real
graph.

Pipeline: build cosine-similarity graph from 6,329 session centroids, keep
edges with `sim >= threshold` (default 0.75), pass to
`louvain_communities(..., resolution=1.0)`.

Measured wall time on the dev-host: 3.6s for 6,329 nodes -> 1,512 communities.
The size distribution is heavy-tailed (top 5: 759, 711, 686, 450, 309; long
tail of size-1 singletons).

### UMAP + HDBSCAN — measured end-to-end

26,158 × 1024d int8-cast-to-float embeddings. Reference times on dev-host:

| Stage | Wall | Config |
|---|---|---|
| UMAP 50d | 78s | `n_neighbors=30, min_dist=0.0, metric='cosine'` |
| UMAP 2d | 25s | Re-fit on 50d output for 2d visualization coords |
| HDBSCAN | 8s | `min_cluster_size=20, min_samples=5, metric='euclidean'` on the 50d coords |
| **Total** | **~110s** | Single-process, no GPU |

299 clusters with 37.7% noise. Textual-embedding HDBSCAN literature puts a
healthy noise band at 25-45%; 37.7% is in the middle — not aggressive
over-clustering, not a dumping ground. Lowering `min_cluster_size` to 10
would push noise down to ~30% but also split several coherent clusters.

### In-house c-TF-IDF

BERTopic pulls in sentence-transformers + UMAP + HDBSCAN + scikit-learn and
pins specific versions — we already have UMAP/HDBSCAN, so we implement
c-TF-IDF directly against CountVectorizer:

- `CountVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))`.
- Pseudo-document per cluster via DuckDB `string_agg(text_content, ' ')`.
- Weight formula: normalized per-cluster term frequency × `log(1 + avg /
  col_sum)` where `avg` is the corpus-average column sum and `col_sum` is
  the per-term column sum across clusters (BERTopic's original formulation).

Output: 299 clusters × 10 terms = 2,990 rows in
`~/.claude/cluster_terms.parquet`. `cluster_top_terms(cid, n)` exposes it
as a macro; `community_top_topics(cid, n)` joins through
`session_communities -> session_clusters -> cluster_terms` to label a
community by the terms of the clusters its sessions' messages land in.

### Stale JSONL bug — graceful skip in build_session_text

`sessions` emits session UUIDs directly from `v_raw_events`. Over the
corpus's lifetime, some JSONL files get deleted (worktree cleanup, failed
rm-rf replays), but the session UUID lingers inside other files that
reference it. The classify pipeline calls `build_session_text(session_id)`
in `session_text.py` per pending session; when it hits a deleted file,
DuckDB's `read_json` raises `duckdb.IOException` and the pipeline used to
abort mid-run, losing every classification after the failure.

Fix: `build_session_text` now catches `duckdb.IOException` and skips the
session with a `logger.warning(...)` — the pipeline continues and the stale
session shows up again on the next run. If the JSONL is still missing, it
gets skipped again (harmless). If it has been restored, it gets classified.

### Carry-forward: dated model IDs in `cost_estimate`

The first v1 backfill surfaced a NULL-cost problem for dated model IDs like
`claude-haiku-4-5-20251001`. `config.DEFAULT_PRICING` keys on the base IDs
(`claude-haiku-4-5`), so the `JOIN` in `cost_estimate(sid)` missed the
dated ones. Fix (shipped in v1.1, carried forward into v2):

```sql
JOIN (VALUES ...) p(model, in_rate, out_rate)
  ON regexp_replace(m.model, '-\d{8}$', '') = p.model
```

The regex strips a trailing `-YYYYMMDD` date, so dated IDs resolve to the
pricing-table base entry. No change to `DEFAULT_PRICING` required when
Anthropic ships a new dated snapshot of an existing base model.

