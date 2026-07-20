# 06 — The Core Engine (`core/`)

> Perspective: **THE CORE ENGINE** — the DuckDB/S3/httpfs zero-copy reader, config,
> schemas, LLM-shared plumbing, and storage/durability layer. This is the foundation
> everything else (`analytics/`, `evals/`, `provenance/`, `app/`) sits on, so its v2
> refactor sequencing matters most.
>
> Source of truth: `src/claude_sql/core/` (Python 3.13). All `path:line` citations are
> against that tree, not the `build/bundle/` copy.

## 0. Module map at a glance

| Module | LOC | Role | v2 classification |
|---|---|---|---|
| `sql_views.py` | 2344 | DuckDB view/macro/VSS registry — the retrieval backbone | Infrastructure (ReaderPort + query layer) |
| `llm_shared.py` | 1343 | Bedrock client + `invoke_model` wrapper + structured output | Infrastructure (LLMPort / Bedrock adapter) |
| `schemas.py` | 597 | Pydantic v2 classification models | **Domain** (closest thing to a v2 domain layer) |
| `session_text.py` | 476 | Per-session transcript assembly for LLM prompts | Application-support (domain service) |
| `config.py` | 411 | `pydantic-settings` runtime config | Infrastructure/config (embedding-switch seam) |
| `checkpointer.py` | 379 | Per-`(session_id, pipeline)` SQLite WAL checkpoint | Infrastructure (CheckpointPort) |
| `parquet_shards.py` | 267 | Sharded parquet append/read helpers | Infrastructure (CachePort) |
| `lance_store.py` | 277 | LanceDB vector store | Infrastructure (VectorStorePort) |
| `output.py` | 254 | Agent-friendly formatting + DuckDB error → exit code | Application-support (CLI adapter) |
| `retry_queue.py` | 210 | Durable retry queue (shares `state.db`) | Infrastructure (RetryPort) |
| `s3_source.py` | 134 | httpfs/S3 secret wiring for remote transcripts | Infrastructure (part of ReaderPort) |
| `home.py` | 93 | `CLAUDE_SQL_HOME` cache-root resolution + legacy migration | Infrastructure (paths) |
| `logging_setup.py` | 95 | loguru config + tenacity `before_sleep` adapter | Infrastructure (cross-cutting) |

`core/__init__.py` is a one-line docstring only (`core/__init__.py:1`) — there is no
package-level API surface; callers import concrete modules.

---

## 1. The zero-copy reader — how a transcript file becomes a row

There is **no export/ingest step**. DuckDB's `read_json` table function reads the
`~/.claude/**/*.jsonl` glob (or `s3://…` URIs) directly, and a stack of
`CREATE OR REPLACE VIEW` DDL normalizes the raw records into business tables. The
corpus is queried *in place*.

### 1.1 The `DEFAULT_GLOB` / Claude-transcript binding (the memory-flagged shape)

The single glob shape that binds the core engine to the Claude Code transcript format
lives in **`sql_views.py:55`**:

```python
# sql_views.py:55-57
DEFAULT_GLOB: str = os.path.expanduser("~/.claude/projects/*/*.jsonl")
SUBAGENT_GLOB: str = os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.jsonl")
SUBAGENT_META_GLOB: str = os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.meta.json")
```

`config.py` carries an independent copy of the same three defaults as factory functions
(`_default_glob` at `config.py:20-23`, `_default_subagent_glob` at `config.py:26-27`,
`_default_subagent_meta_glob` at `config.py:30-31`). **These two definitions are
duplicated** — `sql_views.DEFAULT_GLOB` is the fallback used when
`register_raw(glob=None)`; `config.Settings.default_glob` is what `register_all` actually
threads through (`sql_views.py:2285-2290`). A v2 `AgentReader` protocol should collapse
this duplication into one owner.

The transcript *shape* is pinned by the explicit column projection in
`_RAW_EVENT_COLUMNS` (`sql_views.py:442-453`) and the nested message struct
`_MESSAGE_STRUCT_TYPE` (`sql_views.py:419-432`). This is the hard binding to the Claude
Code JSONL record format — it hard-codes field names `uuid`, `sessionId`, `parentUuid`,
`type`, `timestamp`, `isSidechain`, `isCompactSummary`, `cwd`, `gitBranch`, and a
`message` struct with `role / content / model / stop_reason / usage{input_tokens,
output_tokens, cache_read_input_tokens, cache_creation_input_tokens}`. **This is the
exact surface a v2 `AgentReader` protocol must abstract to support other transcript
shapes** (OpenAI, Gemini, generic agent logs): change these two constants + the
session-id regex and the whole view stack re-targets.

### 1.2 Trace: transcript file → row

1. **`register_all(con, settings=…)`** (`sql_views.py:2238`) is the single entry point.
   Called by `cli._open_connection_full` (`app/cli.py:342`) and `shell` (`app/cli.py:643`).
   Order (load-bearing): `configure_s3` (if needed) → `register_raw` → `register_views`
   → `register_vss` → `register_analytics` → `register_macros` (`sql_views.py:2282-2302`).

2. **`register_raw`** (`sql_views.py:480`) materializes `v_raw_events` as a
   `CREATE OR REPLACE TEMP TABLE` over `read_json` with the resilience flags
   (`sql_views.py:543-551`): `format='newline_delimited', union_by_name=true,
   filename=true, ignore_errors=true, columns={…strict projection…},
   maximum_object_size=67108864`. Using `columns={…}` pays JSON schema inference **once
   per connection** instead of once per view bind — the dominant cost on the live
   ~10K-file/~2GB corpus. `filename=true` unlocks file-level predicate pushdown.

3. **The session-id derivation** — how a *file path* becomes a `session_id` — is a
   `CASE` at **`sql_views.py:538-542`** that handles both on-disk layouts:

   ```sql
   -- sql_views.py:538-542
   CASE
       WHEN regexp_full_match(filename, '.*/part-[^/]*\.jsonl$')
       THEN regexp_extract(filename, '/([^/]+)/part-[^/]*\.jsonl$', 1)   -- S3SessionStore layout
       ELSE regexp_extract(filename, '([^/]+)\.jsonl$', 1)               -- local flat layout
   END AS session_id_file
   ```

   Local: `.../<session_id>.jsonl` → basename. S3: `.../<session_id>/part-<epochMs>-<rand>.jsonl`
   → parent dir. Subagent raw table `v_raw_subagents` uses a sibling regex to recover
   `parent_session_id` + `agent_hex` from `…/<uuid-36>/subagents/agent-<hex>.jsonl`
   (`sql_views.py:564-573`).

4. **`register_views`** (`sql_views.py:626`) builds the derived tables. The pivotal one is
   `sessions` (`sql_views.py:653-668`): `GROUP BY session_id_file` with `min/max(timestamp)`,
   `count(*) FILTER (WHERE type='assistant')`, and `any_value(source_file) AS transcript_path`.
   That is the transcript-file → session-row collapse. Individual JSONL lines become rows in
   `messages` (`sql_views.py:677-704`, filtered to `type IN ('user','assistant')`), then
   fan out into `content_blocks` via `UNNEST(json_extract(content_json, '$[*]'))`
   (`sql_views.py:711-728`).

### 1.3 S3 / remote transcripts (`s3_source.py`)

The remote path is deliberately thin. `settings_need_s3(settings)` (`s3_source.py:52`)
returns true when any of the three globs is an `s3://` URI (`is_s3_uri`, `s3_source.py:43`).
When so, `register_all` calls `configure_s3` (`s3_source.py:73`) which runs
`INSTALL httpfs; LOAD httpfs;` and `CREATE OR REPLACE SECRET claude_sql_s3 (TYPE s3,
PROVIDER credential_chain, REGION …[, ENDPOINT/URL_STYLE/USE_SSL])`. Credentials resolve
via DuckDB's `credential_chain` (the same AWS chain boto3 uses) — **no keys in SQL**. Once
the secret exists, the entire view/macro stack works unchanged against `s3://` because
`read_json` already accepts `s3://` glob args and streams via HTTP range requests. This is
already the seed of a multi-source `ReaderPort` — local vs S3 is a glob + secret decision,
not a code-path fork.

---

## 2. `sql_views.py` deep dive (2344 LOC) — the retrieval backbone

Five registration functions, each idempotent (`CREATE OR REPLACE`):

### 2.1 The view model

**Raw readers** (`register_raw`): `v_raw_events` (TEMP TABLE), `v_raw_subagents`
(TEMP TABLE), `v_raw_subagent_meta` (VIEW — meta files are tiny, one object each).

**v1 transcript-derived views** (`register_views`, enumerated in `VIEW_NAMES`
`sql_views.py:63-94`, schema-pinned in the static `VIEW_SCHEMA` dict `sql_views.py:117-289`):

- **Session/message model**: `sessions` → `messages` → `content_blocks` → `messages_text`
  (one row per *message*, text blocks `string_agg`'d with `HAVING length >= 32` to kill
  tiny-fragment embedding noise, `sql_views.py:738-754`).
- **Turn model**: `turn_window` (`sql_views.py:768-784`) — adjacent-turn `LAG()` window over
  `messages_text` per session, ordered by `(ts, uuid)`, excluding compact-summary rows;
  emits `(prev_uuid, curr_uuid, gap_ms, window_idx)`. This is the pair key the trajectory /
  friction pipelines consume.
- **Tool-use model**: `tool_calls` (`block_type='tool_use'`) and `tool_results`
  (`block_type='tool_result'`), both projections of `content_blocks`.
- **Todo model**: `todo_events` (UNNEST `$.todos[*]` from `TodoWrite`), `todo_state_current`
  (latest per `(session, subject)`).
- **Task model** (Claude Code v2.1.16+ `TaskCreate`/`TaskUpdate` family, distinct from the
  v2.1.63 `Task`→`Agent` subagent rename): `task_creations`, `task_updates`,
  `tasks_state_current` (recovers runtime-assigned `task_id` from tool_result text/JSON, else
  falls back to per-session creation order, `sql_views.py:935-984`).
- **Subagent model**: `subagent_spawns` (`Task`/`Agent` tool calls), `subagent_sessions`
  (from `v_raw_subagents` joined to meta), `subagent_messages`.
- **Skill model**: `skill_invocations` (`sql_views.py:1020-1074`) — unions three shapes:
  the built-in `Skill` tool, list-typed `<command-name>/…` text blocks, and bare-VARCHAR
  slash-command turns. Adds a `context_uuid` subquery resolving each invocation to the
  nearest prior user text message (GH #50).

**v2 analytics views** (`register_analytics`, `sql_views.py:1953`, names in
`ANALYTICS_VIEW_NAMES` `sql_views.py:296-309`): each is a `CREATE OR REPLACE VIEW … FROM
read_parquet([…])` over a parquet cache — `session_classifications`, `session_goals`,
`message_trajectory`, `session_conflicts`, `conflicts_summary`, `message_clusters`,
`cluster_terms`, `session_communities`, `community_profile`, `user_friction`,
`skills_catalog`, `skill_usage`, `ingest_stamps`. **Each view is skipped with a DEBUG log
when its parquet is missing** (`_parquet_is_populated`, `sql_views.py:1938`), so the function
is idempotent on partially-populated systems and a fresh install never crashes. A curated
projection can fall back to bare `SELECT *` when the on-disk schema predates a rewrite
(`sql_views.py:2112-2142`).

### 2.2 Macros (`register_macros`, `sql_views.py:1232`)

18 macros, names in `MACRO_NAMES` (`sql_views.py:316-339`), signatures pinned in the static
`MACRO_SIGNATURES` dict (`sql_views.py:358-380`) because DuckDB's `duckdb_functions()`
returns NULL params for table macros. **v1 always-on**: `ago`, `model_used`, `cost_estimate`
(prefix-match pricing join stripping the `-\d{8}$` dated suffix), `tool_rank`,
`todo_velocity`, `subagent_fanout`, `skill_rank`, `skill_source_mix`, and `semantic_search`.
**v2 analytics macros** gated on parquet existence via `_ANALYTICS_MACRO_REQUIREMENTS`
(`sql_views.py:1171-1188`): `autonomy_trend`, `work_mix`, `success_rate_by_work`,
`cluster_top_terms`, `community_top_topics`, `sentiment_arc`, `friction_counts`,
`friction_rate`, `friction_examples`, `conflicts_over_time`, `unused_skills`,
`canonical_uuid_resolve`. `_safe_macro` (`sql_views.py:1204`) is the defensive backstop for
the gate-check/DDL-bind race.

### 2.3 How SQL search works (`semantic_search` + `register_vss`)

`semantic_search(query_vec, k)` (`sql_views.py:1359-1366`) is a table macro:
`SELECT me.uuid, array_cosine_similarity(…), array_distance(…) FROM message_embeddings
ORDER BY array_distance(me.embedding, query_vec) LIMIT k`. The `ORDER BY array_distance`
triggers the LanceDB IVF_HNSW_SQ index. `register_vss` (`sql_views.py:1812`) `INSTALL
lance; LOAD lance; ATTACH '<uri>' AS lance_store (TYPE LANCE)` and binds
`message_embeddings` as a view casting `embedding` to `FLOAT[dim]`. **Empty-namespace
gate** (`sql_views.py:1884-1902`): it probes `lance_store._has_table` and, when absent,
creates an empty `message_embeddings` *table* so `semantic_search` still binds on a fresh
install (returns `False`). `skip_vss` (threaded from `cli._sql_uses_vss`) skips both the
ATTACH round-trips and the `semantic_search` macro for non-vector queries.

DuckDB errors from the query surface flow through `output.classify_duckdb_error`
(exit 64/65/70) — see §8.

---

## 3. `config.py` — the pydantic-settings model (the embedding-switch seam)

`class Settings(BaseSettings)` (`config.py:126`), `env_prefix="CLAUDE_SQL_"`,
`env_file=".env"`, `extra="ignore"` (`config.py:133-137`). Full field enumeration:

**Data discovery**
- `default_glob: str` — factory `_default_glob` (`config.py:142`)
- `subagent_glob: str` (`config.py:143`)
- `subagent_meta_glob: str` (`config.py:144`)
- `team_corpus_root: Path | None` (`config.py:149`) — when set, rewrites the three globs to
  `<root>/<author>/projects/*` via the `_derive_team_corpus_globs` model-validator
  (`config.py:368-406`); per-glob user pins win.

**S3 transcript source**
- `s3_endpoint: str | None` (`config.py:170`)
- `s3_url_style: Literal["vhost","path"] = "vhost"` (`config.py:177`)
- `s3_use_ssl: bool = True` (`config.py:178`)

**Bedrock / embedding** ← *the block the v2 pluggable-embeddings switch threads through*
- `region: str = "us-east-1"` (`config.py:183`)
- **`model_id: str = "global.cohere.embed-v4:0"`** (`config.py:187`) — the embedding model
- **`output_dimension: Literal[256,512,1024,1536] = 1024`** (`config.py:189`)
- **`embedding_type: Literal["int8","float","uint8","binary","ubinary"] = "int8"`** (`config.py:190`)
- `embed_concurrency: int = 8` (`config.py:194`)
- `llm_concurrency: int = 16` (`config.py:202`)
- `batch_size: int = 96` (`config.py:203`)
- `embeddings_parquet_path: Path` (`config.py:205`) — legacy shard dir (migration only)
- `active_model_id` property (`config.py:408-411`) returns `self.model_id`

**LanceDB embeddings store**
- **`lance_uri: Path`** — factory `_default_lance_uri` = `<home>/embeddings_lance` (`config.py:213`)
- **`hnsw_metric: Literal["cosine","l2","dot"] = "cosine"`** (`config.py:216`)

**Pricing**
- `pricing: dict[str, tuple[float,float]]` — defaults to `DEFAULT_PRICING` (`config.py:117-123`,
  Opus/Sonnet/Haiku 4.x per-MTok in/out rates), `config.py:221`

**v2 LLM classification (Sonnet 4.6)**
- `sonnet_model_id: str = "global.anthropic.claude-sonnet-4-6"` (`config.py:228`)
- `sonnet_pricing: tuple[float,float] = (3.0, 15.0)` (`config.py:230`)
- `classify_thinking: Literal["adaptive","disabled"] = "adaptive"` (`config.py:235`)
- `trajectory_thinking = "disabled"` (`config.py:239`), `friction_thinking = "disabled"` (`config.py:244`)
- `classify_max_tokens: int = 16000` (`config.py:253`)
- `session_text_tool_result_max_chars: int = 50_000` (`config.py:256`)
- `session_text_total_max_chars: int = 800_000` (`config.py:259`)

**v2 parquet output paths** (all default-factory under `claude_sql_home()`):
`classifications_parquet_path` (`262`), `trajectory_parquet_path` (`263`),
`conflicts_parquet_path` (`264`), `clusters_parquet_path` (`265`),
`cluster_terms_parquet_path` (`266`), `communities_parquet_path` (`267`),
`community_profile_parquet_path` (`273`), `user_friction_parquet_path` (`279`),
`friction_max_chars: int = 300` (`284`), `skills_catalog_parquet_path` (`291`),
`user_skills_dir` (`293`), `plugins_cache_dir` (`297`), `checkpoint_db_path`
(`300`, = `state.db`), `ingest_stamps_parquet_path` (`306`).

**v2 UMAP/HDBSCAN/Leiden** (`config.py:311-342`): `umap_n_components_50=50`,
`umap_n_components_2=2`, `umap_n_neighbors=30`, `umap_min_dist_cluster=0.0`,
`umap_min_dist_viz=0.1`, `umap_metric="cosine"`, `hdbscan_min_cluster_size=20`,
`hdbscan_min_samples=5`, `leiden_knn_k=15`, `leiden_edge_floor=0.3`,
`leiden_min_community_size=3`, `leiden_resolution: float | None = None` (auto-γ),
`leiden_resolution_range_lo/hi=0.05/0.95`, `leiden_n_iterations=-1`, `seed=42`.

**v2 TF-IDF** (`config.py:347-351`): `tfidf_min_df=2`, `tfidf_max_df=0.95`,
`tfidf_ngram_min=1`, `tfidf_ngram_max=2`, `tfidf_top_n_terms=10`.

**DuckDB engine tuning** (`config.py:359-366`): `duckdb_threads` (= `os.cpu_count()`),
`duckdb_memory_limit="70%"`, `duckdb_temp_dir` = `<home>/duckdb_tmp`.

> **v2 embedding-provider switch touches exactly**: `model_id`, `output_dimension`,
> `embedding_type`, `embed_concurrency`, `batch_size`, `hnsw_metric`, `lance_uri`, and
> `region`. Today `model_id` is documented as "Cohere Embed v4, no reason to expose the
> knob" (`config.py:184-187`) and `embed_worker` hard-codes the Cohere request body
> (`input_type`, `embedding_types`). A pluggable-embeddings v2 needs an `EmbeddingProvider`
> enum + a provider-shaped request builder so `model_id` no longer implies Cohere's wire
> format.

---

## 4. `schemas.py` — the pydantic domain models (closest to a v2 domain layer)

Seven Pydantic v2 models, each `ConfigDict(extra="forbid")`, each with a module-level
`*_SCHEMA` constant produced by `_bedrock_schema` (`schemas.py:16`) which flattens
`model_json_schema()` (inline `$ref`, inject `additionalProperties:false`, strip the
numeric/string/array constraints Bedrock's Draft-2020-12 subset rejects — `_flatten`,
`schemas.py:39-96`):

1. **`SessionClassification`** (`schemas.py:99`) → `SESSION_CLASSIFICATION_SCHEMA` — fields
   `autonomy_tier` (manual/assisted/autonomous), `work_category` (sde/admin/strategy_business/
   events/thought_leadership/other), `success` (success/partial/failure/unknown),
   `goal: str`, `confidence: float`.
2. **`TrajectoryWindow`** (`schemas.py:173`) — windowed turn-pair: `prev_uuid | None`,
   `curr_uuid`, `prev_sentiment`, `curr_sentiment`, `delta`, `is_transition`,
   `transition_kind` (frustration_spike/resolution/reset/drift/clarification/none), `confidence`.
3. **`TrajectoryArrayResult`** (`schemas.py:266`) → `TRAJECTORY_ARRAY_SCHEMA` — `windows: list[TrajectoryWindow]`.
4. **`ConflictPair`** (`schemas.py:292`) — `turn_a_uuid`, `turn_b_uuid`, `conflict_kind`
   (disagreement/correction/reversal/impasse), `severity` (low/medium/high),
   `agent_position`, `user_position`, `confidence`.
5. **`ConflictsResult`** (`schemas.py:379`) → `SESSION_CONFLICTS_SCHEMA` — `conflicts: list[ConflictPair]`.
6. **`UserFrictionSignal`** (`schemas.py:406`) → `USER_FRICTION_SCHEMA` — `label`
   (status_ping/unmet_expectation/confusion/interruption/correction/frustration/none),
   `rationale`, `confidence`.
7. **`Correction`** (`schemas.py:477`) + **`PRReviewSheet`** (`schemas.py:509`) →
   `PR_REVIEW_SHEET_SCHEMA` — the provenance review-sheet model (`human_intent`,
   `agent_exploration`, `corrections`, `tools_used`, `tools_refused`, `diff_rationale`).

These enums (autonomy tier, work category, success, sentiment, transition kind, conflict
kind, severity, friction label) **are the domain vocabulary of claude-sql** — a v2
hexagonal domain layer would lift these models (stripped of their Bedrock-schema
serialization concern) into the domain package, and the `_bedrock_schema`/`_flatten`
machinery becomes an adapter concern of the LLM port.

---

## 5. `llm_shared.py` (1343 LOC) — the LLM port / Bedrock adapter

Everything every classifier pipeline needs, with **no cross-imports between stage workers** —
all shared symbols live here. Consumers (`analytics/{classify,conflicts,friction,
trajectory,embed}_worker.py`, `provenance/review_sheet_worker.py`, `app/cli.py`) import from it.

### 5.1 Client construction (verbatim signature)

There is **no `get_bedrock_client`** — the actual function is `_build_bedrock_client`
(underscore-private, but imported across the analytics package):

```python
# llm_shared.py:348
def _build_bedrock_client(settings: Settings) -> Any:
```

Process-wide cache keyed on `(region, pool_size)` under a `threading.Lock`
(`llm_shared.py:344-394`); `pool_size = max(32, max(embed_concurrency, llm_concurrency)*2)`;
`BotoConfig(retries={"max_attempts":0,"mode":"adaptive"}, max_pool_connections=…,
connect_timeout=10, read_timeout=600)`. Botocore's own retry loop is disabled
(`max_attempts=0`) so tenacity owns the semantic retry policy.

**Callers (via CodeGraph `callers _build_bedrock_client`, 7 non-test sites)**:
`classify_worker._classify_sessions_async` (`analytics/classify_worker.py:45`),
`conflicts_worker._conflicts_async` (`:167`), `embed_worker.embed_documents_async` (`:268`)
and `embed_query` (`:338`), `friction_worker._classify_async` (`:420`),
`trajectory_worker._trajectory_async` (`:705`), `review_sheet_worker.generate_review_sheet`
(`:342`). **Note the seam gap**: `evals/judge_worker.py` builds its *own* boto3 client
(`evals/judge_worker.py:223-229`), not the shared one — one reason evals is a natural drop
candidate in v2. `embed_worker` uses the shared client but issues Cohere embedding
`invoke_model` calls directly (`analytics/embed_worker.py:235`), not through `classify_one`.

### 5.2 The invoke wrapper (verbatim signature)

```python
# llm_shared.py:397-414 (decorator + def)
@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception(_is_retryable),
    before_sleep=loguru_before_sleep("WARNING"),
    reraise=True,
)
def _invoke_classifier_sync(
    client: Any,
    model_id: str,
    schema: dict[str, Any],
    user_text: str,
    *,
    max_tokens: int,
    thinking_mode: str,
    system: str | None = None,
    pipeline: str = "classifier",
) -> dict[str, Any]:
```

Builds the Bedrock body with `output_config.format` GA structured output
(`{"type":"json_schema","schema":schema}`, `llm_shared.py:443-450`), attaches the system
block with 1h `cache_control` (`build_system_content_block`, `llm_shared.py:104`; used at
`:471`), sets `thinking:{type:"adaptive"}` when requested, calls `client.invoke_model`, and
returns `_parse_structured_payload(payload)`.

### 5.3 Structured-output parse + async dispatch

`_parse_structured_payload` (`llm_shared.py:502`) handles four observed Bedrock response
shapes and raises `BedrockRefusalError` (`llm_shared.py:492`, terminal/non-retryable) on
`stop_reason=="refusal"`. `_is_retryable` (`llm_shared.py:330`) matches SSL/connection/
read-timeout + the `_RETRY_CODES` throttle set (`llm_shared.py:61-75`).

The async entry point:

```python
# llm_shared.py:565-576
async def classify_one(
    client: Any,
    model_id: str,
    schema: dict[str, Any],
    text: str,
    *,
    max_tokens: int,
    thinking_mode: str,
    sem: asyncio.Semaphore | anyio.CapacityLimiter,
    system: str | None = None,
    pipeline: str = "classifier",
) -> dict[str, Any]:
```

Runs under the concurrency limiter and hands the blocking call to
`anyio.to_thread.run_sync` (honors structured-concurrency cancellation). Callers:
classify/conflicts/friction/trajectory workers (CodeGraph `callers classify_one`).

### 5.4 The rest of the module

Four task-framing system prompts (`CLASSIFY_/TRAJECTORY_/CONFLICTS_/USER_FRICTION_SYSTEM_PROMPT`,
`llm_shared.py:614-1247`) each get a shared `_CLASSIFIER_APPENDIX` appended
(`llm_shared.py:1287-1290`); a threadsafe per-pipeline cache-stat accumulator +
`pipeline_cache_stats` context manager (`llm_shared.py:147-282`); and `--dry-run` helpers
`_estimate_cost` / `_count_pending_sessions` (pure-SQL `COUNT(DISTINCT session_id)`,
`llm_shared.py:1293-1343`).

> **v2**: `llm_shared` splits cleanly into an **`LLMPort`** (async `classify_one` /
> `invoke_structured`) with a **Bedrock adapter** (client construction, retry policy, the
> four response shapes, the cache-stat accumulator). The system prompts + `_bedrock_schema`
> flattening are adapter-side serialization concerns. The prompt text arguably belongs in
> the domain/application layer, injected into the port.

---

## 6. Storage & durability

| Store | File/dir | Persists | Idempotency model |
|---|---|---|---|
| **`lance_store`** | `~/.claude/embeddings_lance/` | `FLOAT[dim]` message vectors + IVF_HNSW_SQ index | Append-only fragments via `tbl.add()`; `optimize_if_needed` compacts; `ensure_index` no-ops if index exists; anti-join dedup via `get_embedded_uuids` (`lance_store.py:245`, uuid-projected scan). One-time parquet→Lance migration is row-count-gated (`migrate_from_parquet_shards`, `lance_store.py:167`). |
| **`parquet_shards`** | `<cache>/part-<ts_ns>.parquet` dirs | classify/trajectory/conflicts/friction/skills outputs | `write_part` (`parquet_shards.py:87`) drops a fresh nanosecond-keyed shard (append cost ∝ `len(df)`, no read-rewrite); legacy single-file path still supported (`is_sharded_dir`, `:54`). `replace_sessions` (`:181`) filters-and-rewrites per shard for re-scoring dedup; empty shards unlinked. |
| **`checkpointer`** | `~/.claude/state.db` (SQLite WAL) | `session_checkpoint(session_id, pipeline, last_ts, last_mtime, completed_at)` | UPSERT `ON CONFLICT DO UPDATE` (`mark_completed`, `:330`); `filter_unchanged` (`:286`) skips a session iff *both* `last_ts` and `last_mtime` have not advanced. One-time DuckDB→SQLite migration with a sentinel file (`:92`). Schema bootstrap cached per-process behind a lock to defeat WAL writer races (`:227-262`). |
| **`retry_queue`** | same `state.db` | `retry_queue(pipeline, unit_id, error, attempts, next_attempt_at, completed_at)` | `enqueue` (`:85`) UPSERTs with `attempts+=1` and exponential backoff `2^attempts` min capped at 60 (`_backoff_delta`, `:79`); `drain` (`:131`) returns due, uncompleted, under-max rows; `mark_done` (`:162`) stamps `completed_at` (row kept as audit). `unit_id` = `session_id` for classify/conflicts/friction, message `uuid` for trajectory. Reuses the checkpointer's `_connect`. |

`home.py` (`claude_sql_home`, `:51`) resolves the cache root (`$CLAUDE_SQL_HOME` →
macOS `Application Support` → `$XDG_DATA_HOME` → `~/.claude-sql`) and
`recognized_legacy_caches` (`:75`) drives the RFC-0002 one-time move of caches out of
`~/.claude/`.

---

## 7. `session_text.py` — application-support domain service

Assembles per-session transcripts for the LLM prompts. The design fact that matters: a naive
per-session `SELECT … WHERE session_id=?` is **quadratic** against the zero-copy glob (every
query rescans every JSONL). `session_text_corpus` (`:198`) materializes the three source
views (`messages_text`, `tool_calls`, `tool_results`) in **one glob scan**, registers the
in-window session ids as an Arrow relation (avoids pandas-probe overhead of bound list
params), then slices per-session in Python with a deterministic total-order sort
(`_timeline_sort_key`, `:77`). `SessionTextCorpus.assemble` (`:150`) applies the
`session_text_*` char caps and optionally emits `[uuid=… role ts]` headers (conflicts needs
verbatim turn uuids). `iter_session_texts` (`:454`) is the streaming wrapper. `session_bounds`
(`:299`) returns `(last_ts, transcript_mtime)` per session for the checkpoint skip. Wraps
DuckDB queries in `try/except duckdb.IOException` for stale/rotating JSONLs (`:249`, `:331`).
It is a **domain service** (transcript → prompt text), depending only on the reader views + config.

---

## 8. `output.py` + `logging_setup.py` — application/CLI adapters

`output.py` is the agent-friendly boundary: `OutputFormat` StrEnum (auto/table/json/ndjson/csv,
`:26`), `emit_dataframe`/`emit_json`, `validate_glob` (rejects `>1` `**`, `:156`), and
`classify_duckdb_error` (`:180`) mapping DuckDB exceptions to the stable `EXIT_CODES`
(`:49-57`: parse→64, catalog→65, runtime→70). `run_or_die` (`:228`) wraps subcommand bodies.
`logging_setup.py` is loguru-only (stdlib `logging` is banned repo-wide) and supplies
`loguru_before_sleep` (`:53`) so tenacity `@retry` decorators log natively — used by the
`_invoke_classifier_sync` decorator above.

---

## 9. Hexagonal placement summary (v2)

v2 direction: core largely becomes **domain + infrastructure**; drop evals; pluggable
embeddings; keep retrieval + clustering; stay on 3.13.

| Module | Layer | Implied port |
|---|---|---|
| `schemas.py` | **Domain** | (domain models — the vocabulary) |
| `session_text.py` | Domain service / application-support | (consumes ReaderPort) |
| `sql_views.py` + `s3_source.py` | Infrastructure | **`ReaderPort`** (local + S3 transcript sources) + query/retrieval layer |
| `llm_shared.py` | Infrastructure | **`LLMPort`** + Bedrock adapter |
| `lance_store.py` | Infrastructure | **`VectorStorePort`** |
| `parquet_shards.py` | Infrastructure | **`AnalyticsCachePort`** |
| `checkpointer.py` | Infrastructure | **`CheckpointPort`** |
| `retry_queue.py` | Infrastructure | **`RetryQueuePort`** |
| `config.py` | Infrastructure/config | config seam (embedding-provider switch) |
| `output.py` | Application/CLI adapter | **`PresenterPort`** |
| `logging_setup.py` + `home.py` | Infrastructure (cross-cutting) | — |

**Refactor sequencing (foundation-first)**: (1) unify the duplicated glob defaults
(`sql_views.DEFAULT_GLOB` vs `config._default_glob`) behind a single `ReaderPort` and lift
the `_RAW_EVENT_COLUMNS`/`_MESSAGE_STRUCT_TYPE`/session-id-regex triple into an
`AgentReader` protocol so alternate transcript shapes plug in; (2) split `llm_shared` into
`LLMPort` + Bedrock adapter, and fold `evals`'s private client into (or delete with) evals;
(3) introduce an `EmbeddingProvider` seam over `model_id`/`output_dimension`/`embedding_type`
plus the provider-shaped request builder currently hard-coded in `embed_worker`; (4) the
four storage modules are already port-shaped — formalize their protocols last.
