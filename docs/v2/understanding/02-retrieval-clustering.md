# 02 — The Retrieval + Clustering/Structure Plane (`analytics/`)

**Scope:** the `analytics/` package plus the two `core/` modules that back
retrieval (`core/sql_views.py`, `core/lance_store.py`). This is the **KEEP**
plane for v2 — the reason claude-sql is best-in-class at *finding and reading*
Claude Code transcripts, and the reason the package is pinned to Python 3.13.

Every claim below cites `path:line`. Signatures are quoted verbatim from the
current on-disk source.

> **Terminology correction up front.** The v2 brief calls this the
> "clustering/pagerank/intelligence plane." **There is no PageRank in the
> codebase** — a repo-wide grep for `pagerank|centrality|eigenvector` returns
> zero hits in `src/`. The graph-structure signal is *Leiden + CPM community
> detection* over a mutual-kNN cosine graph, plus per-community **medoid**
> (max-mean-intra-cosine) and **coherence** (mean-intra-cosine). Wherever the
> brief says "pagerank," read "medoid / coherence ranking over Leiden
> communities." This is called out again in §2 and the summary.

---

## 1. The retrieval surface: how you *find* a transcript

Retrieval is a three-modality stack over the same corpus of
`~/.claude/projects/**/*.jsonl` transcripts. All three bind onto **one
in-memory DuckDB connection** opened by `_open_connection_full`
(`app/cli.py:342`), which calls `register_all(...)` to wire raw readers,
18 views, macros, VSS, and analytics views.

### 1a. Lexical / structural: SQL views (`core/sql_views.py`, 2344 lines)

DuckDB reads the JSONL corpus **zero-copy** through `read_json(...)` with the
resilience flags (`union_by_name=true`, `ignore_errors=true`,
`maximum_object_size=67_108_864`) in `register_raw` (`core/sql_views.py:480`,
DDL at `:527`–`:611`). The raw readers `v_raw_events` / `v_raw_subagents` are
`CREATE TEMP TABLE` so JSON schema inference is paid once per connection
(`core/sql_views.py:489`–`495`).

The load-bearing retrieval view is **`messages_text`** (`core/sql_views.py:740`):
one row per message `uuid`, aggregating text blocks with a `HAVING length(...)
>= 32` floor. Nearly every worker's candidate SQL selects from `messages_text`.
Its derived companion **`turn_window`** (`core/sql_views.py:770`) is a `LAG()`
window per session producing `(prev_uuid, curr_uuid, prev_role, curr_role,
gap_ms, window_idx)` — the direct input to the trajectory pipeline.

`register_macros` (`core/sql_views.py:1232`) adds table macros for the
agent-first retrieval surface: `semantic_search`, `cluster_top_terms`
(`:1529`), `community_top_topics` (`:1544`), `sentiment_arc` (`:1583`),
`friction_counts`/`friction_rate` (`:1607`/`:1630`), `conflicts_over_time`
(`:1715`), `skill_rank` (`:1379`), `unused_skills` (`:1753`).

### 1b. Semantic search: LanceDB ANN (`core/lance_store.py` + `embed_query`)

The embeddings live in a **local LanceDB dataset** (`~/.claude/embeddings_lance/`,
`config.py:42`/`:213`), not parquet + `hnsw.duckdb` anymore. Lance holds the
`FLOAT[1024]` vectors AND the `IVF_HNSW_SQ` index in one versioned directory
(`core/lance_store.py:1`–24 module docstring). The Arrow schema pins the vector
column as a **fixed-size** list — required for indexing:

```python
# core/lance_store.py:77
pa.field("embedding", pa.list_(pa.float32(), dim), nullable=False),
```

Index creation uses the unified LanceDB ≥0.30 API:

```python
# core/lance_store.py:144
tbl.create_index("embedding", config=IvfHnswSq(distance_type=metric))
```

DuckDB reads Lance back via the **lance core extension**. `register_vss`
(`core/sql_views.py:1812`) runs `INSTALL lance; LOAD lance` (`:1875`–1876),
probes `lance_store._has_table` (`:1885`) — the empty-namespace gate — then
`ATTACH '<uri>' AS lance_store (TYPE LANCE)` (`:1909`) and binds a view:

```sql
-- core/sql_views.py:1912
CREATE OR REPLACE VIEW message_embeddings AS
SELECT uuid, model, dim,
       CAST(embedding AS FLOAT[1024]) AS embedding, embedded_at
FROM lance_store.main.embeddings;
```

If no Lance table exists yet, `register_vss` creates an **empty** DuckDB
`message_embeddings` table (`:1893`) so the `semantic_search` macro still binds
on a fresh install — returns `False` (`:1902`). This "views register only what
exists, never crash" is the plane's core resilience contract.

The `semantic_search(query_vec, k)` macro (`core/sql_views.py:1359`):

```sql
CREATE OR REPLACE MACRO semantic_search(query_vec, k) AS TABLE (
    SELECT me.uuid,
           array_cosine_similarity(me.embedding, query_vec) AS sim,
           array_distance(me.embedding, query_vec)          AS distance
    FROM message_embeddings me
    ORDER BY array_distance(me.embedding, query_vec)
    LIMIT k
);
```

### 1c. Structure: cluster / terms / community as retrieval signals

Beyond text/vector match, the plane materializes **structure** so an agent can
do progressive disclosure before reading raw transcripts:

- `message_clusters` (UMAP+HDBSCAN) → "which topic blob is this message in."
- `cluster_terms` (c-TF-IDF) → human-readable cluster names via
  `cluster_top_terms(cid, n)` (`core/sql_views.py:1529`).
- `session_communities` (Leiden+CPM) → medoid sessions, coherence, and
  `neighbors_of` lookup so an agent asks "give me the 15 sessions most like
  this one" without reading any of them (`community_worker.neighbors_of`,
  `analytics/community_worker.py:440`).

### 1d. End-to-end trace: a semantic search query

`claude-sql search "<query>"` → `search()` at `app/cli.py:1683`:

1. **Open + gate** — `_open_connection_full(settings)` (`cli.py:1743`). Because
   `search` builds its own VSS SQL after open, VSS is registered.
2. **Guard** — `SELECT count(*) FROM message_embeddings`; exit code 2
   (`no_embeddings`) if 0 (`cli.py:1745`–1749).
3. **Embed the query** — `embed_query(query_text, settings=settings)`
   (`cli.py:1751`), defined at `analytics/embed_worker.py:338`. It calls
   `_invoke_bedrock_sync(...)` with `input_type="search_query"` and forces
   `embedding_type="float"` (`embed_worker.py:358`–365) — HNSW distance math
   needs float, regardless of the corpus `int8` setting. This is a **Bedrock
   Cohere Embed v4** call (`global.cohere.embed-v4:0`, `config.py:187`).
4. **ANN lookup + join** — one DuckDB query (`cli.py:1759`–1776):
   ```sql
   WITH qv AS (SELECT CAST(? AS FLOAT[1024]) AS v)
   SELECT mt.uuid, mt.session_id, mt.role,
          array_cosine_similarity(me.embedding, (SELECT v FROM qv)) AS sim,
          substr(mt.text_content, 1, 200) AS snippet
   FROM message_embeddings me
   JOIN messages_text mt ON CAST(mt.uuid AS VARCHAR) = me.uuid
   ORDER BY array_cosine_distance(me.embedding, (SELECT v FROM qv)) ASC
   LIMIT ?
   ```
   The `ORDER BY array_cosine_distance ASC` is deliberate — it triggers the
   cosine-metric Lance/VSS index. Using L2 `array_distance` would bypass the
   index and mis-rank (doc vectors are int8-cast-to-float with magnitudes in
   the thousands; the query vector is unit-normalized — see `cli.py:1753`–1758).
5. **Emit** — `emit_dataframe(...)` returns `(uuid, session_id, role, sim,
   snippet)`; `--format auto` = table on TTY, JSON on pipe.

Query text → Cohere `search_query` embed → LanceDB ANN via DuckDB → join to
`messages_text` → session rows. Done.

---

## 2. The clustering / structure / intelligence plane

Two distinct sub-planes: **pure-compute** (numeric/graph — cluster, terms,
community, ingest) and **LLM-backed** (Bedrock Sonnet — classify, trajectory,
conflicts, friction). §4 splits them explicitly.

### 2a. `cluster_worker.py` (228 lines) — UMAP + HDBSCAN — PURE COMPUTE

`run_clustering(settings, *, force=False)` (`cluster_worker.py:53`).

- **Input:** the LanceDB embeddings matrix, read directly via the LanceDB
  Python API (`_load_embeddings`, `cluster_worker.py:28`) — `(N, 1024)` float32.
- **Compute:** two UMAP fits — 50d for clustering, 2d for viz — both
  `random_state=settings.seed` (`cluster_worker.py:145`–169), then HDBSCAN on
  the 50d projection (`:178`) with `metric="euclidean"`, `core_dist_n_jobs=-1`.
- **Output:** `clusters.parquet` = `(uuid, cluster_id, x, y, is_noise)`,
  `cluster_id = -1` for noise (`cluster_worker.py:202`–219).
- **Heavy imports (function-local, lazy):**
  ```python
  # cluster_worker.py:131
  import hdbscan
  import umap
  ```
- **Idempotency:** mtime sidecar `clusters.parquet.embeddings_mtime`. It walks
  the Lance dir tree, takes max mtime, and skips the ~40 s refit when unchanged
  (`cluster_worker.py:88`–128). `force=True` always rebuilds.

### 2b. `terms_worker.py` (145 lines) — c-TF-IDF — PURE COMPUTE

`run_terms(con, settings, *, force=False)` (`terms_worker.py:29`).

- **Input:** `clusters.parquet` joined to `messages_text` on uuid
  (`terms_worker.py:70`–79); one pseudo-document per cluster.
- **Compute:** `sklearn.feature_extraction.text.CountVectorizer` (imported
  lazily, `terms_worker.py:65`) with `ngram (1,2)`, `min_df=2`,
  `max_df=0.95`; then **in-house c-TF-IDF** math — per-class TF, L1 norm, IDF
  as `log(1 + avg/col_sum)` (`terms_worker.py:107`–115). No `bertopic` by
  design (CLAUDE.md "c-TF-IDF note").
- **Output:** `cluster_terms.parquet` = `(cluster_id, term, weight, rank)`.

### 2c. `community_worker.py` (686 lines) — Leiden + CPM — PURE COMPUTE

`run_communities(con, settings, *, force, gamma, resolution)`
(`community_worker.py:491`). This is the "graph intelligence" the brief calls
"pagerank." The actual pipeline (module docstring `:1`–52):

1. **Session centroids** — `_load_session_centroids` (`:82`) joins
   `message_embeddings` (Lance-backed) to `messages` on uuid, averages message
   vectors per session via a single `np.add.reduceat` segmented sum, then
   L2-normalizes (`:157`–160). Raises `RuntimeError` if `message_embeddings`
   isn't bound (`:122`–129).
2. **Mutual-kNN cosine graph** — `_build_mutual_knn` (`:166`), k=15, edge floor
   0.3, symmetric so `max(w_ij, w_ji)` is a no-op.
3. **igraph** — `_build_igraph` (`:208`); edge attribute **must** be named
   `"weight"` (leidenalg looks it up by string).
4. **Auto-γ** — `_compute_resolution_profile` (`:221`) runs
   `leidenalg.Optimiser.resolution_profile` over `(0.05, 0.95)`;
   `_pick_zoom` (`:269`) picks the longest-plateau γ (coarse/medium/fine).
   Emits the `community_profile.parquet` sidecar so an agent can ask "what γ
   gives 50 communities" without rerunning Leiden.
5. **Leiden+CPM** — `_run_leiden_cpm` (`:300`):
   `la.find_partition(g, la.CPMVertexPartition, weights="weight",
   resolution_parameter=γ, seed=settings.seed, n_iterations=-1)`.
6. **Medoid + coherence** — `_compute_medoid_and_coherence` (`:350`): medoid =
   node with max mean-intra-community cosine; coherence = mean pairwise
   intra-community cosine. **This is the ranking signal, in place of any
   PageRank.**
7. **Stable relabel** — `_relabel_and_collapse` (`:386`) by descending size;
   communities below `leiden_min_community_size=3` collapse to
   `NOISE_COMMUNITY_ID = -1` (`:75`).
8. **Connectivity check** — `_warn_disconnected` (`:321`) is **warn-only**; it
   never splits (Park et al. 2024 rationale in docstring).

- **Output:** `session_communities.parquet` = `(session_id, community_id, size,
  is_medoid, coherence, gamma_used)` + conditional `community_profile.parquet`.
- **Heavy imports (function-local, lazy):**
  ```python
  # community_worker.py:214
  import igraph as ig
  # community_worker.py:236 and :309
  import leidenalg as la
  ```
- **Determinism:** `settings.seed=42` flows into both
  `find_partition(seed=...)` and `Optimiser.set_rng_seed(...)` (`:239`), so
  same-seed reruns produce byte-identical parquet (CLAUDE.md "Analytics
  pipeline determinism").
- **CPM over modularity** because for cosine edges γ has closed-form density
  semantics (Traag-Van Dooren-Nesterov 2011) and is resolution-limit-free
  (CLAUDE.md "Leiden+CPM note").

### 2d. `ingest.py` (544 lines) — token stamps + SimHash dedup — PURE COMPUTE

`stamp_messages` (`ingest.py:313`) + `resolve_canonicals` (`ingest.py:428`).

- **Compute:** `tiktoken` `cl100k_base` batch encode × 0.78 Anthropic ratio
  (`approx_tokens_batch`, `:95`); vectorized 64-bit **SimHash** over word
  3-grams via `blake2b` + numpy bit-voting (`simhash64`, `:125`); token budget
  buckets (`:197`). `canonical_uuid` near-dup resolution is a pure **DuckDB
  self-join** on `bit_count(xor(...))` gated by a top-16-bit bucket
  (`resolve_canonicals`, SQL at `:484`–509).
- **Output:** `ingest_stamps` sharded parquet = `(uuid, session_id,
  approx_tokens, simhash64, token_budget_bucket, canonical_uuid,
  first_seen_ts, stamped_at)` (schema `:232`). Feeds the **embed anti-join**:
  `discover_unembedded` skips near-dups pointing at a different canonical
  (`embed_worker.py:160`–166).
- **Heavy imports:** `numpy`, `tiktoken` (module-level, `ingest.py:48`/`:50`).
  Hand-rolled SimHash specifically to *avoid* the stale `simhash` PyPI package
  with no 3.13 wheels (`ingest.py:11`–15).

### 2e. `skills_catalog.py` (354 lines) — filesystem walk — PURE COMPUTE

`sync(settings, *, dry_run=False)` (`skills_catalog.py:296`). Walks
`~/.claude/skills/` + `~/.claude/plugins/cache/` for `SKILL.md` frontmatter +
`plugin.json`, plus a constant `BUILTIN_SLASH_COMMANDS` (`:50`). Uses
`packaging.version` to pick newest plugin version (`_version_sort_key`, `:115`).
Output `skills_catalog.parquet` backs the `skill_rank`/`unused_skills` macros.
No Bedrock, no numeric stack — just polars + pyyaml + packaging.

### 2f. LLM-backed workers (see §4 for the split)

- **`classify_worker.py`** (254) — one row/session: autonomy_tier,
  work_category, success, goal, confidence.
- **`trajectory_worker.py`** (1029) — per-session **windowed** sentiment arc
  over `turn_window`; one row per `(prev_uuid, curr_uuid)`.
- **`conflicts_worker.py`** (413) — pair-keyed stance conflicts
  `(turn_a_uuid, turn_b_uuid)`.
- **`friction_worker.py`** (741) — friction labels on short user messages;
  regex → SQL-stamp → LLM cascade.

---

## 3. Data flow: what's materialized where

```
JSONL corpus (~/.claude/projects/**/*.jsonl)
        │  DuckDB read_json (zero-copy, ignore_errors)
        ▼
  v_raw_events / v_raw_subagents  (TEMP TABLE, per-connection)
        ▼  register_views → messages / content_blocks / messages_text / turn_window (18 views)
        │
        ├─ ingest ──────► ingest_stamps/part-*.parquet   (tiktoken+SimHash; DuckDB self-join dedup)
        │
        ├─ embed ───────► ~/.claude/embeddings_lance/     (LanceDB: FLOAT[1024] + IVF_HNSW_SQ index)
        │                   ▲ Cohere Embed v4 on Bedrock
        │                   └ back-read via DuckDB lance ext → message_embeddings view
        │
        ├─ cluster ─────► clusters.parquet  (+ .embeddings_mtime sidecar)   [reads Lance]
        │       └ terms ► cluster_terms.parquet                             [reads clusters+messages_text]
        │
        ├─ community ───► session_communities.parquet (+ community_profile.parquet)  [reads Lance centroids]
        │
        └─ LLM analytics (Bedrock Sonnet 4.6):
             classify ──► session_classifications/part-*.parquet
             trajectory ► message_trajectory/part-*.parquet      [reads turn_window]
             conflicts ─► session_conflicts/part-*.parquet
             friction ──► user_friction/part-*.parquet
```

**Three storage backends:**

1. **DuckDB** — in-memory only (`duckdb.connect(":memory:")`, `cli.py:361`);
   views over JSONL + parquet + Lance. Nothing durable except the corpus it
   reads. Tuning PRAGMAs (threads, `memory_limit='70%'`, temp dir at
   `~/.claude/duckdb_tmp`) applied in `_apply_duckdb_pragmas`.
2. **LanceDB** — the *only* embeddings store; versioned directory with vectors
   + ANN index (`core/lance_store.py`).
3. **Parquet** — every other worker output, mostly **sharded** directories
   (`part-<ts_ns>.parquet` via `core/parquet_shards.write_part`). `clusters`,
   `cluster_terms`, `communities`, `community_profile`, `skills_catalog` are
   single-file; `ingest_stamps`, `embeddings` (legacy), classify/trajectory/
   conflicts/friction are sharded (config defaults `config.py:51`–110).

**Views register only what exists.** `register_analytics`
(`core/sql_views.py:1953`) does one `CREATE OR REPLACE VIEW` per parquet that
passes `_parquet_is_populated` (`:1938`, >16-byte gate); missing parquets warn
and no-op.

**Idempotency / backfill / checkpoint behavior:**

- **embed** — anti-join: `discover_unembedded` reads already-embedded uuids
  from Lance (`get_embedded_uuids`, `lance_store.py:245`, column-projected
  scan) and diffs in Python (`embed_worker.py:148`/`:180`). Chunk checkpoints
  every `max(batch_size*4,256)` rows; `optimize()` every 8 chunks
  (`embed_worker.py:499`); final `ensure_index`.
- **ingest** — SQL anti-join against existing `ingest_stamps` shards
  (`_pending_stamp_sql`, `ingest.py:244`); `resolve_canonicals` truncates +
  rewrites one consolidated shard.
- **cluster** — mtime sidecar skip (§2a).
- **community / terms** — output-parquet-exists short-circuit unless `force`
  (`community_worker.py:528`, `terms_worker.py:60`).
- **classify / trajectory / conflicts / friction** — **SQLite checkpointer**
  (`core/checkpointer.py`, `state.db`) keyed on `(session_id, last_ts, mtime)`
  via `filter_unchanged`, plus a **retry queue** (`core/retry_queue.py`) drained
  each run, plus a per-row anti-join against the output parquet. Reruns on
  untouched sessions are free. `BedrockRefusalError` is terminal → neutral
  placeholder row + retry-queue clear so refusals never cycle.
- **trajectory** additionally `replace_sessions(...)` before writing so a
  growing active session doesn't duplicate `(prev,curr)` pairs on rerun
  (`trajectory_worker.py:922`, GH #45). Stale-schema shards are auto-detected
  via parquet metadata and purged on first run (`_purge_old_shards`, `:365`).
- **conflicts** — `_purge_legacy_shards` (`conflicts_worker.py:88`) nukes the
  cache if any shard carries the v0 `conflict_idx`/`empty` columns.

**The analyze chain** (`app/cli.py:2281`) runs all stages on one shared
connection in order: skills-sync → ingest → embed → cluster → terms →
community → classify → trajectory → conflicts → friction. After embed and
cluster it calls `_rebind_vss` (`cli.py:402`) + `_refresh_analytics_views`
(`cli.py:382`) so downstream stages see freshly written Lance/parquet data
(RFC §9.6 stale-connection fix — community reads `message_embeddings` and would
see 0 rows without the rebind).

---

## 4. LLM-backed vs pure-compute

This split matters for v2: LLM analytics is a **separate concern** that can be
extracted behind a port without touching the retrieval/clustering core.

| Worker | Bedrock? | Model | What it computes |
|---|---|---|---|
| `ingest.py` | **No** | — | tiktoken tokens + blake2b SimHash + DuckDB dedup |
| `embed_worker.py` | **Yes** | Cohere Embed v4 (`global.cohere.embed-v4:0`) | message + query vectors |
| `cluster_worker.py` | **No** | — | UMAP + HDBSCAN topic clusters |
| `terms_worker.py` | **No** | — | c-TF-IDF cluster names |
| `community_worker.py` | **No** | — | Leiden+CPM communities, medoid, coherence |
| `skills_catalog.py` | **No** | — | filesystem catalog walk |
| `classify_worker.py` | **Yes** | Sonnet 4.6 (`global.anthropic.claude-sonnet-4-6`) | session autonomy/category/success/goal |
| `trajectory_worker.py` | **Yes** | Sonnet 4.6 | windowed sentiment arc + transition kinds |
| `conflicts_worker.py` | **Yes** | Sonnet 4.6 | pair-keyed stance conflicts |
| `friction_worker.py` | **Yes** (regex/SQL fast-paths first) | Sonnet 4.6 | short-message friction signals |

**Embed is Bedrock but not "LLM analytics"** — it's the vector-index feeder for
retrieval. The four Sonnet workers are the "LLM analytics" concern proper.
All four share plumbing in `core/llm_shared.py`: `_build_bedrock_client`
(`:348`), `classify_one` (`:565`, async under `anyio.CapacityLimiter` →
`anyio.to_thread`), `_invoke_classifier_sync` with `output_config.format` GA
structured output (`:404`/`:446`), `BedrockRefusalError` (`:492`),
`pipeline_cache_stats` (`:259`) for the 1h system-prompt cache accounting.

Friction is a **cascade** that minimizes Bedrock spend: regex fast-path
(`regex_fast_path`, `friction_worker.py:148`, confidence 0.9) → deterministic
SQL stamps (`sql_stamp`, `:271`, three rules) → Sonnet only for the ambiguous
remainder. Trajectory batches ≤16 windows/call (`MAX_WINDOWS_PER_CHUNK`,
`trajectory_worker.py:74`) with a bounded missing-window retry.

---

## 5. Hexagonal placement for v2

The repo already enforces a **layered** import contract (`pyproject.toml:264`):
`app → (analytics | evals | provenance) → core`, and analytics/evals/provenance
are independent siblings (`:273`). That's a clean starting point; the hexagonal
target refines it into domain / ports / adapters.

### Pure domain (no I/O — keep, wrap in nothing)

The numeric/graph math is already pure and side-effect-free at the function
level — ideal domain logic:

- **Clustering math** — `cluster_worker._load_embeddings` aside, the UMAP+HDBSCAN
  transform is pure numpy in/out.
- **Community math** — `_build_mutual_knn` (`community_worker.py:166`),
  `_compute_resolution_profile` (`:221`), `_pick_zoom` (`:269`),
  `_run_leiden_cpm` (`:300`), `_compute_medoid_and_coherence` (`:350`),
  `_relabel_and_collapse` (`:386`) are all pure `(ndarray|graph) → result`.
  **This is the ranking domain** (medoid/coherence — the "pagerank" analog).
- **c-TF-IDF** — the weighting math in `terms_worker.py:107`–125.
- **SimHash / token / dedup math** — `simhash64`, `approx_tokens_batch`,
  `hamming_distance_64`, `token_budget_bucket` (`ingest.py:95`–221).

These should move to a `domain/` (or stay as pure functions) with **no import
of duckdb, lancedb, or boto3**.

### Infrastructure (adapters — the driven side)

- **`LanceEmbeddingStore`** — `core/lance_store.py` in full. Owns connect,
  schema, add_chunk, ensure_index, optimize, migrate, get_embedded_uuids.
- **`DuckDBCatalog`** — `core/sql_views.py` (`register_raw`/`register_views`/
  `register_macros`/`register_vss`/`register_analytics`). The zero-copy JSONL
  reader + view catalog + the Lance `ATTACH` bridge.
- **`BedrockClient`** — `core/llm_shared._build_bedrock_client` +
  `_invoke_classifier_sync` + `embed_worker._invoke_bedrock_sync`. The retry /
  refusal / cache-stat wrapping.
- **Parquet cache** — `core/parquet_shards`, `core/checkpointer`,
  `core/retry_queue` (durable state adapters).

### Application use-cases (the driving side / orchestrators)

The `run_*` / `detect_* `/ `classify_*` entry points are use-case interactors:
`run_clustering`, `run_terms`, `run_communities`, `neighbors_of`,
`run_backfill`, `stamp_messages`, `resolve_canonicals`, `classify_sessions`,
`trajectory_messages`, `detect_conflicts`, `detect_user_friction`,
`skills_catalog.sync`. They currently take a live `duckdb.DuckDBPyConnection`
and a `Settings` — the seams to invert. The `analyze` chain (`cli.py:2281`) is
the top-level orchestrator.

### Ports to name

Driven (outbound) ports the use-cases should depend on, not the concretes:

- **`EmbeddingPort`** — `embed_documents(texts) -> vectors` /
  `embed_query(text) -> vector`. Today = Cohere-on-Bedrock
  (`embed_worker.embed_documents_async` / `embed_query`). **v2 makes this
  pluggable: Ollama + ONNX bge alongside Cohere.** This is the single most
  important port for v2 — everything downstream (cluster, community, search)
  consumes `message_embeddings` and is embedding-provider-agnostic already.
- **`VectorStorePort`** — `add`, `ann_search(query_vec, k)`, `embedded_ids`,
  `count`. Today = `LanceEmbeddingStore`. The `semantic_search` macro + the
  `search` CLI query are the read side.
- **`CatalogPort`** — register/read the SQL views over the corpus. Today =
  DuckDB. `messages_text` / `turn_window` / `message_embeddings` are the
  domain-facing relations.
- **`AnalyticsSinkPort`** — write/read the derived parquet artifacts
  (clusters, terms, communities, classifications, …). Today = parquet shards +
  `register_analytics` views.
- **`LLMClassifierPort`** — `classify(schema, text) -> dict`. Today = Sonnet on
  Bedrock via `classify_one`. This cleanly isolates the "LLM analytics"
  concern the brief wants separable.
- **`CheckpointPort` / `RetryQueuePort`** — idempotency state. Today = SQLite.

The independence contract (`pyproject.toml:273`) already means the four Sonnet
workers can be lifted behind `LLMClassifierPort` without disturbing cluster /
community / terms — confirming the v2 "drop evals, separate LLM analytics"
direction is structurally low-risk.

---

## 6. The 3.13 pin — concrete evidence

**Python floor is `>=3.13`**, agreed across three files: `pyproject.toml:8`
(`requires-python = ">=3.13"`), `.python-version` (`3.13`), `mise.toml:16`
(`python = "3.13"`).

### What forces it: `hdbscan` (transitively via the clustering stack)

`docs/adr/0015-stack-modernization.md` states it flatly:

> "Every native dep in the closure **except hdbscan** ships cp314 wheels as of
> 2026-05-08 (duckdb, numpy, scipy, scikit-learn, pyarrow, pydantic-core,
> pyyaml, numba, llvmlite). `hdbscan 0.8.42` stops at cp313. On 3.14, pip/uv
> fall back to the sdist — Cython + numpy + C toolchain at install time.
> Unacceptable for the `uv tool install claude-sql` end-user path."

So the pin is a **wheel-availability** constraint, not an API one. The single
package with no cp314 wheel is `hdbscan`, and it is imported at exactly one
site:

```python
# analytics/cluster_worker.py:131 (function-local)
import hdbscan
import umap
```

### The heavy scientific closure (locked versions from `uv.lock`)

| Package | Locked version | Imported at | Role |
|---|---|---|---|
| **hdbscan** | **0.8.44** | `cluster_worker.py:131` | **THE pin — no cp314 wheel** |
| umap-learn | 0.5.12 | `cluster_worker.py:132` | dimensionality reduction |
| numba | 0.66.0 | (transitive, umap/hdbscan) | JIT for umap/hdbscan; **caps numpy<2.5** |
| llvmlite | 0.48.0 | (transitive, numba) | numba's LLVM backend |
| leidenalg | 0.12.0 | `community_worker.py:236`,`:309` | Leiden+CPM community detection |
| igraph | 1.0.0 | `community_worker.py:214` | graph backend for leidenalg |
| scikit-learn | 1.9.0 | `terms_worker.py:65` | CountVectorizer for c-TF-IDF |
| scipy | 1.18.0 | (transitive) | umap/hdbscan/sklearn dep |
| numpy | 2.4.6 | `cluster_worker.py:21`, `community_worker.py:62`, `terms_worker.py:19`, `ingest.py:48` | matrices everywhere |
| lancedb | 0.34.0 | `lance_store.py:32`,`:35` | vector store + ANN |
| pyarrow | 25.0.0 | `lance_store.py:34`, `conflicts_worker.py:40`, `trajectory_worker.py:44` | Arrow bridge |
| duckdb | 1.5.4 | core (SQL) | zero-copy SQL + lance ext |
| tiktoken | 0.13.0 | `ingest.py:50` | token approximation |

Note the **numpy ceiling** in `pyproject.toml:38`:
`"numpy>=2.4.4,<2.5"` — comment explains numba 0.66.0 caps numpy<2.5;
without the ceiling `uv --upgrade` backsolves numpy 2.5 by downgrading numba to
0.53.1, which breaks umap/hdbscan. So numba (a transitive dep of the
clustering stack) also constrains the numpy version even though it's never
imported directly (`grep numba src/` = 0 hits).

The clustering pins are also tightened for **determinism**, not just wheels:
`leidenalg>=0.11.0,<0.13` and `igraph>=1.0.0,<2.0` (`pyproject.toml:34`,`:36`) —
the CLAUDE.md "Leiden+CPM note" pins these to protect against determinism drift
across patch versions.

### What could relax the pin later

- **The named follow-up (ADR 0015):** "flip to 3.14 in a one-line PR once
  `hdbscan 0.8.43+` publishes cp314 wheels." (Note: lock is already at hdbscan
  **0.8.44** — worth re-checking whether cp314 wheels have since shipped; if so
  the pin is stale and 3.14 is a one-line bump.) This is the cheapest path: no
  code change, just wheel availability.
- **Split clustering into an optional extra.** If `cluster` / `community` /
  `terms` moved behind an optional dependency group (`[project.optional-
  dependencies].clustering = ["hdbscan", "umap-learn", "leidenalg", "igraph"]`)
  and the function-local imports already isolate them (they do —
  `cluster_worker.py:131`, `community_worker.py:214`/`:236`), then the **base**
  package (SQL + semantic search + LLM analytics) could target 3.14+ and only
  the clustering extra would stay on 3.13. The retrieval surface (§1) needs
  only duckdb + lancedb + pyarrow + numpy — all of which ship cp314 wheels —
  so semantic search does NOT require the pin. **Only the structure sub-plane
  (cluster/terms/community) does.**
- **Swap hdbscan.** `scikit-learn` has shipped `HDBSCAN` since 1.3; if the
  clustering quality is acceptable, dropping the standalone `hdbscan` package
  (using `sklearn.cluster.HDBSCAN`) would remove the last cp314 blocker
  entirely, since sklearn already ships cp314 wheels. This is a domain-logic
  change (different HDBSCAN implementation) and would need a determinism
  re-validation.

---

## Appendix — key file/line index

- Query trace: `app/cli.py:1683` (`search`), `analytics/embed_worker.py:338`
  (`embed_query`), `core/sql_views.py:1359` (`semantic_search` macro),
  `core/sql_views.py:1812` (`register_vss`), `core/lance_store.py:144`
  (`IvfHnswSq` index).
- Views: `core/sql_views.py:740` (`messages_text`), `:770` (`turn_window`),
  `:1953` (`register_analytics`), `:1938` (`_parquet_is_populated`).
- Clustering: `cluster_worker.py:53`/`:131`, `terms_worker.py:29`/`:65`,
  `community_worker.py:491`/`:214`/`:236`/`:300`/`:350`.
- LLM plumbing: `core/llm_shared.py:348`/`:404`/`:565`/`:492`.
- Orchestration: `app/cli.py:2281` (`analyze`), `:342`
  (`_open_connection_full`), `:402` (`_rebind_vss`), `:382`
  (`_refresh_analytics_views`).
- Config: `core/config.py:187`/`:213`/`:311`–342 (embed model, lance uri,
  UMAP/HDBSCAN/Leiden hyperparameters, seed).
- Pin evidence: `pyproject.toml:8`/`:33`–49, `docs/adr/0015-stack-modernization.md`.
