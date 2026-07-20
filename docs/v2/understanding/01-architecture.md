# claude-sql v2 — Architecture & Hexagonal Layering

**Scope.** Grounded architecture map of the current (`v1.2.1`) codebase, written
to drive the v2 rewrite onto strict hexagonal / ports-and-adapters patterns.
Every structural claim cites `path:line`. Signatures are quoted verbatim from
the on-disk source.

**v2 direction (owner-decided, treated as fixed input):**
- Move onto strict hexagonal layering: `domain / application (ports) /
  infrastructure (adapters) / interfaces`.
- **DROP** the `evals/` plane entirely (judges / freeze / replay / kappa /
  blind-handover / ungrounded).
- **ADD** pluggable embedding providers behind a Protocol. Today it is pinned
  to Cohere-on-Bedrock; add Ollama and a local ONNX BAAI `bge` option.
- **KEEP** the advanced retrieval + clustering/pagerank/community plane
  (analytics). This is the differentiator. Keeping it means staying on
  Python 3.13 (`hdbscan`/`umap`/`leidenalg`/`numba` pin — `pyproject.toml`
  `requires-python = ">=3.13"` and the `numpy<2.5` numba ceiling comment).

---

## 1. Current layering reality

### 1.1 The five sub-packages and what each owns

One distribution, one namespace root `src/claude_sql/`, five sub-packages
(`pyproject.toml` `[tool.uv.build-backend] namespace = true`). LOC measured from
the tree:

| Layer | LOC | Owns | v2 fate |
|---|---|---|---|
| `core/` | 6,881 | DuckDB engine, S3/httpfs reader, LanceDB store, config, schemas, LLM plumbing, cross-cutting (logging, retry queue, checkpointer, output, parquet shards, session-text assembly) | **Split** — this is a mix of domain + infrastructure + application today |
| `analytics/` | 4,910 | Embeddings backfill, UMAP+HDBSCAN clustering, c-TF-IDF terms, Leiden+CPM communities, per-message trajectory, conflicts, friction, ingest/dedup, skills catalog | **KEEP** (the differentiator) |
| `app/` | 3,254 | Cyclopts CLI (`cli.py` 3,175 LOC, 101 symbols), version/install banner | **Becomes `interfaces/cli`** |
| `evals/` | 1,573 | Judge panel, Fleiss/Cohen kappa, freeze/replay studies, blind-handover, ungrounded-claim | **DROP entirely** |
| `provenance/` | 1,376 | Git↔transcript binding (trailer/note), PR review-sheet render + worker | **Drop-candidate — confirmed droppable** (see §1.4) |

Package roots: `src/claude_sql/{core,analytics,app,evals,provenance}/`.

### 1.2 The enforced import DAG (import-linter) vs the real edges

The contract in `pyproject.toml` `[tool.importlinter]` declares a 3-tier layered
DAG plus sibling independence:

```
layers = [
    "claude_sql.app",                                              # L2
    "claude_sql.analytics | claude_sql.evals | claude_sql.provenance",  # L1 siblings
    "claude_sql.core",                                             # L0
]
independence: analytics ⟂ evals ⟂ provenance
```

I ran `lint-imports` against the working tree: **both contracts KEPT, 40 files,
85 dependencies, 0 broken.** The declared DAG is real, not aspirational.

**Actual measured edges (from `grep` over `src/`, cross-checked with CodeGraph
blast-radius):**

- `analytics → core` only. Ten distinct `analytics → core.config`, plus
  `core.parquet_shards` (5), `core.llm_shared` (5), `core.session_text` (4),
  `core.schemas` (4), `core.logging_setup` (1). No `analytics → analytics`
  cross-imports (confirmed by `llm_shared.py:16-18` docstring: "There are NO
  cross-imports between the stage workers themselves").
- `provenance → core` only: `core.schemas`, `core.llm_shared`, `core.config`
  (1 each). No `provenance → analytics`.
- `evals → core` only: `core.logging_setup` (1). `evals/judges.py` imports
  nothing from the package at all (`grep` returns only `from __future__`).
  Evals is almost free-standing — it carries its **own** Bedrock client
  (`evals/judge_worker.py:221` `_bedrock_client`) rather than reusing
  `core.llm_shared._build_bedrock_client`. This is why dropping it is clean.
- `app → {analytics, provenance, evals, core}`. The app layer is the only
  importer of L1 siblings and the composition root.

**Load-bearing detail: the app→L1 imports are almost all _deferred_** (inside
command bodies), not module-top. `cli.py:45-56` documents this: heavy worker
imports each drag a "~0.5–1.4 s import subtree (boto3 via `llm_shared`,
`schemas`, numpy/polars re-exports)". Module-top imports are only the cheap
ones: `evals.{blind_handover, freeze, judges}` and `provenance.binding` +
`review_sheet_render` (`cli.py:91-97`). Every analytics worker and the heavy
eval workers (`judge_worker`, `kappa_worker`, `ungrounded_worker`,
`review_sheet_worker`) are imported lazily (`cli.py:1652,1738,1831,1881,1927,
1979,2037,2092,2124,2199,2348-2361,2638,2691,2732,3073`). This lazy-import
discipline is pinned by `test_cli_import_is_lean`. **v2 must preserve this** —
it is the reason `schema`/`query`/`explain` stay sub-second.

### 1.3 Where domain / application / infrastructure boundaries implicitly fall today

The layering is **technical (by dependency depth)**, not **hexagonal (by
purity)**. `core/` is the biggest problem: it mixes all three hexagonal
concerns in one package.

| Current location | Hexagonal role today (implicit) |
|---|---|
| `core/schemas.py` (pydantic classification models) | **Domain** (pure) — but coupled to Bedrock via `_bedrock_schema` JSON-Schema flattening (`schemas.py:16`) |
| `core/config.py` `Settings` | **Application config** — but leaks infra knobs (Bedrock model IDs, LanceDB URI, DuckDB PRAGMAs, S3 endpoint) into one god-object |
| `core/lance_store.py`, `core/s3_source.py`, `core/sql_views.py` | **Infrastructure (adapters)** — DuckDB, LanceDB, httpfs |
| `core/llm_shared.py` | **Infrastructure (Bedrock adapter) + application (retry/cache policy)** conflated |
| `core/session_text.py`, `core/parquet_shards.py` | **Infrastructure (DuckDB/parquet IO)** with some domain formatting logic mixed in |
| `analytics/*_worker.py` | **Application (use-cases)** — but each reaches directly into infra (DuckDB `con`, LanceDB, Bedrock) with no port indirection |
| `app/cli.py` | **Interface (driving adapter)** + **composition root** (wires connection, settings, workers) |
| `evals/`, `provenance/` | Application use-cases layered as L1 siblings |

**The single deepest hexagonal violation:** every analytics/eval/provenance
use-case takes a live `duckdb.DuckDBPyConnection` as its first parameter and a
concrete `Settings` as its second (`classify_worker.py:195`,
`conflicts_worker.py:358`, `trajectory_worker.py:957`, `friction_worker.py:655`,
`community_worker.py:491`, `terms_worker.py:29`, `embed_worker.py:369`). The
DuckDB connection **is** the de-facto reader port, but it is a concrete
third-party handle, not a domain-owned Protocol. There is no seam to substitute
it.

### 1.4 Provenance drop-decision — confirmed droppable

`provenance/` is a self-contained plane: git trailer/note binding
(`binding.py`), PR review-sheet worker + renderer (`review_sheet_worker.py`,
`review_sheet_render.py`). It backs three CLI commands: `bind`
(`cli.py:2794`), `resolve` (`cli.py:2927`), `review-sheet` (`cli.py:3033`). It
depends only on `core.{schemas, llm_shared, config}` and nothing depends on it
except the CLI. **It is as cleanly severable as evals.** The RFC that motivated
it is `docs/rfc/0001-transcript-pr-binding.md`. Recommendation: treat it the
same way as evals — either drop or park behind an optional `provenance` extra;
it is orthogonal to the "find & read transcripts" core value prop. Flag for
owner confirmation (batched question §5.4).

---

## 2. The core engine

### 2.1 Composition: `register_all` is the engine's assembly line

The engine is a DuckDB in-memory connection with a stack of views + macros
registered against zero-copy readers. The composition root is
`register_all` (`sql_views.py:2238`):

```python
def register_all(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings | None = None,
    include_analytics: bool = True,
    skip_vss: bool = False,
) -> None:
```

Order (`sql_views.py:2282-2302`), each step load-bearing:
1. `if settings_need_s3(settings): configure_s3(con, settings)` — httpfs + S3
   secret, only when a glob is `s3://` (`sql_views.py:2283-2284`).
2. `register_raw(con, glob, subagent_glob, subagent_meta_glob)` — the zero-copy
   `read_json` readers (`sql_views.py:480`).
3. `register_views(con)` — 18 derived views (`sql_views.py:626`).
4. `register_vss(...)` unless `skip_vss` — binds `message_embeddings` over the
   LanceDB dataset (`sql_views.py:1812`).
5. `register_analytics(con, settings)` — parquet-backed analytics views, each
   gated on parquet existence (`sql_views.py:1953`).
6. `register_macros(con, settings, skip_vss)` — 14 macros
   (`sql_views.py:1232`).

### 2.2 The DuckDB connection lifecycle (entry points)

Two connection factories in the CLI, both applying tuning PRAGMAs first:

- `_open_connection_full(settings, *, sql="")` (`cli.py:342`) — full
  `register_all`. Contains a substring optimization: `skip_vss = bool(sql) and
  not _sql_uses_vss(sql)` (`cli.py:363`) so non-vector `query`/`explain` skip
  the `INSTALL vss; LOAD vss; ATTACH` round-trips.
- `_open_connection_introspect(settings)` (`cli.py:368`) — bare connection,
  PRAGMAs only, no registration. For `schema` (reads static `VIEW_SCHEMA`) and
  trivial scalar SQL. Avoids "the ~25 s `register_all` chain entirely".

PRAGMA application is centralized in `_apply_duckdb_pragmas`
(`cli.py:324-339`): threads, memory_limit (percentage resolved to MiB via
`_resolve_memory_limit`, `cli.py:249`), `temp_directory`,
`enable_object_cache`, `preserve_insertion_order = false`. All sourced from
`Settings.duckdb_*` fields (`config.py:359-366`).

**Mid-run rebinding** (the `analyze` chain shares one connection across stages):
`_refresh_analytics_views` (`cli.py:382`) re-runs `register_analytics` after a
stage writes new parquet shards; `_rebind_vss` (`cli.py:402`) re-binds
`message_embeddings` after `embed` populates LanceDB (RFC §9.6
stale-connection bug). This shared-connection-with-rebind pattern is a
stateful coupling the v2 application layer must own explicitly.

### 2.3 The S3/httpfs zero-copy reader

`s3_source.py`. `read_json` already accepts an `s3://` glob; the only work is
loading `httpfs` and creating a `credential_chain` S3 secret on the connection
(`configure_s3`, `s3_source.py:73`). No download step — reads stay zero-copy
over HTTP range requests (`s3_source.py:14-15`). Trigger is
`settings_need_s3(settings)` (`s3_source.py:52`), which checks if any of the
three globs is `s3://` via `is_s3_uri` (`s3_source.py:43`). Credentials never
touch SQL — they resolve through DuckDB's `credential_chain`, the same AWS chain
boto3 uses (`s3_source.py:17-19`). This module is a **clean infra adapter
already** — it takes `(con, settings)` and issues DDL, no business logic.

### 2.4 The LanceDB embeddings store

`lance_store.py` replaced the parquet-shards + `hnsw.duckdb` combo
(`lance_store.py:1-24`). It is the vector-store adapter: `connect_db`
(cached, `:83`), `open_or_create_table` (`:100`), `add_chunk` (`:107`),
`ensure_index` (IVF_HNSW_SQ, `:119`), `get_embedded_uuids`
(projected-scan anti-join, `:245`), `count_rows` (`:235`),
`migrate_from_parquet_shards` (`:167`). DuckDB reads it back via the `lance`
core extension `ATTACH (TYPE LANCE)` (`sql_views.py` register_vss body,
`INSTALL lance; LOAD lance`). The empty-namespace gate — probe
`_has_table(db, TABLE_NAME)` before binding the view — is pinned by
`tests/test_lance_store.py` and documented in CLAUDE.md. **This is the read/
write side of the "search port" and the "embedding store port."**

### 2.5 Config & schemas

`config.py` `Settings(BaseSettings)` (`config.py:126`) — pydantic-settings,
`env_prefix="CLAUDE_SQL_"`, `.env` support (`config.py:133-137`). It is a
single ~90-field god-object spanning data discovery, S3, Bedrock/embedding,
LanceDB, pricing, LLM classification, all parquet output paths, UMAP/HDBSCAN/
Leiden hyperparameters, TF-IDF, DuckDB tuning. It carries a `model_validator`
that rewrites the three globs when `team_corpus_root` is set
(`config.py:368`), and an `active_model_id` property (`config.py:408`) that
exists purely "for call-site stability." **In v2 this must be decomposed** into
per-adapter config objects (embedding config, LLM config, store config, engine
config) so the domain never sees a Bedrock model ID.

`schemas.py` — pydantic v2 domain models for structured output:
`SessionClassification` (`:99`), `TrajectoryWindow`/`TrajectoryArrayResult`
(`:173,:266`), `ConflictPair`/`ConflictsResult` (`:292,:379`),
`UserFrictionSignal` (`:406`), `Correction`/`PRReviewSheet` (`:477,:509`).
These are **the closest thing to pure domain in the codebase** — but each is
paired with a `_bedrock_schema(...)` flattening call (`schemas.py:16`) that
inlines `$ref`, injects `additionalProperties:false`, and strips constraints
Bedrock's JSON-Schema subset rejects (`schemas.py:39-96`). The models are pure;
the flattening is a Bedrock-specific adapter concern that leaked into the
domain module.

---

## 3. Hexagonal gap analysis

### 3.1 Classification of every module by target hexagonal role

**Pure(-ish) domain — moves to `domain/` largely intact:**
- `core/schemas.py` classification models (`:99-580`) — strip the
  `_bedrock_schema` flattening out to an adapter.
- `analytics/kappa_worker.py` math (Cohen/Fleiss kappa, bootstrap CI) — **but
  this is in `evals/`, which is being dropped.**
- `analytics/community_worker.py` graph math: `_build_mutual_knn` (`:166`),
  `_build_igraph` (`:208`), `_run_leiden_cpm` (`:300`),
  `_compute_medoid_and_coherence` (`:350`), `_pick_zoom` (`:269`) — pure numpy/
  igraph given a centroid matrix. The **I/O** (`_load_session_centroids`,
  `:82`, which runs `con.execute` against `message_embeddings`) is infra.
- `analytics/ingest.py` `simhash64` (`:125`), `hamming_distance_64` (`:185`),
  `token_budget_bucket` (`:197`), `approx_tokens_batch` (`:95`, tiktoken) — pure.
- `analytics/friction_worker.py` `regex_fast_path` (`:148`) — pure classifier.
- `cluster_worker` UMAP+HDBSCAN math (given an embedding matrix) — pure; the
  `_load_embeddings` LanceDB read (`cluster_worker.py:28`) is infra.

**Infrastructure (adapters) — moves to `infrastructure/`:**
- `core/sql_views.py` (DuckDB DDL — the whole 2,344-LOC file), `core/s3_source.py`,
  `core/lance_store.py`, `core/session_text.py` (DuckDB queries),
  `core/parquet_shards.py` (parquet IO), `core/checkpointer.py` (SQLite),
  `core/retry_queue.py` (SQLite), `core/llm_shared.py` Bedrock client + invoke
  wrappers (`_build_bedrock_client` `:348`, `_invoke_classifier_sync` `:404`),
  `analytics/embed_worker.py` `_invoke_bedrock_sync` (`:195`).

**Application (use-cases) — moves to `application/`:**
- All analytics worker entrypoints: `classify_sessions`, `detect_conflicts`,
  `trajectory_messages`, `detect_user_friction`, `run_communities`,
  `run_clustering`, `run_terms`, `run_backfill`, `ingest.stamp_messages`,
  `skills_catalog.sync`. Plus the `analyze` orchestration currently living in
  `cli.py:2281-2517`.

**Interface (driving adapter):**
- `app/cli.py` (cyclopts) — plus its embedded composition root and output
  rendering hand-off.

**Cross-cutting (see §4):** logging, retry queue, checkpointer, output.

### 3.2 The Protocols v2 needs to define

These are the ports. Signatures below are derived from the current concrete
call shapes so the refactor is a lift, not a redesign.

**(a) `ReaderPort`** — the transcript/SQL query seam. Today every worker takes
`con: duckdb.DuckDBPyConnection`. v2 wraps this:
```python
class ReaderPort(Protocol):
    def query(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame: ...
    def messages_text(self, *, since_days: int | None, limit: int | None) -> pl.DataFrame: ...
```
The DuckDB adapter owns `register_all`/`register_vss`/`configure_s3` and the
shared-connection + rebind lifecycle (§2.2). The S3-vs-local choice
(`settings_need_s3`) becomes an adapter construction detail, invisible to
use-cases.

**(b) `EmbeddingPort`** — **the headline v2 addition.** Today embedding is
hard-wired to Cohere-on-Bedrock across two functions:
`embed_documents_async(texts, *, settings)` (`embed_worker.py:268`) and
`embed_query(text, *, settings)` (`embed_worker.py:338`), both calling
`_invoke_bedrock_sync` (`:195`) which builds a Cohere-specific JSON body
(`"input_type"`, `"output_dimension"`, `"embedding_types"`, `"truncate"` —
`:226-234`). The port:
```python
class EmbeddingPort(Protocol):
    dim: int
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```
Three adapters: `CohereBedrockEmbedder` (lift of the current path),
`OllamaEmbedder` (HTTP to a local Ollama `/api/embeddings`), `OnnxBgeEmbedder`
(local ONNX runtime running a BAAI `bge` model). Note the two current
asymmetries the port must absorb: documents embed at `embedding_type="int8"`
by default (`config.py:190`) while queries force `"float"`
(`embed_worker.py:341-364`) because "HNSW distance math needs float vectors";
and the `output_dimension` Matryoshka knob (256/512/1024/1536,
`config.py:189`) is Cohere-specific — Ollama/bge have fixed native dims, so
the LanceDB schema `dim` must become adapter-reported, not config-fixed. The
Lance table schema is `pa.list_(pa.float32(), dim)` (`lance_store.py:77`) —
switching providers changes `dim`, which means **the embeddings store is not
portable across providers without a re-embed** (a real migration constraint,
§5.3).

**(c) `SearchPort` / `VectorStorePort`** — the LanceDB read/write + kNN seam.
Today split across `lance_store.py` (write/index/scan) and the DuckDB
`semantic_search` macro + `message_embeddings` view (read via SQL). Port:
```python
class VectorStorePort(Protocol):
    def upsert(self, rows: EmbeddingRows) -> None: ...
    def embedded_uuids(self) -> set[str]: ...        # → get_embedded_uuids
    def search(self, vector: list[float], k: int) -> list[Hit]: ...
    def count(self) -> int: ...
```
Adapter = LanceDB (`lance_store.py`). Keeps the empty-namespace gate.

**(d) `LlmPort`** — structured-output classification seam. Today
`_invoke_classifier_sync(client, model_id, schema, user_text, *, max_tokens,
thinking_mode, system, pipeline)` (`llm_shared.py:404`) plus
`_parse_structured_payload` (`:502`) and `BedrockRefusalError` (`:492`). Port:
```python
class LlmPort(Protocol):
    def classify(self, *, schema: dict, user_text: str, system: str | None,
                 max_tokens: int, thinking: Literal["adaptive","disabled"]) -> dict: ...
```
Adapter = Bedrock Sonnet. The retry policy (tenacity `@retry`, `:397-403`),
client caching (`_CLIENT_CACHE`, `:345`), and prompt-cache accounting
(`maybe_log_bedrock_call`, `:285`) are adapter-internal. `BedrockRefusalError`
generalizes to a domain `RefusalError` (terminal, non-retryable — CLAUDE.md
"BedrockRefusalError is terminal").

**(e) `AnalyticsPort(s)`** — the clustering/community/terms plane. These are
use-cases, not a single port; but the v2 application layer exposes them as a
cohesive analytics facade so the CLI depends on the facade, not on 8 worker
modules. Their only infra needs are `ReaderPort` (centroids/text) +
`VectorStorePort` (embeddings) + a parquet **`CachePort`** for outputs.

**(f) `CachePort`** — parquet-shard artifact store. Today `parquet_shards.py`
(`write_part` `:87`, `read_all` `:131`, `iter_part_files` `:72`,
`replace_sessions` `:181`) + per-worker parquet paths on `Settings`. Every
expensive output is cached; views register only parquets that exist
(parquet-existence gate). Port wraps write/read/exists by artifact name.

**(g) `CheckpointPort` / `RetryQueuePort`** — see §4.

### 3.3 What is already close, what fights the refactor

- **Already close:** `s3_source.py`, `lance_store.py`, `parquet_shards.py`,
  `output.py` are clean adapter shapes. `schemas.py` models are clean domain.
  Community/cluster math is pure once IO is lifted out.
- **Fights the refactor:** (1) the god-`Settings` object threaded everywhere;
  (2) the raw `duckdb.DuckDBPyConnection` passed as an implicit port with a
  stateful rebind lifecycle; (3) `llm_shared.py` conflating Bedrock transport,
  retry policy, prompt-cache accounting, and the four task-framing system
  prompts in one module; (4) `analyze` orchestration (`cli.py:2281`) living in
  the interface layer — it is application logic wearing a CLI costume.

---

## 4. Cross-cutting concerns

| Concern | Current home | Shape | Hexagonal placement in v2 |
|---|---|---|---|
| **Logging** | `core/logging_setup.py` | `configure_logging(verbose, quiet)` (`:27`); loguru-only, stdlib `logging` banned via ruff TID251. `loguru_before_sleep(level)` (`:53`) adapts tenacity retry callbacks. | Stays cross-cutting; a thin logging module usable from every ring. The `loguru_before_sleep` belongs with the infra retry adapters. |
| **Retry queue** | `core/retry_queue.py` | SQLite-backed: `enqueue` (`:85`), `drain` (`:131`), `mark_done` (`:162`), `pending_count` (`:187`), `_backoff_delta` (`:79`). Units that fail LLM calls are parked and retried. | `RetryQueuePort` in application, SQLite adapter in infrastructure. Backoff math (`_backoff_delta`) is domain-ish. |
| **Checkpointer** | `core/checkpointer.py` | SQLite `(session_id, pipeline)` watermark: `load_as_map` (`:265`), `filter_unchanged` (`:286`), `mark_completed` (`:330`). Includes a one-time DuckDB→SQLite migration (`_migrate_from_duckdb_if_present` `:92`). Drives "reruns on untouched sessions are free." | `CheckpointPort` in application, SQLite adapter in infrastructure. The `filter_unchanged`/`_stale_or_equal` (`:317`) staleness logic is domain. |
| **Output rendering** | `core/output.py` | `OutputFormat` StrEnum (`:26`), `EXIT_CODES` (`:49`, parse=64/catalog=65/runtime=70), `emit_dataframe`/`emit_json`/`emit_error` (`:68,:111,:210`), `classify_duckdb_error` (`:180`), `run_or_die` (`:228`), `validate_glob` (`:156`). | This is **interface-adapter** concern (presentation + exit codes), not core. Moves to `interfaces/`. `classify_duckdb_error` couples it to DuckDB exception types — keep that mapping beside the DuckDB adapter. |
| **Session-text assembly** | `core/session_text.py` | `session_text_corpus` (`:198`), `iter_session_texts` (`:454`), the `SessionTextCorpus` dataclass (`:136`); builds LLM-ready text from DuckDB views with per-text clipping. | Straddles: the DuckDB queries are infra; the timeline formatting (`_render_*`, `_timeline_sort_key`) is domain text-shaping. Split accordingly. |
| **Home / cache migration** | `core/home.py` (`claude_sql_home` `:51`, `recognized_legacy_caches` `:75`) + `cli._maybe_migrate_legacy_caches` (`:277`) | Resolves `CLAUDE_SQL_HOME`, one-time legacy cache moves. | Infra (filesystem adapter) + a startup hook the composition root calls once. |

---

## 5. What the v2 rewrite touches

### 5.1 Target package tree

```
src/claude_sql/
├── domain/                     # pure — no boto3, no duckdb, no lancedb imports
│   ├── models.py               # ← schemas.py classification models (flattening removed)
│   ├── clustering.py           # ← cluster/community/terms MATH (umap/hdbscan/leiden/ctfidf)
│   ├── dedup.py                # ← ingest.simhash64 / hamming / token_budget
│   ├── friction_rules.py       # ← friction_worker.regex_fast_path
│   ├── errors.py               # RefusalError (← BedrockRefusalError), staleness rules
│   └── ports.py                # ReaderPort, EmbeddingPort, VectorStorePort, LlmPort,
│                               #   CachePort, CheckpointPort, RetryQueuePort  (Protocols)
├── application/                # use-cases — depend ONLY on domain + ports
│   ├── embed.py                # ← embed_worker.run_backfill (orchestration only)
│   ├── search.py               # semantic search use-case
│   ├── classify.py trajectory.py conflicts.py friction.py   # ← analytics workers
│   ├── cluster.py community.py terms.py ingest.py skills.py
│   ├── analyze.py              # ← the analyze chain lifted OUT of cli.py:2281
│   └── config.py               # decomposed per-use-case config (from god-Settings)
├── infrastructure/             # adapters — the ONLY place with boto3/duckdb/lancedb/onnx
│   ├── duckdb/                 # ← sql_views.py, session_text.py, s3_source.py, pragmas
│   ├── lance/                  # ← lance_store.py  (VectorStorePort impl)
│   ├── embedding/
│   │   ├── cohere_bedrock.py   # ← embed_worker._invoke_bedrock_sync
│   │   ├── ollama.py           # NEW
│   │   └── onnx_bge.py         # NEW (BAAI bge, local ONNX)
│   ├── bedrock_llm.py          # ← llm_shared client/invoke/parse/retry/cache-accounting
│   ├── parquet_cache.py        # ← parquet_shards.py  (CachePort impl)
│   ├── sqlite_checkpoint.py    # ← checkpointer.py
│   ├── sqlite_retry_queue.py   # ← retry_queue.py
│   └── settings.py             # pydantic-settings loader → builds adapter configs
├── interfaces/
│   └── cli/                    # ← app/cli.py (cyclopts) + output.py + install_source.py
│       └── (composition root: wires adapters → use-cases per subcommand)
└── (DROPPED: evals/ entirely; provenance/ pending owner confirm)
```

### 5.2 Migration seams (where to cut, in dependency order)

1. **Define `domain/ports.py`** first (Protocols above). Zero behavior change;
   pure additive.
2. **Extract adapters behind existing concrete calls.** `lance_store.py`,
   `s3_source.py`, `parquet_shards.py`, `checkpointer.py`, `retry_queue.py` are
   already adapter-shaped — wrap each in a class implementing its port. Lowest
   risk.
3. **Split `llm_shared.py`** into `infrastructure/bedrock_llm.py` (transport +
   retry + cache accounting) and keep the four task-framing system prompts as
   application-owned constants (they are prompt content, not transport).
4. **Introduce `EmbeddingPort` + Cohere adapter** as a pure lift of
   `embed_worker._invoke_bedrock_sync`. Only then add Ollama + ONNX bge
   adapters. This is the one seam with new behavior.
5. **Wrap DuckDB as `ReaderPort`.** Highest-risk cut (see §5.5) — the shared
   connection + `_rebind_vss`/`_refresh_analytics_views` lifecycle
   (`cli.py:382-444`) must move into the adapter, and the `analyze` chain
   (`cli.py:2281`) must move to `application/analyze.py` and drive the port,
   not the raw connection.
6. **Move `output.py` + `cli.py` to `interfaces/`.** Composition root becomes
   explicit: each subcommand builds the adapters it needs and injects them into
   the use-case. Preserve the lazy-import discipline (§1.2) by keeping adapter
   construction inside command bodies.
7. **Delete `evals/`** and its tests (`test_judge_worker*.py`,
   `test_kappa_worker.py`, `test_freeze.py`, `test_blind_handover.py`,
   `test_judges*.py`, `test_ungrounded_worker.py`) and CLI commands
   `judges/freeze/replay/blind-handover/judge/ungrounded-claim/kappa`
   (`cli.py:2519-2791`). Update the import-linter contract (drop the `evals`
   term from the layers + independence lists in `pyproject.toml`).

### 5.3 The embeddings-store portability constraint (surfaced early)

The LanceDB table stores a fixed-dim `float32` vector column
(`lance_store.py:77`) tagged with `model` and `dim` per row
(`embed_worker.py:480-491`). Cohere-v4 (Matryoshka, `output_dimension`
256/512/1024/1536) and bge/Ollama (fixed native dims) produce **incompatible
vectors**. Switching the active `EmbeddingPort` requires a full re-embed into a
provider-scoped table (or a `model`-column filter on read). v2 should key the
Lance table/namespace by `(provider, model, dim)` so multiple providers can
coexist and `search`/`cluster` read the active one. This is not optional
polish — mixing dims in one index silently corrupts kNN.

### 5.4 Batched questions for the owner

1. **Provenance:** confirmed cleanly droppable (§1.4) — drop it, or keep behind
   an optional extra? My assumption: drop, same as evals.
2. **Embedding store coexistence:** OK to key the Lance namespace by
   `(provider, model, dim)` and require re-embed on provider switch (§5.3)? My
   assumption: yes — it is the only correct option.
3. **Ollama/bge query-vs-document asymmetry:** Cohere uses distinct
   `input_type` for query vs document; bge uses an instruction prefix for
   queries, Ollama neither. The `EmbeddingPort` will expose separate
   `embed_query`/`embed_documents` so each adapter handles its own convention.
   Confirming this is the desired contract.

### 5.5 The single biggest hexagonal-refactor risk

**Wrapping the DuckDB connection as a port, because the connection is stateful,
long-lived, and shared across analyze stages with an in-place rebind
lifecycle** — not a stateless query executor. `_open_connection_full`
registers 18 views + 14 macros + VSS against parquet path lists that are
**captured and frozen at registration time** (`cli.py:382-399`), and stages
that write new parquet shards or populate LanceDB mid-run require explicit
`_refresh_analytics_views` (`cli.py:382`) and `_rebind_vss` (`cli.py:402`)
calls to see their own writes (the RFC §9.6 stale-connection bug). A naive
`ReaderPort.query(sql)` abstraction that hides the connection will re-break this
unless the port explicitly models the register→write→rebind cycle. This is the
one cut where a clean-looking abstraction can silently reintroduce a known,
already-fixed correctness bug, so it must be sequenced last (§5.2 step 5) and
carry regression tests for the analyze-chain rebind path
(`tests/test_analyze_chain.py` already exists — extend it).
