# claude-sql v2: Design

**Status:** proposed (2026-07-19). This is the target architecture for the v2
rewrite. The shipping package is still v1.2.1 with the 5-package layered
structure; this document describes where it is going, grounded in the
verified understanding pass under `docs/v2/understanding/` (seven perspective
docs, every claim cited to `path:line` against the v1.2.1 tree).

Companion docs:
- `docs/v2/MIGRATION.md`: the ordered cut plan and the definition of done.
- `docs/v2/understanding/*.md`: the grounded reading of the current code.

---

## 1. What v2 is

claude-sql is the tool that turns your own Claude Code transcripts into a live,
queryable corpus with SQL, semantic search, and structural intelligence, with
no export step. v2 keeps that mission and sharpens it to one sentence:
**best-in-class at finding and reading Claude Code transcripts.**

Four decisions define v2 (all owner-decided, treated as fixed input):

1. **Hexagonal architecture.** Reshape the code into `domain / application
   (ports) / infrastructure (adapters) / interfaces`, so the package is both a
   clean importable library and a standalone CLI. Today the code is layered by
   dependency depth (`app > {analytics|evals|provenance} > core`), which is a
   real DAG but mixes domain, infrastructure, and application inside `core/`.
2. **Drop the eval and provenance planes.** Remove `evals/` (judge panel,
   Fleiss/Cohen kappa, freeze/replay studies, blind-handover, ungrounded-claim)
   and `provenance/` (transcript↔PR binding, PR review-sheet). Both are cleanly
   severable: the only cross-boundary importer of either is `app/cli.py`
   (`docs/v2/understanding/05-drop-analysis.md`). The eval gym belongs to a
   dedicated eval project; provenance is audit-adjacent and orthogonal to
   finding and reading transcripts.
3. **Pluggable embeddings.** Put embeddings behind an `EmbeddingProvider` port
   with three adapters: Cohere Embed v4 on Bedrock (the current behavior),
   Ollama (local HTTP), and a local ONNX BGE model (BAAI `bge`). Ollama and
   ONNX ship behind optional extras so the base install stays lean.
4. **Keep retrieval and clustering; stay on Python 3.13.** SQL views, semantic
   search, UMAP+HDBSCAN clustering, c-TF-IDF terms, and Leiden+CPM communities
   all stay. They are the differentiator. `hdbscan` has no cp314 wheel yet, so
   the package stays on Python 3.13 (`docs/adr/0015-stack-modernization.md`).

### 1.1 A terminology correction carried into v2

The roadmap language used "pagerank" for the graph-intelligence plane. The code
implements no PageRank. Graph ranking is **Leiden community detection with the
CPM objective, ranked by community medoid and intra-community coherence**
(`analytics/community_worker.py`, verified: `grep pagerank|centrality` over
`src/` returns zero hits). v2 documentation uses the correct names.

---

## 2. Current shape (the starting point)

One distribution, one namespace root `src/claude_sql/`, five sub-packages
(`pyproject.toml` `[tool.uv.build-backend] namespace = true`):

| Layer | LOC | Owns | v2 fate |
|---|---|---|---|
| `core/` | ~6,900 | DuckDB engine, S3/httpfs reader, LanceDB store, config, schemas, LLM plumbing, cross-cutting (logging, retry queue, checkpointer, output, parquet shards, session-text) | **Split** across domain / infrastructure / application |
| `analytics/` | ~4,900 | Embeddings backfill, UMAP+HDBSCAN clustering, c-TF-IDF terms, Leiden+CPM communities, trajectory, conflicts, friction, ingest/dedup, skills catalog | **Keep** (the differentiator) |
| `app/` | ~3,250 | Cyclopts CLI (`cli.py`), version/install banner | **Becomes `interfaces/cli`** |
| `evals/` | ~1,570 | Judge panel, kappa, freeze/replay, blind-handover, ungrounded | **Drop** |
| `provenance/` | ~1,380 | Git↔transcript binding, PR review-sheet | **Drop** |

The import DAG is enforced by import-linter and verified real (`lint-imports`:
both contracts kept, 0 broken). The measured edges are `analytics → core` only,
`evals → core` only, `provenance → core` only, and `app → everything`. That
sibling independence is exactly why the two drops are clean.

The single deepest hexagonal violation: every analytics/eval/provenance
use-case takes a live `duckdb.DuckDBPyConnection` as its first argument and a
god-`Settings` object as its second. The DuckDB connection is the de-facto
reader port, but it is a concrete third-party handle with no Protocol seam.

---

## 3. Target architecture

### 3.1 Package tree

```
src/claude_sql/
├── domain/                     # pure: no boto3, no duckdb, no lancedb, no onnx
│   ├── models.py               # ← core/schemas.py classification models (Bedrock flattening removed)
│   ├── clustering.py           # ← cluster/community/terms MATH (umap/hdbscan/leiden/c-tf-idf)
│   ├── dedup.py                # ← ingest.simhash64 / hamming / token_budget
│   ├── friction_rules.py       # ← friction_worker.regex_fast_path
│   ├── errors.py               # RefusalError (← BedrockRefusalError), staleness rules
│   └── ports.py                # the Protocols (§3.2)
├── application/                # use-cases: depend only on domain + ports
│   ├── embed.py search.py classify.py trajectory.py conflicts.py friction.py
│   ├── cluster.py community.py terms.py ingest.py skills.py
│   ├── analyze.py              # ← the analyze chain lifted OUT of cli.py
│   └── config.py               # per-use-case config, decomposed from god-Settings
├── infrastructure/             # adapters: the ONLY place with boto3/duckdb/lancedb/onnx
│   ├── duckdb/                 # ← sql_views.py, session_text.py, s3_source.py, pragmas
│   ├── lance/                  # ← lance_store.py  (VectorStorePort)
│   ├── embedding/
│   │   ├── cohere_bedrock.py   # ← embed_worker._invoke_bedrock_sync
│   │   ├── ollama.py           # NEW  ([ollama] extra)
│   │   └── onnx_bge.py         # NEW  ([onnx] extra)
│   ├── bedrock_llm.py          # ← llm_shared client/invoke/parse/retry/cache-accounting
│   ├── parquet_cache.py        # ← parquet_shards.py  (CachePort)
│   ├── sqlite_checkpoint.py    # ← checkpointer.py
│   ├── sqlite_retry_queue.py   # ← retry_queue.py
│   └── settings.py             # pydantic-settings loader → builds per-adapter configs
├── interfaces/
│   └── cli/                    # ← app/cli.py (cyclopts) + output.py + install_source.py
│                               #   composition root: each subcommand wires adapters → use-cases
└── (removed: evals/, provenance/)
```

### 3.2 The ports

Signatures are derived from the current concrete call shapes, so the refactor
is a lift rather than a redesign. Full derivation in
`docs/v2/understanding/01-architecture.md` §3.2.

- **`ReaderPort`**: the transcript/SQL query seam. Wraps the DuckDB
  connection, and owns `register_all` / `register_vss` / `configure_s3` plus
  the shared-connection rebind lifecycle. The S3-vs-local choice becomes an
  adapter construction detail, invisible to use-cases.
- **`EmbeddingProvider`**: the headline v2 port (§4).
- **`VectorStorePort`**: the LanceDB read/write + kNN seam (`upsert`,
  `embedded_uuids`, `search`, `count`). Keeps the empty-namespace gate.
- **`LlmPort`**: structured-output classification (Bedrock Sonnet adapter).
  Retry policy, client caching, and prompt-cache accounting are adapter-internal.
  `BedrockRefusalError` generalizes to a terminal, non-retryable domain
  `RefusalError`.
- **`CachePort`**: the parquet-shard artifact store (write/read/exists by
  artifact name). The parquet-existence gate stays.
- **`CheckpointPort` / `RetryQueuePort`**: the SQLite watermark and retry
  queue; backoff and staleness math are domain.

### 3.3 What lands where, by hexagonal role

- **Domain (pure):** `schemas.py` models (flattening lifted out), community/
  cluster/terms math, `simhash64`/hamming/token-budget, the friction regex fast
  path, refusal and staleness rules.
- **Infrastructure (adapters):** the whole `sql_views.py` DDL layer,
  `s3_source.py`, `lance_store.py`, `session_text.py` queries,
  `parquet_shards.py`, `checkpointer.py`, `retry_queue.py`, the Bedrock client
  and invoke wrappers, and the three embedding adapters.
- **Application (use-cases):** every analytics worker entrypoint plus the
  `analyze` orchestration lifted out of `cli.py`.
- **Interfaces:** the cyclopts CLI, output rendering, and the install banner.

---

## 4. The embedding seam (headline change)

### 4.1 Where it lives today

**Update — Phase B has landed (077f362); this seam is now realized.** The port
below and the three adapters exist. Current locations: the raw `invoke_model`
call `_invoke_bedrock_sync` now lives in the Cohere adapter
(`core/embedding/cohere_bedrock.py:80`); `embed_query`
(`analytics/embed_worker.py:194`) is a thin shim over `build_embedder`;
`embed_documents_async` (`embed_worker.py:165`) and `run_backfill`
(`embed_worker.py:217`) route through the port rather than reading `Settings`
directly. The model is pinned to `global.cohere.embed-v4:0` (`config.py:217`) at
`output_dimension=1024` (`config.py:219`) as the `cohere-bedrock` provider
default. Production callers are small and bounded: `embed_query` feeds
`search` (`cli.py`), `run_backfill` feeds the `embed` and `analyze` commands.
It was a clean seam and lifted cleanly.

### 4.2 The port

```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...        # identity string, stamped into the store
    @property
    def dimension(self) -> int: ...       # fixed width: replaces free settings.output_dimension
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

Two methods rather than Cohere's single `input_type` parameter, because each
backend handles the document-vs-query asymmetry differently: Cohere uses
distinct `input_type` values, BGE uses a query-only instruction prefix, Ollama
uses neither.

### 4.3 The three adapters

| Adapter | Transport | Extra | Notes |
|---|---|---|---|
| `CohereBedrockEmbedder` | Bedrock `invoke_model` | (base) | Lift of the current behavior. Documents embed at `int8`, queries force `float` (`cohere_bedrock.py:157-247`) because HNSW distance math needs float vectors. Matryoshka `output_dimension` (256/512/1024/1536) is Cohere-specific. |
| `OllamaEmbedder` | HTTP POST `/api/embed` | `[ollama]` (`httpx`) | Batch, L2-normalized. Model e.g. `nomic-embed-text` (768) or `bge-m3`. |
| `OnnxBgeEmbedder` | `onnxruntime` + `tokenizers`, no torch | `[onnx]` | BAAI `bge` v1.5 English models use **CLS pooling** (`last_hidden_state[:, 0]`), then L2-normalize. Mean-pooling degrades BGE performance per BAAI's model card. Query-instruction prefix applied to queries only. `numpy` is already core. |

### 4.4 The dimension contract (the migration hazard)

`output_dimension` is declared once and threaded unvalidated into three
consumers:
1. the LanceDB fixed-size schema `pa.list_(pa.float32(), dim)`
   (`lance_store.py:77`), **frozen on first write**: `open_or_create_table`
   ignores the `dim` argument thereafter;
2. the DuckDB `CAST(embedding AS FLOAT[{dim}])` view (`sql_views.py:1964`);
3. the query-time cast (`cli.py`).

Cohere writes 1024; bge-small is 384; bge-base and nomic are 768. Worse than a
dimension mismatch: even at matching dimensions (Cohere-1024 vs bge-large-1024),
the vector spaces are incompatible and produce numerically valid but
semantically garbage cosine scores. v1.2.1 had **zero validation anywhere**
(grep-confirmed). The `model` column exists in the Lance schema
(`lance_store.py:75`, stamped at `embed_worker.py:347`) but was **write-only**
in v1.2.1; Phase B (077f362) added the read-and-enforce guard described below.

**v2 resolution:** key the Lance table/namespace by `(provider, model, dim)` so
providers can coexist, and flip the `model` column into a **read-and-enforce
fail-loud guard** at both `run_backfill` open and `register_vss`. A provider
switch requires a full re-embed: `rm -rf` the store directory and re-run
`claude-sql embed` (empty store triggers a full backfill; the cluster mtime
sidecar auto-refits downstream). This is not optional polish; mixing dims or
providers in one index silently corrupts kNN.

---

## 5. Retrieval and clustering (kept)

Three retrieval modalities over one in-memory DuckDB connection:
- **Lexical / structural SQL** over the derived views (`messages_text`,
  `turn_window`).
- **Semantic ANN**: `search()` embeds the query (Cohere `search_query` float
  vector today) and runs `array_cosine_distance` over the LanceDB-backed
  `message_embeddings` view.
- **Structure**: clusters, terms, and communities as progressive-disclosure
  signals.

The structure plane is pure compute: UMAP+HDBSCAN clustering
(`cluster_worker.py`), in-house c-TF-IDF terms (`terms_worker.py`), and Leiden
with the CPM objective over a mutual-kNN cosine graph, ranked by medoid and
coherence (`community_worker.py`). The LLM-backed workers (classify, trajectory,
conflicts, friction: all Bedrock Sonnet via `llm_shared.classify_one`) are a
separable concern; embedding uses Bedrock but is retrieval infrastructure, not
LLM analytics.

### 5.1 Why Python stays 3.13

`hdbscan` (imported at `cluster_worker.py:131`) is the sole cp314 blocker; the
rest of the clustering closure (`umap-learn`, `leidenalg`, `igraph`,
`scikit-learn`, `scipy`, `numba`, `llvmlite`) has cp314 wheels but travels with
hdbscan. `numba 0.66.0` also caps `numpy < 2.5` (`pyproject.toml:38`).

Relaxation paths, in order of preference: (a) bump to 3.14 once hdbscan ships a
cp314 wheel (`docs/adr/0015` tracks this; the lock is already at 0.8.44, so it
is worth re-checking); (b) move cluster/community/terms behind an optional
extra so the base retrieval + LLM package can target 3.14 while the structure
extra stays on 3.13; (c) swap standalone `hdbscan` for
`sklearn.cluster.HDBSCAN`, which needs determinism re-validation. Semantic
search and LLM analytics do not need 3.13 today; only the structure sub-plane
does.

---

## 6. Interfaces

The CLI is fat today: command bodies own connection creation, PRAGMA tuning,
view rebinding, legacy-cache migration, and the full 10-stage `analyze`
pipeline (`cli.py:2344-2588`). v2 makes `interfaces/cli` a thin cyclopts layer
over `application/use_cases`, plus an importable facade
(`ClaudeSql(embedder=..., reader=...)`) that downstream consumers depend on
instead of reimplementing the reader.

The command surface is 31 commands: **keep 21, change 3, drop 7**. The dropped
7 are exactly the eval plane. The changed 3 (`search`, `embed`, `analyze`) gain
`--embedding-provider`. Provenance's 3 commands (`bind`, `resolve`,
`review-sheet`) drop with that plane. The read-and-find core (`query`,
`explain`, `peek`, `search`) is unaffected. Full inventory in
`docs/v2/understanding/04-cli-interfaces.md`.

Two interface contracts are load-bearing and must survive:
- **Exit codes** (`output.py`): parse=64, catalog=65, runtime=70. Agents match
  on these, so they are a public wire contract.
- **Lazy imports** (`cli.py:45-53`, pinned by `test_cli_import_is_lean`).
  Deferred worker imports are why `schema`/`query`/`explain` stay sub-second.
  The v2 composition root must build adapters inside command bodies, not at
  module top.

---

## 7. The biggest risk

Wrapping the DuckDB connection as a `ReaderPort`. The connection is stateful,
long-lived, and shared across `analyze` stages with an in-place rebind
lifecycle, not a stateless query executor. `register_all` binds 18 views + 14
macros + VSS against parquet path lists **captured and frozen at registration
time**; stages that write new parquet shards or populate LanceDB mid-run need
explicit `_refresh_analytics_views` (`cli.py:418`) and `_rebind_vss`
(`cli.py:438`) calls to see their own writes (the RFC §9.6 stale-connection
bug). A naive `ReaderPort.query(sql)` that hides the connection reintroduces
that already-fixed correctness bug. Sequence this cut last, move the `analyze`
orchestration into `application/analyze.py` first, and extend
`tests/test_analyze_chain.py` to cover the rebind path.

See `docs/v2/MIGRATION.md` for the ordered cut plan.

---

## Wave D: opt-in LLM analytics (GPT-5.6-Luna)

Wave D adds GPT-5.6-Luna as an ALTERNATIVE structured-output provider for the
existing trajectory (sentiment + trajectory) analytics, alongside the current
Sonnet-on-Bedrock path. It is purely additive: the default behavior is
unchanged and Luna is opt-in.

### The provider seam

A narrow structured-output port, `LlmAnalyticsProvider`, lives in `core/`
(the lowest layer, next to the Wave B `EmbeddingProvider` seam under
`core/embedding/`) so `analytics/trajectory_worker.py` can depend on it without
breaking the import-linter DAG (`core (L0) < analytics (L1) < app (L2)`):

```
src/claude_sql/core/llm_analytics/
├── __init__.py          # public exports: the port, factory, terminal error
├── base.py              # LlmAnalyticsProvider Protocol + build_llm_analytics_provider(settings)
├── sonnet_bedrock.py    # default adapter, wraps the existing llm_shared path (no behavior change)
└── strands_luna.py      # opt-in adapter: GPT-5.6-Luna via Strands OpenAIResponsesModel
```

The port is one method:

```python
async def classify_structured(self, *, system: str, prompt: str, schema: type[BaseModel]) -> BaseModel
```

`build_llm_analytics_provider(settings)` returns the `sonnet-bedrock` (default)
or `strands-luna` adapter, importing the concrete adapter lazily inside the
selected branch so a bare import never drags in `strands` / `openai` / `botocore`.
The trajectory worker builds the provider ONCE per run and routes its
per-chunk structured-output call through it, so the SAME worker runs under either
provider and keeps emitting `TrajectoryWindow` / `TrajectoryArrayResult`
unchanged (no duplicated worker). It reuses the existing `"trajectory"`
checkpoint pipeline name, so no `checkpointer.PIPELINE_NAMES` change was needed.

### The Mantle/Responses-only fact

GPT-5.6-Luna (`openai.gpt-5.6-luna`) is served ONLY on the Bedrock **Mantle**
endpoint via the OpenAI-compatible **Responses** API. It is NOT available on
`bedrock-runtime`, NOT via `Converse`, NOT via `InvokeModel`. So it CANNOT reuse
claude-sql's existing `invoke_model` + `output_config.format` Sonnet transport.
The Luna adapter drives it through Strands'
`from strands.models.openai_responses import OpenAIResponsesModel` with
`bedrock_mantle_config={"region": ...}`, which auto-derives the required
`/openai/v1` base URL from the `openai.gpt-5.*` model-id prefix and mints a
fresh `provide_token` bearer per request. It builds a FRESH `Agent` per call
(stateless), then `await agent.invoke_async(prompt, structured_output_model=Schema)`
and returns `result.structured_output`. Mirrors the consumer's structured-output
adapter.

**Fail-open posture (mirrors a fail-open reviewer adapter).** Because the plane
is opt-in, a Luna outage must never crash the core SQL / embedding flows. Any
Strands / transport / structured-output failure (or an empty output) is logged
at WARNING and re-raised as the terminal domain error
`LlmAnalyticsUnavailable`, which the trajectory worker treats as "provider
unavailable" (enqueue the session for a later run) exactly like any other
recoverable per-chunk failure. The `strands` import is lazy inside the adapter;
a missing extra raises `ImportError` with an "install claude-sql[llm-analytics]"
hint (never a bare pass, per CodeQL).

### Opt-in flags and settings

- `config.py`: `llm_analytics_provider: str = "sonnet-bedrock"` (values
  `sonnet-bedrock | strands-luna`), `luna_model_id: str = "openai.gpt-5.6-luna"`
  (BARE id, Luna has no inference profile), `luna_region: str = "us-east-1"`
  (the same region the rest of claude-sql uses). All under the `CLAUDE_SQL_`
  env prefix (e.g. `CLAUDE_SQL_LLM_ANALYTICS_PROVIDER=strands-luna`). The
  defaults reproduce today's behavior exactly.
- CLI: `--llm-analytics-provider {sonnet-bedrock,strands-luna}` on `trajectory`
  and `analyze`, threaded into settings via `_apply_llm_analytics_provider`
  (parallel to the Wave B `_apply_embedding_provider`). Spend is still gated by
  the existing `--dry-run` default. The provider is built inside the command
  body so `strands` never imports on the CLI fast path (pinned by
  `test_cli_import_is_lean` and the forbidden-eager-import guard in
  `tests/test_pr3_perf.py`).
- `pyproject.toml`: optional extra `llm-analytics = ["strands-agents[openai]>=1.46,<2"]`.
  The `[openai]` extra transitively brings `openai` and
  `aws-bedrock-token-generator`; resolved cleanly against the numba/numpy pin
  (numba stays 0.66.0, numpy stays in `>=2.4.4,<2.5`). Install with
  `uv add 'claude-sql[llm-analytics]'`.

### Live smoke test is separate

All tests are hermetic: they mock the Strands / Bedrock surface and NEVER make
a live Strands / Bedrock / Luna call (the Luna-construction test uses
`pytest.importorskip("strands")` so the base env skips cleanly). A real Luna
round-trip is a SEPARATE, opt-in smoke check that is NEVER part of
`mise run check`. Run it by hand against a Mantle-capable region, e.g.:

```bash
uv sync --extra llm-analytics
CLAUDE_SQL_LLM_ANALYTICS_PROVIDER=strands-luna \
  claude-sql trajectory --since-days 1 --limit 1 --no-dry-run
```

(the `analyze` chain accepts the same `--llm-analytics-provider strands-luna`
flag). Keep this out of the hermetic `mise run test` gate.
