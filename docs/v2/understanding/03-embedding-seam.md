# claude-sql v2 — The Embedding Seam

**Scope.** The single most important refactor for v2: making embeddings
pluggable. Today the embedding path is pinned end-to-end to **Cohere Embed v4
on Bedrock** (`global.cohere.embed-v4:0`). v2 must move it behind an
`EmbeddingProvider` port with at least three adapters — the existing Bedrock
Cohere path, a local **Ollama** HTTP server, and a local **ONNX BAAI `bge`**
model (onnxruntime + tokenizers, no torch).

Every structural claim cites `path:line`. Signatures are quoted verbatim from
the on-disk source (`v1.2.1`), cross-checked with CodeGraph
`callers`/`impact`. Ollama and ONNX API shapes are grounded in current vendor
docs (sources at the end of §4).

**The seam in one sentence.** All embedding traffic funnels through two
functions in `src/claude_sql/analytics/embed_worker.py` —
`_invoke_bedrock_sync` (the raw `invoke_model` call, `embed_worker.py:195`) and
`embed_query` (the query-time single-text path, `embed_worker.py:338`) — plus
two orchestrators, `embed_documents_async` (batch, `embed_worker.py:268`) and
`run_backfill` (the corpus loop that writes Lance, `embed_worker.py:369`).
Unpin those four and the seam is open.

---

## 1. The exact current seam

### 1.1 `_invoke_bedrock_sync` — the raw invoke (the true chokepoint)

`src/claude_sql/analytics/embed_worker.py:195-242`. This is the ONE function
that talks to Bedrock for embeddings. Everything else calls it.

```python
def _invoke_bedrock_sync(
    client: Any,
    model_id: str,
    texts: list[str],
    *,
    input_type: str,
    output_dimension: int,
    embedding_type: str,
) -> list[list[int]] | list[list[float]]:
```

It is wrapped by a tenacity `@retry` decorator (`embed_worker.py:185-194`):
`stop_after_attempt(10)`, `wait_exponential(multiplier=2, min=2, max=60)`,
`retry_if_exception(_is_retryable)`, `before_sleep=loguru_before_sleep("WARNING")`,
`reraise=True`. Retryable errors (`embed_worker.py:54-59`, `_is_retryable` at
`:62-76`): `ThrottlingException`, `ServiceUnavailableException`,
`ModelTimeoutException`, `ModelErrorException`, plus SSL / connection /
endpoint / read-timeout network errors.

**The Bedrock request shape** (`embed_worker.py:226-240`) — Cohere Embed v4
native-invoke body:

```python
body = json.dumps(
    {
        "texts": [t[:MAX_CHARS_PER_TEXT] for t in texts],
        "input_type": input_type,
        "output_dimension": output_dimension,
        "embedding_types": [embedding_type],
        "truncate": "RIGHT",
    }
)
resp = client.invoke_model(
    modelId=model_id,
    body=body,
    contentType="application/json",
    accept="application/json",
)
payload = json.loads(resp["body"].read())
return payload["embeddings"][embedding_type]
```

Cohere-specific request/response facts baked into this function:
- `texts` — batch of up to 96 strings, each clipped to `MAX_CHARS_PER_TEXT`
  (`embed_worker.py:51`, `50_000` chars, to stay under Bedrock's 20 MB body
  ceiling with a full batch).
- `input_type` — `"search_document"` for corpus, `"search_query"` for queries.
  This is a **Cohere-specific asymmetric-embedding** concept; not every provider
  has it (Ollama/bge do not, though bge uses a query *prefix* instead — §4.3).
- `output_dimension` — Cohere v4 Matryoshka target: one of `256, 512, 1024,
  1536` (`embed_worker.py:217`). Only Cohere supports on-request dimension
  selection; Ollama/bge emit a **fixed** dimension per model.
- `embedding_types` — list-wrapped single type; one of `"int8"`, `"float"`,
  `"uint8"`, `"binary"`, `"ubinary"` (`embed_worker.py:219`). This is why the
  response is parsed as `payload["embeddings"][embedding_type]` — Cohere keys the
  returned vectors by requested type. **No other provider returns this shape.**
- `truncate: "RIGHT"` — Cohere server-side truncation policy.

The client is a boto3 `bedrock-runtime` built by
`_build_bedrock_client(settings)` (imported at `embed_worker.py:42` from
`core/llm_shared.py:348`). That helper caches one client per `(region,
pool_size)` with `max_pool_connections = max(32, max(embed_concurrency,
llm_concurrency) * 2)`, `connect_timeout=10`, `read_timeout=600`, and
`retries={"max_attempts": 0, "mode": "adaptive"}` (botocore's own retry loop is
disabled so tenacity owns policy) — `llm_shared.py:377-394`.

### 1.2 `embed_documents_async` — batch orchestrator

`src/claude_sql/analytics/embed_worker.py:268-335`.

```python
async def embed_documents_async(
    texts: list[str],
    *,
    settings: Settings,
) -> list[list[float]]:
```

- Short-circuits `[]` on empty input (`:292-293`) — no client built.
- Builds one client (`_build_bedrock_client(settings)`, `:295`), slices `texts`
  into `settings.batch_size`-sized batches (`:297`), gates concurrency with an
  `asyncio.Semaphore(settings.embed_concurrency)` (`:298`).
- Each batch runs through `_embed_one_batch` (`:245-265`) which calls
  `asyncio.to_thread(_invoke_bedrock_sync, ...)` under the semaphore, with
  `input_type="search_document"`, `output_dimension=settings.output_dimension`,
  `embedding_type=settings.embedding_type` (`:310-321`).
- **Casts every value to `float`** on the way out (`:325-327`:
  `[[float(x) for x in v] for batch_vecs in results for v in batch_vecs]`). The
  int8 (or other) Bedrock response is widened to Python float so the downstream
  Lance `FLOAT[dim]` column is directly consumable by the HNSW index.

### 1.3 `embed_query` — query-time single-text path

`src/claude_sql/analytics/embed_worker.py:338-366`.

```python
def embed_query(text: str, *, settings: Settings) -> list[float]:
```

```python
client = _build_bedrock_client(settings)
vectors = _invoke_bedrock_sync(
    client,
    settings.active_model_id,
    [text],
    input_type="search_query",
    output_dimension=settings.output_dimension,
    embedding_type="float",
)
return [float(x) for x in vectors[0]]
```

Two invariants that MUST survive into any adapter:
1. `input_type="search_query"` — asymmetric embedding; the query side of
   Cohere's document/query split.
2. `embedding_type="float"` is **forced regardless of
   `settings.embedding_type`** (`:341-343`). The document backfill may store
   int8, but HNSW distance math at query time needs float. A provider adapter
   must offer the same "give me a float query vector" guarantee, even if its
   documents were stored quantized. Returns a single vector of length
   `settings.output_dimension`.

### 1.4 `run_backfill` — the corpus loop that owns the dimension contract

`src/claude_sql/analytics/embed_worker.py:369-515`.

```python
async def run_backfill(
    *,
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> int | dict[str, Any]:
```

Flow:
1. `discover_unembedded(con, lance_uri=settings.lance_uri, ...)`
   (`:399-404`) — DuckDB `messages_text` LEFT-anti-joined against the already
   embedded uuids read from Lance (`get_embedded_uuids`, `lance_store.py:245`).
2. `dry_run` returns a plan dict `{pipeline, candidates, batches, batch_size,
   concurrency, model, since_days, limit, dry_run}` and never touches Bedrock
   (`:407-441`). `model` is `settings.active_model_id`.
3. Opens the Lance table at the declared dimension:
   `tbl = lance_store.open_or_create_table(db, dim=settings.output_dimension)`
   (`:453`). **This line is where the dimension contract is minted** — see §2.
4. For each `chunk_size = max(settings.batch_size * 4, 256)` slice (`:449`),
   calls `embed_documents_async(texts, settings=settings)` (`:465`), then writes
   a polars frame to Lance (`:477-493`) with the explicit schema:

```python
schema={
    "uuid": pl.Utf8,
    "model": pl.Utf8,
    "dim": pl.Int32,
    "embedding": pl.Array(pl.Float32, settings.output_dimension),
    "embedded_at": pl.Datetime("us", "UTC"),
},
```

   The `model` column is stamped `settings.active_model_id` and `dim` is stamped
   `settings.output_dimension` for **every row** (`:480-481`). These columns are
   written but, as of `v1.2.1`, **never read back for validation** (see §5, the
   fail-loud gap).
5. Periodic `lance_store.optimize_if_needed(tbl)` every 8 chunks (`:499-501`),
   final compaction + `lance_store.ensure_index(tbl, metric=settings.hnsw_metric)`
   (`:505-506`).

### 1.5 Where model_id / dims / type come from config

All in `src/claude_sql/core/config.py`, the `Settings` pydantic-settings class
(`config.py:126`, `env_prefix="CLAUDE_SQL_"` at `:134`):

| Setting | `path:line` | Default | Notes |
|---|---|---|---|
| `region` | `config.py:183` | `"us-east-1"` | Bedrock region |
| `model_id` | `config.py:187` | `"global.cohere.embed-v4:0"` | Cohere v4 global CRIS. Comment: "No reason to expose the knob." |
| `output_dimension` | `config.py:189` | `1024` | `Literal[256, 512, 1024, 1536]` — Cohere Matryoshka domain |
| `embedding_type` | `config.py:190` | `"int8"` | `Literal["int8","float","uint8","binary","ubinary"]` |
| `embed_concurrency` | `config.py:194` | `8` | parallel Bedrock calls |
| `batch_size` | `config.py:203` | `96` | Cohere per-call max |
| `lance_uri` | `config.py:213` | `~/.claude/embeddings_lance` | Lance dataset dir |
| `hnsw_metric` | `config.py:216` | `"cosine"` | `Literal["cosine","l2","dot"]` |
| `active_model_id` (property) | `config.py:408-411` | returns `self.model_id` | call-site stability shim |

`active_model_id` is a property (`config.py:409`) that just returns `model_id` —
a seam already anticipated for indirection.

---

## 2. The dimension contract (the migration hazard)

**The vector dimension is a hard, three-way contract** between the embedder,
the Lance table schema, and the DuckDB `message_embeddings` view / macro. It is
declared in exactly one place — `settings.output_dimension` (`config.py:189`,
default `1024`) — and threaded, unvalidated, into three consumers:

1. **Lance physical schema.** `lance_store.lance_schema(dim)`
   (`lance_store.py:65-80`) builds
   `pa.field("embedding", pa.list_(pa.float32(), dim), nullable=False)` — a
   **fixed-size** float32 list. `open_or_create_table(db, *, dim)`
   (`lance_store.py:100-104`) creates the table at that dim with `mode="create"`
   the first time and thereafter **opens the existing table ignoring `dim`**:

   ```python
   def open_or_create_table(db, *, dim):
       if _has_table(db, TABLE_NAME):
           return db.open_table(TABLE_NAME)   # dim is NOT checked here
       return db.create_table(TABLE_NAME, schema=lance_schema(dim), mode="create")
   ```

   So the **first write** to a fresh Lance dir freezes the dimension. The write
   frame in `run_backfill` uses `pl.Array(pl.Float32, settings.output_dimension)`
   (`embed_worker.py:489`); Lance rejects a variable-size `pl.List`, which is
   why the fixed `pl.Array` is load-bearing (noted in CLAUDE.md "Resilience
   patterns" and `lance_store.py:107-114`).

2. **DuckDB view/table projection.** `register_vss(con, *, ..., dim=1024, ...)`
   (`core/sql_views.py:1812-1930`) casts the Lance column to a fixed shape:
   `CAST(embedding AS FLOAT[{dim_i}]) AS embedding` (`sql_views.py:1917`), and
   the empty-fallback table is `embedding FLOAT[{dim_i}]` (`sql_views.py:1897`).
   `dim` is passed `int(settings.output_dimension)` from all three call sites:
   `register_analytics_views` (`sql_views.py:2297`), the CLI rebind at
   `cli.py:441`, and the search command computes `dim = int(settings.output_dimension)`
   at `cli.py:1752`.

3. **Query-time cast.** The `search` command casts the query vector to the same
   width: `WITH qv AS (SELECT CAST(? AS FLOAT[{dim}]) AS v)` (`cli.py:1762`),
   then `array_cosine_similarity(me.embedding, ...)` /
   `array_cosine_distance(...)` (`cli.py:1766, 1774`). The `semantic_search`
   macro (`sql_views.py:1359-1367`) does the same over `message_embeddings`.
   DuckDB's `array_cosine_similarity` requires **both operands to be `FLOAT[N]`
   of identical N** — a width mismatch is a bind-time / runtime error.

### What breaks when a new provider emits a different dimension

Cohere v4 today writes **1024**. The candidate providers:

| Provider / model | Native dim | On-request dim select? |
|---|---|---|
| Cohere Embed v4 (current) | 256/512/1024/**1024 default**/1536 | Yes (Matryoshka) |
| Ollama `bge-m3` | **1024** | No |
| Ollama `nomic-embed-text` | **768** | No |
| Ollama `mxbai-embed-large` | **1024** | No |
| ONNX `bge-small-en-v1.5` | **384** | No |
| ONNX `bge-base-en-v1.5` | **768** | No |
| ONNX `bge-large-en-v1.5` | **1024** | No |

The hazard is concrete and **silent-until-late** in the current code:

- Switching to `bge-small` (384) while an existing Lance table is 1024 → the
  first `run_backfill` write does `tbl.add()` of a `pl.Array(Float32, 384)`
  frame into a table whose column is `list<float32>[1024]`. Lance rejects the
  arrow append (schema mismatch at `add_chunk`, `lance_store.py:107-116`). This
  one fails relatively loud, but at write time, deep in a backfill.
- Worse: if the Lance dir is deleted/recreated at 384 but a DuckDB connection
  still binds `register_vss(..., dim=1024)` (e.g. a stale `output_dimension` env
  or a cached settings object), the view cast `CAST(embedding AS FLOAT[1024])`
  over 384-wide data throws a DuckDB cast error at SELECT time — not at bind.
- **Cross-provider semantic corruption (the quiet killer):** even at *matching*
  dims (Cohere-1024 vs bge-large-1024 vs bge-m3-1024), the vector *spaces are
  incompatible*. A query embedded with bge against documents embedded with
  Cohere returns cosine scores that are numerically valid but semantically
  garbage. Nothing in the current schema stops this — the `model` column exists
  (`lance_store.py:75`, written at `embed_worker.py:480`) but is **never read to
  detect a provider/model change**. This is the strongest argument for the
  fail-loud guard in §5.

There is **no dimension or model-identity validation anywhere** in `v1.2.1`
(confirmed: no `dim !=` / mismatch guard in `src/claude_sql/`, grep clean). The
`model`/`dim` columns are write-only metadata today. v2 must promote them to a
read-and-enforce contract.

---

## 3. Every call site of the anchor functions

From `codegraph callers` + `codegraph impact`, cross-checked against grep.

### 3.1 `_invoke_bedrock_sync` — 7 callers

The raw invoke. Production callers are entirely **inside `embed_worker.py`** —
it is not imported anywhere else, which makes it a clean thing to replace.

| Caller | `path:line` | Kind | input_type | embedding_type |
|---|---|---|---|---|
| `_embed_one_batch` | `embed_worker.py:245` | batch (via `asyncio.to_thread`) | `search_document` | `settings.embedding_type` (int8) |
| `embed_query` | `embed_worker.py:338` | single query | `search_query` | `"float"` (forced) |
| `test_invoke_bedrock_sync_happy_path` | `tests/test_embed_worker.py:261` | test | doc | int8 |
| `test_invoke_bedrock_sync_clips_long_text` | `tests/test_embed_worker.py:282` | test | doc | int8 |
| `test_invoke_bedrock_sync_retries_throttling` | `tests/test_embed_worker.py:298` | test | doc | int8 |
| `test_invoke_bedrock_sync_body_is_json` | `tests/test_embed_worker.py:470` | test | doc | int8 |
| (`test_embed_worker.py:1`) | file-level | test module | — | — |

Only **two production callers**: `_embed_one_batch` (batch/doc) and `embed_query`
(single/query). Both go through the same body; the only differences are
`input_type`, `embedding_type`, and batch cardinality.

### 3.2 `embed_query` — 3 callers

| Caller | `path:line` | Needs |
|---|---|---|
| `search` | `src/claude_sql/app/cli.py:1683` (call at `:1751`) | single float query vector of length `output_dimension`; feeds the DuckDB HNSW lookup |
| `test_embed_query_uses_search_query_input_type` | `tests/test_embed_worker.py:366` | pins `input_type="search_query"`, `embedding_types=["float"]` |
| (`test_embed_worker.py:1`) | test module | — |

One production caller: the `search` CLI command (imported lazily at
`cli.py:1738`). It needs exactly a `list[float]` it can bind as
`CAST(? AS FLOAT[dim])`.

### 3.3 `embed_documents_async` — 4 callers

| Caller | `path:line` | Needs |
|---|---|---|
| `run_backfill` | `src/claude_sql/analytics/embed_worker.py:369` (call at `:465`) | batch doc embeddings, float, in input order |
| `test_embed_documents_async_empty_short_circuit` | `tests/test_embed_worker.py:322` | `[]` on empty without building a client |
| `test_embed_documents_async_batches_correctly` | `tests/test_embed_worker.py:335` | batch fan-out / order preservation |
| (`test_embed_worker.py:1`) | test module | — |

One production caller: `run_backfill`.

### 3.4 `run_backfill` — 7 callers

| Caller | `path:line` | Needs |
|---|---|---|
| `embed` CLI command | `src/claude_sql/app/cli.py:1606` (call at `:1663`) | corpus backfill; `--dry-run` OFF by default |
| `analyze` CLI command | `src/claude_sql/app/cli.py:2281` (import at `:2353`) | chained embed step; respects `--dry-run`, `--skip-embed` |
| 4× tests | `tests/test_embed_worker.py:385, 404, 415, 444` | dry-run plans, real-run shard writes, no-pending zero |
| (`test_embed_worker.py:1`) | test module | — |

**Consumer summary.** The entire production embedding surface is:
`search` (single query) → `embed_query`; `embed` and `analyze` (batch) →
`run_backfill` → `embed_documents_async`. All four anchors read only from
`Settings`. Downstream, the vectors land in Lance (`lance_store.add_chunk`,
`lance_store.py:107`) and are read back through the DuckDB
`message_embeddings` view (`sql_views.py:1912-1920`) and the `semantic_search`
macro (`sql_views.py:1359`). No embedding code lives outside `embed_worker.py`
except these callers — a tight, well-bounded seam.

---

## 4. Proposed `EmbeddingProvider` protocol

There is **no existing Protocol / port abstraction** for embeddings in the
codebase today (grep for `Protocol` in `src/` finds only unrelated hits). This
is greenfield.

### 4.1 The port

Define a runtime-checkable `Protocol` (structural typing, no inheritance
required) in the future `application/ports` layer — for `v1.2.1` it can live at
`src/claude_sql/core/embedding.py`.

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Port: turn text into vectors. One adapter per backend.

    The document/query asymmetry is preserved as two methods rather than an
    ``input_type`` parameter, because not every backend has Cohere's
    search_document/search_query split — bge uses a query *prefix*, Ollama
    uses neither. Each adapter internalizes its own asymmetry handling.
    """

    @property
    def model_id(self) -> str:
        """Stable identity string stamped into the Lance ``model`` column and
        checked on rebind (e.g. 'bedrock:cohere.embed-v4:0',
        'ollama:bge-m3', 'onnx:bge-small-en-v1.5')."""
        ...

    @property
    def dimension(self) -> int:
        """Fixed output dimension this provider emits. The dimension contract
        (§2) is enforced against this value, NOT against a free-floating
        Settings.output_dimension."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch corpus embedding. Float vectors, input order preserved.
        Empty input returns []. The adapter owns its own batching, concurrency,
        retry, and (if quantized) float-widening."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Single query vector, always float, length == self.dimension.
        Synchronous (query path is one call on the hot path of `search`)."""
        ...
```

Design notes tying it to the current seam:
- `embed_documents` is **async** (mirrors `embed_documents_async`,
  `embed_worker.py:268`); `embed_query` is **sync** (mirrors `embed_query`,
  `embed_worker.py:338`, called synchronously from `search` at `cli.py:1751`).
- `dimension` **replaces** the free `settings.output_dimension` as the source of
  truth for the contract. `run_backfill` becomes
  `open_or_create_table(db, dim=provider.dimension)`;
  `register_vss(..., dim=provider.dimension)`; the `search` cast uses
  `provider.dimension`. One value, provider-owned, checked once.
- `model_id` replaces `settings.active_model_id` in the Lance `model` stamp
  (`embed_worker.py:480`) and becomes the identity checked by the fail-loud
  guard (§5).
- The float-widening currently done inline (`embed_worker.py:325-327, 366`)
  moves **inside** each adapter, so the port always yields floats regardless of
  the backend's native storage type.

`run_backfill` / `embed_documents_async` / `embed_query` collapse to thin
orchestration over `provider`; `_invoke_bedrock_sync`'s body moves wholesale
into the Bedrock adapter.

### 4.2 Adapter A — `BedrockCohereEmbedder` (existing behavior, no new deps)

Wraps exactly what `_invoke_bedrock_sync` + `embed_documents_async` +
`embed_query` do today. `model_id="bedrock:cohere.embed-v4:0"`,
`dimension=settings.output_dimension` (Cohere is the one provider that keeps the
Matryoshka knob). Keeps the `input_type` split internally
(`search_document` in `embed_documents`, `search_query` in `embed_query`), keeps
`embedding_types=[settings.embedding_type]` for documents but **forces
`"float"` for queries** (preserving the `embed_worker.py:341-343` invariant),
keeps the tenacity retry and `_build_bedrock_client` reuse
(`llm_shared.py:348`). **New deps: none** — `boto3` is already core
(`pyproject.toml:29`).

### 4.3 Adapter B — `OllamaEmbedder` (local HTTP, optional `ollama` extra)

Talks to a local Ollama server (default `http://localhost:11434`).

**Use the newer `/api/embed` endpoint, not the legacy `/api/embeddings`.** Two
reasons: (1) `/api/embed` is batch-capable (`input` accepts `str | list[str]`);
(2) `/api/embed` returns **L2-normalized unit-length vectors**, whereas
`/api/embeddings` returns un-normalized vectors — normalized is exactly what the
`cosine`-metric HNSW index wants.

Request / response shapes (grounded in current Ollama API docs):
```python
# POST http://localhost:11434/api/embed
# body:
{"model": "bge-m3", "input": ["text one", "text two"]}   # input: str | list[str]
# response:
{"model": "bge-m3", "embeddings": [[...], [...]], "total_duration": ...}
#                     ^ always list[list[float]], one per input, L2-normalized

# legacy POST /api/embeddings (singular, un-normalized — avoid):
{"model": "bge-m3", "prompt": "one text"}          # -> {"embedding": [...]}
```

Adapter behavior:
- `embed_documents`: POST `/api/embed` with `input=texts` (chunk to a
  configurable batch if desired), return `resp["embeddings"]`.
- `embed_query`: POST `/api/embed` with `input=query`, return
  `resp["embeddings"][0]`. Ollama has **no `input_type` concept**; the same
  endpoint serves both sides. (If serving an mxbai model, that model's own query
  instruction is the only asymmetry and it's applied model-side.)
- `dimension`: fixed per model — `bge-m3`=1024, `nomic-embed-text`=768,
  `mxbai-embed-large`=1024, `all-minilm`=384. Adapter should either hard-map or
  probe once by embedding a sentinel and caching `len(vec)`.
- `model_id`: `f"ollama:{model}"`.
- Optional `dimensions` param exists for Matryoshka-capable models; leave off by
  default. `truncate` defaults to `true` server-side.

**New deps:** `httpx` for the POST. `httpx` is already present transitively
(`uv.lock:769`, pulled by `anthropic`), but for a clean install it should be a
declared dependency of the `[ollama]` extra. No `ollama` Python SDK needed — raw
HTTP keeps the extra tiny.

### 4.4 Adapter C — `OnnxBgeEmbedder` (fully local, optional `onnx` extra)

Runs a BAAI `bge` `.onnx` model with `onnxruntime` + `tokenizers`, no torch.

**CRITICAL CORRECTION grounded in the BAAI model card + FlagEmbedding docs:
BGE v1.5 English models use CLS pooling, NOT mean pooling.** The task brief said
"mean-pool + normalize" — that is wrong for bge and would silently degrade
retrieval. BAAI's card states: *"BGE uses the last hidden state of `[cls]` as
the sentence embedding... If you use mean pooling, there will be a significant
decrease in performance."* Use `last_hidden_state[:, 0]` then L2-normalize.
(Mean pooling is correct only for `nomic` / `all-MiniLM`-style models, which
this adapter does not target.)

Pipeline:
1. Tokenize with `tokenizers.Tokenizer.from_file(tokenizer.json)`
   (`enable_padding()`, `enable_truncation(max_length=512)`).
2. `onnxruntime.InferenceSession(...).run(None, feeds)` where feeds =
   `{"input_ids", "attention_mask"}` and `"token_type_ids"` (zeros) **only if
   `session.get_inputs()` declares it** — BERT-arch exports vary; inspect at
   init.
3. **CLS pool**: `emb = last_hidden_state[:, 0]`.
4. **L2-normalize**: `emb / np.linalg.norm(emb, axis=1, keepdims=True)`.

Query asymmetry: bge v1.5 English models prepend a query instruction to
**queries only** (never documents). Exact string (trailing space intentional):
`"Represent this sentence for searching relevant passages: "`. For v1.5 it is
*optional* ("only a slight degradation" if omitted); apply it in `embed_query`,
not `embed_documents`.

```python
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class OnnxBgeEmbedder:
    def __init__(self, onnx_path, tokenizer_path, dim):
        self._tok = Tokenizer.from_file(tokenizer_path)
        self._tok.enable_padding(); self._tok.enable_truncation(max_length=512)
        self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._dim = dim

    def _run(self, texts):
        encs = self._tok.encode_batch(texts)
        ids  = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        last_hidden = self._sess.run(None, feeds)[0]   # [B, seq, hidden]
        emb = last_hidden[:, 0]                          # CLS pooling — NOT mean
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    async def embed_documents(self, texts):
        if not texts: return []
        return self._run(texts).tolist()

    def embed_query(self, text):
        return self._run([QUERY_PREFIX + text])[0].tolist()
```

- `dimension`: fixed per model — `bge-small-en-v1.5`=384,
  `bge-base-en-v1.5`=768, `bge-large-en-v1.5`=1024.
- Model file: `BAAI/bge-small-en-v1.5` ships `onnx/model.onnx` + `tokenizer.json`
  on HF Hub (or `onnx-community/bge-small-en-v1.5-ONNX`). Download once via
  `huggingface-hub`.
- Since `onnxruntime` is CPU-blocking, wrap `embed_documents` batch runs with
  `asyncio.to_thread` (mirrors the existing `_embed_one_batch` pattern,
  `embed_worker.py:257`).

**New deps (`[onnx]` extra):** `onnxruntime`, `tokenizers`, `huggingface-hub`
(for download). `numpy` is **already core** (`pyproject.toml:38`). Deliberately
**avoid** `transformers`, `optimum`, and `torch` — `tokenizers` alone loads
`tokenizer.json` with zero torch dependency, keeping the extra lean.

### 4.5 Keeping the core install lean — optional extras

Today `dependencies` (`pyproject.toml:27-50`) has **no `[project.optional-dependencies]`
block** (confirmed: none defined). v2 should add extras so the base wheel does
not pull ONNX/HF:

```toml
[project.optional-dependencies]
ollama = ["httpx>=0.28"]
onnx   = ["onnxruntime>=1.19", "tokenizers>=0.20", "huggingface-hub>=0.25"]
```

Bedrock stays in core (`boto3` is already required). Each adapter module
imports its heavy deps **lazily inside the adapter**, and raises a clear
"install `claude-sql[onnx]`" error if missing — matching the existing lazy-import
discipline (e.g. `lance_store` deferred at `embed_worker.py:139, 445` to keep the
2.6s lancedb import off non-embed paths).

---

## 5. Config surface & the fail-loud dimension guard

### 5.1 Provider selection through pydantic-settings

Add a discriminating switch plus per-provider sub-settings to `Settings`
(`config.py:126`), all under the existing `env_prefix="CLAUDE_SQL_"`
(`config.py:134`):

```python
embedding_provider: Literal["bedrock", "ollama", "onnx"] = "bedrock"
#   env: CLAUDE_SQL_EMBEDDING_PROVIDER

# Ollama
ollama_host: str = "http://localhost:11434"       # CLAUDE_SQL_OLLAMA_HOST
ollama_model: str = "bge-m3"                       # CLAUDE_SQL_OLLAMA_MODEL

# ONNX
onnx_model_repo: str = "BAAI/bge-small-en-v1.5"    # CLAUDE_SQL_ONNX_MODEL_REPO
onnx_model_dim: int = 384                          # CLAUDE_SQL_ONNX_MODEL_DIM
```

The existing Bedrock fields (`model_id`, `output_dimension`, `embedding_type`,
`region`, `embed_concurrency`, `batch_size` — `config.py:183-203`) stay as the
`bedrock` provider's config. A factory —
`build_embedder(settings) -> EmbeddingProvider` — switches on
`embedding_provider` and returns the right adapter. `active_model_id`
(`config.py:408`) generalizes to delegate to `provider.model_id`.

Backward compatibility: default `embedding_provider="bedrock"` reproduces
`v1.2.1` behavior byte-for-byte, so existing installs and every current test
keep passing untouched.

### 5.2 Fail-loud on dimension / provider mismatch (the guard that's missing today)

Today there is **zero** validation (§2). v2 must add a guard because switching
providers is otherwise silent-until-corrupt. Two enforcement points:

1. **At `run_backfill` open (`embed_worker.py:453`)** — before appending, read
   the existing Lance table's first row `model` + `dim` columns (both are
   already stored — `lance_store.py:75-77`, written at
   `embed_worker.py:480-481`) and compare to `provider.model_id` /
   `provider.dimension`. On mismatch, raise a typed
   `EmbeddingProviderMismatch` naming both sides and telling the user to
   re-embed (`rm -rf ~/.claude/embeddings_lance/`). Add a
   `lance_store.table_identity(uri) -> (model, dim) | None` helper.

2. **At `register_vss` (`sql_views.py:1812`)** — it already receives `dim`; add
   a check that the attached Lance column width equals the passed `dim` (probe
   `arrow_schema` of `lance_store.main.embeddings`) and that the table's stored
   `model` equals the active provider's `model_id`. Fail with a clear catalog
   error instead of letting the `CAST(embedding AS FLOAT[{dim}])`
   (`sql_views.py:1917`) throw a cryptic width error later.

The `model` column already exists specifically for this — it's just never read.
v2 flips it from write-only metadata to an enforced contract.

---

## 6. Migration / backfill implications

**Switching providers invalidates every existing vector.** Vectors from
different models — even at identical dimension (Cohere-1024 vs bge-large-1024 vs
bge-m3-1024) — live in **incompatible embedding spaces**. A `bge` query against a
Cohere-embedded corpus produces numerically-valid, semantically-meaningless
cosine scores. There is no in-place conversion; the corpus must be re-embedded.

The re-embed path (leaning on mechanisms that already exist):

1. **Drop the old store.** `rm -rf ~/.claude/embeddings_lance/` — the documented
   recovery path (CLAUDE.md "Operational rollback paths"; `lance_uri` default
   `config.py:213`). This clears the frozen dimension (§2) so a
   different-dimension provider can create a fresh table.
2. **Set the provider.** `export CLAUDE_SQL_EMBEDDING_PROVIDER=onnx` (plus the
   per-provider vars from §5.1).
3. **Re-run backfill.** `claude-sql embed --no-dry-run` (or `analyze`). With an
   empty Lance table, `discover_unembedded` (`embed_worker.py:99`) returns the
   full corpus (the `get_embedded_uuids` anti-join, `lance_store.py:245`, yields
   an empty set), so every message is re-embedded under the new provider.
   `run_backfill` stamps the new `model` + `dim` (`embed_worker.py:480-481`) and
   builds a fresh index (`ensure_index`, `embed_worker.py:506`).
4. **Downstream re-derivation.** Embeddings feed UMAP+HDBSCAN clustering and the
   Leiden community graph over centroids. The cluster mtime sidecar
   (`clusters.parquet.embeddings_mtime`, CLAUDE.md "Cluster mtime sidecar")
   detects the moved embeddings and triggers a clustering refit on the next
   `cluster`/`community`/`analyze` run — so structure analytics rebuild
   automatically once vectors change.

**The legacy-parquet migration path is provider-agnostic and unaffected:**
`migrate_from_parquet_shards` (`lance_store.py:167`) copies old
`embeddings/part-*.parquet` into Lance idempotently (row-count guard,
`lance_store.py:188-196`); it does not re-embed, so it only applies to the same
Cohere vectors. A provider switch bypasses it entirely (fresh backfill, not
migration).

**Recommended v2 guardrail:** because a provider switch is destructive, make
`embed` refuse to append into a non-empty store whose stored `model` differs
from the active provider (the §5.2 guard), forcing the explicit `rm -rf` +
re-embed rather than silently mixing two vector spaces in one table.

---

## Appendix — grounding sources for §4

- Ollama API (`/api/embed` vs `/api/embeddings`, request/response, L2
  normalization on `/api/embed`): Ollama API docs
  (`github.com/ollama/ollama/blob/main/docs/api.md`),
  `docs.ollama.com/api/embed`, `docs.ollama.com/capabilities/embeddings`.
- Ollama model dims (`bge-m3`=1024, `nomic-embed-text`=768,
  `mxbai-embed-large`=1024, `all-minilm`=384): `ollama.com/library`.
- BGE CLS pooling (NOT mean), query instruction prefix, dims (384/768/1024):
  BAAI model cards `huggingface.co/BAAI/bge-small-en-v1.5` (and `-base-`,
  `-large-`), FlagEmbedding `github.com/FlagOpen/FlagEmbedding`,
  `bge-model.com/tutorial/1_Embedding/1.2.3.html`.
- ONNX file location + inputs (`input_ids`/`attention_mask`/`token_type_ids`,
  `last_hidden_state`): `BAAI/bge-small-en-v1.5` ONNX PR #9,
  `onnx-community/bge-small-en-v1.5-ONNX`.
