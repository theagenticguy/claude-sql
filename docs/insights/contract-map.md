# claude-sql · Contract map

When one module hands something to another, this file records what the receiver
is really expecting — beyond what the type signature alone says.

**What counts as a contract here.** claude-sql is statically typed (`ty` strict)
and organized as a strict hexagon (`interfaces > application > infrastructure >
domain`, one import-linter layers contract). A "contract" is any declaration in
one module that ≥ 1 other module depends on for its shape. In this codebase they
come in five kinds, all first-class:

1. **Port Protocols** — `@runtime_checkable` `Protocol` classes in
   `application/ports.py` and `domain/ports.py`. These are the seams between a
   use-case and the outside world; concrete adapters in `infrastructure`
   implement them structurally and the composition root injects them.
2. **Structured-output schemas** — pydantic v2 models in `domain/models.py` that
   define the LLM classification wire shape and the parquet cache columns.
3. **The Lance vector schema** — the fixed-size `Array(Float32, dim)` embeddings
   table shape plus its `(model, dim)` identity stamp (the dimension contract).
4. **SQLite DDL** — the `session_checkpoint` and `retry_queue` table shapes that
   back the checkpoint and retry-queue ports.
5. **The consumer collapse byte-parity contract** — `render_turn_text` in
   `domain/transcript.py`, an *external* contract with downstream consumers,
   pinned byte-for-byte by `tests/test_collapse_parity.py`; and the
   **exit-code wire contract** (`EXIT_CODES`) that agents parse from the process.

Contracts are ordered by consumer count descending. Twelve get a full H2; the
rest are one-liners under `## Other contracts`.

## `PortResult[T]` — the uniform result type crossing every port boundary

**Producer:** `src/claude_sql/application/ports.py:60`

**Consumer(s):**
- `src/claude_sql/application/use_cases/__init__.py` — re-exports the port surface for every use-case.
- `src/claude_sql/application/analyze.py` — the 10-stage pipeline threads results between stages.
- `src/claude_sql/application/use_cases/embed.py`, `classify.py`, `trajectory.py`, `conflicts.py`, `friction.py`, `cluster.py`, `ingest.py` — every use-case that names a port return.
- `src/claude_sql/domain/errors.py:37` — declares `DomainError`, the `E` half this type is parameterized over.

**Shape:**
```python
#: The uniform result type crossing a port boundary. A ``Success[T]`` carries
#: the value; a ``Failure[DomainError]`` carries a domain error. See CLAUDE.md
#: "returns discipline": use ``Success``/``Failure``, ``is_successful``,
#: ``.unwrap``, ``.map``, ``.alt``, and ``match`` narrowing ONLY — never
#: ``.bind()``, ``@safe``, ``flow``, ``pipe``, or HKT features (they require the
#: mypy plugin and hard-error under ty).
type PortResult[T] = Result[T, DomainError]
```

**Assumptions consumers make:**
- The error arm is always a `DomainError` subclass, never a raw `Exception` — so a `Failure` match can rely on the domain error hierarchy (`src/claude_sql/domain/errors.py:37-45`).
- Only the `returns` constrained subset is used (`Success`/`Failure`/`is_successful`/`.unwrap`/`.map`/`.alt`/`match`); `.bind()`, `@safe`, `flow`, `pipe`, and HKT features are forbidden because they require the mypy plugin and hard-error under `ty` (`src/claude_sql/application/ports.py:54-60`).

**Drift risk:** If an adapter raises a transport-specific exception (boto3, duckdb) instead of wrapping it in a `DomainError`, the `Failure[DomainError]` guarantee silently breaks and callers matching on the error arm miss it. Mitigation: adapters catch their transport exceptions and re-raise as a `DomainError` subclass at the port boundary, as `errors.py:44` documents.

## `EmbeddingProvider` — the pluggable text→vector port

**Producer:** `src/claude_sql/domain/ports.py:33`

**Consumer(s):**
- `src/claude_sql/application/use_cases/embed.py:31` — the backfill use-case reads `model_id`/`dimension` to stamp rows and `embed_documents` to vectorize.
- `src/claude_sql/infrastructure/session_search.py` — the search adapter calls `embed_query` on the hot path.
- `src/claude_sql/infrastructure/embedding/cohere_bedrock.py`, `ollama.py`, `onnx_bge.py` — the three concrete adapters implementing it.
- `src/claude_sql/composition.py:135` — `build_search` injects an `EmbeddingProvider`.
- `src/claude_sql/infrastructure/embedding/__init__.py:26` — `build_embedder` factory plus the `ensure_store_matches` guard.

**Shape:**
```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...
    @property
    def provider(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

**Assumptions consumers make:**
- `model_id` is globally unique across providers, so it doubles as the provider discriminator in the store guard (`src/claude_sql/domain/ports.py:45-52`).
- `dimension` is the single contract source for the Lance schema, the DuckDB `FLOAT[dim]` view cast, and the query-time cast — it replaced the free `Settings.output_dimension` (`src/claude_sql/domain/ports.py:59-64`).
- `embed_documents` preserves input order, returns `[]` on empty input, and returns float vectors even when the backend stores quantized — the adapter owns batching, concurrency, retry, and float-widening (`src/claude_sql/domain/ports.py:66-70`).
- `embed_query` is synchronous and returns a vector of length `== self.dimension` (`src/claude_sql/domain/ports.py:72-75`).
- The document/query asymmetry is handled inside each adapter, not by a parameter — Cohere uses distinct input types, BGE prepends a query prefix, Ollama uses neither (`src/claude_sql/domain/ports.py:38-43`).

**Drift risk:** A new adapter that returns a vector whose width differs from its own reported `dimension` corrupts the Lance fixed-size column at write and the `FLOAT[dim]` cast at read. Mitigation: `dimension` is read back and enforced by `ensure_store_matches` on every bind (see the Lance vector schema contract below).

## `LlmAnalyticsProvider` — the structured-output classification port

**Producer:** `src/claude_sql/domain/ports.py:78`

**Consumer(s):**
- `src/claude_sql/application/use_cases/trajectory.py:549` — calls `provider.classify_structured(...)` per chunk.
- `src/claude_sql/infrastructure/llm_analytics/sonnet_bedrock.py` — the default Sonnet-on-Bedrock adapter.
- `src/claude_sql/infrastructure/llm_analytics/strands_luna.py` — the opt-in Luna/Strands adapter.
- `src/claude_sql/infrastructure/llm_analytics/__init__.py` — the `build_*` factory.

**Shape:**
```python
@runtime_checkable
class LlmAnalyticsProvider(Protocol):
    @property
    def provider(self) -> str: ...
    async def classify_structured(
        self, *, system: str, prompt: str, schema: type[SchemaT]
    ) -> SchemaT: ...
```

**Assumptions consumers make:**
- The return value is a validated instance of the passed `schema` (bound to `BaseModel`), so the caller skips its own validation (`src/claude_sql/domain/ports.py:95-98`, `SchemaT` at `:30`).
- `system` is byte-stable and cacheable (task framing); `prompt` is the per-call payload — the split lets the adapter drive prompt caching (`src/claude_sql/domain/ports.py:100-103`).
- On failure the adapter raises the terminal signal: `BedrockRefusalError` for a Sonnet content-policy refusal, or `LlmAnalyticsUnavailable` for a Luna transport/structured-output failure — the caller treats these as terminal, not retryable (`src/claude_sql/domain/ports.py:103-106`).

**Drift risk:** If an adapter swallows a refusal and returns an empty/placeholder schema instance instead of raising, the trajectory worker's "stamp neutral placeholder + clear retry queue" path never fires and the unit cycles forever. Mitigation: the refusal/unavailable exceptions are part of the contract (`errors.py:48-106`); the worker matches on them.

## The Lance embeddings vector schema and the dimension contract

**Producer:** `src/claude_sql/infrastructure/lance_store.py:65` (`lance_schema(dim)`), with the identity stamp read back at `:245` (`table_identity`).

**Consumer(s):**
- `src/claude_sql/application/use_cases/embed.py:364-369` — the write side builds the polars frame with the exact matching schema.
- `src/claude_sql/infrastructure/duckdb_views.py:1955-1969` — `register_vss` casts the embedding column to `FLOAT[dim]` and projects `message_embeddings`.
- `src/claude_sql/domain/embedding_guard.py:22` — `ensure_store_matches` compares stored vs active `(model, dim)`.
- `src/claude_sql/infrastructure/adapters.py`, `interfaces/cli/app.py`, `use_cases/cluster.py` — via `TABLE_NAME` / row-count probes.

**Shape:**
```python
def lance_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("uuid", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("dim", pa.int32(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("embedded_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )
```

**Assumptions consumers make:**
- `embedding` is a FIXED-SIZE list of float32 — a variable-size `pa.list_(pa.float32())` (or a polars `pl.List` instead of `pl.Array`) is rejected as a vector column by `create_index` (`src/claude_sql/infrastructure/lance_store.py:65-71`, write side `embed.py:368`).
- Every row stamps the same `model` / `dim`, so reading the first row is sufficient to recover the store identity (`src/claude_sql/infrastructure/lance_store.py:245-264`).
- The `dim` argument to the DuckDB `FLOAT[dim]` cast is overridden by the *stored* dim, so a store written at a different width binds correctly regardless of the caller's `dim` (`src/claude_sql/infrastructure/duckdb_views.py:1937-1948`).
- An absent table reports `0` rows / `None` identity rather than raising — the empty-namespace gate probes `_has_table`, not the filesystem (`src/claude_sql/infrastructure/lance_store.py:53-62`, `duckdb_views.py:1911-1928`).

**Drift risk:** Two different models emitting the same width still live in incompatible vector spaces, so a silent provider switch produces numerically valid but semantically garbage cosine scores. Mitigation: `ensure_store_matches` fails loud on any `(model, dim)` mismatch before either write or bind (`src/claude_sql/domain/embedding_guard.py:47-61`), naming the `rm -rf` + re-embed recovery.

## Structured-output classification schemas (parquet + wire shape)

**Producer:** `src/claude_sql/domain/models.py` — `SessionClassification:25`, `TrajectoryArrayResult:189` (wraps `TrajectoryWindow:96`), `ConflictsResult:299` (wraps `ConflictPair:212`), `UserFrictionSignal:323`.

**Consumer(s):**
- `src/claude_sql/infrastructure/bedrock/structured_output.py:117-120` — flattens each model into the four live `*_SCHEMA` dicts for the Bedrock `output_config.format` field.
- `src/claude_sql/application/use_cases/classify.py:128`, `conflicts.py:261`, `friction.py:482` — pass the flattened schema to the classify call.
- `src/claude_sql/application/use_cases/trajectory.py:549` — passes `TrajectoryArrayResult` as `schema=` to `classify_structured`.
- `src/claude_sql/domain/trajectory.py` — pure trajectory math over the window shape.

**Shape:** (representative — `SessionClassification`)
```python
class SessionClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    autonomy_tier: Literal["manual", "assisted", "autonomous"]
    work_category: Literal["sde", "admin", "strategy_business", "events",
                           "thought_leadership", "other"]
    success: Literal["success", "partial", "failure", "unknown"]
    goal: str = Field(..., min_length=1, max_length=280)
    confidence: float = Field(..., ge=0.0, le=1.0)
```

**Assumptions consumers make:**
- `model_config = ConfigDict(extra="forbid")` on every model — the flattener injects `additionalProperties: false` and the Bedrock structured-output subset relies on it (`src/claude_sql/domain/models.py:32`, flatten at `structured_output.py:117`).
- The `Literal` enums are the exact stored parquet values and the DuckDB views group on them verbatim — e.g. `transition_kind ∈ {frustration_spike, resolution, reset, drift, clarification, none}` (`src/claude_sql/domain/models.py:158-176`).
- `ConflictsResult.conflicts` defaults to an empty list, and an empty list means ZERO rows written for that session (the legacy `empty=True` sentinel is gone) — callers wanting every session must LEFT JOIN and coalesce (`src/claude_sql/domain/models.py:299-320`).
- `TrajectoryWindow.prev_uuid` is `None` on the session-first window and the host pipeline echoes `(prev_uuid, curr_uuid)` back to verify per-window completeness (`src/claude_sql/domain/models.py:111-127`).
- `turn_a_uuid` / `turn_b_uuid` are copied verbatim from the `[uuid=...]` transcript headers, not paraphrased, and must differ (`src/claude_sql/domain/models.py:223-243`).

**Drift risk:** Adding a `Literal` variant (a new `work_category`, a new `transition_kind`) that a DuckDB view or macro doesn't yet group on silently drops those rows from aggregates. Mitigation: extend the corresponding view/macro in `duckdb_views.py` in the same change; the enum is the single source and the flattener strips constraints the validator rejects (`structured_output.py`).

## `render_turn_text` — consumer collapse byte-parity (external contract)

**Producer:** `src/claude_sql/domain/transcript.py:287` (`render_turn_text`), over `TranscriptRow:197`.

**Consumer(s):**
- `src/claude_sql/infrastructure/transcript_reader.py` — `DuckDbTranscriptReader.read_turn_text` projects `read_json` rows into `TranscriptRow` and calls this.
- `src/claude_sql/composition.py:83` — the `ClaudeSql` facade exposes the reader as the importable surface downstream consumers use.
- `tests/test_collapse_parity.py:136,160` — the byte-parity pin (ports the consumer's fixture and expected string verbatim).
- `tests/property/test_transcript_properties.py`, `tests/test_transcript_domain.py` — property + unit coverage.

**Shape:**
```python
def render_turn_text(
    rows: Iterable[TranscriptRow],
    *,
    per_turn_chars: int = 8_000,
    total_chars: int = 200_000,
    truncation_notice: bool = False,
) -> str: ...

@dataclass(slots=True)
class TranscriptRow:
    uuid: str | None
    type: str | None
    timestamp: str | None
    message: Any  # dict | JSON-text str | None
```

**Assumptions consumers make:**
- ONE line per message, shape `[{role} {ts}] {body}`; `role` prefers inner `message.role` and falls back to envelope `type` (so `system` is reachable, tool_result carriers keep their `user` role) (`src/claude_sql/domain/transcript.py:296-309`, `_collapse_role_of:227`).
- `ts` is the RAW timestamp string verbatim (incl. `.000`/`Z`), with no TIMESTAMP round-trip (`src/claude_sql/domain/transcript.py:302-303`).
- Ordering is `(ts, kind_rank, uuid)` with envelope-type rank `user < assistant < tool < system` (`_COLLAPSE_KIND_RANK:49`) — distinct from `assemble`'s block-level `_KIND_RANK` (`:41`).
- `tool_use`/`tool_result` blocks fold INLINE as `[tool_use:{name}]` / `[tool_result]` markers; rows whose collapsed body is empty are dropped (`src/claude_sql/domain/transcript.py:264-284, 307-309`).
- Per-turn cap appends a leading-space `` …`` marker; the whole transcript is hard-sliced at `total_chars` with NO trailing notice unless `truncation_notice=True` — the leading space and the notice-suppression are load-bearing for parity (`src/claude_sql/domain/transcript.py:282-283, 310-312, 330-336`).
- The single-level `part-*.jsonl` glob must never admit `subagents/*` sidecars (`tests/test_collapse_parity.py:148-157`).

**Drift risk:** Any change to spacing, the `role` fallback order, the sort key, or the truncation marker breaks byte-parity with every downstream consumer of the drop-in reader. Mitigation: `tests/test_collapse_parity.py` asserts `text == "\n".join(_COLLAPSE_EXPECTED_LINES)` against the consumer's verbatim fixture; it is wired into `mise run check`.

## `EXIT_CODES` — the CLI exit-code wire contract

**Producer:** `src/claude_sql/domain/errors.py:26`

**Consumer(s):**
- `src/claude_sql/infrastructure/duckdb_errors.py:15,28-42` — `classify_duckdb_error` maps duckdb exceptions to `parse_error` / `catalog_error` / `runtime_error`.
- `src/claude_sql/interfaces/cli/output.py:28,177` — `run_or_die` / `emit_error` exit with the classified code.
- `src/claude_sql/interfaces/cli/app.py:45,234,505` — top-level handlers exit with `invalid_input` and `duckdb_missing`.

**Shape:**
```python
EXIT_CODES: dict[str, int] = {
    "ok": 0,
    "no_embeddings": 2,
    "invalid_input": 64,  # malformed user-supplied flags (e.g. --glob)
    "parse_error": 64,  # malformed SQL
    "catalog_error": 65,  # unknown view/macro/column
    "runtime_error": 70,  # everything else from duckdb.Error
    "duckdb_missing": 127,  # system `duckdb` binary not on PATH
}
```

**Assumptions consumers make:**
- The integer codes are STABLE — agents calling via subprocess branch on them, so they must not renumber (`src/claude_sql/domain/errors.py:24-25`).
- `parse_error` and `invalid_input` deliberately share `64`; catalog is `65`, runtime is `70` — the classifier picks the key, `errors.py` owns the number (`src/claude_sql/infrastructure/duckdb_errors.py:28-42`).
- On non-TTY stderr the error is emitted as `{"error": {"kind", "message", "hint"}}` via `ClassifiedError.to_payload` alongside the exit (`src/claude_sql/domain/errors.py:109-125`, `output.py:142-177`).

**Drift risk:** A new `duckdb.Error` subclass that `classify_duckdb_error` doesn't recognize falls through to `runtime_error` (70) — acceptable, but a genuinely new *catalog* condition mislabeled as runtime would confuse an agent's retry logic. Mitigation: keep `classify_duckdb_error` in sync when new DuckDB exception subclasses land (CLAUDE.md agent-CLI note).

## `SessionTextCorpus.assemble` — the byte-stable internal transcript

**Producer:** `src/claude_sql/domain/transcript.py:126`

**Consumer(s):**
- `src/claude_sql/infrastructure/session_text_loader.py` — materializes the corpus from the `read_json` glob and calls `assemble` per session (via the `TranscriptReaderPort` adapter).
- The four LLM pipelines through the use-cases: `classify.py`, `trajectory.py`, `conflicts.py`, `friction.py` — they checkpoint on this output.

**Shape:**
```python
def assemble(
    self, session_id: str, *, settings: TranscriptCaps, include_uuids: bool = False
) -> str:
    # line shapes:
    #   [uuid=<id> <role> <ts>] <body>   (include_uuids=True, text turns)
    #   [<role> <ts>] <body>             (default, text turns)
    #   [tool_use:<name> <ts>] <input-preview>
    #   [tool_result <tu_id> <ts>] <result-preview>
    #   …(session truncated at <cap> chars, <N> events total)
```

**Assumptions consumers make:**
- Output is byte-stable: the string literals and the tie-break order (`_timeline_sort_key`, block-level `_KIND_RANK` = text < tool_use < tool_result) must not drift because the pipelines checkpoint on it (`src/claude_sql/domain/transcript.py:11-24, 41, 53-63`).
- `settings` is a pure `TranscriptCaps` two-int value-object (`session_text_tool_result_max_chars`, `session_text_total_max_chars`), NOT the god-`Settings`; the keyword name stays `settings` for call-site/test stability (`src/claude_sql/domain/transcript.py:126-145`).
- `include_uuids=True` switches only the text-turn header to `[uuid=<id> role ts]`; the default keeps `[role ts]` so classify/trajectory/friction prompts stay byte-identical (only conflicts needs the uuids — issue #109) (`src/claude_sql/domain/transcript.py:159-161`).

**Drift risk:** This is *not* the collapse-parity contract — it fans a message into per-block rows and uses a different kind-rank. Changing either collapse to "unify" them would break one of the two byte-stable contracts. Mitigation: the two renderers live side by side by design; the module docstring names the split (`src/claude_sql/domain/transcript.py:11-24`).

## `SearchHit` — the semantic-retrieval result value-object

**Producer:** `src/claude_sql/domain/retrieval.py:24`

**Consumer(s):**
- `src/claude_sql/infrastructure/session_search.py` — `DuckDbSessionSearch.search` constructs `SearchHit` rows from the DuckDB+Lance kNN result.
- `src/claude_sql/application/ports.py:43` — re-exported so callers name the whole port surface (ports + row type) from one module; `SessionSearchPort.search` returns `list[SearchHit]` (`:123`).

**Shape:**
```python
@dataclass(frozen=True, slots=True)
class SearchHit:
    uuid: str
    session_id: str
    ts: datetime | None
    role: str
    snippet: str
    cosine_sim: float
```

**Assumptions consumers make:**
- It lives in `domain`, not `application.ports`, so the infrastructure adapter can build it without importing UP into the application layer (`src/claude_sql/domain/retrieval.py:8-12`).
- `ts` is nullable — a hit may lack a resolvable timestamp; callers must handle `None` (`src/claude_sql/domain/retrieval.py:32`).
- `cosine_sim` is a cosine similarity (higher = closer), matching the `array_cosine_similarity` macro, not a distance (`src/claude_sql/domain/retrieval.py:24-30`).

**Drift risk:** No current drift risk — the shape mirrors columns the `search` path has projected since v1; it is a frozen dataclass with a single producer.

## `ReaderPort` — the DuckDB query seam with an explicit rebind lifecycle

**Producer:** `src/claude_sql/application/ports.py:248`

**Consumer(s):**
- `src/claude_sql/application/analyze.py` — the 10-stage pipeline; stages that write parquet/Lance mid-run (`embed`, `cluster`) must rebind before the next read.
- `src/claude_sql/application/use_cases/cluster.py`, and the `community` / `terms` use-cases — read `message_embeddings` after an upstream write.

**Shape:**
```python
@runtime_checkable
class ReaderPort(Protocol):
    def connection(self) -> duckdb.DuckDBPyConnection: ...
    def query(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame: ...
    def refresh_analytics_views(self) -> None: ...
    def rebind_vss(self, stage: str) -> None: ...
```

**Assumptions consumers make:**
- The DuckDB connection is stateful, long-lived, and shared across `analyze` stages — NOT a stateless executor. `register_all` binds views against parquet path lists FROZEN at registration time (`src/claude_sql/application/ports.py:250-259`).
- A stage that writes new shards or populates LanceDB does not see its own writes until the views are re-bound; the write→rebind cycle is explicit and must not be folded into `query` (`src/claude_sql/application/ports.py:259-268`).
- Skipping `rebind_vss` after an embed write reintroduces the RFC §9.6 stale-connection bug — `community` reads `message_embeddings` and gets 0 rows (`src/claude_sql/application/ports.py:260-266`).

**Drift risk:** A future refactor that hides the connection behind a "pure" `query(sql)` and drops `rebind_vss` silently reintroduces the 0-row analyze bug. Mitigation: the Protocol deliberately exposes `connection`, `refresh_analytics_views`, and `rebind_vss` as first-class methods; the docstring documents why (`ports.py:248-290`).

## `CheckpointPort` + `session_checkpoint` DDL — the processing watermark

**Producer:** Port `src/claude_sql/application/ports.py:166`; backing DDL `src/claude_sql/infrastructure/sqlite_state/checkpointer.py:44`.

**Consumer(s):**
- `src/claude_sql/application/use_cases/classify.py`, `trajectory.py`, `conflicts.py`, `friction.py` — each pipeline loads its watermark map and marks completed sessions.
- `src/claude_sql/infrastructure/adapters.py` — `build_checkpoint` wires the SQLite adapter to the port.

**Shape:**
```sql
CREATE TABLE IF NOT EXISTS session_checkpoint (
    session_id            TEXT NOT NULL,
    pipeline              TEXT NOT NULL,
    last_ts_processed     TEXT,
    last_mtime_processed  TEXT,
    completed_at          TEXT NOT NULL,
    PRIMARY KEY (session_id, pipeline)
);
```
```python
def load_as_map(self, pipeline: str) -> dict[str, tuple[datetime | None, datetime | None]]: ...
def mark_completed(self, *, pipeline: str,
                   rows: Iterable[tuple[str, datetime | None, datetime | None]]) -> int: ...
def count_rows(self) -> int: ...
```

**Assumptions consumers make:**
- One row per `(session_id, pipeline)`; the upsert is `INSERT ... ON CONFLICT DO UPDATE` (SQLite 3.24+) (`src/claude_sql/infrastructure/sqlite_state/checkpointer.py:4-6, 44-52`).
- All timestamps are stored as ISO-8601 UTC strings that sort lexicographically (`src/claude_sql/infrastructure/sqlite_state/checkpointer.py:19`).
- `pipeline` is one of `PIPELINE_NAMES = ("classify", "trajectory", "conflicts", "user_friction")` (`src/claude_sql/infrastructure/sqlite_state/checkpointer.py:42`).
- The staleness math (`filter_unchanged` / `_stale_or_equal`) is domain and stays out of the port — the port only loads and marks (`src/claude_sql/application/ports.py:170-173`).

**Drift risk:** Reusing a `pipeline` name for a differently-keyed unit (message uuid vs session id) mixes incompatible watermarks in one table. Mitigation: `unit_id` semantics are documented per pipeline; `retry_queue.py:9-11` records that `trajectory` keys on uuid while the others key on session id.

## `RetryQueuePort` + `retry_queue` DDL — the durable failed-unit queue

**Producer:** Port `src/claude_sql/application/ports.py:193`; backing DDL `src/claude_sql/infrastructure/sqlite_state/retry_queue.py:41`.

**Consumer(s):**
- `src/claude_sql/application/use_cases/trajectory.py`, `classify.py`, `conflicts.py`, `friction.py` — enqueue failures, drain due units, mark done.
- `src/claude_sql/infrastructure/adapters.py` — `build_retry_queue` wires the SQLite adapter.

**Shape:**
```sql
CREATE TABLE IF NOT EXISTS retry_queue (
    pipeline        TEXT    NOT NULL,
    unit_id         TEXT    NOT NULL,
    error           TEXT    NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    completed_at    TEXT,
    PRIMARY KEY (pipeline, unit_id)
);
```
```python
def enqueue(self, *, pipeline: str, unit_id: str, error: str) -> int: ...
def drain(self, *, pipeline: str, max_attempts: int = 5, limit: int | None = None) -> list[str]: ...
def mark_done(self, *, pipeline: str, unit_ids: Iterable[str]) -> int: ...
def pending_count(self, *, pipeline: str) -> int: ...
```

**Assumptions consumers make:**
- One row per `(pipeline, unit_id)`; `unit_id` is `session_id` for classify/conflicts/user_friction and the message `uuid` for trajectory (`src/claude_sql/infrastructure/sqlite_state/retry_queue.py:9-11`).
- Upsert-with-attempt-counter semantics: first failure → attempts=1, next_attempt_at = now + 2 min; retry → attempts += 1, delay = 2^attempts min capped at 60; success → `completed_at` stamped, row kept as audit trail (`src/claude_sql/infrastructure/sqlite_state/retry_queue.py:12-16, 38-39`).
- `drain` returns only units that are not done, under `max_attempts`, and due (`next_attempt_at <= now`) (`src/claude_sql/application/ports.py:205-207`).
- The exponential-backoff math (`_backoff_delta`) is domain; the port surface stays declarative (`src/claude_sql/application/ports.py:198-199`).
- Shares the same `~/.claude/state.db` file and schema-bootstrap lock as the checkpointer (`src/claude_sql/infrastructure/sqlite_state/retry_queue.py:17-19, 28-36`).

**Drift risk:** A `BedrockRefusalError` is terminal and must NOT be re-enqueued or it cycles forever; the workers clear the queue for refused units. Mitigation: the refusal path stamps a neutral placeholder and calls `mark_done` rather than `enqueue` (CLAUDE.md; `errors.py:48-60`).

## Other contracts

- **`Clock` port** (`src/claude_sql/application/ports.py:63`) — a one-method `now() -> datetime` seam over `datetime.now(UTC)` so checkpoint watermarks, retry backoff, and `classified_at` stamps are deterministic under test.
- **`TranscriptReaderPort`** (`src/claude_sql/application/ports.py:77`) — `session_messages` / `read_turn_text` / `session_bounds` / `session_ids`; the DuckDB connection and `read_json` glob are adapter state. `read_turn_text` honors the consumer collapse contract.
- **`SessionSearchPort`** (`src/claude_sql/application/ports.py:115`) — `search(...) -> list[SearchHit]` + `embed_query`; embedder and the Lance-backed view are adapter state.
- **`VectorStorePort`** (`src/claude_sql/application/ports.py:132`) — `add_chunk` / `ensure_index` / `optimize` / `count_rows` / `table_identity` / `get_embedded_uuids`; keeps the empty-namespace gate (absent table → `0` rows / `None` identity).
- **`CachePort`** (`src/claude_sql/application/ports.py:218`) — `write_part` / `read_all` / `count_rows` / `iter_part_files` / `replace_sessions` over the sharded-parquet artifact store; the parquet-existence gate (views register only caches that exist) is preserved by the adapter.
- **`ensure_store_matches`** (`src/claude_sql/domain/embedding_guard.py:22`) — the pure fail-loud guard signature `(*, stored_model, stored_dim, expected_model, expected_dim) -> None`; `None` on either stored value is a no-op (fresh install); raises `EmbeddingProviderMismatch` otherwise.
- **`ClassifiedError`** (`src/claude_sql/domain/errors.py:109`) — the frozen `(kind, exit_code, message, hint)` dataclass with `to_payload()`; the structured shape of a CLI error the non-TTY stderr JSON envelope carries.
- **`ClaudeSql` facade** (`src/claude_sql/composition.py:36`) — the importable entry point (`reader()` / `search()` / `query()`) plus the `build_*` factories; the surface downstream consumers import.

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 17 shared source citations
- [claude-sql · Sequences](../diagrams/behavioral/sequences.md) — 9 shared source citations
- [claude-sql · Public API](../reference/public-api.md) — 9 shared source citations
- [claude-sql · Module map](../architecture/module-map.md) — 8 shared source citations
- [claude-sql · Debugging guide](../insights/debugging-guide.md) — 8 shared source citations
