# claude-sql · Impact analysis

This file answers one question for each of the codebase's most-connected surfaces: *if I touch X, what else do I have to think about?*

**High-impact surface** here means: a symbol (Protocol, class, function, or schema) selected by **inbound reference count** across `src/` and `tests/`, weighted toward the seams of the v2 hexagonal architecture. The hexagon puts the highest-fan-in symbols at two boundaries — the port Protocols in `application/ports.py` (every use-case depends on them) and the infrastructure registration/config surface (`Settings`, `register_all`) that every adapter reads. A third class is the **byte-parity contracts** (`render_turn_text`, `SessionTextCorpus.assemble`, the pydantic schemas): low file-count fan-in but high *behavioral* blast radius, because external consumers and checkpoint caches depend on exact output bytes. The eight surfaces below are the ones where a careless change breaks the most downstream code or silently corrupts cached artifacts.

Each `Type` cell is one of `direct import` / `indirect` / `runtime dispatch` / `test` / `config`. Each `Touch on change` cell is `yes` (must edit), `likely` (review even without a signature change), or `no` (only a behavioral change reaches it).

## The port Protocol surface (`application/ports.py`)

Defined at: `src/claude_sql/application/ports.py:63-306` (9 `@runtime_checkable` Protocols plus re-exported `EmbeddingProvider` / `LlmAnalyticsProvider` / `SearchHit` / `PortResult`).

This is the DIP seam of the whole reshape. Every use-case is typed against these Protocols; every infrastructure adapter must structurally satisfy them. A signature change to any method here ripples to both the adapter that implements it and the use-case that calls it.

| Downstream | Type | Touch on change | Citation |
| --- | --- | --- | --- |
| `infrastructure/adapters.py` (ParquetCache, SqliteCheckpoint, SqliteRetryQueue, LanceVectorStore) | direct import | yes | `src/claude_sql/infrastructure/adapters.py:54-222` |
| `application/use_cases/classify.py` (ReaderPort, CheckpointPort, RetryQueuePort, CachePort) | direct import | yes | `src/claude_sql/application/use_cases/classify.py:44` |
| `application/use_cases/trajectory.py` | direct import | yes | `src/claude_sql/application/use_cases/trajectory.py:75` |
| `application/use_cases/conflicts.py` | direct import | yes | `src/claude_sql/application/use_cases/conflicts.py:64` |
| `application/use_cases/friction.py` | direct import | yes | `src/claude_sql/application/use_cases/friction.py:72` |
| `application/use_cases/embed.py` (VectorStorePort) | direct import | yes | `src/claude_sql/application/use_cases/embed.py:47` |
| `application/use_cases/cluster.py` (VectorStorePort) | direct import | yes | `src/claude_sql/application/use_cases/cluster.py:35` |
| `application/use_cases/ingest.py` (CachePort) | direct import | yes | `src/claude_sql/application/use_cases/ingest.py:50` |
| `composition.py` build_* factories (return the port types) | direct import | likely | `src/claude_sql/composition.py:24-32` |
| `tests/test_ports.py`, `tests/test_port_wiring.py` (import-discipline + structural conformance) | test | yes | `tests/test_port_wiring.py:145` |

Blast-radius notes:
- **`ReaderPort` is stateful, not a stateless query executor.** Its docstring pins the register→write→rebind lifecycle: a use-case that writes parquet or LanceDB mid-run (`embed`, `cluster`) must call `refresh_analytics_views()` / `rebind_vss(stage)` before its next read, or it silently reads 0 rows (the RFC §9.6 stale-connection bug). Do not collapse these methods into a bare `query()`. `src/claude_sql/application/ports.py:248-290`
- **`ports.py` may import only stdlib, `typing`, `returns`, and the pure `domain` modules** (`domain.errors`, `domain.ports`, `domain.retrieval`); heavy types (`duckdb`, `pl.DataFrame`, `datetime`) are `TYPE_CHECKING`-only. Adding a top-level heavy import breaks the layering test. `src/claude_sql/application/ports.py:16-51`
- **Every port method returning a container is contractually total on empty state** — e.g. `VectorStorePort.count_rows()` returns `0` and `table_identity()` returns `None` for an empty store rather than raising (the empty-namespace gate). Changing a method to raise on empty breaks callers that rely on the sentinel. `src/claude_sql/application/ports.py:153-159`

## `Settings` (`infrastructure/settings.py`)

Defined at: `src/claude_sql/infrastructure/settings.py:152` (`class Settings(BaseSettings)`).

The single environment-driven config object, threaded into nearly every adapter and use-case. Adding or renaming a field, or changing a derived method, touches the widest set of files in the repo (~36 src files reference the name).

| Downstream | Type | Touch on change | Citation |
| --- | --- | --- | --- |
| `infrastructure/duckdb_views.py` (`register_all(settings=...)`, glob + identity) | direct import | yes | `src/claude_sql/infrastructure/duckdb_views.py:2285-2333` |
| `infrastructure/duckdb_connection.py` (PRAGMA tuning, glob binding) | direct import | yes | `src/claude_sql/infrastructure/duckdb_connection.py:171` |
| `infrastructure/session_search.py`, `session_text_loader.py` | direct import | yes | `src/claude_sql/infrastructure/session_text_loader.py:49` |
| `infrastructure/embedding/cohere_bedrock.py` (`output_dimension`) | direct import | yes | `src/claude_sql/infrastructure/settings.py:232` |
| `application/use_cases/embed.py` (`expected_embedding_identity`) | direct import | yes | `src/claude_sql/application/use_cases/embed.py:318` |
| `composition.py` / `interfaces/cli/app.py` (composition root) | direct import | yes | `src/claude_sql/composition.py:73-78` |
| `adapters.py` build_checkpoint/build_retry_queue/build_vector_store (take `Settings`) | direct import | yes | `src/claude_sql/infrastructure/adapters.py:230-240` |
| `tests/test_config.py`, `test_config_dir.py` (env override, relocated CONFIG_DIR) | test | likely | `tests/test_config_dir.py:194` |

Blast-radius notes:
- **`output_dimension` (Literal 256/512/1024/1536) is the head of the dimension contract.** It feeds `expected_embedding_identity()` (`settings.py:475-493`), which threads into the Lance fixed-size vector schema, the `register_vss` `FLOAT[dim]` view cast, and the query-time cast. Changing it after a store is written triggers the fail-loud guard — it requires a full re-embed, not a config edit. `src/claude_sql/infrastructure/settings.py:232`
- **The pure value-object projections are the domain firewall.** `transcript_caps()` (:539) hands `domain.transcript` a two-int `TranscriptCaps` slice, and `clustering_config()`/`community_config()`/`terms_config()` (:502-539) hand `domain.structure` their frozen VOs, so domain never imports the god-`Settings`. Widening what a projection returns must not leak infrastructure types into domain. `src/claude_sql/infrastructure/settings.py:502-539`
- **`Settings` lives in `infrastructure`, not `domain`** — importing it into a `domain` or `application` module top-level violates the layering contract (`ports.py` and `composition.py` only reference it under `TYPE_CHECKING`). `src/claude_sql/composition.py:33`

## `render_turn_text` (`domain/transcript.py`)

Defined at: `src/claude_sql/domain/transcript.py:287-336`.

A pure function that collapses raw message-envelope rows into the downstream consumer's transcript text — **one line per message**, with a specific ordering (`(ts, kind_rank, uuid)`), inline `[tool_use:name]`/`[tool_result]` markers, and cap behavior. It is the drop-in retrieval contract downstream consumers depend on, so its output must stay **byte-identical**.

| Downstream | Type | Touch on change | Citation |
| --- | --- | --- | --- |
| `infrastructure/transcript_reader.py` `read_turn_text` (sole src caller) | direct import | yes | `src/claude_sql/infrastructure/transcript_reader.py:543` |
| downstream consumers (external repos consuming the collapse contract) | indirect | likely | `src/claude_sql/domain/transcript.py:184-194` |
| `tests/test_collapse_parity.py` (ported-fixture byte-parity proof) | test | yes | `tests/test_collapse_parity.py:176` |
| `tests/test_transcript_domain.py` (ordering, folding, caps, determinism) | test | yes | `tests/test_transcript_domain.py:87-269` |
| `tests/property/test_transcript_properties.py` (shuffle-invariance, idempotence, caps) | test | yes | `tests/property/test_transcript_properties.py:62-176` |

Blast-radius notes:
- **Byte-parity with the consumer's collapse routine is a hard acceptance gate.** `test_collapse_parity.py` feeds ported fixtures straight to `render_turn_text` and asserts exact string equality. Any change to line shape, ordering, marker text, or the per-turn `` …`` ellipsis (note the leading space) breaks parity even if it "looks equivalent." `src/claude_sql/domain/transcript.py:296-313`
- **Do not conflate with `SessionTextCorpus.assemble`.** That is a *separate* byte-stable contract for the four LLM pipelines (classify/trajectory/conflicts/friction), with a different ordering (block-level `_KIND_RANK`, not envelope `_COLLAPSE_KIND_RANK`) and different header/marker shapes. The two must not be unified. `src/claude_sql/domain/transcript.py:41-49, 126-178`
- **The function is Settings-free by design** — caps arrive as keyword args (`per_turn_chars`, `total_chars`, `truncation_notice`), and it uses the RAW timestamp string verbatim (no TIMESTAMP round-trip). Reintroducing a `Settings` dependency or reformatting the timestamp breaks the pure-domain guarantee and parity. `src/claude_sql/domain/transcript.py:314-336`

## `ClaudeSql` facade + `build_*` factories (`composition.py`)

Defined at: `src/claude_sql/composition.py:36` (`class ClaudeSql`) and `:125-168` (`build_reader` / `build_search` / `build_cache` / `build_checkpoint` / `build_retry_queue` / `build_vector_store`).

The one importable composition root — `from claude_sql import ClaudeSql` (or `from claude_sql.composition import ClaudeSql`). It is the public surface for downstream consumers, so its shape and lazy-import discipline are external contracts.

| Downstream | Type | Touch on change | Citation |
| --- | --- | --- | --- |
| `claude_sql/__init__.py` (re-export via lazy `__getattr__`) | direct import | yes | `src/claude_sql/__init__.py:16-23` |
| `infrastructure/adapters.py` build_* (the real constructors these delegate to) | indirect | likely | `src/claude_sql/infrastructure/adapters.py:225-240` |
| `infrastructure/duckdb_views.py` `register_all` (called by `ClaudeSql.query`) | indirect | likely | `src/claude_sql/composition.py:115-119` |
| downstream consumers (external importers) | runtime dispatch | likely | `src/claude_sql/composition.py:1-6` |
| `tests/test_retrieval_seam.py` (facade + port wiring, no-duckdb-on-import proof) | test | yes | `tests/test_retrieval_seam.py:405-453` |

Blast-radius notes:
- **Zero heavy imports at module top.** `import claude_sql.composition` (and `import claude_sql`) must stay sub-millisecond and dependency-free — every duckdb/polars/adapter import lives inside a method or factory, and port-typed annotations are `TYPE_CHECKING`-only. Adding a top-level heavy import regresses the cold-import cost the siblings depend on. `src/claude_sql/composition.py:8-33`
- **`ClaudeSql.query` binds through `register_all`, not the CLI's private `_open_connection_full`.** This keeps `composition.py` decoupled from the `interfaces` layer while giving identical full-registration semantics. Do not repoint it at a CLI helper. `src/claude_sql/composition.py:107-119`
- **`reader()` / `search()` are lazily built and cached** — construction is cheap and the DuckDB connections are not built until first accessor call. A change making construction eager breaks the "cheap to instantiate" contract. `src/claude_sql/composition.py:80-96`

## `register_all` + the registration chain (`infrastructure/duckdb_views.py`)

Defined at: `src/claude_sql/infrastructure/duckdb_views.py:2285` (`register_all`); sub-steps `register_raw` :490, `register_views` :638, `register_macros` :1244, `register_vss` :1824, `register_analytics` :2000.

The stable infrastructure entrypoint that materializes 18 views + 14 macros + VSS against an open DuckDB connection. Every path that needs a query-ready connection goes through here; its ordering constraints and parameters are load-bearing.

| Downstream | Type | Touch on change | Citation |
| --- | --- | --- | --- |
| `interfaces/cli/app.py` (`_open_connection_full`) | direct import | yes | `src/claude_sql/interfaces/cli/app.py:491` |
| `infrastructure/duckdb_connection.py` (shared connection builder) | direct import | yes | `src/claude_sql/infrastructure/duckdb_connection.py:171` |
| `composition.py` `ClaudeSql.query` | direct import | yes | `src/claude_sql/composition.py:119` |
| `domain/embedding_guard.py` `ensure_store_matches` (called inside `register_vss`) | indirect | likely | `src/claude_sql/infrastructure/duckdb_views.py:1941` |
| `tests/test_config_dir.py`, `test_team_corpus.py`, `test_s3_source.py` | test | yes | `tests/test_config_dir.py:232`, `tests/test_team_corpus.py:277`, `tests/test_s3_source.py:328` |
| `tests/test_pr3_perf.py` (monkeypatches `register_all` as the cold-start sentinel) | test | likely | `tests/test_pr3_perf.py:122-125` |

Blast-radius notes:
- **Registration order is a hard invariant.** `register_vss` must run before `register_macros` (the `semantic_search` macro body references the `message_embeddings` table, resolved at macro-creation time), and `register_analytics` must also precede `register_macros` (analytics macros bind against analytics views at creation). Reordering silently breaks macro binding. `src/claude_sql/infrastructure/duckdb_views.py:2320-2333`
- **`skip_vss=True` must skip both `register_vss` and the `semantic_search` macro**, or the macro references a table that was never bound. The two flags travel together. `src/claude_sql/infrastructure/duckdb_views.py:2308-2319`
- **Missing parquet is a warn-and-skip, never a raise.** Views register only the caches that exist; adding a new analytics view must preserve this so a fresh install doesn't crash. `src/claude_sql/infrastructure/duckdb_views.py:2000` (register_analytics)

## `ensure_store_matches` — the dimension guard (`domain/embedding_guard.py`)

Defined at: `src/claude_sql/domain/embedding_guard.py:22-61`.

The fail-loud provider/dimension guard: a pure rule that refuses to bind or write a vector store whose stamped `(model, dim)` differs from the active embedder. Small surface, high stakes — the alternative is silent kNN corruption.

| Downstream | Type | Touch on change | Citation |
| --- | --- | --- | --- |
| `infrastructure/duckdb_views.py` `register_vss` (read/bind path) | direct import | yes | `src/claude_sql/infrastructure/duckdb_views.py:1941` |
| `infrastructure/embedding/__init__.py` (re-export for the write path) | direct import | yes | `src/claude_sql/infrastructure/embedding/__init__.py:26` |
| `application/use_cases/embed.py` (`store.table_identity()` before backfill) | indirect | likely | `src/claude_sql/application/use_cases/embed.py:318` |
| `infrastructure/lance_store.py` / `adapters.py` `table_identity` (supplies the stored side) | indirect | likely | `src/claude_sql/infrastructure/lance_store.py:245`, `adapters.py:206` |
| `domain/errors.py` `EmbeddingProviderMismatch` (the exception raised) | direct import | yes | `src/claude_sql/domain/errors.py:80` |
| `tests/test_embedding_providers.py` (guard raise + no-op paths) | test | yes | `tests/test_embedding_providers.py:251-283` |

Blast-radius notes:
- **`model_id` is the primary identity; `dim` is checked only when `expected_dim` is supplied.** Cohere is the one provider whose single `model_id` emits multiple Matryoshka widths, so dim is checked there; probe-only providers (ollama/onnx) pass `expected_dim=None` and trust `model_id` alone. Changing the identity-comparison logic must preserve this asymmetry. `src/claude_sql/domain/embedding_guard.py:38-50`
- **`None` on either stored field means empty store — the guard is a no-op** so a fresh install lets any provider claim the store. Making the guard raise on `None` would break first-embed. `src/claude_sql/domain/embedding_guard.py:47-48`
- **This is pure domain (string/int compare, no I/O).** Both the write path (`embed.run_backfill`) and the read path (`register_vss`) must call it; dropping either call reopens the silent-corruption hole. `src/claude_sql/domain/embedding_guard.py:1-15`

## `EXIT_CODES` + the error taxonomy (`domain/errors.py`)

Defined at: `src/claude_sql/domain/errors.py:26` (`EXIT_CODES`) plus `DomainError` :37, `RefusalError` :48, `BedrockRefusalError` :63, `EmbeddingProviderMismatch` :80, `LlmAnalyticsUnavailable` :94, `ClassifiedError` :110, `InputValidationError` :128.

The agent-facing CLI contract: DuckDB errors map to stable exit codes (parse→64, catalog→65, runtime→70) and structured stderr JSON. Changing a code or class hierarchy breaks the documented agent-subprocess surface.

| Downstream | Type | Touch on change | Citation |
| --- | --- | --- | --- |
| `interfaces/cli/output.py` (`EXIT_CODES`, `ClassifiedError`, `InputValidationError`) | direct import | yes | `src/claude_sql/interfaces/cli/output.py:28` |
| `infrastructure/duckdb_errors.py` `classify_duckdb_error` | direct import | yes | `src/claude_sql/infrastructure/duckdb_errors.py:15-42` |
| `domain/embedding_guard.py` (raises `EmbeddingProviderMismatch`) | direct import | yes | `src/claude_sql/domain/embedding_guard.py:19` |
| LLM worker use-cases (catch `BedrockRefusalError` / `RefusalError` as terminal) | indirect | likely | `src/claude_sql/domain/errors.py:48-75` |
| error-path tests (`test_output.py` exit-code mapping, refusal handling) | test | likely | `tests/test_output.py:12-90` |

Blast-radius notes:
- **`RefusalError` is terminal by contract.** Workers stamp a neutral placeholder row and clear the retry queue on it, so refused units don't cycle forever. `BedrockRefusalError` subclasses `RefusalError` so existing `except`/`isinstance` sites keep working — do not flatten the hierarchy. `src/claude_sql/domain/errors.py:48-75`
- **The exit-code numbers are a published agent contract** (64/65/70). `classify_duckdb_error` must stay in sync with new DuckDB exception subclasses, and the numbers must not drift. `src/claude_sql/infrastructure/duckdb_errors.py:18-42`

## The pydantic LLM schemas (`domain/models.py`)

Defined at: `src/claude_sql/domain/models.py:25` (`SessionClassification`), `:189` (`TrajectoryArrayResult`), `:299` (`ConflictsResult`), `:323` (`UserFrictionSignal`).

The structured-output schemas for the four LLM pipelines. They are flattened into Bedrock JSON schemas and echoed in prompts, so a field change ripples to the schema flattener, the prompts, the worker parse logic, and the parquet cache shape.

| Downstream | Type | Touch on change | Citation |
| --- | --- | --- | --- |
| `infrastructure/bedrock/structured_output.py` (`_bedrock_schema` flatten) | direct import | yes | `src/claude_sql/infrastructure/bedrock/structured_output.py:24-27, 117-120` |
| `application/prompts.py` (prompt bodies echo the schema's label semantics; no import, docstring-coupled) | indirect | likely | `src/claude_sql/application/prompts.py:3` |
| `application/use_cases/trajectory.py` (`schema=TrajectoryArrayResult`) | direct import | yes | `src/claude_sql/application/use_cases/trajectory.py:552` |
| analytics parquet caches + `duckdb_views.py` analytics views (column shape) | indirect | likely | `src/claude_sql/infrastructure/duckdb_views.py:2000` |
| `tests/test_schemas.py`, `test_trajectory_windowed.py`, `test_conflicts_storage_v2.py`, `test_llm_analytics.py` | test | yes | `tests/test_schemas.py:8-12` |

Blast-radius notes:
- **A schema field rename/removal changes the parquet column shape**, which the analytics views and downstream macros read. Stale shards from a prior schema are detected via parquet metadata and deleted on first run — but the view SQL and the `sentiment_arc`/`conflicts_summary` macros must be updated in lockstep. `src/claude_sql/domain/models.py:189-321`
- **`output_config.format` structured output rejects parts of the JSON-Schema draft**, so `_bedrock_schema` inlines `$ref`, injects `additionalProperties: false`, and strips numeric/string constraints. Adding a constrained field (regex, min/max) without matching the flattener's stripping breaks the Bedrock call. `src/claude_sql/infrastructure/bedrock/structured_output.py:117-120`

## Other notable surfaces

- **`EmbeddingProvider` / `LlmAnalyticsProvider` Protocols** (`src/claude_sql/domain/ports.py`) — the pluggable-provider seam; implemented by `infrastructure/embedding/{cohere_bedrock,ollama,onnx_bge}.py` and `infrastructure/llm_analytics/{sonnet_bedrock,strands_luna}.py`. 15 / 9 inbound references; a method-signature change touches every adapter under those two packages. Re-exported through `application/ports.py`, so covered by the port-surface section above.
- **`SearchHit`** (`src/claude_sql/domain/retrieval.py`) — the pure value-object returned by `SessionSearchPort.search`; consumed by `infrastructure/session_search.py`, `composition.py`, and CLI output. A field change touches the search adapter and the JSON/table formatters.
- **`interfaces/cli/app.py`** (cyclopts app) — the top-level command surface and composition root; imports every use-case via lazy in-function imports (`app.py:903-2066`). High fan-out rather than fan-in: it is the thing that changes when a subcommand's contract changes, not a thing others import.
- **`infrastructure/lance_store.py`** — the LanceDB read/write module behind `VectorStorePort`; `table_identity` (:245) is the source of the guard's stored side. Covered under the dimension-guard section.

## See also

- [claude-sql · Contract map](../insights/contract-map.md) — 17 shared source citations
- [claude-sql · Module map](../architecture/module-map.md) — 10 shared source citations
- [claude-sql · Processes](../behavior/processes.md) — 10 shared source citations
- [claude-sql · Debugging guide](../insights/debugging-guide.md) — 10 shared source citations
- [claude-sql · Sequences](../diagrams/behavioral/sequences.md) — 9 shared source citations
