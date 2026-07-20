# claude-sql · Module map

`claude-sql` is one namespace root, `src/claude_sql/`, laid out as a strict hexagon — `interfaces` (drives) > `application` (ports + use-cases) > `infrastructure` (adapters) > `domain` (pure core) — with a single `import-linter` layers contract enforcing that dependency direction. The four layer packages plus the root `composition.py` facade are indexed below in dependency-flow order: the entry point first, the pure core last. Modules with fewer than three shortlisted files (`composition.py`, the root `__init__.py`) are collected under `## Supporting code`.

## interfaces

The `interfaces` layer is the only place the outside world touches — a thin cyclopts CLI that wires the `claude-sql` console script to its 17 subcommands and is the composition root that injects concrete adapters into use-cases (`src/claude_sql/interfaces/cli/app.py:1`). `main` is the script entry (`src/claude_sql/interfaces/cli/app.py:2115`); shared flags (`--format`, `--glob`, `--quiet`) live on a flattened `Common` dataclass, and DuckDB errors classify into parse/catalog/runtime exit codes 64/65/70 (`src/claude_sql/interfaces/cli/app.py:8`). `output.py` renders the `{auto,table,json,ndjson,csv}` formats that make every subcommand agent-legible (`src/claude_sql/interfaces/cli/output.py:1`), and `install_source.py` resolves the version/install banner (`src/claude_sql/interfaces/cli/install_source.py:1`).

- `src/claude_sql/interfaces/cli/app.py` (2121 LOC)
- `src/claude_sql/interfaces/cli/output.py` (197 LOC)
- `src/claude_sql/interfaces/cli/install_source.py` (78 LOC)
- `src/claude_sql/interfaces/cli/__init__.py` (13 LOC)
- `src/claude_sql/interfaces/__init__.py` (14 LOC)

## application

The `application` layer holds the port surface and the use-case orchestrations that compose domain math with injected adapters, depending inward on `domain` but never on `interfaces` (`src/claude_sql/application/use_cases/__init__.py:1`). `ports.py` declares nine `@runtime_checkable` Protocols — `TranscriptReaderPort`, `SessionSearchPort`, `VectorStorePort`, `CheckpointPort`, `RetryQueuePort`, `CachePort`, `ReaderPort`, plus a `Clock` — modeled on the existing concrete call shapes so wrapping each adapter is a lift, not a redesign (`src/claude_sql/application/ports.py:64`). `analyze.py` is the composite `embed → structure → LLM analytics` pipeline lifted verbatim out of the CLI, with `run_analyze` preserving the RFC §9.6 two-axis rebind lifecycle byte-for-byte (`src/claude_sql/application/analyze.py:99`). The four LLM workers (`trajectory`, `friction`, `conflicts`, `classify`) and the six structure/ingest use-cases live under `use_cases/`, and `prompts.py` centralizes their system prompts (`src/claude_sql/application/prompts.py:1`).

- `src/claude_sql/application/use_cases/trajectory.py` (903 LOC)
- `src/claude_sql/application/prompts.py` (717 LOC)
- `src/claude_sql/application/use_cases/friction.py` (674 LOC)
- `src/claude_sql/application/use_cases/conflicts.py` (434 LOC)
- `src/claude_sql/application/use_cases/community.py` (434 LOC)
- `src/claude_sql/application/use_cases/embed.py` (394 LOC)
- `src/claude_sql/application/analyze.py` (318 LOC)
- `src/claude_sql/application/ports.py` (306 LOC)

## infrastructure

The `infrastructure` layer holds every concrete adapter — DuckDB, LanceDB, SQLite, Bedrock, parquet — that satisfies an application port (`src/claude_sql/infrastructure/__init__.py:1`). Its center of gravity is `duckdb_views.py`, which registers the zero-copy JSONL readers, 25 SQL views, and 26 analytical macros plus the VSS binding via `register_all` (`src/claude_sql/infrastructure/duckdb_views.py:2285`). `bedrock/client.py` carries the boto3 Converse plumbing, prompt-cache accounting, and tenacity retry policy (`src/claude_sql/infrastructure/bedrock/client.py:1`); `transcript_reader.py` is the importable `TranscriptReaderPort` seam downstream consumers consume (`src/claude_sql/infrastructure/transcript_reader.py:1`); and `settings.py` is the `CLAUDE_SQL_`-prefixed pydantic `BaseSettings`, rehomed here because reading env is I/O (`src/claude_sql/infrastructure/settings.py:152`). The embedding/, llm_analytics/, and sqlite_state/ subpackages carry the pluggable-provider adapters and the checkpoint/retry-queue state store.

- `src/claude_sql/infrastructure/duckdb_views.py` (2394 LOC)
- `src/claude_sql/infrastructure/bedrock/client.py` (601 LOC)
- `src/claude_sql/infrastructure/transcript_reader.py` (559 LOC)
- `src/claude_sql/infrastructure/settings.py` (544 LOC)
- `src/claude_sql/infrastructure/sqlite_state/checkpointer.py` (379 LOC)
- `src/claude_sql/infrastructure/session_text_loader.py` (366 LOC)
- `src/claude_sql/infrastructure/duckdb_connection.py` (338 LOC)
- `src/claude_sql/infrastructure/lance_store.py` (300 LOC)

## domain

The `domain` layer is the innermost hexagon — pure, dependency-free business types that import nothing heavier than stdlib and pydantic, with no duckdb, polars, lancedb, or boto3 (`src/claude_sql/domain/__init__.py:1`). `models.py` holds the pydantic v2 structured-output classification schemas that are the LLM-analytics contract (`src/claude_sql/domain/models.py:1`); `ports.py` declares the two pluggable-provider Protocols, `EmbeddingProvider` and `LlmAnalyticsProvider` (`src/claude_sql/domain/ports.py:34`); `errors.py` owns the `EXIT_CODES` taxonomy and the terminal `RefusalError` (`src/claude_sql/domain/errors.py:26`); and `transcript.py` renders a session's timeline into byte-stable transcript text for the four LLM pipelines (`src/claude_sql/domain/transcript.py:1`). The `structure/` subpackage is the deliberate exception to the stdlib-only rule: it is I/O-free but lazy-imports numpy/igraph/sklearn/umap/hdbscan for the cluster/community/terms math (`src/claude_sql/domain/structure/__init__.py:1`).

- `src/claude_sql/domain/models.py` (398 LOC)
- `src/claude_sql/domain/transcript.py` (343 LOC)
- `src/claude_sql/domain/structure/community.py` (326 LOC)
- `src/claude_sql/domain/dedup.py` (224 LOC)
- `src/claude_sql/domain/trajectory.py` (209 LOC)
- `src/claude_sql/domain/errors.py` (149 LOC)
- `src/claude_sql/domain/friction.py` (119 LOC)
- `src/claude_sql/domain/ports.py` (114 LOC)

## Supporting code

- `src/claude_sql/composition.py` (185 LOC)
- `src/claude_sql/__init__.py` (24 LOC)

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 10 shared source citations
- [claude-sql · Contract map](../insights/contract-map.md) — 8 shared source citations
- [claude-sql · Debugging guide](../insights/debugging-guide.md) — 6 shared source citations
- [claude-sql · Public API](../reference/public-api.md) — 6 shared source citations
- [claude-sql · Sequences](../diagrams/behavioral/sequences.md) — 5 shared source citations
