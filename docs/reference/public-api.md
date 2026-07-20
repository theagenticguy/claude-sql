# claude-sql · Public API

`claude-sql` is a CLI-and-library tool. This page documents the importable Python surface after the v2 hexagonal reshape; the command-line surface lives in `reference/cli.md`. The one convenience re-export is `claude_sql.ClaudeSql`; the full port + domain surface is imported from `claude_sql.composition`, `claude_sql.application.ports`, and the `claude_sql.domain.*` modules. There is no HTTP surface — this is not a server.

### ClaudeSql

```py
class ClaudeSql:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
        config_root: str | Path | None = None,
    ):
```

The composition facade: lazily builds and caches a transcript reader and semantic-search port, and exposes a `query` convenience for ad-hoc SQL over the fully-registered corpus. This is the one importable entry point (`from claude_sql import ClaudeSql`).
`src/claude_sql/composition.py:36`

### render_turn_text

```py
def render_turn_text(
    rows: Iterable[TranscriptRow],
    *,
    per_turn_chars: int = 8_000,
    total_chars: int = 200_000,
    truncation_notice: bool = False,
) -> str:
```

Pure, deterministic port of the downstream consumer's `_collapse` contract that folds raw message-envelope rows into one transcript line per message with inline `[tool_use:name]` / `[tool_result]` markers.
`src/claude_sql/domain/transcript.py:287`

### TranscriptReaderPort

```py
@runtime_checkable
class TranscriptReaderPort(Protocol):
```

Port for reading assembled transcript text and session structure (per-session messages, `read_turn_text`, session bounds, session ids); the DuckDB connection and `read_json` glob are adapter state.
`src/claude_sql/application/ports.py:78`

### SessionSearchPort

```py
@runtime_checkable
class SessionSearchPort(Protocol):
```

Port for semantic top-k search over message embeddings, wrapping the `search` command's embed-query + cosine-kNN path.
`src/claude_sql/application/ports.py:116`

### VectorStorePort

```py
@runtime_checkable
class VectorStorePort(Protocol):
```

Port for the LanceDB embeddings store read/write + kNN seam, preserving the empty-namespace gate (an absent table reports `0` rows / `None` identity rather than raising).
`src/claude_sql/application/ports.py:133`

### CheckpointPort

```py
@runtime_checkable
class CheckpointPort(Protocol):
```

Port for the per-`(session_id, pipeline)` processing watermark backed by the SQLite `state.db`; the staleness math stays in the domain.
`src/claude_sql/application/ports.py:167`

### RetryQueuePort

```py
@runtime_checkable
class RetryQueuePort(Protocol):
```

Port for the durable retry queue of failed LLM units of work; the SQLite path and clock are adapter state and the exponential-backoff math is domain.
`src/claude_sql/application/ports.py:194`

### CachePort

```py
@runtime_checkable
class CachePort(Protocol):
```

Port for the sharded-parquet artifact store of one worker cache, preserving the parquet-existence gate whereby views register only caches that exist.
`src/claude_sql/application/ports.py:219`

### ReaderPort

```py
@runtime_checkable
class ReaderPort(Protocol):
```

Port for the DuckDB query seam with its register→write→rebind lifecycle made explicit, so use-cases that write mid-run re-bind views instead of reintroducing the analyze stale-connection bug.
`src/claude_sql/application/ports.py:249`

### Clock

```py
@runtime_checkable
class Clock(Protocol):
```

Port over `datetime.now(UTC)` so checkpoint watermarks, retry backoff, and `classified_at` stamps are deterministic under test.
`src/claude_sql/application/ports.py:64`

### EmbeddingProvider

```py
@runtime_checkable
class EmbeddingProvider(Protocol):
```

Provider port that turns text into vectors (one adapter per backend), exposing `model_id` / `provider` / `dimension` identity plus async `embed_documents` and sync `embed_query`.
`src/claude_sql/domain/ports.py:34`

### LlmAnalyticsProvider

```py
@runtime_checkable
class LlmAnalyticsProvider(Protocol):
```

Provider port for one structured-output classification call — a system prompt, a user prompt, and a Pydantic schema in; a validated instance out.
`src/claude_sql/domain/ports.py:79`

### SearchHit

```py
@dataclass(frozen=True, slots=True)
class SearchHit:
```

The typed row a `SessionSearchPort` returns: uuid, session_id, timestamp, role, snippet, and cosine similarity, so a use-case gets a typed value instead of a raw DataFrame row.
`src/claude_sql/domain/retrieval.py:24`

### SessionTextCorpus

```py
@dataclass(slots=True)
class SessionTextCorpus:
```

An in-memory corpus of per-session timelines built with one glob scan; its `assemble` method renders one session as a single byte-stable transcript for the four LLM pipelines.
`src/claude_sql/domain/transcript.py:112`

### TranscriptRow

```py
@dataclass(slots=True)
class TranscriptRow:
```

One raw message-envelope row (uuid, type, timestamp, message) — the message-grained input to `render_turn_text`.
`src/claude_sql/domain/transcript.py:198`

### PortResult

```py
type PortResult[T] = Result[T, DomainError]
```

The uniform result type crossing a port boundary: a `Success[T]` carries the value, a `Failure[DomainError]` carries a domain error.
`src/claude_sql/application/ports.py:60`

### build_reader

```py
def build_reader(
    settings: Settings | None = None, *, config_root: str | Path | None = None
) -> TranscriptReaderPort:
```

Construct a standalone `TranscriptReaderPort` adapter.
`src/claude_sql/composition.py:125`

### build_search

```py
def build_search(
    settings: Settings | None = None, *, embedder: EmbeddingProvider | None = None
) -> SessionSearchPort:
```

Construct a standalone `SessionSearchPort` adapter.
`src/claude_sql/composition.py:134`

### build_cache

```py
def build_cache(target: Path) -> CachePort:
```

Construct the default `CachePort` over a parquet cache `target`.
`src/claude_sql/composition.py:143`

### build_checkpoint

```py
def build_checkpoint(settings: Settings) -> CheckpointPort:
```

Construct the default `CheckpointPort` over the SQLite `state.db`.
`src/claude_sql/composition.py:150`

### build_retry_queue

```py
def build_retry_queue(settings: Settings) -> RetryQueuePort:
```

Construct the default `RetryQueuePort` over the SQLite `state.db`.
`src/claude_sql/composition.py:157`

### build_vector_store

```py
def build_vector_store(settings: Settings, *, dim: int | None = None) -> VectorStorePort:
```

Construct the default `VectorStorePort` over `settings.lance_uri`.
`src/claude_sql/composition.py:164`

### build_embedder

```py
def build_embedder(settings: Settings) -> EmbeddingProvider:
```

Return the `EmbeddingProvider` adapter selected by `settings.embedding_provider`, importing each backend's heavy dependency lazily inside its branch.
`src/claude_sql/infrastructure/embedding/__init__.py:34`

### estimate_cost

```py
def estimate_cost(
    n_items: int,
    avg_in_tokens: int,
    avg_out_tokens: int,
    pricing: tuple[float, float],
) -> float:
```

Pure back-of-envelope dollar estimate for `n_items` classification calls given `(input_rate, output_rate)` in $/MTok — the arithmetic every `--dry-run` path uses.
`src/claude_sql/domain/costs.py:21`

### ensure_store_matches

```py
def ensure_store_matches(
    *,
    stored_model: str | None,
    stored_dim: int | None,
    expected_model: str,
    expected_dim: int | None,
) -> None:
```

The fail-loud embedding provider/dimension guard: raises `EmbeddingProviderMismatch` when a Lance store's stamped `(model, dim)` differs from the active embedder's identity.
`src/claude_sql/domain/embedding_guard.py:22`

### classify_duckdb_error

```py
def classify_duckdb_error(exc: duckdb.Error) -> ClassifiedError:
```

Map a concrete `duckdb.Error` onto a stable `ClassifiedError` kind + exit code (parse 64, catalog 65, runtime 70).
`src/claude_sql/infrastructure/duckdb_errors.py:18`

### run_analyze

```py
def run_analyze(
    settings: Settings,
    *,
    since_days: int | None = 30,
    limit: int | None = None,
    dry_run: bool = True,
    no_thinking: bool = False,
    skip_ingest: bool = False,
    skip_embed: bool = False,
    skip_classify: bool = False,
    skip_trajectory: bool = False,
    skip_conflicts: bool = False,
    skip_friction: bool = False,
    skip_cluster: bool = False,
    skip_community: bool = False,
    skip_skills_sync: bool = False,
    force_cluster: bool = False,
    force_community: bool = False,
    open_connection: Callable[..., duckdb.DuckDBPyConnection] | None = None,
    refresh_fn: Callable[[duckdb.DuckDBPyConnection, Settings], None] | None = None,
    rebind_fn: Callable[..., None] | None = None,
) -> dict[str, Any]:
```

Run the full analytics pipeline end-to-end (embed → structure → LLM analytics), honoring `dry_run` on every spending stage.
`src/claude_sql/application/analyze.py:99`

### register_all

```py
def register_all(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings | None = None,
    include_analytics: bool = True,
    skip_vss: bool = False,
) -> None:
```

Register raw views, derived views, VSS, analytics, and macros in order against an open DuckDB connection — the stable full-registration entrypoint the facade's `query` uses.
`src/claude_sql/infrastructure/duckdb_views.py:2285`

### DuckDbTranscriptReader

```py
class DuckDbTranscriptReader:
    def __init__(self, settings: Settings | None = None, *, config_root: str | Path | None = None):
```

The DuckDB adapter implementing `TranscriptReaderPort`; the `read_json` glob triple is adapter state derived from `settings` or an explicit `config_root`.
`src/claude_sql/infrastructure/transcript_reader.py:250`

### DuckDbSessionSearch

```py
class DuckDbSessionSearch:
    def __init__(
        self, settings: Settings | None = None, *, embedder: EmbeddingProvider | None = None
    ):
```

The DuckDB + Lance adapter implementing `SessionSearchPort`; the embedder and bound `message_embeddings` view are adapter state opened lazily on first use.
`src/claude_sql/infrastructure/session_search.py:38`

## See also

- [claude-sql · Contract map](../insights/contract-map.md) — 9 shared source citations
- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 8 shared source citations
- [claude-sql · Module map](../architecture/module-map.md) — 6 shared source citations
- [claude-sql · Sequences](../diagrams/behavioral/sequences.md) — 5 shared source citations
- [claude-sql · Debugging guide](../insights/debugging-guide.md) — 4 shared source citations
