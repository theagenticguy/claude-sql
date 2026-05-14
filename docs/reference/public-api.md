# claude-sql · Public API

`claude-sql` is a CLI tool (entry point `claude-sql = "claude_sql.cli:main"` at `pyproject.toml:55-56`); the package's `__init__.py` re-exports only `__version__` (`src/claude_sql/__init__.py:5`). The "public API" below is the in-process surface that `claude_sql.cli` and downstream callers consume — module-level non-underscore names imported across modules. Symbols are grouped by role; signatures are quoted verbatim from source.

## Configuration

### Settings

```py
class Settings(BaseSettings):
    """Environment-driven settings for claude-sql.

    All fields are overridable via env vars prefixed ``CLAUDE_SQL_`` (e.g.
    ``CLAUDE_SQL_REGION=us-west-2``) or via ``.env`` in the working directory.
    """
```

Environment-driven settings root for every worker, view, and CLI subcommand; every field is overridable via a `CLAUDE_SQL_`-prefixed env var.

`src/claude_sql/config.py:127`

## Pipeline entry points

### classify_sessions

```py
def classify_sessions(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
) -> int | dict[str, Any]:
```

Classify pending sessions and return count of successful classifications (or a plan dict in `--dry-run` mode).

`src/claude_sql/classify_worker.py:195`

### detect_conflicts

```py
def detect_conflicts(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
) -> int | dict[str, Any]:
```

Detect stance conflicts per session and return the count of sessions processed (the v1.0 pair-keyed pipeline).

`src/claude_sql/conflicts_worker.py:286`

### detect_user_friction

```py
def detect_user_friction(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
) -> int | dict[str, Any]:
```

Classify short user messages for friction signals (status_ping, unmet_expectation, confusion, interruption, correction, frustration, none).

`src/claude_sql/friction_worker.py:655`

### run_clustering

```py
def run_clustering(settings: Settings, *, force: bool = False) -> dict[str, int]:
```

Run UMAP + HDBSCAN on the embeddings parquet and return `{"total", "clusters", "noise"}`.

`src/claude_sql/cluster_worker.py:50`

### run_communities

```py
def run_communities(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    force: bool = False,
    gamma: float | None = None,
    resolution: ResolutionLevel = "medium",
) -> dict[str, int | float | str]:
```

Run Leiden+CPM on session centroids and write the primary communities parquet (plus the optional resolution-profile sidecar).

`src/claude_sql/community_worker.py:472`

### run_backfill

```py
async def run_backfill(
    *,
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> int | dict[str, Any]:
```

Discover unembedded messages, embed them via Cohere Embed v4 on Bedrock, and append vectors to the LanceDB embeddings store.

`src/claude_sql/embed_worker.py:391`

### embed_query

```py
def embed_query(text: str, *, settings: Settings) -> list[float]:
```

Embed a single query string for HNSW nearest-neighbor search (forces `embedding_type="float"` regardless of settings).

`src/claude_sql/embed_worker.py:360`

### trajectory_messages

```py
def trajectory_messages(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
) -> int | dict[str, Any]:
```

Per-session windowed sentiment + transition classification; returns the count of windows written (or a plan dict in `--dry-run`).

`src/claude_sql/trajectory_worker.py:933`

### run_terms

```py
def run_terms(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    force: bool = False,
) -> dict[str, int]:
```

Compute c-TF-IDF top terms per cluster and write the `cluster_terms` parquet output.

`src/claude_sql/terms_worker.py:29`

### generate_review_sheet

```py
def generate_review_sheet(
    con: duckdb.DuckDBPyConnection | None,
    settings: Settings,
    *,
    commit_sha: str,
    transcript_uri_override: str | None = None,
    dry_run: bool = True,
    no_thinking: bool = False,
) -> dict[str, Any]:
```

Produce a structured PR review sheet for a merged commit by binding the commit SHA to its transcript and invoking Sonnet 4.6.

`src/claude_sql/review_sheet_worker.py:340`

## Ingest and bootstrap

### stamp_messages

```py
def stamp_messages(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    batch_size: int = 4096,
) -> int:
```

Stamp every `messages_text` row not yet present in `ingest_stamps` with `approx_tokens`, `simhash64`, and a token-budget bucket.

`src/claude_sql/ingest.py:295`

### resolve_canonicals

```py
def resolve_canonicals(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    refresh_view: bool = True,
) -> int:
```

Populate `canonical_uuid` via a DuckDB SQL self-join over `ingest_stamps`, picking the earliest-seen row whose simhash differs by ≤ 3 bits.

`src/claude_sql/ingest.py:410`

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

Register raw views, derived views, VSS, analytics, and macros on a DuckDB connection in the correct order.

`src/claude_sql/sql_views.py:2081`

### claude_sql_home

```py
def claude_sql_home() -> Path:
```

Return (and create on first call) the parent directory for every claude-sql derived cache, resolved from `CLAUDE_SQL_HOME`, platform default, or `XDG_DATA_HOME`.

`src/claude_sql/home.py:51`

## LLM-shared primitives

### classify_one

```py
async def classify_one(
    client: Any,
    model_id: str,
    schema: dict,
    text: str,
    *,
    max_tokens: int,
    thinking_mode: str,
    sem: asyncio.Semaphore | anyio.CapacityLimiter,
    system: str | None = None,
    pipeline: str = "classifier",
) -> dict:
```

Run one Bedrock structured-output classification call under a concurrency limiter, dispatching the blocking `invoke_model` to `anyio.to_thread.run_sync`.

`src/claude_sql/llm_shared.py:563`

### pipeline_cache_stats

```py
@contextmanager
def pipeline_cache_stats(pipeline: str) -> Iterator[None]:
```

Context manager that resets, accumulates, then emits-and-clears the per-pipeline Bedrock cache-stat bucket so each run logs one summary line.

`src/claude_sql/llm_shared.py:259`

### BedrockRefusalError

```py
class BedrockRefusalError(Exception):
    """Bedrock declined to classify the input under its content policy.

    Raised when the response has ``stop_reason == "refusal"`` and no
    content blocks. Callers treat this as a terminal, non-retryable
    outcome and can write a neutral placeholder row so the message is
    not re-tried in every future run.
    """
```

Terminal, non-retryable exception raised when Bedrock returns `stop_reason == "refusal"` with no content blocks.

`src/claude_sql/llm_shared.py:490`

### loguru_before_sleep

```py
def loguru_before_sleep(level: str = "WARNING") -> Callable[[RetryCallState], None]:
```

Return a tenacity `before_sleep` callback that logs retry state via loguru, replacing the historical stdlib-`logging` `before_sleep_log` shape.

`src/claude_sql/logging_setup.py:53`

## Parquet-shards primitives

### iter_part_files

```py
def iter_part_files(target: Path) -> list[Path]:
```

Return a sorted list of parquet files backing `target` — every `part-*.parquet` for a sharded directory, or `[target]` for a legacy single-file path.

`src/claude_sql/parquet_shards.py:72`

### write_part

```py
def write_part(target: Path, df: pl.DataFrame) -> Path:
```

Write a polars DataFrame as a new shard under a sharded cache directory, or rewrite the legacy single-file parquet for older callers.

`src/claude_sql/parquet_shards.py:87`

### read_all

```py
def read_all(target: Path, *, dtypes: dict[str, Any] | None = None) -> pl.DataFrame | None:
```

Return the union of all part files (or the legacy single file) as a polars DataFrame, or `None` when the cache is empty or missing.

`src/claude_sql/parquet_shards.py:131`

## Session-text primitives

### iter_session_texts

```py
def iter_session_texts(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings,
    since_days: int | None = None,
    limit: int | None = None,
) -> Iterator[tuple[str, str]]:
```

Yield `(session_id, text)` newest-first for every session with at least one text block, materializing one `SessionTextCorpus` for the whole window.

`src/claude_sql/session_text.py:371`

### session_bounds

```py
def session_bounds(
    con: duckdb.DuckDBPyConnection,
    *,
    since_days: int | None = None,
    limit: int | None = None,
) -> dict[str, tuple[datetime | None, datetime | None]]:
```

Return `{session_id: (last_ts, transcript_mtime)}` per session so the LLM workers can drive mtime-based checkpoint skip.

`src/claude_sql/session_text.py:224`

## CLI / output infrastructure

### OutputFormat

```py
class OutputFormat(StrEnum):
    """Supported output formats for tabular and structured CLI output.

    ``AUTO`` resolves to ``TABLE`` when stdout is a TTY and ``JSON`` otherwise.
    Keeping it a string Enum lets cyclopts parse ``--format json`` without any
    custom converter.

    Markdown rendering is intentionally absent: only ``review-sheet`` emits
    human prose, and it owns its own ``--render`` flag (see
    :class:`claude_sql.cli.RenderFormat`). Pulling markdown into this enum
    advertised the format on every subcommand even though no other command
    knows how to produce it.
    """
```

String enum for the `--format` CLI flag values: `AUTO`, `TABLE`, `JSON`, `NDJSON`, `CSV`.

`src/claude_sql/output.py:26`

### configure_logging

```py
def configure_logging(verbose: bool = False, quiet: bool = False) -> None:  # noqa: FBT001, FBT002 — CLI flag pass-through
```

Install the stderr loguru handler for claude-sql at DEBUG / ERROR / `LOGURU_LEVEL`-driven INFO depending on the `--verbose` / `--quiet` flags.

`src/claude_sql/logging_setup.py:27`

### format_version

```py
def format_version() -> str:
```

Return `"claude-sql X.Y.Z"` plus an install-source line (PyPI, git, project venv) when known.

`src/claude_sql/install_source.py:65`

## Structured-output schemas

### SESSION_CLASSIFICATION_SCHEMA

```py
SESSION_CLASSIFICATION_SCHEMA: dict = _bedrock_schema(SessionClassification)
```

Flattened JSON schema dict (Bedrock-`output_config.format`-compatible) for the per-session classifier output: autonomy tier, work category, success, goal.

`src/claude_sql/schemas.py:170`

### TRAJECTORY_ARRAY_SCHEMA

```py
TRAJECTORY_ARRAY_SCHEMA: dict = _bedrock_schema(TrajectoryArrayResult)
```

Flattened JSON schema dict for the windowed trajectory classifier — up to 16 `(prev_uuid, curr_uuid)` window rows per Sonnet call.

`src/claude_sql/schemas.py:289`

### USER_FRICTION_SCHEMA

```py
USER_FRICTION_SCHEMA: dict = _bedrock_schema(UserFrictionSignal)
```

Flattened JSON schema dict for the user-friction classifier output (one of seven labels plus confidence).

`src/claude_sql/schemas.py:474`

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 13 shared citations
- [claude-sql · Processes](../behavior/processes.md) — 9 shared citations
- [claude-sql · Contract map](../insights/contract-map.md) — 7 shared citations
- [claude-sql · Risk hotspots](../analysis/risk-hotspots.md) — 5 shared citations
