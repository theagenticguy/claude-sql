# claude-sql · Public API

`claude-sql` is a virtual uv workspace whose five members (`core`, `analytics`, `evals`, `provenance`, `app`) carry docstring-only barrel `__init__.py` files with no re-exports. The public surface below is the set of non-underscore symbols ranked by inbound cross-module import count across `packages/**/*.py`; ties are broken alphabetically. The `claude-sql` distribution also ships a cyclopts CLI binary (`claude-sql = claude_sql.app.cli:main`, `packages/app/pyproject.toml:37`); its command surface is documented in `reference/cli.md`. No HTTP routes exist, so no `## HTTP` section is rendered.

### Settings

```py
class Settings(BaseSettings):
    """Environment-driven settings for claude-sql.

    All fields are overridable via env vars prefixed ``CLAUDE_SQL_`` (e.g.
    ``CLAUDE_SQL_REGION=us-west-2``) or via ``.env`` in the working directory.
    """
```

Environment-driven settings model (env prefix `CLAUDE_SQL_`) that every package threads through; the dominant public type by inbound references.
`packages/core/src/claude_sql/core/config.py:126`

### iter_part_files

```py
def iter_part_files(target: Path) -> list[Path]:
```

Returns a sorted list of parquet files backing a sharded cache directory or a legacy single-file path.
`packages/core/src/claude_sql/core/parquet_shards.py:72`

### write_part

```py
def write_part(target: Path, df: pl.DataFrame) -> Path:
```

Writes a DataFrame as a new shard in a sharded directory, or appends-and-rewrites the legacy single parquet file.
`packages/core/src/claude_sql/core/parquet_shards.py:87`

### read_all

```py
def read_all(target: Path, *, dtypes: dict[str, Any] | None = None) -> pl.DataFrame | None:
```

Returns the union of all part files (or the legacy single file), or `None` when the cache is empty or missing.
`packages/core/src/claude_sql/core/parquet_shards.py:131`

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

Runs one structured-output Bedrock classification call under a concurrency limiter, returning the parsed payload dict.
`packages/core/src/claude_sql/core/llm_shared.py:563`

### pipeline_cache_stats

```py
@contextmanager
def pipeline_cache_stats(pipeline: str) -> Iterator[None]:
```

Context manager that resets, accumulates, then emits-and-clears Bedrock cache-token statistics for a named pipeline.
`packages/core/src/claude_sql/core/llm_shared.py:259`

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

Terminal, non-retryable exception raised when Bedrock declines to classify input under its content policy.
`packages/core/src/claude_sql/core/llm_shared.py:490`

### loguru_before_sleep

```py
def loguru_before_sleep(level: str = "WARNING") -> Callable[[RetryCallState], None]:
```

Returns a tenacity `before_sleep` callback that logs retry attempts via loguru instead of stdlib logging.
`packages/core/src/claude_sql/core/logging_setup.py:53`

### session_bounds

```py
def session_bounds(
    con: duckdb.DuckDBPyConnection,
    *,
    since_days: int | None = None,
    limit: int | None = None,
) -> dict[str, tuple[datetime | None, datetime | None]]:
```

Returns `{session_id: (last_ts, transcript_mtime)}` for the requested time window, used to detect which sessions need reprocessing.
`packages/core/src/claude_sql/core/session_text.py:224`

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

Yields `(session_id, text)` newest-first for every session with at least one text block, from a single glob scan.
`packages/core/src/claude_sql/core/session_text.py:371`

### claude_sql_home

```py
def claude_sql_home() -> Path:
```

Returns (creating on first call) the parent directory for every claude-sql derived cache, resolved per-platform from env vars.
`packages/core/src/claude_sql/core/home.py:51`

### run_or_die

```py
def run_or_die(
    fn: Any,
    *args: Any,
    fmt: OutputFormat | str = OutputFormat.AUTO,
    **kwargs: Any,
) -> Any:
```

Invokes a callable and translates DuckDB and input-validation errors into classified errors with matching process exit codes.
`packages/core/src/claude_sql/core/output.py:228`

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

Registers raw views, derived views, VSS, analytics, and macros on a DuckDB connection in dependency order.
`packages/core/src/claude_sql/core/sql_views.py:2078`

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

Classifies pending sessions and returns the count of successful classifications (or a plan dict in dry-run mode).
`packages/analytics/src/claude_sql/analytics/classify_worker.py:195`

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

Detects stance conflicts per session and returns the count of sessions processed.
`packages/analytics/src/claude_sql/analytics/conflicts_worker.py:286`

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

Classifies short user messages for friction signals across the session corpus.
`packages/analytics/src/claude_sql/analytics/friction_worker.py:655`

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

Computes per-session windowed sentiment and transition classifications and returns the count of windows written.
`packages/analytics/src/claude_sql/analytics/trajectory_worker.py:933`

### run_clustering

```py
def run_clustering(settings: Settings, *, force: bool = False) -> dict[str, int]:
```

Runs UMAP plus HDBSCAN over the embeddings parquet and returns a counts summary.
`packages/analytics/src/claude_sql/analytics/cluster_worker.py:50`

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

Runs Leiden plus CPM community detection on session centroids and writes the primary parquet output.
`packages/analytics/src/claude_sql/analytics/community_worker.py:472`

### neighbors_of

```py
def neighbors_of(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    session_id: str,
    *,
    top_k: int = 15,
) -> pl.DataFrame:
```

Returns the top-k cosine neighbors of a session in centroid space, bypassing Leiden.
`packages/analytics/src/claude_sql/analytics/community_worker.py:421`

### run_terms

```py
def run_terms(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    force: bool = False,
) -> dict[str, int]:
```

Computes c-TF-IDF top terms per cluster and writes the parquet output.
`packages/analytics/src/claude_sql/analytics/terms_worker.py:29`

### embed_query

```py
def embed_query(text: str, *, settings: Settings) -> list[float]:
```

Embeds a single query string for HNSW nearest-neighbor search, forcing a float embedding vector.
`packages/analytics/src/claude_sql/analytics/embed_worker.py:334`

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

Discovers unembedded messages, embeds them, and appends the vectors to the embeddings parquet.
`packages/analytics/src/claude_sql/analytics/embed_worker.py:365`

### Judge

```py
@dataclass(frozen=True)
class Judge:
    """One Bedrock foundation model wired into the judge panel."""
```

Frozen dataclass describing one Bedrock foundation model wired into the eval judge panel.
`packages/evals/src/claude_sql/evals/judges.py:24`

### resolve

```py
def resolve(name: str) -> Judge:
```

Resolves a judge shortname or model ID to a `Judge`, raising `KeyError` with the full catalog on an unknown name.
`packages/evals/src/claude_sql/evals/judges.py:198`

### freeze

```py
def freeze(
    rubric_path: Path,
    panel_shortnames: tuple[str, ...],
    embed_model_id: str = "global.cohere.embed-v4:0",
    session_scope: SessionScope | None = None,
    seed: int = 42,
    repo: Path | None = None,
) -> Study:
```

Creates and persists a self-contained study manifest (rubric copy plus panel and scope) and returns the `Study`.
`packages/evals/src/claude_sql/evals/freeze.py:115`

### replay

```py
def replay(manifest_sha: str) -> Study:
```

Loads a previously-frozen study by its manifest SHA.
`packages/evals/src/claude_sql/evals/freeze.py:151`

### resolve_commit_to_transcript

```py
def resolve_commit_to_transcript(
    commit_sha: str,
    *,
    repo: Path | None = None,
    all_sources: bool = False,
) -> TranscriptBinding:
```

Resolves a commit SHA to its bound transcript via the RFC 0001 trailer-first, note-fallback precedence.
`packages/provenance/src/claude_sql/provenance/binding.py:674`

### render_markdown

```py
def render_markdown(sheet: dict[str, Any], metadata: dict[str, Any]) -> str:
```

Renders a structured PR review-sheet dict into the canonical Markdown shape.
`packages/provenance/src/claude_sql/provenance/review_sheet_render.py:68`

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

Produces a PR review sheet for a commit SHA by resolving its bound transcript and running an LLM pass.
`packages/provenance/src/claude_sql/provenance/review_sheet_worker.py:342`

## See also

- [claude-sql · Contract map](../insights/contract-map.md) — 13 shared source files
- [claude-sql · Module map](../architecture/module-map.md) — 13 shared source files
- [claude-sql · Processes](../behavior/processes.md) — 13 shared source files
- [claude-sql · Debugging guide](../insights/debugging-guide.md) — 10 shared source files
- [claude-sql · Tech debt](../insights/tech-debt.md) — 10 shared source files
