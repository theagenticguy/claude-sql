# claude-sql · Contract map

This file enumerates the load-bearing contracts between modules in
`src/claude_sql/`. The codebase is fully typed Python with `ty` strict
(`pyproject.toml` declares `[tool.ty.rules] all = "error"`), so a
"contract" here is **any public symbol — type, function, dataclass, or
module-level constant — declared in one module and imported by ≥ 1
other module under `src/claude_sql/`**. Where the load-bearing
behaviour exceeds what the type signature says (call ordering, file-vs-
directory overload, structured-output JSON-Schema rules, refusal
sentinels), the per-contract `Assumptions consumers make` field
captures it.

Ranking is by unique-consumer-module count descending. The CLI
(`claude_sql.cli`) is treated as a single consumer even though it
imports from every worker; it's the public binary boundary, so
contracts whose only consumer is `cli` are still load-bearing per the
packet's "structurally important entry point" rule.

## `Settings` (config)

**Producer:** `src/claude_sql/config.py:127`

**Consumer(s):**
- `src/claude_sql/cli.py:63` — top-level `Settings()` constructor under every CLI subcommand.
- `src/claude_sql/sql_views.py:47` — `register_all` / `register_macros` / `register_analytics` thread it through the DuckDB binding pass.
- `src/claude_sql/embed_worker.py:44` — used to size Bedrock client pool, region, embedding model.
- `src/claude_sql/community_worker.py:66` — reads every `leiden_*` field plus `seed`.
- `src/claude_sql/cluster_worker.py:26` — reads UMAP/HDBSCAN hyperparameters and the clusters parquet path.
- `src/claude_sql/skills_catalog.py:44` — reads `user_skills_dir` / `plugins_cache_dir` / `skills_catalog_parquet_path`.
- `src/claude_sql/ingest.py:52` — reads `ingest_stamps_parquet_path`.
- `src/claude_sql/llm_shared.py:58` — typed `TYPE_CHECKING` import for `_build_bedrock_client(settings)`.
- `src/claude_sql/classify_worker.py:42`, `src/claude_sql/conflicts_worker.py:62`, `src/claude_sql/trajectory_worker.py:61`, `src/claude_sql/friction_worker.py:66`, `src/claude_sql/review_sheet_worker.py:54` — `TYPE_CHECKING` imports for orchestration entry points.
- `src/claude_sql/session_text.py:41`, `src/claude_sql/terms_worker.py:26` — `TYPE_CHECKING` imports for `Settings`-driven cap fields.

**Shape:**
```python
class Settings(BaseSettings):
    """Environment-driven settings for claude-sql.

    All fields are overridable via env vars prefixed ``CLAUDE_SQL_`` (e.g.
    ``CLAUDE_SQL_REGION=us-west-2``) or via ``.env`` in the working directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_SQL_",
        env_file=".env",
        extra="ignore",
    )
    # ... (130+ fields covering data discovery, Bedrock model IDs / pricing,
    # parquet cache paths, UMAP+HDBSCAN+Leiden hyperparameters, c-TF-IDF
    # parameters, DuckDB engine PRAGMAs)
```
(Full field list: `src/claude_sql/config.py:140`-`src/claude_sql/config.py:343`.)

**Assumptions consumers make:**
- `Settings` is constructed once at process start and passed by reference; consumers never re-read env vars. `src/claude_sql/cli.py:63` instantiates it once per subcommand and threads it through. The `pydantic_settings` env hierarchy is read at construction time only.
- The deprecated `concurrency` field aliases onto `embed_concurrency` and `llm_concurrency` only when the per-pipeline fields are still at their factory defaults — see the model validator at `src/claude_sql/config.py:384`. Consumers that read `settings.embed_concurrency` directly inherit this aliasing transparently (`src/claude_sql/llm_shared.py:375`).
- Path fields ending in `_parquet_path` may be either a sharded directory or a legacy single file at runtime; consumers must hand the path to `parquet_shards.write_part` or `iter_part_files` rather than `polars.write_parquet` directly (`src/claude_sql/embed_worker.py` and every worker that writes a cache parquet).
- `default_glob` / `subagent_glob` / `subagent_meta_glob` get rewritten by `_derive_team_corpus_globs` (`src/claude_sql/config.py:344`) when `team_corpus_root` is set; consumers that read these fields after construction get the rewritten values, not the originals.

**Drift risk:** Adding a new `Settings` field with no factory default breaks every consumer that constructs `Settings()` with zero args (the dominant pattern). Mitigation: every new field must carry a `Field(default=...)` or `Field(default_factory=...)`.

## Sharded parquet I/O (`parquet_shards`)

**Producer:** `src/claude_sql/parquet_shards.py:54`-`src/claude_sql/parquet_shards.py:243`

**Consumer(s):**
- `src/claude_sql/classify_worker.py:35` — `read_all`, `write_part`.
- `src/claude_sql/conflicts_worker.py:53` — `iter_part_files`, `read_all`, `write_part`.
- `src/claude_sql/friction_worker.py:61` — `read_all`, `write_part`.
- `src/claude_sql/trajectory_worker.py:54` — `iter_part_files`, `replace_sessions`, `write_part`.
- `src/claude_sql/ingest.py:53` — `iter_part_files`, `write_part`.
- `src/claude_sql/sql_views.py:48` — `iter_part_files` (used to glob shards into DuckDB views).
- `src/claude_sql/cli.py:87` — `count_rows`, `is_sharded_dir`, `iter_part_files` for `list-cache` and `cache compact`.

**Shape:**
```python
def is_sharded_dir(path: Path) -> bool: ...

def iter_part_files(target: Path) -> list[Path]: ...

def write_part(target: Path, df: pl.DataFrame) -> Path: ...

def read_all(target: Path, *, dtypes: dict[str, Any] | None = None) -> pl.DataFrame | None: ...

def count_rows(target: Path) -> int: ...

def replace_sessions(
    target: Path,
    *,
    key_column: str,
    session_ids: Iterable[str],
) -> int: ...

PART_GLOB: str = "part-*.parquet"
```

**Assumptions consumers make:**
- `target` may be either a directory (sharded layout) or a `.parquet` file (legacy layout). `is_sharded_dir` (`src/claude_sql/parquet_shards.py:54`) overloads on the suffix when the path is missing on disk, so a brand-new `tmp_path / "x.parquet"` fixture takes the legacy path while a brand-new `tmp_path / "x"` fixture takes the sharded path. Workers inherit this overload via `Settings.<x>_parquet_path` without saying so locally (`src/claude_sql/classify_worker.py:35`).
- `write_part` does not enforce schema across shards; readers use `pl.read_parquet([list_of_paths])` which uses vertical-relaxed concat (`src/claude_sql/parquet_shards.py:144`). A schema drift across two shards surfaces as a polars error at `read_all` time, not at write time.
- `replace_sessions` deletes empty shards in place (`src/claude_sql/parquet_shards.py:222`); the only worker using it is `trajectory_worker` which relies on this for the v1.0 windowed rewrite-on-rerun path. Other workers are not allowed to call it without checkpointing, since they'd lose unique rows.
- `count_rows` uses `pyarrow.parquet.ParquetFile.metadata.num_rows` (`src/claude_sql/parquet_shards.py:163`) — correct only because every cache parquet currently has a flat row schema. A future struct-of-arrays cache would have to invalidate this assumption.

**Drift risk:** A worker that writes a cache parquet via `polars.DataFrame.write_parquet(target)` directly (instead of `write_part(target, df)`) silently bypasses the sharded layout. Mitigation: every cache write in a worker module goes through `write_part`; the test suite under `tests/test_parquet_shards.py` enforces this for the five worker caches.

## Bedrock LLM plumbing (`llm_shared`)

**Producer:** `src/claude_sql/llm_shared.py:259`-`src/claude_sql/llm_shared.py:596`, plus the four task-framing system prompts (`src/claude_sql/llm_shared.py:612`, `:741`, `:895`, `:1119`).

**Consumer(s):**
- `src/claude_sql/classify_worker.py:27` — `CLASSIFY_SYSTEM_PROMPT`, `_build_bedrock_client`, `_count_pending_sessions`, `_estimate_cost`, `classify_one`, `pipeline_cache_stats`.
- `src/claude_sql/conflicts_worker.py:45` — `CONFLICTS_SYSTEM_PROMPT`, `_build_bedrock_client`, `_count_pending_sessions`, `_estimate_cost`, `classify_one`, `pipeline_cache_stats`.
- `src/claude_sql/trajectory_worker.py:47` — `BedrockRefusalError`, `_build_bedrock_client`, `_estimate_cost`, `classify_one`, `pipeline_cache_stats`, `TRAJECTORY_SYSTEM_PROMPT` (also imported).
- `src/claude_sql/friction_worker.py:53` — `USER_FRICTION_SYSTEM_PROMPT`, `BedrockRefusalError`, `_build_bedrock_client`, `_estimate_cost`, `classify_one`, `pipeline_cache_stats`.
- `src/claude_sql/review_sheet_worker.py:44` — `BedrockRefusalError`, `_build_bedrock_client`, `_invoke_classifier_sync`.

**Shape:**
```python
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
) -> dict: ...

class BedrockRefusalError(Exception): ...

@contextmanager
def pipeline_cache_stats(pipeline: str) -> Iterator[None]: ...

def _build_bedrock_client(settings: Settings) -> Any: ...

def _invoke_classifier_sync(
    client: Any,
    model_id: str,
    schema: dict,
    user_text: str,
    *,
    max_tokens: int,
    thinking_mode: str,
    system: str | None = None,
    pipeline: str = "classifier",
) -> dict: ...

def _estimate_cost(
    n_items: int,
    avg_in_tokens: int,
    avg_out_tokens: int,
    pricing: tuple[float, float],
) -> float: ...
```

**Assumptions consumers make:**
- `BedrockRefusalError` is **terminal** — workers stamp a neutral placeholder row and clear the retry queue rather than retrying (`src/claude_sql/trajectory_worker.py:47`, `src/claude_sql/friction_worker.py:53` import it for this branch). Re-raising or catching `Exception` in the worker would silently re-queue refusals into an infinite retry loop.
- `pipeline_cache_stats(pipeline)` clears any stale accumulator on entry (`src/claude_sql/llm_shared.py:277`); two pipelines using the same string name would collide. Workers use stable string keys: `"classify"`, `"trajectory"`, `"conflicts"`, `"friction"` (`src/claude_sql/llm_shared.py:265`).
- `_build_bedrock_client` is process-cached on `(region, pool_size)` (`src/claude_sql/llm_shared.py:343`); changing `Settings.embed_concurrency` after first build does NOT shrink the connection pool. Consumers that vary concurrency at runtime must build a fresh `Settings` first.
- `classify_one` accepts both `asyncio.Semaphore` and `anyio.CapacityLimiter` (`src/claude_sql/llm_shared.py:571`); callers using `anyio.to_thread.run_sync` must pass the latter or cancellation does not cascade properly per the project's anyio structured-concurrency lesson.

**Drift risk:** Adding a new `_RETRY_CODES` entry (`src/claude_sql/llm_shared.py:61`) is silently consumed; removing one breaks every worker's failure surface. Mitigation: tests in `tests/test_llm_shared.py` cover each code's retry path.

## Bedrock structured-output schemas (`schemas`)

**Producer:** `src/claude_sql/schemas.py:99`-`src/claude_sql/schemas.py:580`. Every pydantic model is paired with a flattened `*_SCHEMA` dict that consumers actually pass to Bedrock.

**Consumer(s):**
- `src/claude_sql/classify_worker.py:36` — `SESSION_CLASSIFICATION_SCHEMA`.
- `src/claude_sql/conflicts_worker.py:54` — `SESSION_CONFLICTS_SCHEMA`.
- `src/claude_sql/trajectory_worker.py:55` — `TRAJECTORY_ARRAY_SCHEMA`.
- `src/claude_sql/friction_worker.py:62` — `USER_FRICTION_SCHEMA`.
- `src/claude_sql/review_sheet_worker.py:49` — `PR_REVIEW_SHEET_SCHEMA`.

**Shape:**
```python
class SessionClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    autonomy_tier: Literal["manual", "assisted", "autonomous"]
    work_category: Literal["sde", "admin", "strategy_business",
                            "events", "thought_leadership", "other"]
    success: Literal["success", "partial", "failure", "unknown"]
    goal: str = Field(..., min_length=1, max_length=280)
    confidence: float = Field(..., ge=0.0, le=1.0)


SESSION_CLASSIFICATION_SCHEMA: dict = _bedrock_schema(SessionClassification)
# ... same shape for TrajectoryArrayResult / ConflictsResult /
# UserFrictionSignal / PRReviewSheet (src/claude_sql/schemas.py:266,
# :379, :406, :509)
```

**Assumptions consumers make:**
- `_bedrock_schema(model)` (`src/claude_sql/schemas.py:16`) flattens `$ref`/`$defs`, injects `additionalProperties: false` at every object level, and strips numeric/string/array constraint keywords that Bedrock's Draft-2020-12 subset rejects. Consumers pass the flattened dict to Bedrock's `output_config.format.schema` field (`src/claude_sql/llm_shared.py:444`); the unflattened pydantic model is only used for response-side validation.
- The schema is computed at import time, not per-call. A model edit requires a process restart to land in Bedrock requests (`src/claude_sql/schemas.py:170` and the four sibling bindings).
- `extra="forbid"` (`src/claude_sql/schemas.py:106` and every other model) means a Bedrock response with a field the model didn't declare raises `ValidationError` rather than silently dropping the field. Consumers handle this by enqueueing the offending unit on the retry queue.

**Drift risk:** Adding a new field with a default value to a pydantic model breaks the next live Bedrock call until both ends redeploy together — `additionalProperties: false` on the flattened schema rejects any field the schema doesn't whitelist, even if pydantic would accept it. Mitigation: ship every schema change behind a CLI flag toggle so the old schema is reachable at call time during the transition.

## Session text assembly (`session_text`)

**Producer:** `src/claude_sql/session_text.py:84`-`src/claude_sql/session_text.py:387`

**Consumer(s):**
- `src/claude_sql/classify_worker.py:37` — `iter_session_texts`, `session_bounds`.
- `src/claude_sql/conflicts_worker.py:55` — `iter_session_texts`, `session_bounds`.
- `src/claude_sql/trajectory_worker.py:56` — `session_bounds`.
- `src/claude_sql/friction_worker.py:63` — `session_bounds`.

**Shape:**
```python
@dataclass(slots=True)
class SessionTextCorpus:
    """An in-memory corpus of per-session timelines, built with one glob scan."""

    texts_by_session: dict[str, list[_TimelineRow]]
    order: list[str]

    def assemble(self, session_id: str, *, settings: Settings) -> str: ...


def iter_session_texts(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings,
    since_days: int | None = None,
    limit: int | None = None,
) -> Iterator[tuple[str, str]]: ...


def session_bounds(
    con: duckdb.DuckDBPyConnection,
    *,
    since_days: int | None = None,
    limit: int | None = None,
) -> dict[str, tuple[datetime | None, datetime | None]]: ...
```

**Assumptions consumers make:**
- `iter_session_texts` is a generator over a *single* materialized `SessionTextCorpus`; the underlying `read_json` glob scan happens once even though the iterator may be consumed lazily (`src/claude_sql/session_text.py:383`). Two parallel iterators would force two scans — workers always wrap iteration in a single async loop.
- `session_bounds` returns `(last_ts, transcript_mtime)` where the mtime is computed via `Path(transcript_path).stat()` at call time (`src/claude_sql/session_text.py:266`). The checkpoint pipeline (`checkpointer.filter_unchanged`) treats `None` mtime as "advance" (assume changed); consumers must pass the unmodified tuple straight through.
- Both functions accept `since_days=None` (corpus-wide) but `iter_session_texts` is unbounded by default; a backfill caller MUST pass `limit` or wrap the iterator with their own counter to avoid spending the whole corpus's Bedrock budget (`src/claude_sql/classify_worker.py:37`).

**Drift risk:** A schema change to the `messages_text` view breaks `_load_messages_text` silently — the SQL is hardcoded at `src/claude_sql/session_text.py:280`. Mitigation: `VIEW_SCHEMA` (`src/claude_sql/sql_views.py:117`) is the static contract for the view shape; CI test `test_view_schema_matches_describe_all` flags drift.

## SQL view + macro registration (`sql_views`)

**Producer:** `src/claude_sql/sql_views.py:62` (`VIEW_NAMES`), `:117` (`VIEW_SCHEMA`), `:325` (`MACRO_NAMES`), `:366` (`MACRO_SIGNATURES`), `:622` (`register_views`), `:1216` (`register_macros`), `:1832` (`register_analytics`), `:1691` (`register_vss`), `:2081` (`register_all`).

**Consumer(s):**
- `src/claude_sql/cli.py:94` — `MACRO_NAMES`, `MACRO_SIGNATURES`, `VIEW_NAMES`, `VIEW_SCHEMA`, `_parquet_is_populated`, `register_all`, `register_raw`, `register_views`, `register_vss`.
- `src/claude_sql/cli.py:393` — `register_analytics` (deferred import inside CLI body).

**Shape:**
```python
VIEW_NAMES: tuple[str, ...] = (
    "sessions", "messages", "content_blocks", "messages_text",
    "turn_window", "tool_calls", "tool_results", "todo_events",
    "todo_state_current", "subagent_spawns", "task_creations",
    "task_updates", "tasks_state_current", "task_spawns",
    "skill_invocations", "subagent_sessions", "subagent_messages",
    # v2 analytics views ...
    "session_classifications", "session_goals", "message_trajectory",
    "session_conflicts", "conflicts_summary", "message_clusters",
    "cluster_terms", "session_communities", "community_profile",
    "user_friction", "skills_catalog", "skill_usage", "ingest_stamps",
)

VIEW_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = { ... }

MACRO_NAMES: tuple[str, ...] = (
    "ago", "model_used", "cost_estimate", "tool_rank", "todo_velocity",
    "subagent_fanout", "semantic_search", "skill_rank", ...,
    "canonical_uuid_resolve",
)

MACRO_SIGNATURES: dict[str, tuple[str, ...]] = { ... }


def register_all(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings | None = None,
    include_analytics: bool = True,
    skip_vss: bool = False,
) -> None: ...
```

**Assumptions consumers make:**
- `register_all` enforces an internal call ordering: `register_raw` → `register_views` → optional `register_vss` → optional `register_analytics` → `register_macros` (`src/claude_sql/sql_views.py:2126`). DuckDB resolves macro bodies at creation time, so `register_macros` must run last or `semantic_search` references an unbound `message_embeddings` table.
- `VIEW_SCHEMA` only covers v1 (transcript-derived) views (`src/claude_sql/sql_views.py:106`); v2 analytics views (`session_classifications`, `message_trajectory`, etc.) are intentionally absent because their schemas live in the parquet metadata. Consumers iterating `VIEW_NAMES` for schema dumps must tolerate missing entries.
- `MACRO_SIGNATURES` is hand-maintained because DuckDB's `duckdb_functions()` returns NULL `parameters` for table macros (`src/claude_sql/sql_views.py:354`); consumers like `claude-sql schema` rely on this dict for signature display.
- `_parquet_is_populated` (`src/claude_sql/sql_views.py:1817`) is private but imported by `cli.py:99` — it returns `True` for a sharded directory iff at least one part file exists. CLI gating (e.g., the `analyze` chain) uses this to decide whether to register the corresponding analytics view.

**Drift risk:** Adding a view name to `VIEW_NAMES` without updating `VIEW_SCHEMA` for v1 views surfaces as a CI-test failure (`test_view_schema_matches_describe_all` per `src/claude_sql/sql_views.py:112`). Adding a macro without updating `MACRO_SIGNATURES` surfaces via `test_macro_signatures_match_ddl`. No mitigation needed beyond keeping the tests in CI.

## Output formatting + error classification (`output`)

**Producer:** `src/claude_sql/output.py:26`-`src/claude_sql/output.py:254`

**Consumer(s):**
- `src/claude_sql/cli.py:75` — `EXIT_CODES`, `ClassifiedError`, `InputValidationError`, `OutputFormat`, `emit_dataframe`, `emit_error`, `emit_json`, `resolve_format`, `run_or_die`, `validate_glob`.

**Shape:**
```python
class OutputFormat(StrEnum):
    AUTO = "auto"
    TABLE = "table"
    JSON = "json"
    NDJSON = "ndjson"
    CSV = "csv"


EXIT_CODES: dict[str, int] = {
    "ok": 0,
    "no_embeddings": 2,
    "invalid_input": 64,
    "parse_error": 64,
    "catalog_error": 65,
    "runtime_error": 70,
    "duckdb_missing": 127,
}


@dataclass(frozen=True, slots=True)
class ClassifiedError:
    kind: str
    exit_code: int
    message: str
    hint: str | None = None

    def to_payload(self) -> dict[str, Any]: ...


class InputValidationError(ValueError):
    def __init__(self, message: str, *, hint: str | None = None) -> None: ...


def classify_duckdb_error(exc: duckdb.Error) -> ClassifiedError: ...
def emit_dataframe(df: pl.DataFrame, fmt: OutputFormat | str = OutputFormat.AUTO,
                   *, table_rows: int = 100, table_str_len: int = 120) -> None: ...
def emit_json(payload: Any, fmt: OutputFormat | str = OutputFormat.AUTO) -> None: ...
def emit_error(err: ClassifiedError, fmt: OutputFormat | str = OutputFormat.AUTO) -> None: ...
def run_or_die(fn: Any, *args: Any, fmt: OutputFormat | str = OutputFormat.AUTO,
               **kwargs: Any) -> Any: ...
def validate_glob(pattern: str | None, *, flag: str = "--glob") -> None: ...
```

**Assumptions consumers make:**
- `OutputFormat.AUTO` is resolved per-call against `sys.stdout.isatty()` (`src/claude_sql/output.py:60`); a subcommand that captures stdout for its own use must pass an explicit format or it gets the wrong shape.
- `EXIT_CODES["parse_error"] == EXIT_CODES["invalid_input"] == 64` — two distinct kinds collapse to the same code on purpose (`src/claude_sql/output.py:49`). Consumers who care about distinguishing them must inspect the JSON payload's `kind` field, not the exit code.
- `run_or_die` catches **only** `InputValidationError` and `duckdb.Error` (`src/claude_sql/output.py:242`); any other exception propagates and crashes the CLI process. Consumers wrapping a Bedrock call in `run_or_die` must catch `BedrockRefusalError` themselves first or it bypasses the structured error envelope.
- `emit_error` writes to stderr, never stdout (`src/claude_sql/output.py:220`); consumers piping `claude-sql ... | jq` get clean JSON on stdout even when an error occurs.

**Drift risk:** A new DuckDB exception subclass (e.g., `duckdb.SerializationException`) would silently route through the `runtime_error` branch (`src/claude_sql/output.py:202`). Mitigation: track DuckDB release notes; extend `classify_duckdb_error` when new subclasses ship.

## Cache home resolution (`home`)

**Producer:** `src/claude_sql/home.py:51`, `:75`, `:33`

**Consumer(s):**
- `src/claude_sql/config.py:18` — `claude_sql_home` (every default factory builds paths off it).
- `src/claude_sql/cli.py:67` — `claude_sql_home`, `recognized_legacy_caches` (used by the legacy migration on first connect).

**Shape:**
```python
_LEGACY_CACHE_NAMES: tuple[str, ...] = (
    "embeddings_lance", "embeddings", "message_trajectory",
    "session_classifications", "session_conflicts", "user_friction",
    "clusters.parquet", "cluster_terms.parquet",
    "session_communities.parquet", "community_profile.parquet",
    "state.db", "duckdb_tmp", "profiling", "claude_sql.duckdb",
)


def claude_sql_home() -> Path: ...
def recognized_legacy_caches(legacy_root: Path | None = None) -> dict[str, Path]: ...
```

**Assumptions consumers make:**
- `claude_sql_home()` reads `os.environ` on every call (`src/claude_sql/home.py:62`) — by design, so that test fixtures using `monkeypatch.setenv("CLAUDE_SQL_HOME", ...)` see the new value without needing module reloads.
- The function creates the directory with `mkdir(parents=True, exist_ok=True)` (`src/claude_sql/home.py:71`); consumers can rely on it existing immediately. A read-only-FS environment would fail here at the first `Settings()` instantiation, since `config.py`'s default factories call into it.
- `recognized_legacy_caches` returns only paths that actually exist on disk (`src/claude_sql/home.py:91`); the migration code in `cli.py:67` treats an empty dict as "no migration needed" and short-circuits.

**Drift risk:** Adding a new analytics parquet without adding its filename to `_LEGACY_CACHE_NAMES` means future legacy-tree migrations leave it under `~/.claude/`. Mitigation: every new cache field in `Settings` must be paired with an entry in `_LEGACY_CACHE_NAMES`.

## Per-pipeline checkpointing (`checkpointer`)

**Producer:** `src/claude_sql/checkpointer.py:41` (`PIPELINE_NAMES`), `:262` (`load_as_map`), `:283` (`filter_unchanged`), `:327` (`mark_completed`), `:367` (`count_rows`), `:62` (`_SCHEMA_BOOTSTRAPPED`/`_SCHEMA_BOOTSTRAP_LOCK`).

**Consumer(s):**
- `src/claude_sql/retry_queue.py:28` — `_SCHEMA_BOOTSTRAP_LOCK`, `_SCHEMA_BOOTSTRAPPED`, `PIPELINE_NAMES`, `_connect`, `_from_iso`, `_legacy_duckdb_path`, `_to_iso` (intentional internal sharing — both tables live in the same SQLite file).
- Every LLM worker calls `load_as_map`, `filter_unchanged`, `mark_completed` indirectly through orchestration logic (e.g., `src/claude_sql/trajectory_worker.py` and the other workers consume the API surface even though they don't import the symbols by name; the entry-point modules wire it).

**Shape:**
```python
PIPELINE_NAMES: tuple[str, ...] = ("classify", "trajectory", "conflicts", "user_friction")


def load_as_map(db_path: Path, pipeline: str) -> dict[str, tuple[datetime | None, datetime | None]]: ...

def filter_unchanged(
    candidates: Iterable[tuple[str, datetime | None, datetime | None]],
    *,
    pipeline: str,
    checkpoint_db_path: Path,
) -> tuple[list[str], int]: ...

def mark_completed(
    db_path: Path,
    *,
    pipeline: str,
    rows: Iterable[tuple[str, datetime | None, datetime | None]],
) -> int: ...
```

**Assumptions consumers make:**
- `PIPELINE_NAMES` is the closed set of valid `pipeline` strings (`src/claude_sql/checkpointer.py:41`); consumers that mistype the name (e.g., `"frictions"` instead of `"user_friction"`) silently get an empty checkpoint and re-process every session.
- `filter_unchanged` returns `(pending_ids, skipped_count)` where pending preserves input order; consumers that re-shuffle the candidate list lose the newest-first guarantee from `session_bounds`.
- The `_SCHEMA_BOOTSTRAPPED` set + `_SCHEMA_BOOTSTRAP_LOCK` pair is per-process state (`src/claude_sql/checkpointer.py:62`); `retry_queue.py` reuses both via underscore-prefixed import to avoid a second WAL-pragma race on the same SQLite file. This is the only sanctioned cross-module use of underscore-prefixed symbols in the codebase.
- All timestamps are tz-aware UTC datetimes in Python and ISO-8601 strings on disk (`src/claude_sql/checkpointer.py:19`); consumers passing naive datetimes get them coerced to UTC at write time.

**Drift risk:** Adding a fifth pipeline name (e.g., `"review_sheet"`) requires updating `PIPELINE_NAMES` *and* `retry_queue.py`'s reuse path. Mitigation: a CI test enumerating workers and asserting each name is in `PIPELINE_NAMES` would close the gap; today the responsibility lives with the contributor.

## Loguru / tenacity glue (`logging_setup`)

**Producer:** `src/claude_sql/logging_setup.py:27`, `:53`

**Consumer(s):**
- `src/claude_sql/cli.py:74` — `configure_logging` (called once at top-level entry).
- `src/claude_sql/llm_shared.py:53` — `loguru_before_sleep` (passed to every `@retry` decorator).
- `src/claude_sql/embed_worker.py:45` — `loguru_before_sleep`.
- `src/claude_sql/judge_worker.py:50` — `loguru_before_sleep`.

**Shape:**
```python
def configure_logging(verbose: bool = False, quiet: bool = False) -> None: ...

def loguru_before_sleep(level: str = "WARNING") -> Callable[[RetryCallState], None]: ...
```

**Assumptions consumers make:**
- `configure_logging` is called exactly once per process (it calls `logger.remove()` first — `src/claude_sql/logging_setup.py:37`); a worker that calls it would clobber the CLI's handler. Workers use `from loguru import logger` directly and rely on the parent process's configuration.
- `loguru_before_sleep("LEVEL")` is the **only** sanctioned shape for tenacity `before_sleep` callbacks; the project bans stdlib `logging` (per `pyproject.toml`'s `flake8-tidy-imports.banned-api`), so a contributor reaching for `before_sleep_log(stdlib_logger, level)` gets a hard ruff failure.

**Drift risk:** None — the contract is two functions, both stable since the loguru-only migration. No current drift risk.

## Community detection (`community_worker`)

**Producer:** `src/claude_sql/community_worker.py:79` (`ResolutionLevel`), `:421` (`neighbors_of`), `:472` (`run_communities`), `:75` (`NOISE_COMMUNITY_ID`).

**Consumer(s):**
- `src/claude_sql/cli.py:58` — `ResolutionLevel`, `neighbors_of`, `run_communities`.

**Shape:**
```python
ResolutionLevel = Literal["coarse", "medium", "fine"]
NOISE_COMMUNITY_ID: int = -1


def run_communities(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    force: bool = False,
    gamma: float | None = None,
    resolution: ResolutionLevel = "medium",
) -> dict[str, int | float | str]: ...


def neighbors_of(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    session_id: str,
    *,
    top_k: int = 15,
) -> pl.DataFrame: ...
```

**Assumptions consumers make:**
- `run_communities` writes `session_communities.parquet` directly using `polars.write_parquet` (not via `parquet_shards.write_part`); the file is intentionally a single artefact, not sharded — see the design note at `src/claude_sql/community_worker.py:26`. CLI consumers passing `--force` rely on this for rebuild semantics.
- An explicit `gamma` value skips the resolution-profile sidecar (`src/claude_sql/community_worker.py:493`); the CLI surfaces this in `claude-sql community --gamma 0.4` so an agent can skip the auto-tune cost.
- `neighbors_of` raises `ValueError` when the session id is absent from the embeddings corpus (`src/claude_sql/community_worker.py:437`); CLI converts it to `EXIT_CODES["invalid_input"]` via `run_or_die`.
- A missing `session_communities.parquet` produces a *partial* result from `neighbors_of` — only `(neighbor_session_id, weight)` columns, no `community_id` / `is_medoid` (`src/claude_sql/community_worker.py:464`). Consumers must check for the optional columns rather than assume them.

**Drift risk:** Renaming a `ResolutionLevel` literal without updating the CLI's `--resolution` argument string surfaces only at runtime as a cyclopts validation error. Mitigation: keep `ResolutionLevel` and the CLI option in lock-step — both live in modules touched by the same PR.

## Ingest stamping (`ingest`)

**Producer:** `src/claude_sql/ingest.py:274` (`count_pending`), `:295` (`stamp_messages`), `:410` (`resolve_canonicals`).

**Consumer(s):**
- `src/claude_sql/cli.py:68` — `count_pending` (aliased `_ingest_count_pending`), `resolve_canonicals` (aliased `_ingest_resolve_canonicals`), `stamp_messages` (aliased `_ingest_stamp_messages`).

**Shape:**
```python
def count_pending(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
) -> int: ...

def stamp_messages(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    since_days: int | None = None,
    limit: int | None = None,
    batch_size: int = 4096,
) -> int: ...

def resolve_canonicals(...) -> int: ...
```

**Assumptions consumers make:**
- `count_pending` is pure SQL (`src/claude_sql/ingest.py:288`); the dry-run path uses it specifically to avoid materializing every message text. A consumer who calls `stamp_messages` and inspects the return without calling `count_pending` first will spend Bedrock tokens on potentially zero rows.
- `stamp_messages` writes to `Settings.ingest_stamps_parquet_path` via `parquet_shards.write_part`; the path resolves to a sharded directory in new installs (`src/claude_sql/config.py:282`). Consumers that copy the parquet for inspection must either glob the directory or run `claude-sql cache compact` first.
- `resolve_canonicals` runs after at least one `stamp_messages` pass — there must be a non-empty `ingest_stamps` view for the self-join to bind. CLI orchestration enforces this ordering inside the `ingest` subcommand.

**Drift risk:** The `_ANTHROPIC_RATIO = 0.78` constant (`src/claude_sql/ingest.py:73`) is recomputed against fresh cache receipts twice per minor release per the docstring; a stale ratio drifts the dry-run cost estimate without breaking anything obvious. Mitigation: the recomputation cadence is captured in `.erpaval/solutions/api-patterns/anthropic-prompt-cache-tokenizer-gap.md`.

## Other contracts

- **`embed_worker.{embed_query, run_backfill, _build_bedrock_client}`** — `src/claude_sql/embed_worker.py:360` and `:182`; consumed only by `cli.py:65`. The local `_build_bedrock_client` mirrors `llm_shared._build_bedrock_client` because Cohere Embed v4 needs a different pool size at high `embed_concurrency`.
- **`cluster_worker.run_clustering`** — `src/claude_sql/cluster_worker.py`; consumed only by `cli.py:57`. Self-contained UMAP+HDBSCAN runner.
- **`classify_worker.classify_sessions`**, **`conflicts_worker.detect_conflicts`**, **`trajectory_worker.trajectory_messages`**, **`friction_worker.detect_user_friction`**, **`terms_worker.run_terms`**, **`review_sheet_worker.generate_review_sheet`** — each is a single-symbol contract whose only consumer is `cli.py:56`/`:64`/`:106`/`:66`/`:105`/`:93`. These are entry points to the per-stage pipelines; the contract is stable because the CLI is the only caller.
- **`review_sheet_render.{render_markdown, render_refusal_markdown}`** — `src/claude_sql/review_sheet_render.py`; consumed only by `cli.py:92`.
- **`install_source.format_version`** — `src/claude_sql/install_source.py`; consumed only by `cli.py:73`. Formats a `--version` string with provenance (PyPI / git / wheel).
- **`judges.Judge`** — `src/claude_sql/judges.py`; consumed only by `judge_worker.py:49`. Internal data class for the judge-panel pipeline.
- **`retry_queue.{enqueue, drain, mark_done, pending_count}`** — `src/claude_sql/retry_queue.py:85`-`:187`; orchestrated by the LLM workers via the checkpointer's shared SQLite connection helper.

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 22 shared citations
- [claude-sql · Public API](../reference/public-api.md) — 7 shared citations
- [claude-sql · Business logic](../insights/business-logic.md) — 6 shared citations
- [claude-sql · Risk hotspots](../analysis/risk-hotspots.md) — 4 shared citations
