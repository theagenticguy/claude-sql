# claude-sql · Contract map

`claude-sql` is one package with five layer sub-packages (the `claude_sql.*` layers) laid out as a strict import DAG: `core` (L0) is depended on by the three independent siblings `analytics`, `evals`, `provenance` (L1), which are in turn depended on by `app` (L2). The DAG is enforced in CI by import-linter (`pyproject.toml:261-280`: a `layers` contract plus an `independence` contract over the three siblings).

A **contract** in this codebase is one of two things:

1. A type, schema, or constant declared in `claude_sql.core` (the L0 producer) and imported by a downstream layer — the layered DAG funnels almost every cross-layer dependency through core.
2. A runtime **wire format** that crosses a process or storage boundary: the Bedrock `invoke_model` request body, the on-disk parquet shard column schema, the LanceDB table schema, the git trailer / git-note binding (RFC 0001), and the CLI's stable exit codes. These are contracts even when one side is an LLM, a file, or a git subprocess rather than a Python import.

The most load-bearing and drift-prone contracts here are the ones where the same shape is asserted in three independent places — a pydantic model, a polars parquet schema, and a hand-maintained DuckDB column dict — and nothing but a CI test keeps them aligned. Those are called out under Drift risk.

Contracts are ordered below by consumer count, descending. Overflow beyond the top 12 is in `## Other contracts`.

## Bedrock `output_config.format` request body

The structured-output envelope every LLM worker sends to Bedrock `invoke_model`. The producer builds one body shape; five workers feed it a different `*_SCHEMA`.

**Producer:** `src/claude_sql/core/llm_shared.py:441-448` (`_invoke_structured_output` builds the `body` dict).

**Consumer(s):**
- `src/claude_sql/analytics/classify_worker.py:109` — passes `SESSION_CLASSIFICATION_SCHEMA`.
- `src/claude_sql/analytics/trajectory_worker.py:637` — passes `TRAJECTORY_ARRAY_SCHEMA`.
- `src/claude_sql/analytics/conflicts_worker.py:191` — passes `SESSION_CONFLICTS_SCHEMA`.
- `src/claude_sql/analytics/friction_worker.py:554` — passes `USER_FRICTION_SCHEMA`.
- `src/claude_sql/provenance/review_sheet_worker.py:430` — passes `PR_REVIEW_SHEET_SCHEMA`.

**Shape:**
```python
body: dict[str, Any] = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": max_tokens,
    "output_config": {
        "format": {"type": "json_schema", "schema": schema},
    },
    "messages": [{"role": "user", "content": user_text}],
}
```
The `schema` value must be a flattened JSON Schema Draft 2020-12 *subset*: no `$ref`/`$defs`, `additionalProperties: false` on every object, and no numeric/string/array range constraints (`_bedrock_schema` strips these, `src/claude_sql/core/schemas.py:39-96`).

**Assumptions consumers make:**
- The body uses `output_config.format`, never `tool_use`/`tool_choice`; tests assert `"tool_choice" not in body` (`tests/test_v2_analytics.py:123`).
- `thinking` is added as a sibling key only when `thinking_mode == "adaptive"` (`src/claude_sql/core/llm_shared.py:470-471`); workers assume Bedrock accepts thinking combined with `output_config` and expose a `disabled` escape hatch (`src/claude_sql/core/config.py:212`).
- The response is unwrapped by `_parse_structured_payload`, which assumes one of four observed envelope shapes and raises `RuntimeError` otherwise (`src/claude_sql/core/llm_shared.py:500-516`); a `stop_reason == "refusal"` is a terminal non-retryable outcome (`src/claude_sql/core/llm_shared.py:490-518`).

**Drift risk:** Bedrock's accepted JSON Schema subset is undocumented and has narrowed before (range constraints now rejected); a new pydantic `Field` constraint that `_bedrock_schema` does not strip would 400 at runtime for whichever worker owns that schema. Mitigation: extend the strip lists in `_flatten` (`schemas.py:82-92`) and keep the per-schema subset tests in `tests/test_schemas.py` / `tests/test_review_sheet_worker.py:49-61` green.

## Sharded parquet I/O API (`parquet_shards`)

The append-by-part / read-the-union cache contract shared by every analytics worker that persists rows, and by the CLI's `cache` subcommands.

**Producer:** `src/claude_sql/core/parquet_shards.py:87-242` (`write_part`, `read_all`, `iter_part_files`, `replace_sessions`, `count_rows`).

**Consumer(s):**
- `src/claude_sql/analytics/classify_worker.py:35` — `read_all`, `write_part`.
- `src/claude_sql/analytics/conflicts_worker.py:53` — `iter_part_files`, `read_all`, `write_part`.
- `src/claude_sql/analytics/trajectory_worker.py:54` — `iter_part_files`, `replace_sessions`, `write_part`.
- `src/claude_sql/analytics/ingest.py:53` — `iter_part_files`, `write_part`.
- `src/claude_sql/core/sql_views.py:1805` — `iter_part_files` (the `_parquet_is_populated` gate).
- `src/claude_sql/app/cli.py:83,598,1310` — `iter_part_files` (cache info/migrate subcommands).

**Shape:**
```python
def write_part(target: Path, df: pl.DataFrame) -> Path: ...
def read_all(target: Path, *, dtypes: dict[str, Any] | None = None) -> pl.DataFrame | None: ...
def iter_part_files(target: Path) -> list[Path]: ...
def replace_sessions(target: Path, *, key_column: str, session_ids: Iterable[str]) -> int: ...
def count_rows(target: Path) -> int: ...
```

**Assumptions consumers make:**
- A `target` `Path` is overloaded: a directory (or a non-`.parquet` path that does not yet exist) is a sharded cache, an existing `*.parquet` file is the legacy single-file shape (`src/claude_sql/core/parquet_shards.py:54-69`). Workers pass `Settings.*_parquet_path` and never branch on which mode is live.
- `read_all` returns `None` for an empty/missing cache; consumers null-check before reading columns (e.g. `src/claude_sql/analytics/conflicts_worker.py:146-148`).
- `iter_part_files` is sorted, and the sort is chronological because part names are `part-<time.time_ns()>.parquet` (`src/claude_sql/core/parquet_shards.py:118`); the analytics views rely on this when applying the `classified_at DESC` QUALIFY dedup (`src/claude_sql/core/sql_views.py:1921-1928`).
- `write_part` does not dedup; trajectory's at-least-once re-scoring depends on calling `replace_sessions(key_column="session_id", ...)` first to drop prior rows (`src/claude_sql/analytics/trajectory_worker.py:896-903`).

**Drift risk:** A worker that writes a `Settings.*_parquet_path` ending in `.parquet` silently takes the legacy read-then-rewrite branch (O(file) per chunk) instead of the sharded append, reintroducing the IO cost the module was built to remove. Mitigation: keep new cache defaults directory-shaped in `config.py` and assert sharded layout in worker tests.

## CLI output format + exit-code contract

The agent-facing surface: how every subcommand renders results and which integer it exits with. Agents pattern-match on these instead of parsing tracebacks.

**Producer:** `src/claude_sql/core/output.py:26-57` (`OutputFormat`, `EXIT_CODES`) and `output.py:125-141` (`ClassifiedError`).

**Consumer(s):**
- `src/claude_sql/app/cli.py:69-72` — imports `EXIT_CODES`, `ClassifiedError`, `OutputFormat`; used in every subcommand (e.g. `cli.py:692,1537,2877-2899`).
- `src/claude_sql/core/output.py:228-254` — `run_or_die` maps `InputValidationError`/`duckdb.Error` to `EXIT_CODES` and `sys.exit`.
- RFC 0001 reader semantics reuse these codes: exit 2 for "no binding", exit 70 for trailer/note disagreement (`docs/rfc/0001-transcript-pr-binding.md:425-429`), realized at `src/claude_sql/app/cli.py:2879-2899`.

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
    "invalid_input": 64,   # malformed user-supplied flags (e.g. --glob)
    "parse_error": 64,     # malformed SQL
    "catalog_error": 65,   # unknown view/macro/column
    "runtime_error": 70,   # everything else from duckdb.Error
    "duckdb_missing": 127, # system `duckdb` binary not on PATH
}
```

**Assumptions consumers make:**
- `OutputFormat.AUTO` resolves to `TABLE` on a TTY and `JSON` otherwise (`src/claude_sql/core/output.py:60-65`); subcommands assume piped/agent invocation gets machine-readable output with no flag.
- Markdown is deliberately *not* in the enum; only `review-sheet` emits prose and owns a separate `RenderFormat` (`src/claude_sql/core/output.py:33-38`, `src/claude_sql/app/cli.py:2913-2933`).
- `invalid_input` and `parse_error` share exit code 64 — a consumer cannot distinguish them by exit code alone and must read the `error.kind` field of the JSON payload (`src/claude_sql/core/output.py:134-141`).

**Drift risk:** Adding a new format to `OutputFormat` lights it up on every subcommand's `--format` even if only one command can produce it; the docstring at `output.py:33-38` records this as the reason markdown stays out. Mitigation: keep one-off render modes in a command-local enum, not the shared one.

## `VIEW_SCHEMA` — static v1 view column catalog

The hand-maintained source of truth for the 18 transcript-derived view schemas, traded against catalog introspection for a sub-50ms `schema` command.

**Producer:** `src/claude_sql/core/sql_views.py:116-287` (`VIEW_SCHEMA`).

**Consumer(s):**
- `src/claude_sql/app/cli.py:88-89,961,967` — the `schema` subcommand renders `VIEW_SCHEMA` for both JSON and table output.
- `tests/test_turn_window.py:23,339` — drift test asserts `turn_window` is present with the right columns.
- `tests/test_sql_views.py` (`test_view_schema_matches_describe_inline`, referenced at `sql_views.py:111-115`) — registers each v1 view over the fixture corpus, runs inline `DESCRIBE`, and asserts column-level equality with this dict.

**Shape (excerpt — `sessions`; full dict covers 18 views):**
```python
VIEW_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "sessions": (
        ("session_id", "VARCHAR"),
        ("cwd", "VARCHAR"),
        ("git_branch", "VARCHAR"),
        ("started_at", "TIMESTAMP"),
        ("ended_at", "TIMESTAMP"),
        ("assistant_messages", "BIGINT"),
        ("record_count", "BIGINT"),
        ("transcript_path", "VARCHAR"),
    ),
    # ... messages, content_blocks, messages_text, turn_window, tool_calls,
    #     tool_results, todo_events, todo_state_current, subagent_spawns,
    #     task_creations, task_updates, tasks_state_current, skill_invocations,
    #     subagent_sessions, subagent_messages
}
```

**Assumptions consumers make:**
- Only v1 views appear; v2 analytics views are intentionally omitted because their schema lives in parquet metadata (`src/claude_sql/core/sql_views.py:105-109`). The `schema` command thus advertises v1 columns even on a system with no analytics parquets.
- The dict order is the registration order in `register_views`; the table renderer at `cli.py:967` iterates it directly.
- Every column listed must also appear in `_RAW_EVENT_COLUMNS` (the strict `read_json` projection at `sql_views.py:438-449`) or it silently NULLs — the projection filter is a second, coupled contract (`sql_views.py:401-414`).

**Drift risk:** Editing a view's DDL without updating `VIEW_SCHEMA` makes the `schema` command lie to agents; a dropped top-level field in `_RAW_EVENT_COLUMNS` makes a column silently NULL. Both are caught by `test_view_schema_matches_describe_inline` as a hard CI failure (`sql_views.py:111-115`). No mitigation beyond keeping that test green.

## `MACRO_SIGNATURES` — static macro parameter catalog

Parameter names for all 21 macros, kept static because DuckDB returns NULL `parameters` for table macros and cannot be introspected at runtime.

**Producer:** `src/claude_sql/core/sql_views.py:355-376` (`MACRO_SIGNATURES`).

**Consumer(s):**
- `src/claude_sql/app/cli.py:87,963,972` — `schema` subcommand emits `{"name": n, "params": list(p)}` per macro.
- `src/claude_sql/core/sql_views.py:2182` — `list_macros` joins the live catalog against this dict to recover params.
- `tests/test_sql_views.py` (`test_macro_signatures_match_ddl`, referenced at `sql_views.py:349-354`) — regex-parses the `CREATE OR REPLACE MACRO <name>(<args>)` strings out of `register_macros` and asserts equality.

**Shape (excerpt; full dict has 21 entries):**
```python
MACRO_SIGNATURES: dict[str, tuple[str, ...]] = {
    "ago": ("interval_text",),
    "model_used": ("sid",),
    "cost_estimate": ("sid",),
    "tool_rank": ("last_n_days",),
    "semantic_search": ("query_vec", "k"),
    "sentiment_arc": ("sid",),
    "friction_examples": ("label_name", "n"),
    "canonical_uuid_resolve": (),
    # ... 13 more
}
```

**Assumptions consumers make:**
- `list_macros` returns an empty params tuple for any macro absent from this dict (`src/claude_sql/core/sql_views.py:2157-2182`); a new macro registered without a `MACRO_SIGNATURES` entry shows blank params rather than erroring.
- `MACRO_NAMES` (`sql_views.py:314-336`), `MACRO_SIGNATURES`, and `_ANALYTICS_MACRO_REQUIREMENTS` (`sql_views.py:1133-1149`) must list the same analytics-macro names; the docstring at `sql_views.py:1129-1132` notes adding a macro means touching all three.

**Drift risk:** Renaming a macro arg in the DDL without updating `MACRO_SIGNATURES` makes the `schema` command misreport the call signature to agents. Mitigation: the regex drift test `test_macro_signatures_match_ddl` (`sql_views.py:349-354`) fails CI on mismatch.

## `SessionClassification` — per-session classifier schema

One row per session: autonomy tier, work category, success, goal, confidence.

**Producer:** `src/claude_sql/core/schemas.py:99-167` (model) and `schemas.py:170` (`SESSION_CLASSIFICATION_SCHEMA`).

**Consumer(s):**
- `src/claude_sql/analytics/classify_worker.py:36,109` — sends the flattened schema to Bedrock and builds the parquet rows.
- `src/claude_sql/core/sql_views.py:1894-1895` — the `session_classifications` analytics view adds `autonomy`/`success_outcome`/`category` aliases over these columns; `session_goals` (`sql_views.py:1996-1998`) and `work_mix`/`autonomy_trend`/`success_rate_by_work` macros (`sql_views.py:1399-1442`) bind to `autonomy_tier`/`work_category`/`success`/`goal`.

**Shape:**
```python
class SessionClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    autonomy_tier: Literal["manual", "assisted", "autonomous"] = Field(...)
    work_category: Literal[
        "sde", "admin", "strategy_business", "events",
        "thought_leadership", "other",
    ] = Field(...)
    success: Literal["success", "partial", "failure", "unknown"] = Field(...)
    goal: str = Field(..., min_length=1, max_length=280)
    confidence: float = Field(..., ge=0.0, le=1.0)
```

**Assumptions consumers make:**
- The classify worker materializes a parquet row with exactly `session_id, autonomy_tier, work_category, success, goal, confidence, classified_at` (`src/claude_sql/analytics/classify_worker.py:138-160`) — it adds `session_id` + `classified_at` and drops nothing; the DuckDB macros assume those column names verbatim.
- The `work_mix` / `success_rate_by_work` macros assume the `Literal` label sets above are exhaustive (e.g. `WHERE success = 'failure'` at `sql_views.py:1434`); a new label appears in counts but no macro special-cases it.

**Drift risk:** The pydantic model, the worker's parquet `schema=` dict, and the DuckDB macro column references are three separate assertions of one shape with no shared constant. Adding a field to the model without adding it to `classify_worker.py:152-160` drops it before it reaches parquet. Mitigation: change all three together; `tests/test_v2_analytics.py:122` pins the Bedrock schema half.

## `TrajectoryArrayResult` / `TrajectoryWindow` — windowed sentiment schema

Sonnet returns an array of per-window turn-pair classifications; the host verifies completeness by echoing `(prev_uuid, curr_uuid)`.

**Producer:** `src/claude_sql/core/schemas.py:173-263` (`TrajectoryWindow`), `schemas.py:266-286` (`TrajectoryArrayResult`), `schemas.py:289` (`TRAJECTORY_ARRAY_SCHEMA`).

**Consumer(s):**
- `src/claude_sql/analytics/trajectory_worker.py:55,637` — sends the array schema, echoes `(prev_uuid, curr_uuid)` keys (`trajectory_worker.py:656`), retries on missing windows, stamps neutral placeholders on persistent misses.
- `src/claude_sql/core/sql_views.py:1903` — `message_trajectory` view adds `sentiment`/`transition` aliases; `sentiment_arc` macro joins on `mt.curr_uuid` (`sql_views.py:1513`).

**Shape:**
```python
class TrajectoryWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prev_uuid: str | None = Field(...)
    curr_uuid: str = Field(..., min_length=1)
    prev_sentiment: Literal["negative", "neutral", "positive"] | None = Field(...)
    curr_sentiment: Literal["negative", "neutral", "positive"] = Field(...)
    delta: float | None = Field(...)
    is_transition: bool = Field(...)
    transition_kind: Literal[
        "frustration_spike", "resolution", "reset",
        "drift", "clarification", "none",
    ] = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)

class TrajectoryArrayResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    windows: list[TrajectoryWindow] = Field(...)
```

**Assumptions consumers make:**
- `curr_uuid` must match exactly one `curr_uuid` supplied in the request `<window>` payload; the worker keys returned windows on `(prev_uuid, curr_uuid)` to detect drops (`src/claude_sql/analytics/trajectory_worker.py:656`).
- The parquet row is `session_id, prev_uuid, curr_uuid, prev_sentiment, curr_sentiment, delta, is_transition, transition_kind, confidence, classified_at` (`src/claude_sql/analytics/trajectory_worker.py:590-601`); the worker assumes `delta` widens from the model's integer-encoded `{-2..2}` into `pl.Float64`.
- The `sentiment_arc` macro joins `messages.uuid` to `message_trajectory.curr_uuid` (the per-window anchor, not the old per-message `uuid`) — assumes the v1.0 windowed rewrite happened (`src/claude_sql/core/sql_views.py:1500-1516`).

**Drift risk:** Read-side dedup keys on `(session_id, prev_uuid, curr_uuid)` with `classified_at DESC` (`sql_views.py:1921-1928`); if a future write path or legacy shard lands rows without `classified_at`, the QUALIFY is dropped and duplicate windows leak into `sentiment_arc`. Mitigation: keep the writer-side `replace_sessions` call and the read-side QUALIFY both in place.

## `ConflictsResult` / `ConflictPair` — stance-conflict schema

Zero or more pair-keyed stance conflicts per session; an empty list writes zero rows.

**Producer:** `src/claude_sql/core/schemas.py:292-376` (`ConflictPair`), `schemas.py:379-400` (`ConflictsResult`), `schemas.py:403` (`SESSION_CONFLICTS_SCHEMA`).

**Consumer(s):**
- `src/claude_sql/analytics/conflicts_worker.py:54,191,225-249` — sends the schema, builds one parquet row per pair.
- `src/claude_sql/core/sql_views.py:1910` — `session_conflicts` view (no projection aliases); `conflicts_summary` view counts rows per session (`sql_views.py:2015-2019`).

**Shape:**
```python
class ConflictPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_a_uuid: str = Field(..., min_length=1, max_length=64)
    turn_b_uuid: str = Field(..., min_length=1, max_length=64)
    conflict_kind: Literal["disagreement", "correction", "reversal", "impasse"] = Field(...)
    severity: Literal["low", "medium", "high"] = Field(...)
    agent_position: str = Field(..., min_length=1, max_length=280)
    user_position: str = Field(..., min_length=1, max_length=280)
    confidence: float = Field(..., ge=0.0, le=1.0)

class ConflictsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conflicts: list[ConflictPair] = Field(default_factory=list)
```

**Assumptions consumers make:**
- An empty `conflicts` list writes zero rows; the legacy `empty=True` sentinel is gone, so a session with no conflicts simply does not appear in `session_conflicts` or `conflicts_summary` (`src/claude_sql/core/schemas.py:379-400`, `sql_views.py:2005-2010`). Callers wanting every session must `LEFT JOIN conflicts_summary` and coalesce to 0.
- The parquet row is `session_id, turn_a_uuid, turn_b_uuid, conflict_kind, severity, agent_position, user_position, confidence, detected_at` (`src/claude_sql/analytics/conflicts_worker.py:70-80`) — the timestamp column is `detected_at` here, not `classified_at` as in the other workers.
- `turn_a_uuid`/`turn_b_uuid` are copied verbatim from `[uuid=...]` headers in the bound transcript and must differ (`schemas.py:303-323`).

**Drift risk:** No current drift risk on the empty-list semantics — the v1.0 sentinel removal is settled and the timestamp-column name difference (`detected_at`) is intentional and stable per RFC 0002 §3.4.

## `UserFrictionSignal` — per-message friction schema

Classifies one short user message for friction; applied only below `friction_max_chars`.

**Producer:** `src/claude_sql/core/schemas.py:406-471` (model), `schemas.py:474` (`USER_FRICTION_SCHEMA`).

**Consumer(s):**
- `src/claude_sql/analytics/friction_worker.py:62,554` — LLM path sends the schema; a regex fast-path also writes rows tagged `source='regex'` (`friction_worker.py:482-512`).
- `src/claude_sql/core/sql_views.py` — `user_friction` view backs the `friction_counts`/`friction_rate`/`friction_examples` macros (`sql_views.py:1527-1599`), which filter on `label`, `source`, `confidence`, `text_snippet`.

**Shape:**
```python
class UserFrictionSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal[
        "status_ping", "unmet_expectation", "confusion", "interruption",
        "correction", "frustration", "none",
    ] = Field(...)
    rationale: str = Field(..., min_length=1, max_length=200)
    confidence: float = Field(..., ge=0.0, le=1.0)
```

**Assumptions consumers make:**
- The parquet row is `uuid, session_id, ts, text_snippet, label, rationale, source, confidence, classified_at` (`src/claude_sql/analytics/friction_worker.py:320-330`) — the model contributes only `label, rationale, confidence`; the worker adds `uuid, session_id, ts, text_snippet` (truncated to 200 chars) and a `source` in `{regex, llm, refused}`.
- The macros assume `label != 'none'` is the interesting slice and exclude the `'none'` majority class (`sql_views.py:1536`); `friction_counts` further assumes `source` is exactly `'regex'` or `'llm'` for its `n_regex`/`n_llm` splits (`sql_views.py:1532-1533`).

**Drift risk:** A new friction `label` appears in `friction_counts` but `friction_rate` has per-label hard-coded columns (`n_status`, `n_unmet`, ... at `sql_views.py:1554-1559`); a new label is silently absent from the rate breakdown. Mitigation: add the matching `count(*) FILTER` column to `friction_rate` whenever a `label` literal is added.

## `lance_schema` — embeddings table schema

The pyarrow schema for the single LanceDB embeddings table; the fixed-size vector column is what makes it indexable.

**Producer:** `src/claude_sql/core/lance_store.py:60-75` (`lance_schema`), `lance_store.py:38` (`TABLE_NAME = "embeddings"`).

**Consumer(s):**
- `src/claude_sql/analytics/embed_worker.py:469-482` — builds the polars frame with `embedding: pl.Array(pl.Float32, dim)` matching the schema, then `add_chunk`.
- `src/claude_sql/analytics/cluster_worker.py:42` — reads the table via `lance_store._has_table(db, lance_store.TABLE_NAME)`.
- `src/claude_sql/core/sql_views.py:1716-1777` — `register_vss` ATTACHes the Lance dir and projects it as the `message_embeddings` DuckDB view, casting `embedding` to `FLOAT[dim]`; the `semantic_search` macro binds to it (`sql_views.py:1320-1327`).

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
- `embedding` must be a *fixed-size* list (`pa.list_(pa.float32(), dim)`); a variable-size `pl.List` is rejected by Lance for `create_index` and the docstring warns the polars side must be `pl.Array(pl.Float32, dim)` (`src/claude_sql/core/lance_store.py:102-109`).
- `register_vss` casts `embedding` to `FLOAT[dim]` so the cosine/distance macros keep working unchanged (`src/claude_sql/core/sql_views.py:1766-1777`); `dim` must match `Settings.output_dimension` (default 1024, `sql_views.py:1675`).
- The empty-dataset path creates an equivalent DuckDB-side `message_embeddings` table (uuid PK, model, dim, `embedding FLOAT[dim]`, embedded_at) so `semantic_search` still binds (`sql_views.py:1747-1758`).

**Drift risk:** The legacy parquet shards used `dim: pl.UInt16` while Lance wants `Int32`; migration recasts `dim` but deliberately lets embedding-column type drift (List vs Array) raise loudly at `tbl.add()` (`src/claude_sql/core/lance_store.py:193-204`). Mitigation: do not silently recast the embedding column — that masking hid drift bugs before.

## RFC 0001 transcript binding — git trailer + git-note wire format

The host-neutral pointer from a merged commit to the transcript that produced it: three commit trailers plus a JSON git note.

**Producer:** spec `docs/rfc/0001-transcript-pr-binding.md:130-261`; reference impl `src/claude_sql/provenance/binding.py:61-197` (constants, `TranscriptBinding`, `to_note_payload`).

**Consumer(s):**
- `src/claude_sql/app/cli.py:2877,2992` — `resolve` and `review-sheet` subcommands call `resolve_commit_to_transcript` and map `LookupError` to exit 2, `BindingMismatchError` to exit 70 (`cli.py:2879-2899`).
- `src/claude_sql/provenance/binding.py:515-645` — `read_trailer` / `read_note` parse the two surfaces; `resolve_commit_to_transcript` (`binding.py:674-743`) enforces trailer-wins precedence.
- `src/claude_sql/provenance/review_sheet_worker.py` / `review_sheet_render.py` — consume the resolved transcript to build the PR review sheet.

**Shape (trailers + note JSON):**
```
Claude-Transcript-Digest: sha256:<64-hex-chars>
Claude-Transcript-URI: <file://… | s3://… | git-notes://…>
Claude-Agent-Runtime: <vendor/version>
```
```json
{
  "uri": "file:///…/2026-05-09T03-48-12.jsonl",
  "digest": "sha256:9d4c0…",
  "agent_runtime": "claude-code/0.42.1",
  "transcript_id": "01HXYZ-…-session-id",
  "captured_at": "2026-05-09T03:51:08+00:00"
}
```

**Assumptions consumers make:**
- Resolution precedence is trailer-first, note-fallback, loud failure on `digest` disagreement (`docs/rfc/0001-transcript-pr-binding.md:312-331`); `read_trailer` returns the binding only when all three trailers are present and leaves `transcript_id`/`captured_at` empty since those are note-only (`src/claude_sql/provenance/binding.py:560-572`).
- Readers tolerate duplicate trailers (rebase/fixup squash) by taking the first occurrence per key (`src/claude_sql/provenance/binding.py:498-512`).
- Unknown trailers and unknown JSON keys are ignored for forward compatibility; only `digest`/`uri`/`agent_runtime` are required in the note (`src/claude_sql/provenance/binding.py:575-596`).
- `git notes show` exiting non-zero with "no note found" is the absence signal, not an error (`src/claude_sql/provenance/binding.py:628-635`).
- The digest is over raw file bytes with a `sha256:` prefix, recomputable in one line; canonicalization is explicitly rejected (`docs/rfc/0001-transcript-pr-binding.md:541-567`, `binding.py:232-246`).

**Drift risk:** Removing or renaming any of the three trailer keys or the five note fields is a breaking change requiring RFC supersession (`docs/rfc/0001-transcript-pr-binding.md:520-523`); adding fields is forward-compatible by construction. Mitigation: only add, never rename, and keep the ten `tests/test_binding.py` round-trip tests green.

## Layered import DAG (import-linter contracts)

The architectural contract every import is checked against: core is depended on by all, the three L1 siblings never import each other, and only app sits above them.

**Producer:** `pyproject.toml:261-280` (`[tool.importlinter]` root package + two contracts).

**Consumer(s):**
- Enforced in CI by `import-linter` (declared as a dev dependency, `pyproject.toml:97`); every module's import statements are the implicit consumer.
- `src/claude_sql/core/sql_views.py:1716` defers `from claude_sql.core import lance_store` to a function body — an in-layer import that keeps module-load-time clean; the DAG is the reason cross-layer imports always point down a layer.

**Shape:**
```toml
[[tool.importlinter.contracts]]
name = "claude-sql layered architecture"
type = "layers"
layers = [
    "claude_sql.app",
    "claude_sql.analytics | claude_sql.evals | claude_sql.provenance",
    "claude_sql.core",
]

[[tool.importlinter.contracts]]
name = "analytics / evals / provenance are independent siblings"
type = "independence"
modules = [
    "claude_sql.analytics",
    "claude_sql.evals",
    "claude_sql.provenance",
]
```

**Assumptions consumers make:**
- Shared logic that two siblings both need must live in `core`; it cannot be imported sibling-to-sibling because the independence contract forbids it. This is why `schemas.py`, `llm_shared.py`, `parquet_shards.py`, `lance_store.py`, and `output.py` all live in core.
- `app` is the only layer allowed to import from the L1 siblings (e.g. `cli.py` imports binding, the workers, and the render module).

**Drift risk:** A convenience import added sibling-to-sibling (e.g. `provenance` importing an `analytics` helper) compiles and runs locally but fails the `independence` contract in CI. Mitigation: hoist the shared code into `core` rather than crossing siblings; import-linter fails the build on violation.

## Other contracts

- **`Settings.*_parquet_path` config fields** — `src/claude_sql/core/config.py` declares the per-cache paths and pricing; consumed by every worker and by `register_analytics` / `_ANALYTICS_MACRO_REQUIREMENTS` (`sql_views.py:1133-1149,1856-1886`). One producer, many consumers, but it is configuration plumbing rather than a shape contract.
- **`Correction` (nested in `PRReviewSheet`)** — `src/claude_sql/core/schemas.py:477-506`; only consumed transitively via `PRReviewSheet` by the review-sheet worker, so it rides that contract rather than standing alone.
- **`PRReviewSheet`** — `src/claude_sql/core/schemas.py:509-577`, `PR_REVIEW_SHEET_SCHEMA` at `schemas.py:580`; single consumer `src/claude_sql/provenance/review_sheet_worker.py:430` plus the renderer `review_sheet_render.py:75`. Six-field schema budgeted for a ~1K-token Markdown render; structurally a public boundary (the `review-sheet` CLI surface) but only one worker produces it.
- **`ANALYTICS_VIEW_NAMES` / `VIEW_NAMES`** — `src/claude_sql/core/sql_views.py:62-93,294-307`; consumed by `cli.py:483` as the analytics-view name guard and by smoke tests; a naming subset of the view contracts above.
- **`ClassifiedError.to_payload` JSON error shape** — `src/claude_sql/core/output.py:134-141`; the `{"error": {"kind", "message", "hint"}}` envelope agents read on stderr; consumed by `emit_error` and every `run_or_die` failure path.

## See also

- [claude-sql · Processes](../behavior/processes.md) — 13 shared source files
- [claude-sql · Public API](../reference/public-api.md) — 13 shared source files
- [claude-sql · Debugging guide](debugging-guide.md) — 12 shared source files
- [claude-sql · Module map](../architecture/module-map.md) — 12 shared source files
- [claude-sql · Tech debt](tech-debt.md) — 10 shared source files
