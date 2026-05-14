# claude-sql · Impact analysis

This file answers the reader question: *"If I touch X, what else do I have
to think about?"*

A **high-impact surface** in this codebase is the public symbol from a
module whose name appears in 8 or more inbound `from claude_sql.<module>`
or `from claude_sql import <module>` statements across `src/` and
`tests/`. Module-level inbound count is the ranking criterion; within each
module the H2 names the load-bearing symbol (a function, class, or
exported constant) whose signature change would propagate into the
listed downstream consumers. The cap is 8 H2 surfaces; everything below
the cutoff but still load-bearing is captured in the trailing
*Other notable surfaces* section.

`Type` cells use the closed vocabulary
`direct import` / `indirect` / `runtime dispatch` / `test` / `config`.
`Touch on change` is `yes` (consumer must be touched), `likely`
(probably needs review even without a signature change), or `no`
(only behavioral change would propagate).

## `Settings`

Defined at: `src/claude_sql/config.py:127`.

Pydantic v2 `BaseSettings` reading `CLAUDE_SQL_*` env vars. Every CLI
subcommand and every worker takes a `Settings` instance; field renames
or removals propagate everywhere.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `cli._resolve_settings` constructs `Settings()` and passes it everywhere | direct import | yes | `src/claude_sql/cli.py:217` |
| `sql_views.register_macros` / `register_vss` / `register_analytics` accept `settings: Settings` | direct import | yes | `src/claude_sql/sql_views.py:1218`, `src/claude_sql/sql_views.py:1835`, `src/claude_sql/sql_views.py:2084` |
| `cluster_worker.run_clustering(settings: Settings, ...)` | direct import | yes | `src/claude_sql/cluster_worker.py:50` |
| `community_worker` `_load_session_centroids` / `run_communities` take `settings` | direct import | yes | `src/claude_sql/community_worker.py:423`, `src/claude_sql/community_worker.py:474` |
| `classify_worker._classify_sessions_async(settings: Settings, ...)` | direct import | yes | `src/claude_sql/classify_worker.py:47` |
| `trajectory_worker._classify_trajectory_async(settings: Settings, ...)` | direct import | yes | `src/claude_sql/trajectory_worker.py:611` |
| `conflicts_worker._classify_conflicts_async(settings: Settings, ...)` | direct import | yes | `src/claude_sql/conflicts_worker.py:133` |
| `friction_worker._classify_friction_async(settings: Settings, ...)` | direct import | yes | `src/claude_sql/friction_worker.py:422` |
| `review_sheet_worker.run_review_sheet(settings: Settings, ...)` | direct import | yes | `src/claude_sql/review_sheet_worker.py:342` |
| `terms_worker.run_terms(settings: Settings, ...)` | direct import | yes | `src/claude_sql/terms_worker.py:31` |
| `ingest._existing_parts_clause` / `_ingest_*` chain takes `Settings` | direct import | yes | `src/claude_sql/ingest.py:264`, `src/claude_sql/ingest.py:276` |
| `session_text.SessionTextCorpus.assemble(settings: Settings)` and `iter_session_texts(settings, ...)` | direct import | yes | `src/claude_sql/session_text.py:99`, `src/claude_sql/session_text.py:374` |
| `llm_shared._build_bedrock_client(settings)` keys client cache on `(region, llm_concurrency)` | direct import | yes | `src/claude_sql/llm_shared.py:346` |
| `embed_worker` reads `settings.lance_uri` / `settings.output_dimension` / `settings.hnsw_metric` | direct import | yes | `src/claude_sql/embed_worker.py:470`, `src/claude_sql/embed_worker.py:524` |
| `skills_catalog` walks `settings.user_skills_dir` / `settings.plugins_cache_dir` | direct import | yes | `src/claude_sql/config.py:269`, `src/claude_sql/config.py:273` |
| `tests/test_config.py` exercises every default + env override | test | yes | `tests/test_config.py:25` |
| `tests/test_team_corpus.py` re-reads `Settings()` after env mutations | test | yes | `tests/test_team_corpus.py:239` |
| `tests/test_home.py` builds `Settings()` to verify `claude_sql_home` plumbing | test | yes | `tests/test_home.py:127` |
| `CLAUDE_SQL_*` env vars in user shells / CI | config | likely | `src/claude_sql/config.py:134` |
| (15 more `from claude_sql.config import Settings` sites under `src/claude_sql/`, see `grep -l "from claude_sql.config import"`) | direct import | yes | `src/claude_sql/config.py:127` |

### Blast-radius notes

- The `concurrency` field is a deprecated alias that mirrors itself onto
  `embed_concurrency` *and* `llm_concurrency` only when those still hold
  their default values; renaming or removing it requires deleting the
  `_resolve_concurrency_alias` validator in lockstep, otherwise existing
  `CLAUDE_SQL_CONCURRENCY=...` shells silently no-op (`src/claude_sql/config.py:384`).
- `team_corpus_root` rewrites `default_glob` / `subagent_glob` /
  `subagent_meta_glob` only when no per-glob user pin is detected. The
  detection compares against the factory defaults, so renaming
  `_default_glob()` *and* the `default_glob` field at the same time
  silently breaks the rewrite (`src/claude_sql/config.py:344`).
- Path-typed fields default to functions that resolve `claude_sql_home()`
  at call time, not module-import time. Tests that monkeypatch
  `CLAUDE_SQL_HOME` rely on this — switching to module-level constants
  would re-introduce the test-pollution class fixed in v1.0
  (`src/claude_sql/config.py:35`).

## `register_views` / `register_macros` / `register_vss` / `register_raw`

Defined at: `src/claude_sql/sql_views.py:487` (`register_raw`),
`src/claude_sql/sql_views.py:622` (`register_views`),
`src/claude_sql/sql_views.py:1216` (`register_macros`),
`src/claude_sql/sql_views.py:1691` (`register_vss`),
`src/claude_sql/sql_views.py:1832` (`register_analytics`),
`src/claude_sql/sql_views.py:2081` (`register_all`).

The DuckDB binding seam. Every CLI subcommand opens a connection and
runs one of these to materialize 18 views + 14 macros + the LanceDB VSS
attach. View names, column shapes, and macro signatures are the
contract every analytics consumer reads.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `cli._open_connection_full` runs `register_all(con, settings, skip_vss=skip_vss)` | direct import | yes | `src/claude_sql/cli.py:360` |
| `cli._refresh_analytics_views` re-runs `register_analytics` after worker writes | direct import | yes | `src/claude_sql/cli.py:393` |
| `cli._rebind_vss` calls `register_vss` after embeddings advance | direct import | yes | `src/claude_sql/cli.py:433` |
| `cli.bind_cmd` re-binds `register_raw` + `register_views` against a custom glob | direct import | yes | `src/claude_sql/cli.py:1447` |
| `embed_worker` notes `register_analytics_views` only binds `ingest_stamps` when its parquet exists | indirect | likely | `src/claude_sql/embed_worker.py:84` |
| `community_worker._load_session_centroids` reads `message_embeddings` (registered by `register_vss`) | indirect | yes | `src/claude_sql/community_worker.py:87` |
| `trajectory_worker` reads the `turn_window` view (registered by `register_views`) | indirect | yes | `src/claude_sql/trajectory_worker.py:398` |
| `friction_worker` joins `messages_text` + `friction_*` macros bound by `register_macros` | indirect | yes | `src/claude_sql/friction_worker.py:53` |
| `tests/conftest.py` `registered_con` fixture runs `register_raw` + `register_views` | test | yes | `tests/conftest.py:297` |
| `tests/test_embed_canonical_skip.py` mounts `register_raw` + `register_views` directly | test | yes | `tests/test_embed_canonical_skip.py:117` |
| `tests/test_llm_worker_pipelines.py` builds its own `register_raw` + `register_views` con | test | yes | `tests/test_llm_worker_pipelines.py:46` |
| `tests/test_sql_views.py` covers macro shapes + view DDL contract | test | yes | `tests/test_sql_views.py:1` |
| `tests/test_v2_analytics.py` covers per-pipeline analytics views | test | yes | `tests/test_v2_analytics.py:1` |
| `claude-sql shell` REPL drops into a connection with `register_all` already run | runtime dispatch | yes | `src/claude_sql/cli.py:651` |

### Blast-radius notes

- View registration is **conditional on parquet presence**. Renaming any
  `Settings.*_parquet_path` field requires updating the
  `_analytics_macro_ready` gate so missing parquets keep no-opping
  instead of failing (`src/claude_sql/sql_views.py:1175`).
- `register_vss` issues `CREATE OR REPLACE VIEW` against the LanceDB
  ATTACH'd alias and may be called multiple times in one process. The
  alias name and table name (`embeddings`) are load-bearing — renaming
  the LanceDB table breaks every re-bind path (`src/claude_sql/cli.py:419`).
- `register_macros` must run *after* `register_vss` because macros that
  consume `message_embeddings` bind eagerly and fail with a catalog
  error otherwise. The `register_all` orchestrator sequences these
  intentionally; do not reorder (`src/claude_sql/sql_views.py:2081`).

## `parquet_shards.write_part` / `read_all` / `iter_part_files`

Defined at: `src/claude_sql/parquet_shards.py:87` (`write_part`),
`src/claude_sql/parquet_shards.py:131` (`read_all`),
`src/claude_sql/parquet_shards.py:72` (`iter_part_files`).

Sharded-cache I/O contract. Five worker outputs (embeddings,
classifications, trajectory, conflicts, friction) are written as
`<dir>/part-<ts_ns>.parquet` files; `read_all` is the canonical reader.
The directory-vs-file shape is detected by `is_sharded_dir`; both
branches must stay correct because legacy single-file caches still
exist on user disks until they run `claude-sql cache migrate`.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `classify_worker` reads + appends classifications via `read_all` / `write_part` | direct import | yes | `src/claude_sql/classify_worker.py:35` |
| `trajectory_worker` reads + appends trajectory rows; calls `replace_sessions` to drop stale shards | direct import | yes | `src/claude_sql/trajectory_worker.py:47` |
| `conflicts_worker` reads + appends conflicts; calls `replace_sessions` on rerun | direct import | yes | `src/claude_sql/conflicts_worker.py:45` |
| `friction_worker` reads + appends user_friction rows | direct import | yes | `src/claude_sql/friction_worker.py:53` |
| `ingest._existing_parts_clause` joins `iter_part_files` paths into a `read_parquet` glob | direct import | yes | `src/claude_sql/ingest.py:264` |
| `cli.cache_compact` consolidates shards into a single file | direct import | yes | `src/claude_sql/cli.py:1108` |
| `cli.cache_migrate` migrates legacy single-file caches to sharded directories | direct import | yes | `src/claude_sql/cli.py:1206` |
| `sql_views` view-readiness gate uses `count_rows` + `iter_part_files` to gate analytics binding | direct import | yes | `src/claude_sql/sql_views.py:1175` |
| `tests/test_parquet_shards.py` covers `is_sharded_dir`, `write_part`, `read_all`, `replace_sessions` | test | yes | `tests/test_parquet_shards.py:1` |
| `tests/test_trajectory_windowed.py` exercises `replace_sessions` for rerun semantics | test | yes | `tests/test_trajectory_windowed.py:1` |
| `tests/test_conflicts_storage_v2.py` exercises pair-keyed shard rewrites | test | yes | `tests/test_conflicts_storage_v2.py:1` |
| `tests/test_ingest.py` covers ingest stamp shard pruning | test | yes | `tests/test_ingest.py:1` |
| `tests/test_friction_sql_stamps.py` covers friction shard layout | test | yes | `tests/test_friction_sql_stamps.py:1` |
| `tests/test_friction_worker_llm.py` exercises `read_all` + `write_part` end-to-end | test | yes | `tests/test_friction_worker_llm.py:1` |

### Blast-radius notes

- `is_sharded_dir` returns *True* for any non-existent path that does
  not end in `.parquet`. New `Settings.*_parquet_path` defaults must
  not accidentally end in `.parquet` if they want the sharded layout
  (`src/claude_sql/parquet_shards.py:54`).
- `replace_sessions` deletes shards that empty out after a filter —
  callers that read shard counts before and after a rerun must accept
  shrinking shard sets (`src/claude_sql/parquet_shards.py:222`).
- Shard filenames are nanosecond-resolution (`part-<time.time_ns()>.parquet`);
  a writer that races and produces sub-nanosecond colliding writes
  would silently overwrite — concurrency above the per-pipeline
  semaphore in `llm_shared` is therefore the load-bearing invariant
  (`src/claude_sql/parquet_shards.py:118`).

## `llm_shared.classify_one` and the Bedrock client cache

Defined at: `src/claude_sql/llm_shared.py:563` (`classify_one`),
`src/claude_sql/llm_shared.py:346` (`_build_bedrock_client`),
`src/claude_sql/llm_shared.py:259` (`pipeline_cache_stats`),
`src/claude_sql/llm_shared.py:490` (`BedrockRefusalError`).

Single-entry async path through Bedrock for every Sonnet 4.6
classification. The boto3 client is process-cached on
`(region, max_pool_connections)`; `pipeline_cache_stats` is the
context manager every worker wraps its run in to surface 1 h
prompt-cache hit/write totals.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `classify_worker` imports `classify_one`, `_build_bedrock_client`, `_count_pending_sessions`, `_estimate_cost`, `pipeline_cache_stats`, `CLASSIFY_SYSTEM_PROMPT` | direct import | yes | `src/claude_sql/classify_worker.py:27` |
| `trajectory_worker` imports the same set + uses `BedrockRefusalError` for terminal-refusal rows | direct import | yes | `src/claude_sql/trajectory_worker.py:47` |
| `conflicts_worker` imports the same set + uses `BedrockRefusalError` | direct import | yes | `src/claude_sql/conflicts_worker.py:45` |
| `friction_worker` imports the same set + uses `BedrockRefusalError` | direct import | yes | `src/claude_sql/friction_worker.py:53` |
| `review_sheet_worker` imports `_invoke_classifier_sync` for sync PR-review path | direct import | yes | `src/claude_sql/review_sheet_worker.py:44` |
| `tests/test_llm_worker.py` covers `classify_one` happy path + retry + refusal | test | yes | `tests/test_llm_worker.py:1` |
| `tests/test_llm_worker_cache.py` covers prompt-cache stat plumbing | test | yes | `tests/test_llm_worker_cache.py:1` |
| `tests/test_llm_worker_parser.py` covers `_parse_structured_payload` shapes | test | yes | `tests/test_llm_worker_parser.py:1` |
| `tests/test_llm_worker_pipelines.py` exercises `_build_bedrock_client` cache reuse + `_count_pending_sessions` | test | yes | `tests/test_llm_worker_pipelines.py:501` |
| `tests/test_trajectory_coverage.py` autouse fixture clears `_CLIENT_CACHE` per test | test | yes | `tests/test_trajectory_coverage.py:45` |
| `tests/test_conflicts_coverage.py` autouse fixture clears `_CLIENT_CACHE` | test | yes | `tests/test_conflicts_coverage.py:48` |
| `tests/test_trajectory_windowed.py` autouse fixture clears `_CLIENT_CACHE` | test | yes | `tests/test_trajectory_windowed.py:51` |

### Blast-radius notes

- The client cache key is `(region, pool_size)` where pool size is
  derived from `settings.llm_concurrency`. Renaming or splitting
  `llm_concurrency` requires updating both `_build_bedrock_client` *and*
  every test that monkeypatches `_CLIENT_CACHE` between runs
  (`src/claude_sql/llm_shared.py:343`).
- `BedrockRefusalError` is **terminal**: every worker stamps a neutral
  placeholder row and clears the retry queue when it fires. Renaming
  the class or its semantics breaks the
  refused-session-doesn't-cycle-forever invariant called out in
  CLAUDE.md (`src/claude_sql/llm_shared.py:490`).
- `_parse_structured_payload` accepts four observed Bedrock response
  shapes; adding a fifth (e.g., a future tool-use envelope) requires
  the new branch *and* coverage in `test_llm_worker_parser.py` so
  `RuntimeError("Unexpected response shape")` doesn't start firing on
  fresh model versions (`src/claude_sql/llm_shared.py:500`).

## `schemas.SessionClassification` / `TrajectoryArrayResult` / `ConflictsResult` / `UserFrictionSignal` / `PRReviewSheet` and their `_SCHEMA` flatten dicts

Defined at: `src/claude_sql/schemas.py:99` (`SessionClassification`),
`src/claude_sql/schemas.py:266` (`TrajectoryArrayResult`),
`src/claude_sql/schemas.py:379` (`ConflictsResult`),
`src/claude_sql/schemas.py:406` (`UserFrictionSignal`),
`src/claude_sql/schemas.py:509` (`PRReviewSheet`); flatten dicts at
`src/claude_sql/schemas.py:170`, `:289`, `:403`, `:474`, `:580`.

Pydantic v2 models that double as Bedrock structured-output schemas
(via `_bedrock_schema(model)`). A field-rename on any model touches
the worker's parsing code, the parquet schema written by
`write_part`, and every analytics view that reads those columns.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `classify_worker` consumes `SESSION_CLASSIFICATION_SCHEMA` | direct import | yes | `src/claude_sql/classify_worker.py:36` |
| `trajectory_worker` consumes `TRAJECTORY_ARRAY_SCHEMA` | direct import | yes | `src/claude_sql/trajectory_worker.py:47` |
| `conflicts_worker` consumes `SESSION_CONFLICTS_SCHEMA` | direct import | yes | `src/claude_sql/conflicts_worker.py:45` |
| `friction_worker` consumes `USER_FRICTION_SCHEMA` | direct import | yes | `src/claude_sql/friction_worker.py:53` |
| `review_sheet_worker` + `review_sheet_render` consume `PR_REVIEW_SHEET_SCHEMA` | direct import | yes | `src/claude_sql/review_sheet_worker.py:44`, `src/claude_sql/review_sheet_render.py:1` |
| `tests/test_schemas.py` exercises every flatten dict's shape | test | yes | `tests/test_schemas.py:1` |
| `tests/test_conflicts_storage_v2.py` covers v1.0 pair-keyed shape | test | yes | `tests/test_conflicts_storage_v2.py:1` |
| `tests/test_trajectory_windowed.py` covers windowed trajectory shape | test | yes | `tests/test_trajectory_windowed.py:1` |
| `tests/test_friction_worker.py` + `_llm.py` cover friction shape | test | yes | `tests/test_friction_worker.py:1` |
| `tests/test_review_sheet_worker.py` covers PR review sheet shape | test | yes | `tests/test_review_sheet_worker.py:1` |
| `tests/test_v2_analytics.py` covers downstream analytics views over the v2 parquets | test | yes | `tests/test_v2_analytics.py:1` |
| `tests/test_llm_worker.py` exercises round-trip with each schema | test | yes | `tests/test_llm_worker.py:1` |
| Bedrock structured-output API contract (Draft 2020-12 subset) | runtime dispatch | likely | `src/claude_sql/schemas.py:16` |
| Per-pipeline parquet caches whose columns mirror these models | indirect | yes | `src/claude_sql/config.py:238` |

### Blast-radius notes

- `_flatten` strips numeric range constraints (`minimum`, `maximum`),
  string format constraints, and array length constraints because
  Bedrock's JSON Schema subset rejects them; the pydantic models still
  enforce these at parse time. Adding a constraint that Bedrock now
  accepts (a future schema-subset expansion) requires updating
  `_flatten` *and* a guard test (`src/claude_sql/schemas.py:79`).
- The pair-keyed conflicts shape (`turn_a_uuid`, `turn_b_uuid`) and the
  windowed trajectory shape (`prev_uuid`, `curr_uuid`) are referenced
  by parquet columns *and* by SQL macros (`sentiment_arc`,
  `conflicts_summary`). Renaming a key column requires the macro
  rewrites called out in CLAUDE.md alongside the schema bump.
- `additionalProperties: false` is injected at every object level. A
  pydantic model that later adds a field without a default will
  silently fail Bedrock's schema validation in production — every
  field add must ship with a default *and* a regression test
  (`src/claude_sql/schemas.py:76`).

## `cli.app`, `cli._open_connection_full`, `cli.main`

Defined at: `src/claude_sql/cli.py:164` (`app`),
`src/claude_sql/cli.py:338` (`_open_connection_full`),
`src/claude_sql/cli.py:2911` (`main`).

The cyclopts typer-style entry point and the canonical connection
factory. Tests, the `claude-sql` console script, and several
external fixtures call directly into private helpers like
`_rebind_vss`, `_resolve_memory_limit`, and `_capture_profile`.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `claude-sql` console script wired to `cli:main` | runtime dispatch | yes | `pyproject.toml [project.scripts]` |
| `tests/test_cli.py` imports the `cli` module + `Common` dataclass | test | yes | `tests/test_cli.py:24` |
| `tests/test_cli_coverage.py` imports `cli` + `Common` for subcommand coverage | test | yes | `tests/test_cli_coverage.py:32` |
| `tests/test_analyze_chain.py` imports `_rebind_vss` + accesses `cli` module directly | test | yes | `tests/test_analyze_chain.py:39` |
| `tests/test_binding.py` imports `bind_cmd` + `resolve_cmd` | test | yes | `tests/test_binding.py:419` |
| `tests/test_config.py` imports `_resolve_memory_limit` | test | yes | `tests/test_config.py:20` |
| `tests/test_pr3_perf.py` imports `cli`, `sql_views`, and `Common` | test | yes | `tests/test_pr3_perf.py:28` |
| `tests/test_profile_json.py` imports `_capture_profile`, `_profile_path_for` | test | yes | `tests/test_profile_json.py:17` |
| `tests/test_community_worker_coverage.py` imports `cli` + `Common` | test | yes | `tests/test_community_worker_coverage.py:22` |
| Every `@app.command`-decorated subcommand (shell, query, embed, search, classify, trajectory, conflicts, friction, ingest, cluster, terms, community, analyze, judges, freeze, ...) | runtime dispatch | yes | `src/claude_sql/cli.py:638` |

### Blast-radius notes

- `Common` is the cyclopts shared-flags dataclass mounted on every
  subcommand via `Parameter(name="*")`. Adding a flag here changes the
  signature of every test that constructs `Common(...)` directly —
  every test fixture that mocks subcommand invocation must be updated
  in the same change (`src/claude_sql/cli.py:171`).
- `_open_connection_full` runs `register_all`, which is the
  load-bearing path for view DDL. Tests that need a faster connection
  reach for `_open_connection_introspect` instead — those two factories
  must remain feature-paired so introspect callers don't see analytics
  views (`src/claude_sql/cli.py:364`).
- Private helpers (`_rebind_vss`, `_capture_profile`, `_resolve_memory_limit`)
  are imported by name from tests; renaming any of them is a
  test-breaking change even though the underscore prefix suggests
  otherwise (`tests/test_analyze_chain.py:39`,
  `tests/test_profile_json.py:17`, `tests/test_config.py:20`).

## `lance_store.connect_db` / `_has_table` / `TABLE_NAME` / `add_chunk` / `ensure_index`

Defined at: `src/claude_sql/lance_store.py:78` (`connect_db`),
`src/claude_sql/lance_store.py:48` (`_has_table`),
`src/claude_sql/lance_store.py:38` (`TABLE_NAME`),
`src/claude_sql/lance_store.py:102` (`add_chunk`),
`src/claude_sql/lance_store.py:114` (`ensure_index`).

The LanceDB embeddings store. Replaces the pre-v1.0
`embeddings/part-*.parquet` + `hnsw.duckdb` combo. Holds the
FLOAT[1024] vectors + IVF_HNSW_SQ index in one versioned directory;
DuckDB reads it back via `INSTALL lance; LOAD lance; ATTACH (TYPE LANCE)`.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `embed_worker` writes vectors via `connect_db` + `open_or_create_table` + `add_chunk` + `ensure_index` | direct import | yes | `src/claude_sql/embed_worker.py:470` |
| `embed_worker.get_embedded_uuids` filters out already-embedded UUIDs | direct import | yes | `src/claude_sql/embed_worker.py:145` |
| `cluster_worker` opens the table and pulls FLOAT[dim] vectors | direct import | yes | `src/claude_sql/cluster_worker.py:36` |
| `sql_views.register_vss` probes via `_has_table` then runs ATTACH + view DDL | direct import | yes | `src/claude_sql/sql_views.py:1763` |
| `cli._describe_lance_entry` reports row counts via `count_rows` | direct import | yes | `src/claude_sql/cli.py:580` |
| `tests/test_lance_store.py` covers the empty-namespace gate + table CRUD | test | yes | `tests/test_lance_store.py:1` |
| `tests/test_embed_worker.py` end-to-end exercises `connect_db` + `add_chunk` | test | yes | `tests/test_embed_worker.py:55` |
| `tests/test_community_worker.py` builds an embedding table to feed centroids | test | yes | `tests/test_community_worker.py:136` |
| `tests/test_cluster_worker.py` covers cluster reuse via Lance mtime | test | yes | `tests/test_cluster_worker.py:1` |
| `tests/test_hnsw_persistence.py` covers IVF_HNSW_SQ index persistence | test | yes | `tests/test_hnsw_persistence.py:1` |
| `tests/test_embed_canonical_skip.py` covers canonical-skip behavior over Lance | test | yes | `tests/test_embed_canonical_skip.py:93` |
| LanceDB on-disk dataset at `~/.claude/embeddings_lance/` | config | likely | `src/claude_sql/config.py:43` |

### Blast-radius notes

- `TABLE_NAME = "embeddings"` is the hard-coded table name referenced
  by both the producer (`embed_worker`) and the SQL ATTACH
  (`sql_views.register_vss` SELECTs from `lance_store.main.embeddings`).
  Renaming it is a coordinated three-file change
  (`src/claude_sql/lance_store.py:38`, `src/claude_sql/sql_views.py:1798`,
  `src/claude_sql/cluster_worker.py:39`).
- `lance_schema(dim)` declares `embedding` as a *fixed-size* list of
  float32. A regular `pa.list_(pa.float32())` breaks `create_index`
  with no clear error. Tests that synthesize embedding DataFrames must
  use `pl.Array(pl.Float32, dim)`, not `pl.List`
  (`src/claude_sql/lance_store.py:60`).
- `connect_db` caches connections on resolved-path key. Tests that
  rotate `lance_uri` between runs without clearing `_DB_CACHE` will
  see stale handles — every Lance-touching test must either use
  `tmp_path` exclusively or wipe the cache explicitly
  (`src/claude_sql/lance_store.py:43`).

## `checkpointer.mark_completed` / `filter_unchanged` / `load_as_map` / `count_rows`

Defined at: `src/claude_sql/checkpointer.py:327` (`mark_completed`),
`src/claude_sql/checkpointer.py:283` (`filter_unchanged`),
`src/claude_sql/checkpointer.py:262` (`load_as_map`),
`src/claude_sql/checkpointer.py:367` (`count_rows`).

Per-`(session_id, pipeline)` SQLite checkpoint. Every classifier
worker (classify, trajectory, conflicts, friction) calls
`filter_unchanged` to drop sessions whose `(last_ts, last_mtime)` has
not advanced, then calls `mark_completed` after the chunk lands.
`PIPELINE_NAMES` is the closed enum of valid `pipeline` values.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `classify_worker` uses `checkpointer.filter_unchanged` + `mark_completed` | direct import | yes | `src/claude_sql/classify_worker.py:61`, `src/claude_sql/classify_worker.py:169` |
| `trajectory_worker` uses `filter_unchanged` + `mark_completed` (+ partial in-loop checkpointing) | direct import | yes | `src/claude_sql/trajectory_worker.py:696`, `src/claude_sql/trajectory_worker.py:914` |
| `conflicts_worker` uses `filter_unchanged` + `mark_completed` | direct import | yes | `src/claude_sql/conflicts_worker.py:151`, `src/claude_sql/conflicts_worker.py:256` |
| `friction_worker` uses `filter_unchanged` + `mark_completed` (+ partial in-loop) | direct import | yes | `src/claude_sql/friction_worker.py:437`, `src/claude_sql/friction_worker.py:532` |
| `retry_queue` imports the SQLite `_connect` helpers from `checkpointer` | direct import | yes | `src/claude_sql/retry_queue.py:28` |
| `cli.list_cache` reports checkpoint row count via `count_rows` | direct import | likely | `src/claude_sql/cli.py:546` |
| `tests/test_checkpointer.py` covers happy + edge cases | test | yes | `tests/test_checkpointer.py:1` |
| `tests/test_checkpointer_extras.py` covers `_to_iso` / `_from_iso` round-trips | test | yes | `tests/test_checkpointer_extras.py:24` |
| `tests/test_conflicts_coverage.py` exercises `mark_completed` for conflicts | test | yes | `tests/test_conflicts_coverage.py:575` |
| `tests/test_friction_worker_llm.py` reads `count_rows` to assert checkpoint advanced | test | yes | `tests/test_friction_worker_llm.py:602` |
| `tests/test_retry_queue.py` re-imports `mark_completed` for cross-table assertions | test | yes | `tests/test_retry_queue.py:116` |

### Blast-radius notes

- `PIPELINE_NAMES` is a closed tuple at module top
  (`src/claude_sql/checkpointer.py:41`); adding a new pipeline (e.g.,
  ungrounded, judge) requires appending here, then wiring the worker
  to call `mark_completed(pipeline=<new name>)` so reruns skip
  unchanged sessions.
- `filter_unchanged` returns `(pending_session_ids, skipped_count)`.
  The skipped count is a load-bearing observability metric — every
  worker logs it. Tightening the function to return only the pending
  list silently drops the metric (`src/claude_sql/checkpointer.py:283`).
- `_migrate_from_duckdb_if_present` runs once per process+path on
  first connect and writes a sentinel. A future SQLite schema bump
  must also bump the sentinel filename, otherwise legacy DuckDB rows
  are not re-migrated against the new shape
  (`src/claude_sql/checkpointer.py:91`).

## Other notable surfaces

- `session_text.iter_session_texts` / `session_bounds`
  (`src/claude_sql/session_text.py:371`, `src/claude_sql/session_text.py:224`):
  consumed by all four classifiers + tests; sits just below the
  cutoff at 6 inbound imports.
- `retry_queue.enqueue` / `drain` / `mark_done`
  (`src/claude_sql/retry_queue.py:85`, `src/claude_sql/retry_queue.py:131`,
  `src/claude_sql/retry_queue.py:162`): per-pipeline retry persistence;
  every classifier worker drains its queue at the start of a run and
  enqueues failures at the end.
- `output.OutputFormat` / `emit_dataframe` / `emit_json` / `classify_duckdb_error`
  (`src/claude_sql/output.py:26`, `src/claude_sql/output.py:68`,
  `src/claude_sql/output.py:111`, `src/claude_sql/output.py:180`):
  agent-friendly CLI plumbing — every subcommand renders through
  these. New DuckDB exception subclasses require updating
  `classify_duckdb_error` per CLAUDE.md.
- `home.claude_sql_home` (`src/claude_sql/home.py:1`): every default
  parquet path resolves through it; rerouting the corpus root
  changes every `Settings.*_parquet_path` factory in lockstep.
- `logging_setup.configure_logging` / `loguru_before_sleep`
  (`src/claude_sql/logging_setup.py:1`): the loguru-only logging seam
  documented in CLAUDE.md as the reason `import logging` is banned;
  every tenacity `@retry` decorator must use `loguru_before_sleep`.

## See also

- [claude-sql · Contract map](../insights/contract-map.md) — 22 shared citations
- [claude-sql · Public API](../reference/public-api.md) — 13 shared citations
- [claude-sql · Processes](../behavior/processes.md) — 10 shared citations
- [claude-sql · Risk hotspots](../analysis/risk-hotspots.md) — 7 shared citations
- [claude-sql · Tech debt](../insights/tech-debt.md) — 6 shared citations
