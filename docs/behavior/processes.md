# claude-sql · Processes

Inventory of "what runs when." Every process listed below is a Cyclopts subcommand on the single `app` instance declared at `src/claude_sql/cli.py:164`. The console-script entry point is `main()` at `src/claude_sql/cli.py:2911`, which delegates to `app()`.

Two helpers thread through nearly every process:
- `_resolve_settings(common)` builds the `Settings` from env + CLI overrides at `src/claude_sql/cli.py:209`.
- `_open_connection_full(settings, sql=...)` opens an in-memory DuckDB and calls `register_all`, optionally skipping VSS when the SQL doesn't need it, at `src/claude_sql/cli.py:338`.

## analyze

Entry point: `src/claude_sql/cli.py:2046`

1. Configure logging and resolve settings via `_configure(common)` + `_resolve_settings(common)` at `src/claude_sql/cli.py:2113`.
2. Stage 0 — skills sync via `_skills_catalog.sync(settings)` (filesystem walk, zero cost) at `src/claude_sql/cli.py:2120`.
3. Open one shared DuckDB connection with full catalog registration: `_open_connection_full(settings)` at `src/claude_sql/cli.py:2137`.
4. Stage 1 — ingest stamps: `_ingest_count_pending` for dry-run, else `_ingest_stamp_messages` + `_refresh_analytics_views` + `_ingest_resolve_canonicals` at `src/claude_sql/cli.py:2144`.
5. Stage 2 — embed via `asyncio.run(run_backfill(...))`, then `_rebind_vss(con, settings, stage="embed")` + `_refresh_analytics_views` at `src/claude_sql/cli.py:2156`.
6. Stage 3 — cluster via `run_clustering(settings, force=...)` and stage 3b — `run_terms(con, settings, force=...)` at `src/claude_sql/cli.py:2176`.
7. Stage 4 — community detection via `_rebind_vss` + `run_communities(con, settings, force=...)` at `src/claude_sql/cli.py:2200`.
8. Stages 5-8 — `classify_sessions`, `trajectory_messages`, `detect_conflicts`, `detect_user_friction` in order, each followed by `_refresh_analytics_views` at `src/claude_sql/cli.py:2210`.

### Related

- `src/claude_sql/cli.py:378` (`_refresh_analytics_views`)
- `src/claude_sql/cli.py:398` (`_rebind_vss`)
- `src/claude_sql/skills_catalog.py:1`
- `src/claude_sql/embed_worker.py:391` (`run_backfill`)
- `src/claude_sql/classify_worker.py:195` (`classify_sessions`)
- `src/claude_sql/community_worker.py:472` (`run_communities`)

## query

Entry point: `src/claude_sql/cli.py:728`

1. Configure logging and resolve `Settings` via `_configure(common)` + `_resolve_settings(common)` at `src/claude_sql/cli.py:786`.
2. Pick the connection helper via `_sql_uses_catalog(sql)`: full registration when the SQL touches a known view/macro/VSS token, bare connection otherwise at `src/claude_sql/cli.py:789`.
3. When `--profile-json` is set, install DuckDB profiling PRAGMAs via `_capture_profile(con, label="query")` at `src/claude_sql/cli.py:797`.
4. Execute the SQL through the error-classifying wrapper `run_or_die(lambda: con.execute(sql).pl(), fmt=fmt)` at `src/claude_sql/cli.py:798`.
5. Emit the result with `emit_dataframe(df, fmt)` (table on TTY, JSON on pipe) at `src/claude_sql/cli.py:799`.
6. If a profile path was captured, log its location at `src/claude_sql/cli.py:800`.
7. Always close the connection in `finally` at `src/claude_sql/cli.py:803`.

### Related

- `src/claude_sql/cli.py:469` (`_sql_uses_catalog`)
- `src/claude_sql/cli.py:709` (`_capture_profile`)
- `src/claude_sql/output.py:1` (`run_or_die` / `emit_dataframe`)
- `src/claude_sql/sql_views.py:2081` (`register_all`)
- `src/claude_sql/cli.py:338` (`_open_connection_full`)

## embed

Entry point: `src/claude_sql/cli.py:1397`

1. Lazy-import `asyncio`, configure logging, resolve settings at `src/claude_sql/cli.py:1441`.
2. Open a bare in-memory DuckDB connection at `src/claude_sql/cli.py:1445`.
3. Register raw views (`register_raw`) and derived views (`register_views`) so `messages_text` is available at `src/claude_sql/cli.py:1447`.
4. `asyncio.run(run_backfill(...))` discovers unembedded messages, batches them, and writes to LanceDB at `src/claude_sql/cli.py:1454`.
5. Inside `run_backfill`, `discover_unembedded(con, lance_uri=...)` produces the pending list at `src/claude_sql/embed_worker.py:421`.
6. Each chunk goes through `embed_documents_async(texts, settings=settings)` then is appended to the Lance table opened by `lance_store.connect_db` + `lance_store.open_or_create_table` at `src/claude_sql/embed_worker.py:470`.
7. Return value is normalized to stdout JSON via `_emit_worker_result(result, common, pipeline="embed")` at `src/claude_sql/cli.py:1464`.

### Related

- `src/claude_sql/embed_worker.py:101` (`discover_unembedded`)
- `src/claude_sql/embed_worker.py:290` (`embed_documents_async`)
- `src/claude_sql/embed_worker.py:267` (`_embed_one_batch`)
- `src/claude_sql/lance_store.py:1`
- `src/claude_sql/cli.py:488` (`_emit_worker_result`)

## classify

Entry point: `src/claude_sql/cli.py:1568`

1. Configure logging and resolve settings at `src/claude_sql/cli.py:1616`.
2. Open the full catalog connection via `_open_connection_full(settings)` at `src/claude_sql/cli.py:1618`.
3. Dispatch to `classify_sessions(con, settings, ...)` which forks dry-run vs. real-run at `src/claude_sql/cli.py:1620`.
4. Real-run path: load `already`-classified set via `read_all(settings.classifications_parquet_path)`, then filter pending sessions through `checkpointer.filter_unchanged` and the `retry_queue.drain` set at `src/claude_sql/classify_worker.py:54`.
5. Build the Bedrock client and capacity limiter, then iterate chunks of `max(batch_size * 4, 256)` sessions, gathering `classify_one(...)` coroutines per chunk at `src/claude_sql/classify_worker.py:89`.
6. Successful results are written as a fresh part shard via `write_part(settings.classifications_parquet_path, df)` and checkpointed via `checkpointer.mark_completed` + `retry_queue.mark_done` at `src/claude_sql/classify_worker.py:162`.
7. Emit the worker result through `_emit_worker_result(result, common, pipeline="classify")` at `src/claude_sql/cli.py:1629`.

### Related

- `src/claude_sql/classify_worker.py:45` (`_classify_sessions_async`)
- `src/claude_sql/llm_shared.py:1` (`classify_one`, `_build_bedrock_client`, `pipeline_cache_stats`)
- `src/claude_sql/checkpointer.py:1`
- `src/claude_sql/retry_queue.py:1`
- `src/claude_sql/parquet_shards.py:1` (`write_part`, `read_all`)
- `src/claude_sql/session_text.py:1` (`iter_session_texts`, `session_bounds`)

## trajectory

Entry point: `src/claude_sql/cli.py:1635`

1. Configure logging and resolve settings at `src/claude_sql/cli.py:1664`.
2. Open the full catalog connection via `_open_connection_full(settings)` at `src/claude_sql/cli.py:1666`.
3. Call `trajectory_messages(con, settings, ...)` which selects dry-run vs. real-run inside one entry at `src/claude_sql/cli.py:1668`.
4. Dry-run path counts text turns directly off `messages_text` and estimates Sonnet calls as `ceil(turns / MAX_WINDOWS_PER_CHUNK)` at `src/claude_sql/trajectory_worker.py:948`.
5. Real-run path enters `pipeline_cache_stats("trajectory")` then `_trajectory_async(con, settings, ...)` to load windows, chunk them, classify each chunk via Sonnet, and persist part-shards at `src/claude_sql/trajectory_worker.py:1004`.
6. Emit the worker result through `_emit_worker_result(result, common, pipeline="trajectory")` at `src/claude_sql/cli.py:1677`.

### Related

- `src/claude_sql/trajectory_worker.py:681` (`_trajectory_async`)
- `src/claude_sql/trajectory_worker.py:385` (`_load_windows`)
- `src/claude_sql/trajectory_worker.py:445` (`_chunk_windows`)
- `src/claude_sql/trajectory_worker.py:609` (`_classify_chunk`)
- `src/claude_sql/schemas.py:1` (`TrajectoryArrayResult`)

## friction

Entry point: `src/claude_sql/cli.py:1727`

1. Configure logging and resolve settings at `src/claude_sql/cli.py:1758`.
2. Open the full catalog connection via `_open_connection_full(settings)` at `src/claude_sql/cli.py:1760`.
3. Dispatch to `detect_user_friction(con, settings, ...)` at `src/claude_sql/cli.py:1762`.
4. Dry-run path runs `_candidate_sql(settings.friction_max_chars, since_days)` to count short-message candidates and estimate cost at `src/claude_sql/friction_worker.py:695`.
5. Real-run path enters `pipeline_cache_stats("friction")` then `_classify_async(con, settings, ...)` which pulls candidates, runs `regex_fast_path` for cheap labels, and dispatches the rest to Sonnet 4.6 at `src/claude_sql/friction_worker.py:729`.
6. The async worker writes part shards to `settings.user_friction_parquet_path` with a `source ∈ {regex, llm, refused}` column at `src/claude_sql/friction_worker.py:420`.
7. Emit the worker result through `_emit_worker_result(result, common, pipeline="friction")` at `src/claude_sql/cli.py:1771`.

### Related

- `src/claude_sql/friction_worker.py:148` (`regex_fast_path`)
- `src/claude_sql/friction_worker.py:271` (`sql_stamp`)
- `src/claude_sql/friction_worker.py:349` (`_candidate_sql`)
- `src/claude_sql/friction_worker.py:420` (`_classify_async`)
- `src/claude_sql/schemas.py:1` (`USER_FRICTION_SCHEMA`)

## community

Entry point: `src/claude_sql/cli.py:1908`

1. Configure logging, resolve settings, and resolve the format at `src/claude_sql/cli.py:1966`.
2. Validate that `--neighbors-of` is mutually exclusive with `--gamma`, `--force`, `--dry-run`; emit a classified `invalid_input` error and exit 64 on conflict at `src/claude_sql/cli.py:1970`.
3. Open the full catalog connection via `_open_connection_full(settings)` at `src/claude_sql/cli.py:1983`.
4. Early-return path: `--neighbors-of <sid>` calls `neighbors_of(con, settings, sid, top_k=...)` and emits a dataframe at `src/claude_sql/cli.py:1985`.
5. Dry-run path counts candidate sessions via a `COUNT(DISTINCT m.session_id)` join over `message_embeddings` and emits a plan JSON at `src/claude_sql/cli.py:1990`.
6. Real-run path calls `run_communities(con, settings, force=..., gamma=..., resolution=...)` which loads centroids, builds a mutual-kNN igraph, optionally runs the resolution profile, then `_run_leiden_cpm` at `src/claude_sql/cli.py:2020`.
7. Emit the worker result through `_emit_worker_result(stats, common, pipeline="community")` at `src/claude_sql/cli.py:2040`.

### Related

- `src/claude_sql/community_worker.py:421` (`neighbors_of`)
- `src/claude_sql/community_worker.py:472` (`run_communities`)
- `src/claude_sql/community_worker.py:147` (`_build_mutual_knn`)
- `src/claude_sql/community_worker.py:202` (`_compute_resolution_profile`)
- `src/claude_sql/community_worker.py:281` (`_run_leiden_cpm`)
- `src/claude_sql/community_worker.py:331` (`_compute_medoid_and_coherence`)

## review-sheet

Entry point: `src/claude_sql/cli.py:2777`

1. Configure logging, resolve render format via `_review_sheet_format(common)`, resolve error format and settings at `src/claude_sql/cli.py:2817`.
2. Pre-resolve the commit's bound transcript via `_binding.resolve_commit_to_transcript(commit_sha, repo=repo_path)` at `src/claude_sql/cli.py:2830`.
3. Map `BindingMismatchError` → exit 70, `LookupError` → exit 2, `GitInvocationError` → exit 65 via `emit_error` at `src/claude_sql/cli.py:2831`.
4. Hand the resolved URI to `generate_review_sheet(None, settings, commit_sha=..., transcript_uri_override=binding.uri, dry_run=..., no_thinking=...)` at `src/claude_sql/cli.py:2861`.
5. Worker resolves transcript URI, flattens the JSONL via `_flatten_jsonl_to_text`, computes a digest, and either returns a plan dict or invokes the Sonnet structured-output classifier at `src/claude_sql/review_sheet_worker.py:384`.
6. Dry-run branch emits the plan JSON via `emit_json(plan, fmt=OutputFormat.JSON)` at `src/claude_sql/cli.py:2870`.
7. Refusal branch routes to `render_refusal_markdown` (TTY) or JSON; success branch routes to `render_markdown(sheet, metadata)` (TTY) or JSON at `src/claude_sql/cli.py:2878`.

### Related

- `src/claude_sql/binding.py:671` (`resolve_commit_to_transcript`)
- `src/claude_sql/review_sheet_worker.py:340` (`generate_review_sheet`)
- `src/claude_sql/review_sheet_worker.py:192` (`_flatten_jsonl_to_text`)
- `src/claude_sql/review_sheet_render.py:1` (`render_markdown`, `render_refusal_markdown`)
- `src/claude_sql/schemas.py:1` (`PR_REVIEW_SHEET_SCHEMA`)

## Minor flows

- shell — entry at `src/claude_sql/cli.py:639`. Materializes a temp DuckDB file, runs `register_all`, then execs the system `duckdb` REPL against the file.
- explain — entry at `src/claude_sql/cli.py:807`. Wraps the user SQL with `EXPLAIN` (or `EXPLAIN ANALYZE` when `--analyze`) and highlights pushdown markers in green on a TTY.
- schema — entry at `src/claude_sql/cli.py:917`. Reads the static `VIEW_SCHEMA` + `MACRO_SIGNATURES` dicts, joins a `cached` map from `_compute_cached_map(settings)`, emits per-format.
- list-cache — entry at `src/claude_sql/cli.py:980`. Composes `{name, path, exists, bytes, mtime, rows}` rows via `_describe_cache_entry`, `_describe_checkpoint_entry`, `_describe_lance_entry`.
- search — entry at `src/claude_sql/cli.py:1470`. Embeds the query via `embed_query`, then runs an HNSW cosine-distance lookup over `message_embeddings` joined to `messages_text`.
- ingest — entry at `src/claude_sql/cli.py:1777`. Stamps `approx_tokens` / `simhash64` / `canonical_uuid` via `_ingest_stamp_messages` + `_ingest_resolve_canonicals` (`src/claude_sql/ingest.py:295` / `src/claude_sql/ingest.py:410`).
- cluster — entry at `src/claude_sql/cli.py:1845`. Runs `run_clustering(settings, force=...)` (`src/claude_sql/cluster_worker.py:50`) which fits UMAP + HDBSCAN with mtime-sidecar skip.
- terms — entry at `src/claude_sql/cli.py:1876`. Runs `run_terms(con, settings, force=...)` to compute c-TF-IDF labels for clusters.
- conflicts — entry at `src/claude_sql/cli.py:1683`. Dispatches to `detect_conflicts(con, settings, ...)` (`src/claude_sql/conflicts_worker.py:286`); pair-keyed `(turn_a_uuid, turn_b_uuid)` shape.
- cache compact — entry at `src/claude_sql/cli.py:1108`. Walks each sharded cache directory, reads every `part-*.parquet`, writes one `part-compacted-<ts_ns>.parquet`, deletes originals on success.
- cache migrate — entry at `src/claude_sql/cli.py:1206`. Renames legacy `~/.claude/<name>.parquet` files into `~/.claude/<name>/part-<original_mtime_ns>.parquet`.
- skills sync — entry at `src/claude_sql/cli.py:1309`. Walks `~/.claude/skills` and `~/.claude/plugins/cache` via `_skills_catalog.sync(settings, dry_run=...)`.
- skills ls — entry at `src/claude_sql/cli.py:1349`. Reads `settings.skills_catalog_parquet_path` and filters by `--kind` / `--plugin`.
- judges — entry at `src/claude_sql/cli.py:2270`. Lists the cross-provider Bedrock judge catalog via `_judge_catalog.catalog()`.
- freeze — entry at `src/claude_sql/cli.py:2290`. Pre-registers a study via `_freeze.freeze(rubric_path=..., panel_shortnames=..., session_scope=..., seed=...)`.
- replay — entry at `src/claude_sql/cli.py:2332`. Loads and echoes a frozen study via `_freeze.replay(manifest_sha)`.
- judge — entry at `src/claude_sql/cli.py:2372`. Replays a study, validates the sessions parquet, dispatches `_judge_worker.run(...)`; defaults to `--dry-run`.
- ungrounded-claim — entry at `src/claude_sql/cli.py:2426`. Replays a study, validates the turns parquet, runs `_ungrounded_worker.detect(turns, freeze_sha=...)`.
- kappa — entry at `src/claude_sql/cli.py:2463`. Computes Cohen's + Fleiss' κ with bootstrapped 95% CI; exits 66 when any axis is below `--floor` or the delta gate trips.
- blind-handover — entry at `src/claude_sql/cli.py:2341`. Strips identity markers from a `(session_id, text)` parquet via `_blind_handover.strip_text`; adds `original_hash`.
- bind — entry at `src/claude_sql/cli.py:2538`. Resolves the active transcript via `_binding.find_active_transcript`, builds a binding, writes trailers via `_binding.write_trailer` and a JSON note via `_binding.write_note` when `--no-dry-run`.
- resolve — entry at `src/claude_sql/cli.py:2671`. Reads `Claude-Transcript-*` trailers first, falls back to the `refs/notes/transcripts` JSON note, raises on digest disagreement; supports `--all-sources` for diagnostics.

## See also

- [claude-sql · CLI](../reference/cli.md) — 31 shared citations
- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 10 shared citations
- [claude-sql · Risk hotspots](../analysis/risk-hotspots.md) — 9 shared citations
- [claude-sql · Public API](../reference/public-api.md) — 9 shared citations
- [claude-sql · Module map](../architecture/module-map.md) — 7 shared citations
