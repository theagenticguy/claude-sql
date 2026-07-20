# claude-sql · Processes

`claude-sql` has no HTTP routes, RPC handlers, or scheduled jobs. Every process
is initiated by a cyclopts CLI subcommand registered on the `app` object in
`src/claude_sql/interfaces/cli/app.py`; the console-script entry point is
`main` (`src/claude_sql/interfaces/cli/app.py:2115`), bound to
`claude_sql.interfaces.cli.app:main`. Command bodies are thin: they configure
logging, resolve `Settings`, open a DuckDB connection, and delegate to an
application-layer use case under `src/claude_sql/application/`. The eight
full-treatment processes below are the load-bearing analytics and read paths;
the rest are collected under `## Minor flows`.

A shared connection lifecycle underlies most processes — `open_connection_full`
(`src/claude_sql/infrastructure/duckdb_connection.py:149`) registers raw views,
derived views, VSS, and analytics via `register_all`
(`src/claude_sql/infrastructure/duckdb_views.py:2285`); `refresh_analytics_views`
(`src/claude_sql/infrastructure/duckdb_connection.py:189`) and `rebind_vss`
(`src/claude_sql/infrastructure/duckdb_connection.py:207`) re-bind parquet- and
Lance-backed views after a stage writes new data (RFC §9.6).

## analyze

Entry point: `src/claude_sql/interfaces/cli/app.py:1992` (`analyze` command) → `run_analyze` (`src/claude_sql/application/analyze.py:99`)

1. Apply `--embedding-provider` / `--llm-analytics-provider` overrides to `Settings` before delegating `src/claude_sql/interfaces/cli/app.py:2069`.
2. Stage 0 — skills catalog sync (filesystem walk, zero-cost) `src/claude_sql/application/analyze.py:169`.
3. Open one shared DuckDB connection for every catalog-reading stage `src/claude_sql/application/analyze.py:187`.
4. Stage 1 — ingest stamps, then double-refresh bracketing `resolve_canonicals` so `embed` sees canonical UUIDs `src/claude_sql/application/analyze.py:195`.
5. Stage 2 — embed backfill, then `rebind_vss("embed")` + `refresh_analytics_views` so downstream stages see the fresh vectors `src/claude_sql/application/analyze.py:207`.
6. Stage 3 — cluster (UMAP+HDBSCAN), refresh, then terms (c-TF-IDF) `src/claude_sql/application/analyze.py:227`.
7. Stage 5 — community detection, preceded by the RFC §9.6 `rebind_vss("cluster_terms")` + refresh (the named stale-connection bug site) `src/claude_sql/application/analyze.py:251`.
8. Stages 6–9 — classify, trajectory, conflicts, friction (each LLM, each `refresh_analytics_views` after), then `con.close()` in `finally` `src/claude_sql/application/analyze.py:262`.

### Related

- `src/claude_sql/application/analyze.py:56` (`DuckDbReader` — ReaderPort impl holding the shared connection)
- `src/claude_sql/infrastructure/duckdb_connection.py:207` (`rebind_vss`)
- `src/claude_sql/infrastructure/duckdb_connection.py:189` (`refresh_analytics_views`)
- `src/claude_sql/application/use_cases/embed.py:222` (`run_backfill`)
- `src/claude_sql/application/use_cases/community.py:228` (`run_communities`)

## query

Entry point: `src/claude_sql/interfaces/cli/app.py:540` (`query` command)

1. Configure logging and resolve `Settings` with `--glob` validation `src/claude_sql/interfaces/cli/app.py:599`.
2. Route connection: full-catalog if the SQL references a view/macro, else a bare introspection connection, via `sql_uses_catalog` `src/claude_sql/interfaces/cli/app.py:602`.
3. Optionally set the DuckDB profiling PRAGMAs when `--profile-json` is passed `src/claude_sql/interfaces/cli/app.py:609`.
4. Execute the SQL through `run_or_die` (classifies DuckDB errors into exit 64/65/70) and materialize a polars frame `src/claude_sql/interfaces/cli/app.py:611`.
5. Emit the frame in the resolved format (table on TTY, JSON on pipe) `src/claude_sql/interfaces/cli/app.py:612`.
6. Close the connection in `finally` `src/claude_sql/interfaces/cli/app.py:616`.

### Related

- `src/claude_sql/infrastructure/duckdb_connection.py:281` (`sql_uses_catalog`)
- `src/claude_sql/infrastructure/duckdb_connection.py:149` (`open_connection_full`)
- `src/claude_sql/infrastructure/duckdb_connection.py:175` (`open_connection_introspect`)
- `src/claude_sql/interfaces/cli/output.py` (`run_or_die`, `emit_dataframe`, `resolve_format`)
- `src/claude_sql/infrastructure/duckdb_views.py:2285` (`register_all`)

## embed

Entry point: `src/claude_sql/interfaces/cli/app.py:1306` (`embed` command) → `run_backfill` (`src/claude_sql/application/use_cases/embed.py:222`)

1. Apply the `--embedding-provider` override and open an in-memory connection with raw + derived views registered `src/claude_sql/interfaces/cli/app.py:1361`.
2. Discover unembedded `(uuid, text)` pairs by anti-joining the Lance store's embedded UUIDs (and the optional `ingest_stamps` canonical-dedup gate) `src/claude_sql/application/use_cases/embed.py:258`.
3. Short-circuit to a plan dict when `--dry-run` or nothing is pending `src/claude_sql/application/use_cases/embed.py:264`.
4. Build the embedding provider once; capture its `model_id` and `dimension` as the vector-space contract `src/claude_sql/application/use_cases/embed.py:304`.
5. Fail loud if the store's stamped `(model, dim)` differs from the provider's, via `ensure_store_matches` `src/claude_sql/application/use_cases/embed.py:318`.
6. Embed each chunk, write fixed-size `Array(Float32, dim)` rows to LanceDB, and periodically `store.optimize()` `src/claude_sql/application/use_cases/embed.py:334`.
7. Final `store.optimize()` + `store.ensure_index(metric=...)` so `search` hits the HNSW index `src/claude_sql/application/use_cases/embed.py:384`.

### Related

- `src/claude_sql/application/use_cases/embed.py:82` (`discover_unembedded`)
- `src/claude_sql/infrastructure/embedding/__init__.py` (`build_embedder`, `ensure_store_matches`)
- `src/claude_sql/infrastructure/adapters.py` (`build_vector_store`, `LanceVectorStore`)
- `src/claude_sql/infrastructure/embedding/cohere_bedrock.py` (`_embed_one_batch`, `_invoke_bedrock_sync`)
- `src/claude_sql/infrastructure/settings.py` (`expected_embedding_identity`)

## search

Entry point: `src/claude_sql/interfaces/cli/app.py:1388` (`search` command) → `DuckDbSessionSearch.search` (`src/claude_sql/infrastructure/session_search.py:129`)

1. Apply the `--embedding-provider` override and subclass the search adapter so the query embedding routes through the `embed_query` use case (monkeypatch seam) `src/claude_sql/interfaces/cli/app.py:1464`.
2. Open + minimally register the connection (raw + views + VSS only) on first `search` call `src/claude_sql/infrastructure/session_search.py:87`.
3. Guard on an empty embeddings store — return `[]` rather than run the kNN `src/claude_sql/infrastructure/session_search.py:137`.
4. Embed the query string to a float vector at the store's native width `src/claude_sql/infrastructure/session_search.py:142`.
5. Run the inline cosine-kNN SQL: `ORDER BY array_cosine_distance ASC` to trigger the HNSW cosine index, joined back to `messages_text` for a 200-char snippet `src/claude_sql/infrastructure/session_search.py:160`.
6. Map rows to typed `SearchHit`s; the CLI emits them as a frame and exits code 2 on an empty store `src/claude_sql/interfaces/cli/app.py:1471`.

### Related

- `src/claude_sql/infrastructure/session_search.py:38` (`DuckDbSessionSearch`)
- `src/claude_sql/application/use_cases/embed.py:199` (`embed_query`)
- `src/claude_sql/infrastructure/duckdb_views.py:1824` (`register_vss` — fail-loud provider/dim guard)
- `src/claude_sql/domain/retrieval.py` (`SearchHit`)

## classify

Entry point: `src/claude_sql/interfaces/cli/app.py:1491` (`classify` command) → `classify_sessions` (`src/claude_sql/application/use_cases/classify.py:204`)

1. Resolve thinking mode and default the parquet `CachePort`; short-circuit to a cost-estimate plan dict on `--dry-run` `src/claude_sql/application/use_cases/classify.py:231`.
2. Read already-classified session IDs from the cache for the anti-join `src/claude_sql/application/use_cases/classify.py:66`.
3. Compute session bounds and skip unchanged sessions via the checkpointer `src/claude_sql/application/use_cases/classify.py:75`.
4. Drain the retry queue and merge failed sessions back into the keep set `src/claude_sql/application/use_cases/classify.py:89`.
5. Assemble pending session texts via `iter_session_texts` `src/claude_sql/application/use_cases/classify.py:95`.
6. Dispatch parallel `classify_one` structured-output calls under a `CapacityLimiter`, gathering results per chunk `src/claude_sql/application/use_cases/classify.py:124`.
7. Write OK rows to a parquet shard, mark sessions completed in the checkpoint, and clear them from the retry queue `src/claude_sql/application/use_cases/classify.py:176`.

### Related

- `src/claude_sql/application/use_cases/classify.py:53` (`_classify_sessions_async`)
- `src/claude_sql/infrastructure/bedrock/client.py` (`classify_one`, `_build_bedrock_client`, `pipeline_cache_stats`)
- `src/claude_sql/infrastructure/session_text_loader.py` (`iter_session_texts`, `session_bounds`)
- `src/claude_sql/infrastructure/sqlite_state/checkpointer.py` (`filter_unchanged`)
- `src/claude_sql/infrastructure/adapters.py` (`build_cache`, `build_checkpoint`, `build_retry_queue`)

## community

Entry point: `src/claude_sql/interfaces/cli/app.py:1859` (`community` command) → `run_communities` (`src/claude_sql/application/use_cases/community.py:228`)

1. Reject `--neighbors-of` combined with partition flags (exit 64); route the early-return neighbors path when requested `src/claude_sql/interfaces/cli/app.py:1928`.
2. On `--dry-run`, count candidate sessions with embeddings via pure SQL and emit a plan `src/claude_sql/interfaces/cli/app.py:1948`.
3. Return cached stats if `session_communities.parquet` exists and `--force` is off `src/claude_sql/application/use_cases/community.py:265`.
4. Load per-session centroids: join `message_embeddings` to `messages`, mean-reduce with `np.add.reduceat`, L2-normalize `src/claude_sql/application/use_cases/community.py:290`.
5. Build the mutual-kNN cosine graph and the igraph object `src/claude_sql/application/use_cases/community.py:297`.
6. Auto-pick CPM γ from the resolution profile (unless `--gamma` given), then run `_run_leiden_cpm` `src/claude_sql/application/use_cases/community.py:310`.
7. Compute medoid + coherence, relabel by descending size and collapse sub-min-size to noise `src/claude_sql/application/use_cases/community.py:361`.
8. Write `session_communities.parquet` and, when auto-γ ran, the `community_profile.parquet` sidecar `src/claude_sql/application/use_cases/community.py:384`.

### Related

- `src/claude_sql/application/use_cases/community.py:73` (`_load_session_centroids`)
- `src/claude_sql/application/use_cases/community.py:177` (`neighbors_of`)
- `src/claude_sql/domain/structure/community.py` (`_build_mutual_knn`, `_run_leiden_cpm`, `_compute_resolution_profile`, `_relabel_and_collapse`)
- `src/claude_sql/infrastructure/settings.py` (`community_config`)

## friction

Entry point: `src/claude_sql/interfaces/cli/app.py:1666` (`friction` command) → `detect_user_friction` (`src/claude_sql/application/use_cases/friction.py:572`)

1. On `--dry-run`, count short user-message candidates and emit a cost-estimate plan `src/claude_sql/application/use_cases/friction.py:616`.
2. Read already-classified UUIDs, filter unchanged sessions via the checkpointer, and drain retries `src/claude_sql/application/use_cases/friction.py:354`.
3. Pull user-role messages under `friction_max_chars`, excluding Claude Code system markers `src/claude_sql/application/use_cases/friction.py:378`.
4. Regex fast-path: stamp unambiguous `status_ping` / `interruption` / `correction` at 0.9 confidence `src/claude_sql/application/use_cases/friction.py:404`.
5. SQL stamp layer: three deterministic DuckDB rules (repeated body, imperative revert, trailing-`?` after an error) tag remaining rows without Bedrock `src/claude_sql/application/use_cases/friction.py:425`.
6. Write the fast-path rows to a parquet shard; return early if nothing needs the LLM `src/claude_sql/application/use_cases/friction.py:454`.
7. LLM path: parallel `classify_one` calls with `USER_FRICTION_SCHEMA`; refusals become neutral `none` rows, failures enqueue for retry `src/claude_sql/application/use_cases/friction.py:474`.
8. Write OK rows, mark done in the retry queue, and checkpoint processed sessions `src/claude_sql/application/use_cases/friction.py:539`.

### Related

- `src/claude_sql/application/use_cases/friction.py:341` (`_classify_async`)
- `src/claude_sql/application/use_cases/friction.py:192` (`sql_stamp`)
- `src/claude_sql/domain/friction.py` (`regex_fast_path`, `_REGEX_BANK`)
- `src/claude_sql/infrastructure/bedrock/structured_output.py` (`USER_FRICTION_SCHEMA`)
- `src/claude_sql/infrastructure/bedrock/client.py` (`classify_one`, `pipeline_cache_stats`)

## ingest

Entry point: `src/claude_sql/interfaces/cli/app.py:1718` (`ingest` command)

1. On `--dry-run`, count pending rows via pure SQL and emit a plan `src/claude_sql/interfaces/cli/app.py:1766`.
2. Stamp messages: pull `messages_text` rows not yet in `ingest_stamps`, compute `approx_tokens` + `simhash64` + `token_budget_bucket`, write shards `src/claude_sql/application/use_cases/ingest.py:142`.
3. Re-bind the analytics views so the freshly written stamp shards are visible `src/claude_sql/interfaces/cli/app.py:1784`.
4. Resolve canonical UUIDs: DuckDB SimHash self-join gated on the top-16-bit bucket, `bit_count(xor(...)) <= 3` `src/claude_sql/application/use_cases/ingest.py:252`.
5. Truncate the cache and rewrite one consolidated shard with `canonical_uuid` populated `src/claude_sql/application/use_cases/ingest.py:344`.
6. Emit the stamped-row count as the worker result `src/claude_sql/interfaces/cli/app.py:1787`.

### Related

- `src/claude_sql/application/use_cases/ingest.py:121` (`count_pending`)
- `src/claude_sql/application/use_cases/ingest.py:73` (`_pending_stamp_sql`)
- `src/claude_sql/domain/dedup.py` (`simhash64`, `approx_tokens_batch`, `token_budget_bucket`, `NEAR_DUP_HAMMING_THRESHOLD`)
- `src/claude_sql/infrastructure/duckdb_connection.py:189` (`refresh_analytics_views`)

## Minor flows

- trajectory — entry at `src/claude_sql/interfaces/cli/app.py:1560` → `trajectory_messages` (`src/claude_sql/application/use_cases/trajectory.py:808`). Per-session windowed sentiment + transition classification; purges stale shards, checkpoints, loads windows, batches ≤16 windows per structured-output call via `build_llm_analytics_provider`, echoes uuid-pairs to verify completeness.
- conflicts — entry at `src/claude_sql/interfaces/cli/app.py:1620` → `detect_conflicts` (`src/claude_sql/application/use_cases/conflicts.py:366`). Pair-keyed stance-conflict detection over sessions; same checkpoint/retry/`classify_one` shape as `classify`, validating turn UUIDs via `_valid_turn_uuids` before writing shards.
- cluster — entry at `src/claude_sql/interfaces/cli/app.py:1792` → `run_clustering` (`src/claude_sql/application/use_cases/cluster.py:63`). UMAP+HDBSCAN over the Lance embeddings; mtime-sidecar fast path skips the refit, writes `clusters.parquet`.
- terms — entry at `src/claude_sql/interfaces/cli/app.py:1825` → `run_terms` (`src/claude_sql/application/use_cases/terms.py:31`). c-TF-IDF per-cluster term labels via `compute_ctfidf`; writes `cluster_terms.parquet`.
- skills sync — entry at `src/claude_sql/interfaces/cli/app.py:1216` → `sync` (`src/claude_sql/application/use_cases/skills.py:72`). Filesystem walk of user skills + plugin cache + builtins via `_collect_rows`; atomic tmp-then-replace write of `skills_catalog.parquet`.
- skills ls — entry at `src/claude_sql/interfaces/cli/app.py:1258`. Reads and filters `skills_catalog.parquet` by `--kind` / `--plugin`; exits code 2 if the catalog is missing.
- schema — entry at `src/claude_sql/interfaces/cli/app.py:729`. Emits the static `VIEW_SCHEMA` + `MACRO_SIGNATURES` plus a parquet-existence `cached` map; no DuckDB connection.
- list-cache — entry at `src/claude_sql/interfaces/cli/app.py:792`. Reports `{name, path, exists, bytes, mtime, rows}` for every parquet cache, the Lance store, and the checkpoint DB.
- peek — entry at `src/claude_sql/interfaces/cli/app.py:877` → `peek_session` (`src/claude_sql/application/use_cases/peek.py:31`). One-shot per-session summary (roles, top tools, samples); maps unknown session to exit 65.
- explain — entry at `src/claude_sql/interfaces/cli/app.py:619`. Prints the DuckDB plan (`EXPLAIN`, or `EXPLAIN ANALYZE` under `--analyze`) with pushdown operators highlighted.
- shell — entry at `src/claude_sql/interfaces/cli/app.py:451`. Creates a temp on-disk DuckDB, runs `register_all`, and execs the system `duckdb` binary (exit 127 if absent).
- cache compact — entry at `src/claude_sql/interfaces/cli/app.py:1015`. Consolidates `part-*.parquet` shards of the five worker-append caches into one compacted file; defaults to `--dry-run`.
- cache migrate — entry at `src/claude_sql/interfaces/cli/app.py:1113`. Moves legacy single-file caches into the sharded directory layout, preserving mtime; defaults to `--dry-run`.

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 10 shared source citations
- [claude-sql · Tech debt](../insights/tech-debt.md) — 10 shared source citations
- [claude-sql · Data flow](../architecture/data-flow.md) — 5 shared source citations
- [claude-sql · Sequences](../diagrams/behavioral/sequences.md) — 5 shared source citations
- [claude-sql · Contract map](../insights/contract-map.md) — 5 shared source citations
