# claude-sql · Processes

`claude-sql` is a single-binary `cyclopts` CLI; it has no HTTP routes, cron jobs, or queue consumers. Every process is initiated by a subcommand registered on the `cyclopts` `App` at `src/claude_sql/app/cli.py:164`, dispatched through the `[project.scripts]` entry point `main()` at `src/claude_sql/app/cli.py:3073`. The eight processes below cover the load-bearing flows (the composite pipeline, the cheap read path, the embedding/search loop, the LLM analytics workers, graph community detection, ingest stamping, and provenance review). Every other subcommand appears under `## Minor flows`.

## analyze

The composite end-to-end pipeline. Opens one shared DuckDB connection and threads it through every stage, re-binding the VSS view and refreshing analytics views between stages so each stage sees the prior stage's freshly-written parquet/Lance data.

Entry point: `src/claude_sql/app/cli.py:2207`

1. Sync the skills catalog (filesystem walk, zero cost) unless `--skip-skills-sync` `src/claude_sql/app/cli.py:2281-2282`.
2. Open one shared full connection registering all views, macros, and VSS `src/claude_sql/app/cli.py:2299`.
3. Ingest stamps (tiktoken + blake2b SimHash), then refresh views and resolve canonical UUIDs unless `--skip-ingest` `src/claude_sql/app/cli.py:2306-2315`.
4. Embed pending messages via Cohere v4, then re-bind VSS and refresh views so later stages see the new vectors unless `--skip-embed` `src/claude_sql/app/cli.py:2318-2335`.
5. Cluster embeddings (UMAP+HDBSCAN), refresh views, then compute c-TF-IDF terms unless `--skip-cluster` `src/claude_sql/app/cli.py:2338-2354`.
6. Re-bind VSS and run Leiden+CPM community detection unless `--skip-community` `src/claude_sql/app/cli.py:2362-2370`.
7. Run the four Sonnet 4.6 workers in sequence — classify, trajectory, conflicts, friction — each followed by a view refresh `src/claude_sql/app/cli.py:2373-2424`.
8. Close the shared connection in the `finally` block and log completion `src/claude_sql/app/cli.py:2425-2428`.

### Related
- `src/claude_sql/app/cli.py:378` (`_refresh_analytics_views`)
- `src/claude_sql/app/cli.py:398` (`_rebind_vss`)
- `src/claude_sql/app/cli.py:338` (`_open_connection_full`)
- `src/claude_sql/analytics/skills_catalog.py:296` (`sync`)
- `src/claude_sql/core/sql_views.py:2078` (`register_all`)

## query

The cheapest read path: a single SQL statement against the registered catalog, with a connection chosen by a substring pre-flight so trivial scalar queries skip the ~25s view-registration chain.

Entry point: `src/claude_sql/app/cli.py:727`

1. Configure logging and resolve `Settings` (validating any `--glob`) `src/claude_sql/app/cli.py:786-788`.
2. Pre-flight the SQL text: route to the full connection only when it references a registered view/macro/VSS token, else to a bare introspect connection `src/claude_sql/app/cli.py:789-793`.
3. Optionally arm DuckDB JSON profiling before the query runs when `--profile-json` is set `src/claude_sql/app/cli.py:795-797`.
4. Execute the statement into a Polars DataFrame under `run_or_die`, which classifies DuckDB errors into parse/catalog/runtime exit codes `src/claude_sql/app/cli.py:798`.
5. Emit the DataFrame in the resolved format (table on TTY, JSON on pipe) `src/claude_sql/app/cli.py:799`.
6. Close the connection in the `finally` block `src/claude_sql/app/cli.py:802-803`.

### Related
- `src/claude_sql/app/cli.py:469` (`_sql_uses_catalog`)
- `src/claude_sql/app/cli.py:364` (`_open_connection_introspect`)
- `src/claude_sql/core/output.py` (`run_or_die`, `emit_dataframe`, imported at `cli.py:68-79`)
- `src/claude_sql/core/sql_views.py:2078` (`register_all`)

## embed

Cohere Embed v4 backfill: discover unembedded messages, embed them in parallel batches, and append the float vectors to the LanceDB HNSW store.

Entry point: `src/claude_sql/app/cli.py:1558`

1. Open a bare in-memory DuckDB connection and register the raw transcript scans plus the v1 views (no VSS) `src/claude_sql/app/cli.py:1607-1615`.
2. Discover `(uuid, text)` pairs absent from the Lance store, dropping near-dup rows when `ingest_stamps` is bound `src/claude_sql/analytics/embed_worker.py:395-400` → `embed_worker.py:100`.
3. Return a plan dict early under `--dry-run`, otherwise proceed to embed `src/claude_sql/analytics/embed_worker.py:425-437`.
4. Open or create the Lance table and loop chunk-by-chunk for crash-resilient checkpointing `src/claude_sql/analytics/embed_worker.py:444-447`.
5. Embed each chunk's texts in parallel `search_document` batches via Bedrock under a semaphore, casting int8 to float `src/claude_sql/analytics/embed_worker.py:457` → `embed_worker.py:264`.
6. Write each chunk to a fixed-size `FLOAT[]` Array column and add it to the Lance table `src/claude_sql/analytics/embed_worker.py:469-488`.
7. Compact periodically, then run a final compaction and ensure the HNSW index on the way out `src/claude_sql/analytics/embed_worker.py:491-498`.
8. Emit the row count (or plan dict) as machine-readable JSON `src/claude_sql/app/cli.py:1626`.

### Related
- `src/claude_sql/analytics/embed_worker.py:365` (`run_backfill`)
- `src/claude_sql/analytics/embed_worker.py:191` (`_invoke_bedrock_sync`, tenacity-retried)
- `src/claude_sql/core/lance_store.py` (`connect_db`, `open_or_create_table`, `add_chunk`, `ensure_index`)
- `src/claude_sql/core/llm_shared.py` (`_build_bedrock_client`)
- `src/claude_sql/core/sql_views.py:476` (`register_raw`)

## search

Semantic top-k nearest-neighbor search: embed the query string, run a DuckDB VSS HNSW cosine lookup, and join back to message text for snippets.

Entry point: `src/claude_sql/app/cli.py:1631`

1. Open a full connection (registers VSS / `message_embeddings`) `src/claude_sql/app/cli.py:1690`.
2. Guard on embedding availability: count rows in `message_embeddings` and exit code 2 when empty `src/claude_sql/app/cli.py:1692-1696`.
3. Embed the query string with Cohere v4 `search_query` mode, forcing float vectors `src/claude_sql/app/cli.py:1698` → `src/claude_sql/analytics/embed_worker.py:334`.
4. Run the HNSW lookup: `ORDER BY array_cosine_distance` ASC to trigger the cosine index, joining to `messages_text` for a 200-char snippet `src/claude_sql/app/cli.py:1706-1723`.
5. Emit the result DataFrame capped at `k` rows `src/claude_sql/app/cli.py:1724`.
6. Close the connection in the `finally` block `src/claude_sql/app/cli.py:1725-1726`.

### Related
- `src/claude_sql/analytics/embed_worker.py:334` (`embed_query`)
- `src/claude_sql/app/cli.py:456` (`_sql_uses_vss`)
- `src/claude_sql/core/sql_views.py:1668` (`register_vss`)
- `src/claude_sql/core/output.py` (`run_or_die`, `emit_dataframe`)

## classify

Representative Sonnet 4.6 structured-output worker (the same pull-once/write-many shape drives `trajectory`, `conflicts`, and `friction`). Classifies whole sessions into autonomy tier, work category, success, and goal.

Entry point: `src/claude_sql/app/cli.py:1729`

1. Resolve the thinking mode and, under `--dry-run`, count pending sessions and return a cost-estimate plan dict `src/claude_sql/analytics/classify_worker.py:211-243`.
2. Anti-join the existing classifications parquet to find sessions already done `src/claude_sql/analytics/classify_worker.py:54-57`.
3. Apply the checkpointer skip — drop sessions whose `(last_ts, mtime)` is unchanged since the last run `src/claude_sql/analytics/classify_worker.py:60-67`.
4. Drain the retry queue and union those session ids back into the keep set `src/claude_sql/analytics/classify_worker.py:69-73`.
5. Assemble pending `(session_id, text)` pairs from `iter_session_texts` `src/claude_sql/analytics/classify_worker.py:75-81`.
6. Dispatch parallel `classify_one` Bedrock calls per chunk under an anyio capacity limiter `src/claude_sql/analytics/classify_worker.py:101-119`.
7. Split successes from exceptions, enqueueing failures for retry, and write OK rows to a fresh parquet shard `src/claude_sql/analytics/classify_worker.py:123-162`.
8. Mark completed sessions in the checkpointer and clear them from the retry queue `src/claude_sql/analytics/classify_worker.py:167-178`.

### Related
- `src/claude_sql/analytics/classify_worker.py:195` (`classify_sessions`)
- `src/claude_sql/core/llm_shared.py` (`classify_one`, `CLASSIFY_SYSTEM_PROMPT`, `pipeline_cache_stats`)
- `src/claude_sql/core/checkpointer.py` (`filter_unchanged`, `mark_completed`)
- `src/claude_sql/core/retry_queue.py` (`drain`, `enqueue`, `mark_done`)
- `src/claude_sql/core/session_text.py` (`iter_session_texts`, `session_bounds`)

## community

Session-level Leiden+CPM community detection over a mutual-kNN cosine graph of session centroids, with auto-γ selection from a resolution profile.

Entry point: `src/claude_sql/app/cli.py:2069`

1. Reject `--neighbors-of` combined with partition flags, then open a full connection `src/claude_sql/app/cli.py:2132-2145`.
2. Take the early-return `--neighbors-of` path (top-k cosine neighbors, no Leiden) when requested `src/claude_sql/app/cli.py:2147-2149` → `src/claude_sql/analytics/community_worker.py:421`.
3. Under `--dry-run`, count candidate sessions through the `message_embeddings` view and emit a plan `src/claude_sql/app/cli.py:2152-2180`.
4. Load and L2-normalize per-session centroids by averaging message embeddings grouped by session `src/claude_sql/analytics/community_worker.py:527` → `community_worker.py:82`.
5. Build the symmetric mutual-kNN graph: cosine matrix, top-k mask, edge floor, then an igraph `src/claude_sql/analytics/community_worker.py:531-546`.
6. Pick γ — explicit value, configured resolution, or the longest-plateau midpoint of `Optimiser.resolution_profile` `src/claude_sql/analytics/community_worker.py:548-576`.
7. Run `leidenalg.find_partition` with `CPMVertexPartition`, warn on disconnected communities, and compute medoid + coherence `src/claude_sql/analytics/community_worker.py:578-602`.
8. Stable-relabel by descending size (collapsing sub-min-size communities to noise) and write the primary parquet plus optional resolution-profile sidecar `src/claude_sql/analytics/community_worker.py:604-649`.

### Related
- `src/claude_sql/analytics/community_worker.py:472` (`run_communities`)
- `src/claude_sql/analytics/community_worker.py:421` (`neighbors_of`)
- `src/claude_sql/analytics/community_worker.py:147` (`_build_mutual_knn`)
- `src/claude_sql/analytics/community_worker.py:202` (`_compute_resolution_profile`)
- `src/claude_sql/analytics/community_worker.py:367` (`_relabel_and_collapse`)

## ingest

Zero-cost per-message stamping: `approx_tokens` (cl100k × 0.78), `simhash64`, `token_budget_bucket`, then a DuckDB self-join that resolves near-duplicate messages to a canonical UUID.

Entry point: `src/claude_sql/app/cli.py:1938`

1. Open a full connection `src/claude_sql/app/cli.py:1978`.
2. Under `--dry-run`, count pending rows via pure SQL and emit a plan dict `src/claude_sql/app/cli.py:1980-1994` → `src/claude_sql/analytics/ingest.py:274`.
3. Pull `messages_text` rows absent from the existing `ingest_stamps` shards via an anti-join `src/claude_sql/analytics/ingest.py:331-333` → `ingest.py:226`.
4. Compute `approx_tokens`, `simhash64`, and `token_budget_bucket` per chunk and write each chunk to a fresh part-file shard `src/claude_sql/analytics/ingest.py:342-369`.
5. Re-bind the analytics views so the resolve pass sees the newly-written shards `src/claude_sql/app/cli.py:1998`.
6. Resolve canonical UUIDs via a top-16-bit-bucketed `bit_count(xor) <= 3` self-join, writing the earliest-seen near-dup as canonical `src/claude_sql/app/cli.py:1999` → `src/claude_sql/analytics/ingest.py:410`.
7. Truncate the cache and rewrite one consolidated shard with the resolved `canonical_uuid` column `src/claude_sql/analytics/ingest.py:500-504`.
8. Emit the stamped row count as JSON `src/claude_sql/app/cli.py:2001`.

### Related
- `src/claude_sql/analytics/ingest.py:295` (`stamp_messages`)
- `src/claude_sql/analytics/ingest.py:410` (`resolve_canonicals`)
- `src/claude_sql/analytics/ingest.py:115` (`simhash64`)
- `src/claude_sql/analytics/ingest.py:85` (`approx_tokens_batch`)
- `src/claude_sql/core/parquet_shards.py` (`write_part`, `iter_part_files`)

## review-sheet

Provenance flow: resolve a merged commit's bound transcript, flatten it to review text, and ask Sonnet 4.6 to compress it into a structured PR review sheet.

Entry point: `src/claude_sql/app/cli.py:2938`

1. Pick the review render format (markdown on TTY, JSON off-TTY) and resolve `Settings` `src/claude_sql/app/cli.py:2980-2985`.
2. Resolve the commit's bound transcript via RFC 0001 precedence (trailer first, note fallback), mapping `LookupError` / mismatch / git failures to canonical exit codes `src/claude_sql/app/cli.py:2992-3019` → `src/claude_sql/provenance/binding.py:674`.
3. Translate the resolved `file://` URI to a path and flatten the JSONL into one-event-per-line review text `src/claude_sql/provenance/review_sheet_worker.py:387-396`.
4. Under `--dry-run`, emit a plan dict (commit, URI, transcript digest, model id, prompt size) and stop `src/claude_sql/app/cli.py:3032-3038` → `src/claude_sql/provenance/review_sheet_worker.py:399-409`.
5. Build the user prompt and call the structured-output classifier against `PR_REVIEW_SHEET_SCHEMA` with the XML-tagged system prompt `src/claude_sql/provenance/review_sheet_worker.py:412-435`.
6. Return `{"refused": True, ...}` on a Bedrock refusal, else `{"sheet": ..., "metadata": ...}` `src/claude_sql/provenance/review_sheet_worker.py:436-459`.
7. Render the refusal or the sheet as markdown or JSON per the resolved format `src/claude_sql/app/cli.py:3040-3053`.

### Related
- `src/claude_sql/provenance/review_sheet_worker.py:342` (`generate_review_sheet`)
- `src/claude_sql/provenance/binding.py:674` (`resolve_commit_to_transcript`)
- `src/claude_sql/provenance/review_sheet_render.py` (`render_markdown`, `render_refusal_markdown`)
- `src/claude_sql/core/schemas.py` (`PR_REVIEW_SHEET_SCHEMA`)
- `src/claude_sql/core/llm_shared.py` (`_invoke_classifier_sync`, `BedrockRefusalError`)

## Minor flows

- shell — entry at `src/claude_sql/app/cli.py:639`. Materializes a temp on-disk DuckDB with all views/macros/HNSW registered, then execs the system `duckdb` REPL against it (exit 127 if the binary is missing).
- explain — entry at `src/claude_sql/app/cli.py:806`. Runs `EXPLAIN` / `EXPLAIN ANALYZE` and highlights pushdown / HNSW / hash operators in the plan text.
- schema — entry at `src/claude_sql/app/cli.py:916`. Emits the static `VIEW_SCHEMA` + `MACRO_SIGNATURES` plus a parquet-existence `cached` map; no DuckDB connection.
- list-cache — entry at `src/claude_sql/app/cli.py:979`. Reports presence/bytes/mtime/rows for every parquet cache, the Lance store, and the SQLite checkpoint.
- peek — entry at `src/claude_sql/app/cli.py:1070`. One-shot session summary: line count, role mix, top tools, and first/last message samples.
- cluster — entry at `src/claude_sql/app/cli.py:2006`. UMAP (8D) + HDBSCAN over embeddings → `clusters.parquet` (`run_clustering` at `src/claude_sql/analytics/cluster_worker.py:50`).
- terms — entry at `src/claude_sql/app/cli.py:2037`. c-TF-IDF per-cluster term labels → `cluster_terms.parquet` (`run_terms` at `src/claude_sql/analytics/terms_worker.py:29`).
- trajectory — entry at `src/claude_sql/app/cli.py:1796`. Regex prefilter then Sonnet 4.6 per-message sentiment + topic-transition (`trajectory_messages` at `src/claude_sql/analytics/trajectory_worker.py:933`).
- conflicts — entry at `src/claude_sql/app/cli.py:1844`. Sonnet 4.6 per-session stance-conflict detection (`detect_conflicts` at `src/claude_sql/analytics/conflicts_worker.py:286`).
- friction — entry at `src/claude_sql/app/cli.py:1888`. Regex fast-path then Sonnet 4.6 friction labels for short user messages (`detect_user_friction` at `src/claude_sql/analytics/friction_worker.py:655`).
- skills sync — entry at `src/claude_sql/app/cli.py:1470`. Filesystem walk of `~/.claude/skills` + plugins cache → `skills_catalog.parquet` (`sync` at `src/claude_sql/analytics/skills_catalog.py:296`).
- skills ls — entry at `src/claude_sql/app/cli.py:1510`. Lists the skills catalog parquet, filterable by `--kind` / `--plugin`.
- cache compact — entry at `src/claude_sql/app/cli.py:1269`. Consolidates many `part-*.parquet` shards into one compacted part (defaults to `--dry-run`).
- cache migrate — entry at `src/claude_sql/app/cli.py:1367`. Moves legacy single-file caches into the sharded directory layout, preserving mtime.
- judges — entry at `src/claude_sql/app/cli.py:2431`. Lists the cross-provider Bedrock judge catalog (`catalog` at `src/claude_sql/evals/judges.py:237`).
- freeze — entry at `src/claude_sql/app/cli.py:2451`. Pre-registers an eval study, writing an immutable manifest under `~/.claude/studies/<sha>/` (`freeze` at `src/claude_sql/evals/freeze.py:115`).
- replay — entry at `src/claude_sql/app/cli.py:2493`. Loads and echoes a frozen study manifest by SHA (`replay` at `src/claude_sql/evals/freeze.py:151`).
- judge — entry at `src/claude_sql/app/cli.py:2533`. Dispatches a frozen study's judge panel over a sessions parquet (`run` at `src/claude_sql/evals/judge_worker.py:430`); defaults to `--dry-run`.
- ungrounded-claim — entry at `src/claude_sql/app/cli.py:2587`. Runs the ungrounded-claim detector over a turns parquet (`detect` at `src/claude_sql/evals/ungrounded_worker.py:133`).
- kappa — entry at `src/claude_sql/app/cli.py:2624`. Computes Cohen's + Fleiss' kappa with bootstrapped CI; exits 66 when an axis is below floor or the delta gate trips (`compute_fleiss` at `src/claude_sql/evals/kappa_worker.py:186`).
- blind-handover — entry at `src/claude_sql/app/cli.py:2502`. Strips identity markers from a sessions parquet and adds an `original_hash` column (`strip_text` at `src/claude_sql/evals/blind_handover.py:81`).
- bind — entry at `src/claude_sql/app/cli.py:2699`. Pre-commit-hook flow attaching transcript-PR trailers + a git-notes JSON to a commit (`find_active_transcript` at `src/claude_sql/provenance/binding.py:265`, `build_binding` at `binding.py:320`).
- resolve — entry at `src/claude_sql/app/cli.py:2832`. Resolves a commit's bound transcript per RFC 0001 precedence, raising loudly on trailer/note digest disagreement (`resolve_commit_to_transcript` at `src/claude_sql/provenance/binding.py:674`).

## See also

- [claude-sql · Contract map](../insights/contract-map.md) — 13 shared source files
- [claude-sql · Public API](../reference/public-api.md) — 13 shared source files
- [claude-sql · Module map](../architecture/module-map.md) — 12 shared source files
- [claude-sql · Tech debt](../insights/tech-debt.md) — 10 shared source files
- [claude-sql · Business logic](../insights/business-logic.md) — 7 shared source files
