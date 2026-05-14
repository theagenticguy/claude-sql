# claude-sql · Business logic

claude-sql is a single-user CLI that reads JSONL transcripts under
`~/.claude/projects/` and writes derived parquets under `~/.claude/`. There is
no database, no auth surface, no UI, and no rate-limit middleware. "Business
logic" here therefore means:

- Application-layer **validations** of user-supplied flags, env-driven
  Settings fields, LLM responses, and on-disk parquet schemas.
- **Invariants** the codebase enforces against its own caches, retry queue,
  checkpointer, and git binding artifacts. There is no DBMS — every invariant
  is application-side.
- **Calculations** that derive analytics outputs (cost, kappa, friction rate,
  community medoid/coherence, sentiment-arc delta, c-TF-IDF, embedding
  L2-norm). DuckDB SQL macros and Python helpers are both in scope.
- **Policy and gates** that ship as default behavior — dry-run cost guard,
  refusal-as-terminal, retry backoff, optional-dep silent fallthrough,
  cache-mtime skips, schema-drift purge.

UI form-validation, third-party library internals, and DB constraints are out
of scope (there is no DB).

## Validations

| Rule | Domain | Citation | Failure mode |
|---|---|---|---|
| `team_corpus_root`, when set, rewrites all three transcript globs unless any glob is user-pinned. | Settings | `src/claude_sql/config.py:344` | Silent: model_validator returns Self with three globs replaced. |
| Deprecated `concurrency` field aliases onto both pipelines and emits `DeprecationWarning`. | Settings | `src/claude_sql/config.py:384` | `DeprecationWarning`; original explicit `embed_concurrency`/`llm_concurrency` are preserved. |
| `--glob`/`--subagent-glob` rejects patterns with more than one `**`. | CLI input | `src/claude_sql/output.py:156` | Raises `InputValidationError`; CLI exits 64 (`invalid_input`). |
| `output_dimension` must be one of `{256, 512, 1024, 1536}`. | Settings | `src/claude_sql/config.py:168` | Pydantic raises at model construction. |
| `embedding_type` must be one of `{int8, float, uint8, binary, ubinary}`. | Settings | `src/claude_sql/config.py:169` | Pydantic raises at model construction. |
| `hnsw_metric` must be one of `{cosine, l2, dot}`. | Settings | `src/claude_sql/config.py:199` | Pydantic raises at model construction. |
| `classify_thinking`, `trajectory_thinking`, `friction_thinking` must be `{adaptive, disabled}`. | Settings | `src/claude_sql/config.py:218`, `:222`, `:227` | Pydantic raises at model construction. |
| `register_vss(metric=...)` rejects metrics outside `{cosine, l2}`. | DuckDB binding | `src/claude_sql/sql_views.py:1737` | Raises `ValueError`. |
| `register_vss` requires either `lance_uri` or `embeddings_parquet`. | DuckDB binding | `src/claude_sql/sql_views.py:1733` | Raises `ValueError`. |
| `SessionClassification.autonomy_tier` ∈ `{manual, assisted, autonomous}`. | LLM output schema | `src/claude_sql/schemas.py:108` | Pydantic v2 forbids extras (`extra="forbid"`); Bedrock `output_config.format` rejects at the model. |
| `SessionClassification.work_category` ∈ `{sde, admin, strategy_business, events, thought_leadership, other}`. | LLM output schema | `src/claude_sql/schemas.py:118` | Pydantic raises at parse. |
| `SessionClassification.success` ∈ `{success, partial, failure, unknown}`. | LLM output schema | `src/claude_sql/schemas.py:139` | Pydantic raises at parse. |
| `SessionClassification.goal` length ∈ [1, 280]. | LLM output schema | `src/claude_sql/schemas.py:149` | Pydantic raises at parse (Bedrock pre-strips numeric/length constraints from the schema; pydantic enforces post-parse). |
| `SessionClassification.confidence` ∈ [0.0, 1.0]. | LLM output schema | `src/claude_sql/schemas.py:159` | Pydantic raises at parse. |
| `TrajectoryWindow.curr_uuid` non-empty. | LLM output schema | `src/claude_sql/schemas.py:196` | Pydantic raises at parse. |
| `TrajectoryWindow.prev_sentiment` ∈ `{negative, neutral, positive, null}`. | LLM output schema | `src/claude_sql/schemas.py:205` | Pydantic raises at parse. |
| `TrajectoryWindow.transition_kind` ∈ `{frustration_spike, resolution, reset, drift, clarification, none}`. | LLM output schema | `src/claude_sql/schemas.py:235` | Pydantic raises at parse. |
| `TrajectoryWindow.confidence` ∈ [0.0, 1.0]. | LLM output schema | `src/claude_sql/schemas.py:254` | Pydantic raises at parse. |
| `ConflictPair.turn_a_uuid` and `turn_b_uuid` length ∈ [1, 64], distinct (description-level). | LLM output schema | `src/claude_sql/schemas.py:303`, `:314` | Pydantic raises at parse on length; identity guard is prompt-level. |
| `ConflictPair.conflict_kind` ∈ `{disagreement, correction, reversal, impasse}`. | LLM output schema | `src/claude_sql/schemas.py:324` | Pydantic raises at parse. |
| `ConflictPair.severity` ∈ `{low, medium, high}`. | LLM output schema | `src/claude_sql/schemas.py:337` | Pydantic raises at parse. |
| `ConflictPair.agent_position` and `user_position` length ∈ [1, 280]. | LLM output schema | `src/claude_sql/schemas.py:348`, `:358` | Pydantic raises at parse. |
| `UserFrictionSignal.label` ∈ `{status_ping, unmet_expectation, confusion, interruption, correction, frustration, none}`. | LLM output schema | `src/claude_sql/schemas.py:423` | Pydantic raises at parse. |
| `UserFrictionSignal.rationale` length ∈ [1, 200]. | LLM output schema | `src/claude_sql/schemas.py:453` | Pydantic raises at parse. |
| `Correction.what_agent_did` and `correction` length ∈ [5, 300]. | LLM output schema | `src/claude_sql/schemas.py:489`, `:498` | Pydantic raises at parse. |
| `PRReviewSheet.human_intent` length ∈ [10, 600]; `agent_exploration` length ∈ [1, 8]; `corrections` length ≤ 5; `tools_used` length ≤ 20; `tools_refused` length ≤ 10; `diff_rationale` length ∈ [20, 800]. | LLM output schema | `src/claude_sql/schemas.py:520`-`:577` | Pydantic raises at parse. |
| Every Bedrock-bound schema flattens `$ref` inline and injects `additionalProperties: false`. | LLM output | `src/claude_sql/schemas.py:39`-`:96` | Bedrock 400 if not flattened (Bedrock's JSON Schema subset rejects `$ref`/`$defs`, numeric ranges, length, pattern). |
| Trailer URI scheme must be `file://` for the review-sheet worker. | Trailer-resolved transcript | `src/claude_sql/review_sheet_worker.py:184` | Raises `ValueError`; CLI maps to runtime exit 70. |
| Kappa input parquet must carry `{session_id, axis, judge_shortname, score}`. | Kappa input | `src/claude_sql/kappa_worker.py:147`-`:152`, `:255`-`:256` | Raises `ValueError(f"parquet is missing columns: ...")`. |
| Rubric YAML must list ≥1 axis; each axis must be a mapping with a `name`. | Rubric input | `src/claude_sql/judge_worker.py:140`-`:158` | Raises `TypeError` / `ValueError`. |
| `retry_queue.enqueue(pipeline=...)` rejects pipelines outside `PIPELINE_NAMES = (classify, trajectory, conflicts, user_friction)`. | Retry queue | `src/claude_sql/retry_queue.py:97`-`:98` | Raises `ValueError(f"unknown pipeline: ...")`. |
| `OutputFormat` enum closed-set; unknown format raises in `emit_dataframe`. | CLI output | `src/claude_sql/output.py:108` | Raises `ValueError` (defensive — unreachable while enum stays closed). |
| `resolve_commit_to_transcript(all_sources=True)` is rejected; callers must use `resolve_all_sources()` for the dict shape. | Git binding | `src/claude_sql/binding.py:703`-`:707` | Raises `TypeError`. |
| `JSONL` raw-event reader caps per-object size at `_MAX_OBJECT_SIZE` (64 MiB). | JSONL ingestion | `src/claude_sql/sql_views.py:547` | DuckDB error from `read_json` with `maximum_object_size` exceeded. Per-line failures are absorbed via `ignore_errors=true`. |
| Per-text payload to Cohere Embed v4 truncated to `MAX_CHARS_PER_TEXT = 50_000`. | Embed worker | `src/claude_sql/embed_worker.py:53`, `:250` | Silent right-truncation; Bedrock `truncate: "RIGHT"` is also passed for safety. |
| Per-`tool_result` body capped to `session_text_tool_result_max_chars` (default 50K). | Session-text assembly | `src/claude_sql/config.py:232`, `src/claude_sql/session_text.py:57`-`:65` | Silent truncation with `…(truncated, N chars dropped)` marker. |
| Total session-text capped to `session_text_total_max_chars` (default 800K, ~200K tokens). | Session-text assembly | `src/claude_sql/config.py:235`, `src/claude_sql/session_text.py:128`-`:130` | Silent truncation with `…(session truncated at N chars)` marker. |
| Friction worker only classifies user-role messages ≤ `friction_max_chars` (default 300). | Friction worker | `src/claude_sql/config.py:256`-`:260` | Long messages skipped — never reach Bedrock. |
| Friction parquet entries flagged with `source ∈ {regex, sql, llm, refused}`. | Friction worker | `src/claude_sql/friction_worker.py:31`-`:36` | Schema-level convention; downstream filters rely on it. |
| `_parse_structured_payload` raises `RuntimeError` with observed top-level keys when no known Bedrock response shape matches. | LLM dispatcher | `src/claude_sql/llm_shared.py:514`-`:560` | Caller enqueues onto retry queue. |

## Invariants

| Invariant | Where enforced | Citation |
|---|---|---|
| Trailer/note must agree on digest. Disagreement raises `BindingMismatchError`; resolution prefers the trailer for wire fields and the note for forensic fields. | Application — `resolve_commit_to_transcript` | `src/claude_sql/binding.py:715`-`:733` |
| Transcript JSONL digest is SHA-256 of raw bytes prefixed with `sha256:`. Recomputable in any language. | Application — `compute_digest` | `src/claude_sql/binding.py:229`-`:243` |
| Trailer block written via `git interpret-trailers --in-place --if-exists replace` so re-running on amend is idempotent. | Application — `write_trailer` | `src/claude_sql/binding.py:401`-`:439` |
| Note body is single-line JSON written with `git notes --ref=transcripts add -f`; re-runs replace, never duplicate. | Application — `write_note` | `src/claude_sql/binding.py:442`-`:474` |
| Repeated trailer keys: first occurrence wins (rebase / fixup-squash duplication tolerance). | Application — `_parse_trailer_block` | `src/claude_sql/binding.py:485`-`:509` |
| `(session_id, pipeline)` is the primary key in `session_checkpoint`; UPSERT semantics on conflict. | SQLite checkpointer | `src/claude_sql/checkpointer.py:43`-`:51` |
| `(pipeline, unit_id)` is the primary key in `retry_queue`; upsert with attempt counter. | SQLite retry queue | `src/claude_sql/retry_queue.py:41`-`:52` |
| Schema bootstrap (`CREATE TABLE IF NOT EXISTS`) runs once per process per file path under a process-local lock so concurrent opens never double-issue DDL. | SQLite checkpointer | `src/claude_sql/checkpointer.py:57`-`:64`, `src/claude_sql/retry_queue.py:62`-`:75` |
| Every timestamp persisted to SQLite is ISO-8601 UTC; sorting is lexicographic. | SQLite checkpointer | `src/claude_sql/checkpointer.py:19`, `:66`-`:80` |
| Resolve precedence: trailer wins on `(digest, uri, agent_runtime)`; note supplements `(transcript_id, captured_at)` when both exist. | Application — `resolve_commit_to_transcript` | `src/claude_sql/binding.py:725`-`:733` |
| `_stale_old_shape` purges trajectory shards whose schema lacks `curr_uuid` or carries `(uuid, sentiment_delta)`; runs once per pipeline invocation. | Trajectory cache | `src/claude_sql/trajectory_worker.py:329`-`:377` |
| `_purge_legacy_shards` deletes the entire conflicts cache directory the first time any shard carries `conflict_idx` or `empty` columns (legacy v0 → v1.0 schema migration). | Conflicts cache | `src/claude_sql/conflicts_worker.py:82`-`:143` |
| Embeddings parquet is written with explicit `pl.Array(pl.Float32, output_dimension)` schema (otherwise polars infers `Object` and the roundtrip breaks). | Embed worker / Lance store | `src/claude_sql/lance_store.py:194`-`:204` (legacy migration drops unknown columns; type drift surfaces loudly at `tbl.add()` rather than silently re-cast). |
| Cluster IDs are stable: relabel by descending size, ties broken by smallest node index. | Community detection | `src/claude_sql/community_worker.py:387`-`:418` |
| Communities below `leiden_min_community_size` (default 3) collapse to `NOISE_COMMUNITY_ID = -1`. | Community detection | `src/claude_sql/community_worker.py:75`, `:399`-`:402` |
| Mutual-kNN graph: edges are sorted `(u, v)` with `u < v` so igraph never sees a duplicate; weight is `sim[u, v]` (symmetric). | Community detection | `src/claude_sql/community_worker.py:147`-`:186` |
| `igraph` edge attribute MUST be named `"weight"` — leidenalg looks it up by string in `find_partition(weights="weight")`. | Community detection | `src/claude_sql/community_worker.py:189`-`:198` |
| Determinism: `Optimiser.set_rng_seed(seed)` keys bisection RNG; `find_partition(seed=settings.seed)` keys community detection. Same seed + same input ⇒ byte-equal parquet. | Community detection | `src/claude_sql/community_worker.py:46`-`:51`, `:215`-`:220`, `:281`-`:298` |
| Cohen's kappa returns 0.0 (not NaN) when `pe == 1.0` so downstream stats stay valid; same for Fleiss'. | Kappa worker | `src/claude_sql/kappa_worker.py:60`-`:75`, `:96`-`:98` |
| Cohen's kappa input shape: both arrays equal length (`assert a.shape == b.shape`). | Kappa worker | `src/claude_sql/kappa_worker.py:63` |
| Fleiss' kappa requires ≥ 3 judges per axis; otherwise the axis is skipped. | Kappa worker | `src/claude_sql/kappa_worker.py:189`-`:193` |
| `BedrockRefusalError` is terminal and non-retryable; callers stamp a neutral placeholder so the unit is never re-tried. | LLM dispatcher | `src/claude_sql/llm_shared.py:490`-`:518` |
| Neither trailer nor note present ⇒ `LookupError` (CLI maps to exit 2). | Git binding | `src/claude_sql/binding.py:712`-`:713` |
| `embeddings_mtime` sidecar tracks the deepest mtime across the Lance directory tree; clusters skip recompute when it matches. | Cluster worker | `src/claude_sql/cluster_worker.py:76`-`:115`, `:207` |

## Calculations

| Calculation | Inputs | Output | Citation |
|---|---|---|---|
| `cost_estimate(sid)` | `messages.input_tokens`, `messages.cache_write`, `messages.output_tokens`, pricing table | USD per session | `src/claude_sql/sql_views.py:1288`-`:1299` |
| `_estimate_cost(n_items, avg_in, avg_out, pricing)` (Python) | item count, average in/out tokens, `(in_rate, out_rate)` tuple | USD estimate | `src/claude_sql/llm_shared.py:1291`-`:1299` |
| `cohens_kappa(a, b)` | two equal-length rater arrays | Cohen's κ float | `src/claude_sql/kappa_worker.py:57`-`:75` |
| `fleiss_kappa(ratings)` | `(n_items, n_categories)` count matrix with equal row sums | Fleiss' κ float | `src/claude_sql/kappa_worker.py:78`-`:98` |
| `bootstrap_kappa_ci` / `bootstrap_fleiss_ci` | rater arrays / count matrix, `n_bootstrap=1000`, `confidence=0.95`, `seed=42` | `(ci_low, ci_high)` | `src/claude_sql/kappa_worker.py:101`-`:139` |
| `delta_gate_excludes_zero(current, prior)` | two `FleissKappa` records | bool — does 95% CI on κ-delta exclude zero? | `src/claude_sql/kappa_worker.py:225`-`:247` |
| `friction_rate(since_days)` macro | `user_friction`, `messages_text` | per-session `(label counts, n_user_msgs, rate)` | `src/claude_sql/sql_views.py:1573`-`:1607` |
| `friction_counts(since_days)` macro | `user_friction` | per-label `(n, sessions, avg_confidence, n_regex, n_llm)` | `src/claude_sql/sql_views.py:1550`-`:1563` |
| `success_rate_by_work(since_days)` macro | `session_classifications` | per work_category `(sessions, success_rate, failure_rate, partial_rate)` | `src/claude_sql/sql_views.py:1451`-`:1466` |
| `autonomy_trend(window_days)` macro | `session_classifications` | per `(week, autonomy_tier)` count | `src/claude_sql/sql_views.py:1422`-`:1432` |
| `work_mix(since_days)` macro | `session_classifications` | per `work_category` count | `src/claude_sql/sql_views.py:1438`-`:1444` |
| `todo_velocity(sid)` macro | `todo_state_current` | completed-todo / distinct-subject ratio per session | `src/claude_sql/sql_views.py:1316`-`:1322` |
| `subagent_fanout(sid)` macro | `subagent_sessions` | child-session count per parent session | `src/claude_sql/sql_views.py:1327`-`:1332` |
| `tool_rank(last_n_days)` macro | `tool_calls` | per-tool count, descending | `src/claude_sql/sql_views.py:1303`-`:1311` |
| `skill_rank(last_n_days)` / `skill_source_mix(last_n_days)` / `unused_skills(last_n_days)` macros | `skill_invocations`, `skills_catalog` | per-skill or per-source counts; unused-skill list | `src/claude_sql/sql_views.py:1363`-`:1387`, `:1632`-`:1648` |
| `sentiment_arc(sid)` macro | `messages` ⨝ `message_trajectory.curr_uuid` | per-window `(ts, role, curr_sentiment, delta, transition_kind, is_transition, confidence)` | `src/claude_sql/sql_views.py:1526`-`:1539` |
| `cluster_top_terms(cid, n)` / `community_top_topics(cid, n)` macros | `cluster_terms`, `message_clusters`, `messages`, `session_communities` | top-N TF-IDF terms or top-N cluster ids per community | `src/claude_sql/sql_views.py:1472`-`:1510` |
| `canonical_uuid_resolve()` macro | `ingest_stamps.simhash64` | `(uuid, canonical_uuid)` mapping for near-duplicate dedup | `src/claude_sql/sql_views.py:1661`-`:1671` |
| `semantic_search(query_vec, k)` macro | LanceDB-backed `message_embeddings` | top-k by `array_distance` (HNSW-rewritten) | `src/claude_sql/sql_views.py:1343`+ |
| Session centroid embedding | message embeddings → mean per `session_id`, then L2-normalize | `(session_ids, centroids)` (N, dim) float32 | `src/claude_sql/community_worker.py:82`-`:144` |
| Mutual-kNN cosine adjacency | session-centroid similarity matrix, `k=15`, edge floor 0.3 | sorted edge list with weights | `src/claude_sql/community_worker.py:147`-`:186` |
| Resolution-profile longest-plateau pick | `Optimiser.resolution_profile` over `[range_lo, range_hi]` | γ at the longest-plateau community count | `src/claude_sql/community_worker.py:202`-`:279` |
| Per-community medoid + coherence | per-community node indices, similarity matrix | `(medoid_indices, {community_id: coherence})` | `src/claude_sql/community_worker.py:331`-`:364` |
| Retry backoff | attempt count `n` | `min(2^n, 60)` minutes | `src/claude_sql/retry_queue.py:79`-`:82` |
| `_backoff_delta` schedule (literal) | first 5 attempts | `2, 4, 8, 16, 32` minutes, then capped at 60 | `src/claude_sql/retry_queue.py:79`-`:82` |
| `c-TF-IDF` (per-class TF, IDF, L1 norm; ngram (1,2); min_df=2) | message clusters + raw text | top-N terms per cluster | `src/claude_sql/terms_worker.py` (per `CLAUDE.md` ‘c-TF-IDF note’) |
| `delta` field — sentiment encoded as `{negative=-1, neutral=0, positive=1}` then subtracted | prev/curr sentiment | integer in `{-2,-1,0,1,2}` (null when prev null) | `src/claude_sql/schemas.py:218`-`:226` |

Two calculations whose multi-step shape doesn't fit a one-row prose summary:

**Cost estimation pipeline** (`cost_estimate` macro). For each message in
session `sid`, sum
`(coalesce(input_tokens, 0) + coalesce(cache_write, 0)) * in_rate +
coalesce(output_tokens, 0) * out_rate`, then divide by `1e6` to convert per-MTok
rates to absolute USD. Pricing rows come from `Settings.pricing` (default
`DEFAULT_PRICING`, src/claude_sql/config.py:118-124). The pricing join uses a
prefix match: `regexp_replace(m.model, '-\d{8}$', '')` strips a dated model-ID
suffix (e.g. `claude-haiku-4-5-20251001`) so it resolves to the base entry
`claude-haiku-4-5`. `cache_read_input_tokens` is intentionally excluded — cache
reads are billed at 0.1× input and not yet modeled.

**Community detection pipeline** (`run_communities`).
1. `_load_session_centroids` joins LanceDB-backed embeddings to `messages`,
   averages per session, L2-normalizes the centroids
   (`src/claude_sql/community_worker.py:82`).
2. `_build_mutual_knn` constructs a mutual-kNN cosine adjacency at `k=15`,
   drops edges below `leiden_edge_floor=0.3`
   (`src/claude_sql/community_worker.py:147`).
3. `_compute_resolution_profile` runs `Optimiser.resolution_profile` over
   `[leiden_resolution_range_lo, leiden_resolution_range_hi]` (default
   `[0.05, 0.95]`) seeded with `Optimiser.set_rng_seed(settings.seed)`
   (`src/claude_sql/community_worker.py:202`).
4. `_pick_zoom` chooses γ — `medium` = midpoint of the longest plateau,
   `coarse` = lowest n_communities ≥ 2, `fine` = highest n_communities
   (`src/claude_sql/community_worker.py:250`).
5. `_run_leiden_cpm` runs `find_partition(g, CPMVertexPartition,
   weights="weight", resolution_parameter=γ, seed=settings.seed,
   n_iterations=settings.leiden_n_iterations)`
   (`src/claude_sql/community_worker.py:281`).
6. `_warn_disconnected` warns on multi-component induced subgraphs without
   splitting (`src/claude_sql/community_worker.py:302`).
7. `_compute_medoid_and_coherence` picks each community's medoid (max mean
   intra-community cosine) and coherence (mean intra-community cosine)
   (`src/claude_sql/community_worker.py:331`).
8. `_relabel_and_collapse` relabels by descending size; communities below
   `leiden_min_community_size` collapse to `NOISE_COMMUNITY_ID = -1`
   (`src/claude_sql/community_worker.py:367`).

## Policy and gates

- **Dry-run-by-default cost guard:** every command that calls Bedrock for real
  money (`embed`, `classify`, `trajectory`, `conflicts`, `friction`, `analyze`,
  `judge`, `review-sheet`) defaults to `dry_run=True`; `--no-dry-run` is
  required to spend. Dry-run emits a plan JSON with `pipeline`, candidate
  count, estimated cost, and `dry_run: True`. `src/claude_sql/cli.py:118`,
  `:131`, `:151`-`:153`, `:492`-`:500`; per-worker branches at
  `src/claude_sql/conflicts_worker.py:292`-`:331`,
  `src/claude_sql/friction_worker.py:661`-`:726`,
  `src/claude_sql/judge_worker.py:437`-`:456`,
  `src/claude_sql/review_sheet_worker.py:346`-`:405`.
- **Refusal-as-terminal:** Bedrock `stop_reason == "refusal"` raises
  `BedrockRefusalError`; callers stamp a neutral placeholder row and clear the
  retry queue so refused units do not cycle. `src/claude_sql/llm_shared.py:490`-`:518`,
  `src/claude_sql/trajectory_worker.py:645`,
  `src/claude_sql/friction_worker.py:572`,
  `src/claude_sql/review_sheet_worker.py:434`.
- **Tenacity retries on Bedrock transients only:** `ClientError` codes
  `{ThrottlingException, ServiceUnavailableException, ModelTimeoutException,
  ModelErrorException}` plus network-layer `SSL/Connection/Endpoint/ReadTimeout`.
  Botocore's own retry layer is disabled (`max_attempts=0`) so tenacity owns
  the policy. Up to 10 attempts, exponential 2s → 60s. `src/claude_sql/embed_worker.py:55`-`:78`,
  `:195`-`:216`.
- **Retry-queue backoff cap:** `min(2^attempts, 60)` minutes; up to
  `MAX_ATTEMPTS_DEFAULT = 5` attempts before drain skips. `src/claude_sql/retry_queue.py:38`-`:82`,
  `:131`-`:159`.
- **JSONL fail-soft:** every `read_json` over the transcript glob uses
  `union_by_name=true, ignore_errors=true, sample_size=-1, maximum_object_size=64 MiB`
  so truncated or growing files don't abort the query.
  `src/claude_sql/sql_views.py:540`-`:609`.
- **Stale glob entries:** `session_text.build_session_text` and
  `session_bounds` wrap their DuckDB query in
  `try/except duckdb.IOException`; stale JSONLs warn and skip with one
  retry. `src/claude_sql/session_text.py:177`, `:254`-`:260`.
- **Optional analytics views:** `register_analytics_views` registers each
  parquet-backed view only if its parquet exists on disk; missing parquets
  log at DEBUG and no-op. `src/claude_sql/sql_views.py:1848`-`:1996`.
- **Optional dependencies:** `igraph` and `leidenalg` are imported lazily
  inside community-detection helpers so a fresh install lacking them still
  loads the rest of the CLI. `src/claude_sql/community_worker.py:189`-`:300`.
- **Cluster mtime sidecar skip:** UMAP+HDBSCAN refit (~40s) is skipped when
  the Lance dataset's deepest mtime equals the value stored in
  `<clusters.parquet>.embeddings_mtime`. `force=True` overrides.
  `src/claude_sql/cluster_worker.py:76`-`:115`, `:207`.
- **Stale-shard purge:** trajectory cache is scanned for legacy
  `(uuid, sentiment_delta)` schema and purged once per run; conflicts cache
  is wholly deleted on first detection of `conflict_idx` or `empty` columns.
  `src/claude_sql/trajectory_worker.py:329`-`:377`,
  `src/claude_sql/conflicts_worker.py:82`-`:143`.
- **Friction regex fast-path skips Bedrock:** unambiguous patterns
  (status pings, hard interruptions, explicit corrections) get
  `confidence=0.9` from regex alone. Ambiguous shapes
  (e.g. `screenshot?`) deliberately fall through to Sonnet 4.6.
  `src/claude_sql/friction_worker.py:79`-`:160`.
- **Friction short-message gate:** only user-role messages ≤
  `friction_max_chars` (default 300) are eligible; long turns are almost
  always genuine instructions and bypass the classifier entirely.
  `src/claude_sql/config.py:256`-`:260`,
  `src/claude_sql/friction_worker.py:17`.
- **Connectivity post-check is warn-only:** Leiden communities whose induced
  subgraph has multiple weakly-connected components emit a warning but are
  not split. `src/claude_sql/community_worker.py:302`-`:328`.
- **DuckDB engine PRAGMAs from Settings:** `duckdb_threads`,
  `duckdb_memory_limit` (default `'70%'`, supports absolute `'4GB'`),
  `duckdb_temp_dir` (default `~/.claude/duckdb_tmp`, not `/tmp`).
  `src/claude_sql/config.py:330`-`:342`; applied in
  `cli._open_connection_full` per CLAUDE.md.
- **Determinism:** `CLAUDE_SQL_SEED=42` (default) seeds UMAP, HDBSCAN, Leiden
  `find_partition` and `Optimiser` bisection RNG. Same seed + same input
  produces byte-equal parquets. `src/claude_sql/config.py:318`,
  `src/claude_sql/community_worker.py:46`-`:51`, `:215`-`:220`.
- **Single-glob recursive-segment cap:** DuckDB's `read_json` rejects
  patterns with more than one `**`; CLI catches up front via `validate_glob`
  with a fix-it hint, exit 64. `src/claude_sql/output.py:156`-`:177`.
- **CLI exit-code contract:** `0=ok`, `2=no_embeddings`, `64=invalid_input`
  (parse / glob), `65=catalog_error` (unknown view/column), `70=runtime_error`
  (everything else from `duckdb.Error`), `127=duckdb_missing`. Stable for
  agent subprocess callers. `src/claude_sql/output.py:49`-`:57`,
  `:180`-`:225`.
- **Auto output format:** `--format auto` resolves to `TABLE` on a TTY and
  `JSON` on a pipe so agent subprocesses get JSON without a flag.
  `src/claude_sql/output.py:60`-`:65`.
- **Ingest dedup gate:** when `ingest_stamps` is registered, the embed
  candidate query left-joins it and skips messages whose `canonical_uuid`
  points elsewhere; absent view ⇒ unfiltered candidate set (fresh-install
  default). `src/claude_sql/embed_worker.py:101`-`:179`.
- **`extra="forbid"` on every Sonnet schema:** Bedrock rejects unknown
  fields up front so the model can't drift the parquet shape.
  `src/claude_sql/schemas.py:106`, `:186`, `:301`, `:421`, `:487`, `:518`.
- **Bedrock schema constraint stripping:** numeric `minimum/maximum`,
  string `minLength/maxLength/pattern/format`, and array
  `minItems/maxItems/uniqueItems` are stripped from the schema dict
  because Bedrock's JSON Schema subset rejects them. Pydantic still
  enforces the constraints at parse time. `src/claude_sql/schemas.py:79`-`:96`.
- **Connection-mode separation:** `cli._open_connection_full` (registers
  every view) vs `_open_connection_introspect` (no view registration) so
  `claude-sql schema` doesn't pay for analytics view binding. Per
  CLAUDE.md *Agent-friendly CLI surface*.
- **`stdlib logging` is banned;** every module logs via loguru, enforced by
  `flake8-tidy-imports.banned-api` raising `TID251` on `import logging`.
  `pyproject.toml`, per CLAUDE.md *Logging* section.
- **Tenacity-loguru bridge:** every `@retry` uses
  `claude_sql.logging_setup.loguru_before_sleep("LEVEL")` so retry-state
  lines route through loguru. `src/claude_sql/logging_setup.py:83`,
  `src/claude_sql/embed_worker.py:214`.

## See also

- [claude-sql · Contract map](../insights/contract-map.md) — 6 shared citations
- [claude-sql · Processes](../behavior/processes.md) — 5 shared citations
- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 4 shared citations
- [claude-sql · Risk hotspots](../analysis/risk-hotspots.md) — 3 shared citations
