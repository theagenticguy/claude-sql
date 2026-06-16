## v1.1.8 (2026-06-16)

### Fix

- **conflicts**: add conflicts_over_time macro for a real conversation-time axis (#110)

## v1.1.7 (2026-06-15)

### Fix

- **conflicts**: raise classify_max_tokens 2048 → 16000 to stop max_tokens truncation (#106)

## v1.1.6 (2026-06-15)

### Perf

- **session-text**: format ts.isoformat() engine-side in the timeline loaders (#100)

## v1.1.5 (2026-06-14)

### Perf

- **trajectory**: JOIN the session-id window instead of binding a list param (#98)

## v1.1.4 (2026-06-13)

### Perf

- **session-text**: JOIN the session-id window instead of binding a list param (#95)
- **kappa**: vectorize Fleiss bootstrap CI over the resample axis (~4x) (#94)

## v1.1.3 (2026-06-11)

### Perf

- **kappa**: vectorize bootstrap CI over the resample axis (~9.5x) (#91)

## v1.1.2 (2026-06-10)

### Perf

- **cli**: defer heavy worker imports out of the CLI module-load path (~0.9s) (#86)

## v1.1.1 (2026-06-09)

### Fix

- **release**: bundle 5 workspace members into one publishable claude-sql wheel (#80)

### Refactor

- **workspace**: collapse 5-package workspace into one claude-sql package (#81)

## v1.1.0 (2026-06-09)

### Feat

- **s3**: read transcripts from an S3 source (claude-agent-sdk SessionStore) (#66)
- **cli**: add peek subcommand for session introspection (#51) (#59)

### Fix

- **types**: parameterize bare dict/list/tuple for ty 0.0.46 (#78)
- **sql_views**: cast parent_uuid to VARCHAR so recursive CTEs work (#47) (#58)
- **sql_views**: read-side dedup for message_trajectory (#46) (#57)

### Refactor

- **workspace**: split into 5 PEP 420 namespace packages under uv workspace (#60)

### Perf

- **cluster**: hand polars numpy arrays directly in the clusters output frame (#74)
- **cli**: scope peek's UNNEST to one session, not the whole corpus (~5x) (#72)
- **workers**: project read_all to key column in LLM-worker anti-joins (#70)
- **lance**: column-project the uuid anti-join scan in get_embedded_uuids (#71)
- **kappa**: hoist category support out of the Cohen's-kappa bootstrap loop (#73)
- **cli**: defer lancedb import out of the CLI module-load path (~1.5s) (#77)
- **community**: zero-copy embedding extraction in session centroids (#68)
- **community**: segmented numpy mean for session centroids (#65)

## v1.0.1 (2026-05-14)

### Fix

- **trajectory**: replace prior session shards on rerun (#45) (#54)

### Refactor

- **config**: drop Settings.concurrency + _resolve_concurrency_alias
- **views**: drop describe_all and migrate drift test inline
- **views**: drop deprecated task_spawns view
- **embed**: consolidate _build_bedrock_client

## v1.0.0 (2026-05-13)

### Feat

- v1.0 windowed pipelines (turn_window, ingest stamps, trajectory rewrite, CLAUDE_SQL_HOME) (#42)

## v0.7.0 (2026-05-12)

### Feat

- lance-backed embeddings store + sqlite WAL checkpointer (#37)

## v0.6.0 (2026-05-11)

### Feat

- **perf**: two-tier connections + parquet-gated macros + static schema (#31)
- **sql_views**: static catalog + ago() macro (#30)
- **cli**: drop --format markdown from public OutputFormat (#29)

## v0.5.0 (2026-05-10)

### BREAKING CHANGE

- Settings field rename — every louvain_* config knob and
CLAUDE_SQL_LOUVAIN_* env var is gone. Replace with leiden_* equivalents.
Parquet schema gains is_medoid, coherence, gamma_used columns; removes
nothing. New community_profile.parquet sidecar appears at
~/.claude/community_profile.parquet on auto-γ runs.

### Feat

- **sql_views**: register community_profile view + thread parquet path
- **cli**: agent-first community subcommand with --gamma, --resolution, --neighbors-of, --dry-run

### Refactor

- **communities**: replace Louvain with Leiden+CPM

## v0.4.0 (2026-05-09)

### Feat

- **ci**: PyPI Trusted Publishing workflow + cz changelog drift gate (#23)

### Fix

- **publish**: switch to uv-native publish + add missing packaging dep (#25)
- **ci**: scope changelog gate to PRs + normalize squash-merge suffix
- **ci**: correct cyclonedx-py environment flags in SBOM workflow (#20)

## v0.3.0 (2026-05-09)

### Feat

- **views**: split task views for v2.1.16/v2.1.63 Claude Code taxonomy (#18)
- **review-sheet**: PRReviewSheet schema + worker + claude-sql review-sheet CLI (#13)
- **binding**: RFC 0001 — transcript-PR binding (commit-trailer + git notes) (#12)
- **team-corpus**: Settings.team_corpus_root + 2-user fixture smoke test (#11)
- **cli**: add --profile-json flag to query and explain subcommands
- **perf**: skip UMAP+HDBSCAN refit when embeddings mtime is unchanged
- **perf**: replace community_worker centroid Python loop with DuckDB SQL CTE
- **perf**: shard parquet caches to drop the read-rewrite tax
- **perf**: persist HNSW index in ~/.claude/hnsw.duckdb across CLI runs
- **perf**: add DuckDB tuning PRAGMAs and split Bedrock concurrency
- **skills**: add skill usage counts + seedable skills catalog

### Fix

- **hooks**: restore pre-push files fallback for first-push branches
- **llm**: pad system prompts past the empirical Anthropic cache threshold
- **llm**: system prompts + prompt caching + filter Claude Code system markers

### Perf

- **llm**: shared boto3 client, anyio limiter, concurrency 16, XML prompts

## v0.2.4 (2026-04-22)

### Fix

- **judges**: use global CRIS profile for Anthropic holdouts

## v0.2.3 (2026-04-22)

### Refactor

- **judges**: drop Mistral Large 3 and Magistral-Small from primary panel

## v0.2.2 (2026-04-22)

### Fix

- **judge**: add tenacity retries to Converse calls

## v0.2.1 (2026-04-22)

### Fix

- **judge**: remove rationale truncation, raise output cap to 4096 tokens

## v0.2.0 (2026-04-21)

### Feat

- **judges**: cross-provider IRR study toolchain (judges, judge, ungrounded-claim, kappa, freeze, replay, blind-handover)

## v0.1.1 (2026-04-21)

### Fix

- **friction**: repair malformed SQL in dry-run and active-session branches
- **hooks**: pre-push pytest now fires on every push touching python
