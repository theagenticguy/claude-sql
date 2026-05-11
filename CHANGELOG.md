## Unreleased

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
