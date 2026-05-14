# claude-sql · CLI

The `claude-sql` CLI is a [cyclopts](https://cyclopts.readthedocs.io) `App` wired in `src/claude_sql/cli.py:164` and registered as the `claude-sql` console script in `pyproject.toml:55`.

Every subcommand inherits a flattened `Common` flag block (`src/claude_sql/cli.py:171`): `--verbose` / `--quiet` (logging), `--glob` / `--subagent-glob` (transcript scoping), and `--format {auto,table,json,ndjson,csv}` (output). These are documented once here and elided from each subcommand's flag list.

## shell

```
claude-sql shell
```

Launches the interactive duckdb REPL with every view, macro, and the HNSW index pre-registered.
`src/claude_sql/cli.py:639`.

(No subcommand-specific flags.)

## query

```
claude-sql query <SQL> [--profile-json]
```

Runs one SQL query against the claude-sql catalog and emits results.
`src/claude_sql/cli.py:728`.

Flags:
- `<SQL>` — single SQL statement (positional, required). `src/claude_sql/cli.py:729`.
- `--profile-json` — write a DuckDB JSON profile under `~/.claude/profiling/` and log the path. `src/claude_sql/cli.py:732`.

## explain

```
claude-sql explain <SQL> [--analyze] [--profile-json]
```

Shows the DuckDB query plan and highlights pushdown / noteworthy operators.
`src/claude_sql/cli.py:807`.

Flags:
- `<SQL>` — single SQL statement (positional, required). `src/claude_sql/cli.py:808`.
- `--analyze` — run `EXPLAIN ANALYZE` (executes the query); off by default so probing is free. `src/claude_sql/cli.py:811`.
- `--profile-json` — write a DuckDB JSON profile under `~/.claude/profiling/` and log the path. `src/claude_sql/cli.py:812`.

## schema

```
claude-sql schema
```

Lists every registered view (with columns) and every macro signature, plus a `cached` map of which analytics parquets are populated.
`src/claude_sql/cli.py:917`.

(No subcommand-specific flags.)

## list-cache

```
claude-sql list-cache
```

Reports each parquet cache's presence, size, freshness, and row count, plus the LanceDB embeddings store and the persistent SQLite checkpointer.
`src/claude_sql/cli.py:980`.

(No subcommand-specific flags.)

## cache compact

```
claude-sql cache compact [--name <cache>] [--dry-run|--no-dry-run]
```

Consolidates `part-*.parquet` shards into a single compacted part file under each sharded cache directory.
`src/claude_sql/cli.py:1108`.

Flags:
- `--name <cache>` — restrict to one of `embeddings`, `session_classifications`, `message_trajectory`, `session_conflicts`, `user_friction`. `src/claude_sql/cli.py:1110`.
- `--dry-run` / `--no-dry-run` — default `True`; pass `--no-dry-run` to actually rewrite. `src/claude_sql/cli.py:1111`.

## cache migrate

```
claude-sql cache migrate [--dry-run|--no-dry-run]
```

Moves legacy single-file caches into the sharded directory layout, preserving the original mtime as `part-<original_mtime_ns>.parquet`.
`src/claude_sql/cli.py:1206`.

Flags:
- `--dry-run` / `--no-dry-run` — default `True`; pass `--no-dry-run` to actually move files. `src/claude_sql/cli.py:1208`.

## skills sync

```
claude-sql skills sync [--dry-run]
```

Walks `~/.claude/skills` and `~/.claude/plugins/cache` to write `skills_catalog.parquet` (zero cost, pure filesystem walk).
`src/claude_sql/cli.py:1309`.

Flags:
- `--dry-run` — count rows without writing the parquet. `src/claude_sql/cli.py:1311`.

## skills ls

```
claude-sql skills ls [--kind <value>] [--plugin <value>]
```

Lists entries from the skills catalog parquet (run `claude-sql skills sync` first).
`src/claude_sql/cli.py:1349`.

Flags:
- `--kind <value>` — filter by `source_kind` (`user-skill`, `plugin-skill`, `plugin-command`, `builtin`). `src/claude_sql/cli.py:1351`.
- `--plugin <value>` — filter by plugin name (exact match). `src/claude_sql/cli.py:1352`.

## embed

```
claude-sql embed [--since-days N] [--limit N] [--dry-run|--no-dry-run]
```

Embeds new messages with Cohere Embed v4 (`global.cohere.embed-v4:0`) and appends to the embeddings parquet / Lance store.
`src/claude_sql/cli.py:1397`.

Flags:
- `--since-days N` — only consider messages newer than N days. `src/claude_sql/cli.py:1399`.
- `--limit N` — cap the number of messages embedded this run. `src/claude_sql/cli.py:1400`.
- `--dry-run` / `--no-dry-run` — default `False` (this command spends by default, unlike LLM workers); pass `--dry-run` to preview. `src/claude_sql/cli.py:1401`.

## search

```
claude-sql search <QUERY_TEXT> [--k N]
```

Semantic top-k nearest-neighbor search over message embeddings via the DuckDB VSS HNSW cosine index.
`src/claude_sql/cli.py:1470`.

Flags:
- `<QUERY_TEXT>` — natural-language query string (positional, required). `src/claude_sql/cli.py:1471`.
- `--k N` — top-k (default 10). `src/claude_sql/cli.py:1474`.

## classify

```
claude-sql classify [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
```

Classifies sessions with Sonnet 4.6 into autonomy tier, work category, success, and goal.
`src/claude_sql/cli.py:1568`.

Flags:
- `--since-days N` — only classify sessions newer than N days. `src/claude_sql/cli.py:1570`.
- `--limit N` — cap at N sessions this run. `src/claude_sql/cli.py:1571`.
- `--dry-run` / `--no-dry-run` — default `True`; pass `--no-dry-run` to spend real money. `src/claude_sql/cli.py:1572`.
- `--no-thinking` — disable Sonnet adaptive thinking (cheaper, less precise). `src/claude_sql/cli.py:1573`.

## trajectory

```
claude-sql trajectory [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
```

Per-message sentiment and topic-transition classification (regex prefilter then Sonnet 4.6) over `(prev_uuid, curr_uuid)` windows.
`src/claude_sql/cli.py:1635`.

Flags:
- `--since-days N` — restrict to messages newer than N days. `src/claude_sql/cli.py:1637`.
- `--limit N` — cap candidate messages this run. `src/claude_sql/cli.py:1638`.
- `--dry-run` / `--no-dry-run` — default `True`. `src/claude_sql/cli.py:1639`.
- `--no-thinking` — disable Sonnet adaptive thinking. `src/claude_sql/cli.py:1640`.

## conflicts

```
claude-sql conflicts [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
```

Per-session stance-conflict detection via Sonnet 4.6, pair-keyed on `(turn_a_uuid, turn_b_uuid)`.
`src/claude_sql/cli.py:1683`.

Flags:
- `--since-days N` — restrict to sessions newer than N days. `src/claude_sql/cli.py:1685`.
- `--limit N` — cap sessions this run. `src/claude_sql/cli.py:1686`.
- `--dry-run` / `--no-dry-run` — default `True`. `src/claude_sql/cli.py:1687`.
- `--no-thinking` — disable Sonnet adaptive thinking. `src/claude_sql/cli.py:1688`.

## friction

```
claude-sql friction [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
```

Classifies short user messages (<= 300 chars) for friction signals (regex fast-path, Sonnet 4.6 fallback).
`src/claude_sql/cli.py:1727`.

Flags:
- `--since-days N` — restrict to messages newer than N days. `src/claude_sql/cli.py:1729`.
- `--limit N` — cap candidate messages this run. `src/claude_sql/cli.py:1730`.
- `--dry-run` / `--no-dry-run` — default `True`. `src/claude_sql/cli.py:1731`.
- `--no-thinking` — disable Sonnet adaptive thinking. `src/claude_sql/cli.py:1732`.

## ingest

```
claude-sql ingest [--since-days N] [--limit N] [--dry-run|--no-dry-run]
```

Stamps every message with `approx_tokens` / `simhash64` / `canonical_uuid` (CPU-only, zero Bedrock cost).
`src/claude_sql/cli.py:1777`.

Flags:
- `--since-days N` — only stamp messages newer than N days. `src/claude_sql/cli.py:1779`.
- `--limit N` — cap stamped rows this run. `src/claude_sql/cli.py:1780`.
- `--dry-run` / `--no-dry-run` — default `True`; pass `--no-dry-run` to stamp and resolve. `src/claude_sql/cli.py:1781`.

## cluster

```
claude-sql cluster [--force]
```

Clusters message embeddings with UMAP (8D) and HDBSCAN, writing `clusters.parquet`.
`src/claude_sql/cli.py:1845`.

Flags:
- `--force` — re-cluster even if `clusters.parquet` already exists. `src/claude_sql/cli.py:1845`.

## terms

```
claude-sql terms [--force]
```

Computes c-TF-IDF per-cluster term labels and writes `cluster_terms.parquet`.
`src/claude_sql/cli.py:1876`.

Flags:
- `--force` — recompute even if `cluster_terms.parquet` already exists. `src/claude_sql/cli.py:1876`.

## community

```
claude-sql community [--force] [--gamma <float>] [--resolution {coarse,medium,fine}]
                    [--neighbors-of <session_id>] [--top-k N] [--dry-run]
```

Session-level Leiden+CPM community detection over a mutual-kNN cosine graph of session centroids.
`src/claude_sql/cli.py:1908`.

Flags:
- `--force` — re-detect even if `session_communities.parquet` exists. `src/claude_sql/cli.py:1910`.
- `--gamma <float>` — explicit CPM gamma; skips the resolution profile and sidecar. `src/claude_sql/cli.py:1911`.
- `--resolution {coarse,medium,fine}` — pick a gamma plateau (default `medium` = longest plateau). `src/claude_sql/cli.py:1912`.
- `--neighbors-of <session_id>` — early-return path: skip Leiden, return top-k cosine neighbors of the session. `src/claude_sql/cli.py:1913`.
- `--top-k N` — top-k for `--neighbors-of` (default 15). `src/claude_sql/cli.py:1914`.
- `--dry-run` — plan-only: count candidate sessions via SQL, do not run Leiden. `src/claude_sql/cli.py:1915`.

## analyze

```
claude-sql analyze [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
                   [--skip-ingest] [--skip-embed] [--skip-classify] [--skip-trajectory]
                   [--skip-conflicts] [--skip-friction] [--skip-cluster]
                   [--skip-community] [--skip-skills-sync]
                   [--force-cluster] [--force-community]
```

Runs the full analytics pipeline end-to-end: skills sync, ingest, embed, cluster, terms, community, classify, trajectory, conflicts, friction.
`src/claude_sql/cli.py:2046`.

Flags:
- `--since-days N` — scope all stages to the last N days (default `30`). `src/claude_sql/cli.py:2048`.
- `--limit N` — cap each LLM stage at N items. `src/claude_sql/cli.py:2049`.
- `--dry-run` / `--no-dry-run` — default `True`. `src/claude_sql/cli.py:2050`.
- `--no-thinking` — disable Sonnet adaptive thinking across all stages. `src/claude_sql/cli.py:2051`.
- `--skip-ingest` — drop the ingest stamping stage. `src/claude_sql/cli.py:2052`.
- `--skip-embed` — drop the Cohere Embed v4 stage. `src/claude_sql/cli.py:2053`.
- `--skip-classify` — drop the Sonnet classification stage. `src/claude_sql/cli.py:2054`.
- `--skip-trajectory` — drop the trajectory stage. `src/claude_sql/cli.py:2055`.
- `--skip-conflicts` — drop the conflicts stage. `src/claude_sql/cli.py:2056`.
- `--skip-friction` — drop the friction stage. `src/claude_sql/cli.py:2057`.
- `--skip-cluster` — drop the cluster + terms stage pair. `src/claude_sql/cli.py:2058`.
- `--skip-community` — drop the Leiden+CPM community stage. `src/claude_sql/cli.py:2059`.
- `--skip-skills-sync` — drop the skills catalog sync. `src/claude_sql/cli.py:2060`.
- `--force-cluster` — rebuild `clusters.parquet` (and terms) even if present. `src/claude_sql/cli.py:2061`.
- `--force-community` — rebuild `session_communities.parquet` even if present. `src/claude_sql/cli.py:2062`.

## judges

```
claude-sql judges
```

Lists the cross-provider Bedrock judge catalog (shortname, model ID, family, notes).
`src/claude_sql/cli.py:2270`.

(No subcommand-specific flags.)

## freeze

```
claude-sql freeze <RUBRIC> --panel <s1,s2,...> [--embed-model ID] [--seed N]
                  [--min-turns N] [--max-turns N]
```

Pre-registers a study: writes an immutable manifest under `~/.claude/studies/<sha>/`.
`src/claude_sql/cli.py:2290`.

Flags:
- `<RUBRIC>` — rubric YAML path (positional, required). `src/claude_sql/cli.py:2291`.
- `--panel <s1,s2,...>` — comma-separated list of judge shortnames. `src/claude_sql/cli.py:2294`.
- `--embed-model <id>` — embedding model id (default `global.cohere.embed-v4:0`). `src/claude_sql/cli.py:2295`.
- `--seed N` — RNG seed for the panel-fingerprint (default 42). `src/claude_sql/cli.py:2296`.
- `--min-turns N` — minimum session turns in scope (default 10). `src/claude_sql/cli.py:2297`.
- `--max-turns N` — maximum session turns in scope (default 40). `src/claude_sql/cli.py:2298`.

## replay

```
claude-sql replay <MANIFEST_SHA>
```

Loads and echoes a frozen study manifest by SHA.
`src/claude_sql/cli.py:2332`.

Flags:
- `<MANIFEST_SHA>` — frozen study manifest SHA (positional, required). `src/claude_sql/cli.py:2332`.

## blind-handover

```
claude-sql blind-handover <INPUT_PATH> <OUTPUT_PATH>
```

Strips identity markers from a parquet of sessions for grader-safe handover.
`src/claude_sql/cli.py:2341`.

Flags:
- `<INPUT_PATH>` — input parquet with `(session_id, text)` columns (positional, required). `src/claude_sql/cli.py:2342`.
- `<OUTPUT_PATH>` — output parquet path (positional, required). `src/claude_sql/cli.py:2344`.

## judge

```
claude-sql judge <MANIFEST_SHA> --sessions-parquet <PATH> --output-parquet <PATH>
                 [--dry-run|--no-dry-run] [--concurrency N] [--region <region>]
```

Dispatches a frozen study's judge panel over a sessions parquet.
`src/claude_sql/cli.py:2372`.

Flags:
- `<MANIFEST_SHA>` — frozen study manifest SHA (positional, required). `src/claude_sql/cli.py:2373`.
- `--sessions-parquet <path>` — input sessions parquet with `(session_id, text)` columns. `src/claude_sql/cli.py:2376`.
- `--output-parquet <path>` — output parquet for grades. `src/claude_sql/cli.py:2377`.
- `--dry-run` / `--no-dry-run` — default `True`. `src/claude_sql/cli.py:2378`.
- `--concurrency N` — parallel judge calls (default 4). `src/claude_sql/cli.py:2379`.
- `--region <region>` — Bedrock region (default `us-east-1`). `src/claude_sql/cli.py:2380`.

## ungrounded-claim

```
claude-sql ungrounded-claim <MANIFEST_SHA> --turns-parquet <PATH> --output-parquet <PATH>
```

Runs the ungrounded-claim detector over a turns parquet, writing per-claim grounded flags.
`src/claude_sql/cli.py:2426`.

Flags:
- `<MANIFEST_SHA>` — frozen study manifest SHA (positional, required). `src/claude_sql/cli.py:2427`.
- `--turns-parquet <path>` — input turns parquet with `(session_id, turn_idx, assistant_text, tool_output_text)`. `src/claude_sql/cli.py:2430`.
- `--output-parquet <path>` — output parquet path. `src/claude_sql/cli.py:2431`.

## kappa

```
claude-sql kappa <SCORES_PARQUET> [--bootstrap N] [--floor F] [--delta-gate <prior.parquet>]
```

Computes Cohen's and Fleiss' kappa with bootstrapped 95% CI; exits 66 if any axis falls below `--floor` or the delta-gate excludes zero.
`src/claude_sql/cli.py:2463`.

Flags:
- `<SCORES_PARQUET>` — input scores parquet (positional, required). `src/claude_sql/cli.py:2464`.
- `--bootstrap N` — bootstrap samples for the CI (default 1000). `src/claude_sql/cli.py:2467`.
- `--floor F` — Fleiss kappa floor; below trips exit 66 (default 0.6). `src/claude_sql/cli.py:2468`.
- `--delta-gate <prior.parquet>` — pre-registered stopping rule: trip exit 66 if delta-kappa CI excludes zero against this prior. `src/claude_sql/cli.py:2469`.

## bind

```
claude-sql bind [--repo <path>] [--commit-msg <path>] [--dry-run|--no-dry-run]
```

Attaches the transcript-PR binding (trailers + `refs/notes/transcripts` JSON) to a commit (RFC 0001 entry point).
`src/claude_sql/cli.py:2538`.

Flags:
- `--repo <path>` — repository root (default: `git rev-parse --show-toplevel`). `src/claude_sql/cli.py:2540`.
- `--commit-msg <path>` — explicit path to the commit-message file (overrides `CLAUDE_SQL_BIND_COMMIT_MSG` and `.git/COMMIT_EDITMSG`). `src/claude_sql/cli.py:2541`.
- `--dry-run` / `--no-dry-run` — default `False`; with `--dry-run` prints the planned binding as JSON without touching git. `src/claude_sql/cli.py:2542`.

## resolve

```
claude-sql resolve <COMMIT_SHA> [--repo <path>] [--all-sources]
```

Resolves a commit's bound transcript per RFC 0001 §Resolution precedence (trailer first, note fallback, loud failure on disagreement).
`src/claude_sql/cli.py:2671`.

Flags:
- `<COMMIT_SHA>` — commit SHA to resolve (positional, required). `src/claude_sql/cli.py:2672`.
- `--repo <path>` — repository root (default: `git rev-parse --show-toplevel`). `src/claude_sql/cli.py:2675`.
- `--all-sources` — return `{"trailer": ..., "note": ...}` instead of merging; never raises on disagreement. `src/claude_sql/cli.py:2676`.

## review-sheet

```
claude-sql review-sheet <COMMIT_SHA> [--repo <path>] [--no-thinking] [--dry-run|--no-dry-run]
```

Renders a compressed PR review sheet for a merged commit, resolving its bound transcript and asking Sonnet 4.6 to populate the `PRReviewSheet` schema.
`src/claude_sql/cli.py:2777`.

Flags:
- `<COMMIT_SHA>` — commit SHA to render (positional, required). `src/claude_sql/cli.py:2778`.
- `--repo <path>` — repository root (default: `git rev-parse --show-toplevel`). `src/claude_sql/cli.py:2781`.
- `--no-thinking` — disable Sonnet adaptive thinking. `src/claude_sql/cli.py:2782`.
- `--dry-run` / `--no-dry-run` — default `True`; dry-run prints the plan dict and skips the Bedrock call. `src/claude_sql/cli.py:2783`.

## See also

- [claude-sql · Processes](../behavior/processes.md) — 31 shared citations
- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 4 shared citations
- [claude-sql · Data flow](../architecture/data-flow.md) — 3 shared citations
