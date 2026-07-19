# claude-sql · CLI

The `claude-sql` CLI is a [cyclopts](https://cyclopts.readthedocs.io) `App` wired in `src/claude_sql/app/cli.py:164` and registered as the `claude-sql` console script in `pyproject.toml:53`.

Every subcommand inherits a flattened `Common` flag block (`src/claude_sql/app/cli.py:171`): `--verbose` / `--quiet` (`src/claude_sql/app/cli.py:184-185`), `--glob` / `--subagent-glob` (`src/claude_sql/app/cli.py:186-187`), and `--format {auto,table,json,ndjson,csv}` (`src/claude_sql/app/cli.py:188`). These shared flags are documented once here and elided from each subcommand's flag list below. The router exposes 32 subcommands, ordered as they appear in the source.

For a machine-readable equivalent of this whole page (command names, parameter types/choices/defaults, exit codes, output conventions), run `claude-sql manifest` (see below) or read the generated [`cli-manifest.md`](cli-manifest.md), which is derived from the same command tree by introspection and cannot drift from the code.

## shell

```
claude-sql shell
```

Launch the interactive duckdb REPL with every view, macro, and the HNSW index pre-registered.
`src/claude_sql/app/cli.py:638`.

## query

```
claude-sql query SQL [--profile-json]
```

Run one SQL query against the claude-sql catalog and emit results.
`src/claude_sql/app/cli.py:727`.

Flags:

- `--profile-json` — Write a DuckDB JSON profile of the executed query under the profiling dir. `src/claude_sql/app/cli.py:732`.

## explain

```
claude-sql explain SQL [--analyze] [--profile-json]
```

Show the DuckDB query plan and highlight pushdown / noteworthy operators.
`src/claude_sql/app/cli.py:806`.

Flags:

- `--analyze` — Run `EXPLAIN ANALYZE` (executes the query and reports real timings); off by default so probing is free. `src/claude_sql/app/cli.py:811`.
- `--profile-json` — Write a DuckDB JSON profile of the plan run under the profiling dir. `src/claude_sql/app/cli.py:812`.

## schema

```
claude-sql schema
```

List every registered view (with columns) and every macro signature, plus a `cached` map of which analytics parquets are populated.
`src/claude_sql/app/cli.py:916`.

## list-cache

```
claude-sql list-cache
```

Report each parquet cache's presence, size, freshness, and row count, plus the LanceDB store and the persistent SQLite checkpointer.
`src/claude_sql/app/cli.py:979`.

## peek

```
claude-sql peek SESSION_ID
```

One-shot summary of a session: lines, role mix, top tools, and message samples.
`src/claude_sql/app/cli.py:1070`.

## cache compact

```
claude-sql cache compact [--name CACHE] [--dry-run|--no-dry-run]
```

Consolidate `part-*.parquet` shards into a single compacted part file under each sharded cache directory.
`src/claude_sql/app/cli.py:1269`.

Flags:

- `--name CACHE` — Restrict to one of `embeddings`, `session_classifications`, `message_trajectory`, `session_conflicts`, `user_friction`. `src/claude_sql/app/cli.py:1272`.
- `--dry-run` / `--no-dry-run` — Default `True`; pass `--no-dry-run` to actually rewrite. `src/claude_sql/app/cli.py:1273`.

## cache migrate

```
claude-sql cache migrate [--dry-run|--no-dry-run]
```

Move legacy single-file caches into the sharded directory layout, preserving the original mtime as `part-<original_mtime_ns>.parquet`.
`src/claude_sql/app/cli.py:1367`.

Flags:

- `--dry-run` / `--no-dry-run` — Default `True`; pass `--no-dry-run` to actually move files. `src/claude_sql/app/cli.py:1370`.

## skills sync

```
claude-sql skills sync [--dry-run]
```

Walk `~/.claude/skills` and `~/.claude/plugins/cache` to write `skills_catalog.parquet` (zero cost, pure filesystem walk).
`src/claude_sql/app/cli.py:1470`.

Flags:

- `--dry-run` — Count rows without writing the parquet. `src/claude_sql/app/cli.py:1473`.

## skills ls

```
claude-sql skills ls [--kind VALUE] [--plugin VALUE]
```

List entries from the skills catalog parquet (run `claude-sql skills sync` first).
`src/claude_sql/app/cli.py:1510`.

Flags:

- `--kind VALUE` — Filter by `source_kind` (`user-skill`, `plugin-skill`, `plugin-command`, `builtin`). `src/claude_sql/app/cli.py:1513`.
- `--plugin VALUE` — Filter by plugin name (exact match). `src/claude_sql/app/cli.py:1514`.

## embed

```
claude-sql embed [--since-days N] [--limit N] [--dry-run|--no-dry-run]
```

Embed new messages with Cohere Embed v4 (`global.cohere.embed-v4:0`) and append to the embeddings parquet / Lance store.
`src/claude_sql/app/cli.py:1558`.

Flags:

- `--since-days N` — Only consider messages newer than N days. `src/claude_sql/app/cli.py:1561`.
- `--limit N` — Cap the number of messages embedded this run. `src/claude_sql/app/cli.py:1562`.
- `--dry-run` / `--no-dry-run` — Default `False` (this command spends by default, unlike LLM workers); pass `--dry-run` to preview the plan. `src/claude_sql/app/cli.py:1563`.

## search

```
claude-sql search QUERY_TEXT [--k N]
```

Semantic top-k nearest-neighbor search over message embeddings via the DuckDB VSS HNSW cosine index.
`src/claude_sql/app/cli.py:1631`.

Flags:

- `--k N` — Top-k results (default 10). `src/claude_sql/app/cli.py:1636`.

## classify

```
claude-sql classify [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
```

Classify sessions with Sonnet 4.6 into autonomy tier, work category, success, and goal.
`src/claude_sql/app/cli.py:1729`.

Flags:

- `--since-days N` — Only classify sessions newer than N days. `src/claude_sql/app/cli.py:1732`.
- `--limit N` — Cap at N sessions this run. `src/claude_sql/app/cli.py:1733`.
- `--dry-run` / `--no-dry-run` — Default `True`; pass `--no-dry-run` to spend real money. `src/claude_sql/app/cli.py:1734`.
- `--no-thinking` — Disable Sonnet adaptive thinking (cheaper, less precise). `src/claude_sql/app/cli.py:1735`.

## trajectory

```
claude-sql trajectory [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
```

Per-message sentiment and topic-transition classification (regex prefilter then Sonnet 4.6).
`src/claude_sql/app/cli.py:1796`.

Flags:

- `--since-days N` — Restrict to messages newer than N days. `src/claude_sql/app/cli.py:1799`.
- `--limit N` — Cap candidate messages this run. `src/claude_sql/app/cli.py:1800`.
- `--dry-run` / `--no-dry-run` — Default `True`. `src/claude_sql/app/cli.py:1801`.
- `--no-thinking` — Disable Sonnet adaptive thinking. `src/claude_sql/app/cli.py:1802`.

## conflicts

```
claude-sql conflicts [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
```

Per-session stance-conflict detection via Sonnet 4.6.
`src/claude_sql/app/cli.py:1844`.

Flags:

- `--since-days N` — Restrict to sessions newer than N days. `src/claude_sql/app/cli.py:1847`.
- `--limit N` — Cap sessions this run. `src/claude_sql/app/cli.py:1848`.
- `--dry-run` / `--no-dry-run` — Default `True`. `src/claude_sql/app/cli.py:1849`.
- `--no-thinking` — Disable Sonnet adaptive thinking. `src/claude_sql/app/cli.py:1850`.

## friction

```
claude-sql friction [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
```

Classify short user messages (<= 300 chars) for friction signals (regex fast-path, Sonnet 4.6 fallback).
`src/claude_sql/app/cli.py:1888`.

Flags:

- `--since-days N` — Restrict to messages newer than N days. `src/claude_sql/app/cli.py:1891`.
- `--limit N` — Cap candidate messages this run. `src/claude_sql/app/cli.py:1892`.
- `--dry-run` / `--no-dry-run` — Default `True`. `src/claude_sql/app/cli.py:1893`.
- `--no-thinking` — Disable Sonnet adaptive thinking. `src/claude_sql/app/cli.py:1894`.

## ingest

```
claude-sql ingest [--since-days N] [--limit N] [--dry-run|--no-dry-run]
```

Stamp every message with `approx_tokens` / `simhash64` / `canonical_uuid` (CPU-only, zero Bedrock cost).
`src/claude_sql/app/cli.py:1938`.

Flags:

- `--since-days N` — Only stamp messages newer than N days. `src/claude_sql/app/cli.py:1941`.
- `--limit N` — Cap stamped rows this run. `src/claude_sql/app/cli.py:1942`.
- `--dry-run` / `--no-dry-run` — Default `True`; pass `--no-dry-run` to stamp and resolve. `src/claude_sql/app/cli.py:1943`.

## cluster

```
claude-sql cluster [--force]
```

Cluster message embeddings with UMAP (8D) and HDBSCAN, writing `clusters.parquet`.
`src/claude_sql/app/cli.py:2006`.

Flags:

- `--force` — Re-cluster even if `clusters.parquet` already exists. `src/claude_sql/app/cli.py:2007`.

## terms

```
claude-sql terms [--force]
```

Compute c-TF-IDF per-cluster term labels and write `cluster_terms.parquet`.
`src/claude_sql/app/cli.py:2037`.

Flags:

- `--force` — Recompute even if `cluster_terms.parquet` already exists. `src/claude_sql/app/cli.py:2038`.

## community

```
claude-sql community [--force] [--gamma FLOAT] [--resolution {coarse,medium,fine}]
                    [--neighbors-of SID] [--top-k N] [--dry-run]
```

Session-level Leiden+CPM community detection over a mutual-kNN cosine graph of session centroids.
`src/claude_sql/app/cli.py:2069`.

Flags:

- `--force` — Re-detect even if `session_communities.parquet` exists. `src/claude_sql/app/cli.py:2072`.
- `--gamma FLOAT` — Explicit CPM gamma; skips the resolution profile and sidecar. `src/claude_sql/app/cli.py:2073`.
- `--resolution {coarse,medium,fine}` — Pick a gamma plateau without specifying a value (default `medium`). `src/claude_sql/app/cli.py:2074`.
- `--neighbors-of SID` — Early-return path: skip Leiden, return top-k cosine neighbors of the session. `src/claude_sql/app/cli.py:2075`.
- `--top-k N` — Top-k for `--neighbors-of` (default 15). `src/claude_sql/app/cli.py:2076`.
- `--dry-run` — Plan-only: count candidate sessions via SQL, do not run Leiden. `src/claude_sql/app/cli.py:2077`.

## analyze

```
claude-sql analyze [--since-days N] [--limit N] [--dry-run|--no-dry-run] [--no-thinking]
                   [--skip-ingest] [--skip-embed] [--skip-classify] [--skip-trajectory]
                   [--skip-conflicts] [--skip-friction] [--skip-cluster]
                   [--skip-community] [--skip-skills-sync]
                   [--force-cluster] [--force-community]
```

Run the full analytics pipeline end-to-end: skills sync, ingest, embed, cluster, terms, community, classify, trajectory, conflicts, friction.
`src/claude_sql/app/cli.py:2207`.

Flags:

- `--since-days N` — Scope all stages to the last N days (default `30`). `src/claude_sql/app/cli.py:2210`.
- `--limit N` — Cap each LLM stage at N items. `src/claude_sql/app/cli.py:2211`.
- `--dry-run` / `--no-dry-run` — Default `True`; pass `--no-dry-run` to execute. `src/claude_sql/app/cli.py:2212`.
- `--no-thinking` — Disable Sonnet adaptive thinking across all stages. `src/claude_sql/app/cli.py:2213`.
- `--skip-ingest` — Drop the ingest stamping stage. `src/claude_sql/app/cli.py:2214`.
- `--skip-embed` — Drop the Cohere Embed v4 stage. `src/claude_sql/app/cli.py:2215`.
- `--skip-classify` — Drop the Sonnet classification stage. `src/claude_sql/app/cli.py:2216`.
- `--skip-trajectory` — Drop the trajectory stage. `src/claude_sql/app/cli.py:2217`.
- `--skip-conflicts` — Drop the conflicts stage. `src/claude_sql/app/cli.py:2218`.
- `--skip-friction` — Drop the friction stage. `src/claude_sql/app/cli.py:2219`.
- `--skip-cluster` — Drop the cluster + terms stage pair. `src/claude_sql/app/cli.py:2220`.
- `--skip-community` — Drop the Leiden+CPM community stage. `src/claude_sql/app/cli.py:2221`.
- `--skip-skills-sync` — Drop the skills catalog sync. `src/claude_sql/app/cli.py:2222`.
- `--force-cluster` — Rebuild `clusters.parquet` (and terms) even if present. `src/claude_sql/app/cli.py:2223`.
- `--force-community` — Rebuild `session_communities.parquet` even if present. `src/claude_sql/app/cli.py:2224`.

## judges

```
claude-sql judges
```

List the cross-provider Bedrock judge catalog (shortname, model ID, family, notes).
`src/claude_sql/app/cli.py:2431`.

## manifest

```
claude-sql manifest
```

Emit a machine-readable manifest of every command, flag, and exit code as JSON (always JSON, regardless of `--format`). Derived from the cyclopts command tree by introspection (`src/claude_sql/core/manifest.py`), so it cannot describe a flag that doesn't exist. `src/claude_sql/app/cli.py:2540`.

## freeze

```
claude-sql freeze RUBRIC --panel s1,s2,... [--embed-model ID] [--seed N]
                  [--min-turns N] [--max-turns N]
```

Pre-register a study: write an immutable manifest under `~/.claude/studies/<sha>/`.
`src/claude_sql/app/cli.py:2451`.

Flags:

- `--panel s1,s2,...` — Comma-separated list of judge shortnames (required). `src/claude_sql/app/cli.py:2456`.
- `--embed-model ID` — Embedding model id (default `global.cohere.embed-v4:0`). `src/claude_sql/app/cli.py:2457`.
- `--seed N` — RNG seed for the panel fingerprint (default 42). `src/claude_sql/app/cli.py:2458`.
- `--min-turns N` — Minimum session turns in scope (default 10). `src/claude_sql/app/cli.py:2459`.
- `--max-turns N` — Maximum session turns in scope (default 40). `src/claude_sql/app/cli.py:2460`.

## replay

```
claude-sql replay MANIFEST_SHA
```

Load and echo a frozen study manifest by SHA.
`src/claude_sql/app/cli.py:2493`.

## blind-handover

```
claude-sql blind-handover INPUT_PATH OUTPUT_PATH
```

Strip identity markers from a parquet of sessions for grader-safe handover.
`src/claude_sql/app/cli.py:2502`.

## judge

```
claude-sql judge MANIFEST_SHA --sessions-parquet PATH --output-parquet PATH
                 [--dry-run|--no-dry-run] [--concurrency N] [--region REGION]
```

Dispatch a frozen study's judge panel over a sessions parquet.
`src/claude_sql/app/cli.py:2533`.

Flags:

- `--sessions-parquet PATH` — Input sessions parquet with `(session_id, text)` columns. `src/claude_sql/app/cli.py:2538`.
- `--output-parquet PATH` — Output parquet for grades. `src/claude_sql/app/cli.py:2539`.
- `--dry-run` / `--no-dry-run` — Default `True`. `src/claude_sql/app/cli.py:2540`.
- `--concurrency N` — Parallel judge calls (default 4). `src/claude_sql/app/cli.py:2541`.
- `--region REGION` — Bedrock region (default `us-east-1`). `src/claude_sql/app/cli.py:2542`.

## ungrounded-claim

```
claude-sql ungrounded-claim MANIFEST_SHA --turns-parquet PATH --output-parquet PATH
```

Run the ungrounded-claim detector over a turns parquet, writing per-claim grounded flags.
`src/claude_sql/app/cli.py:2587`.

Flags:

- `--turns-parquet PATH` — Input turns parquet with `(session_id, turn_idx, assistant_text, tool_output_text)` columns. `src/claude_sql/app/cli.py:2592`.
- `--output-parquet PATH` — Output parquet path. `src/claude_sql/app/cli.py:2593`.

## kappa

```
claude-sql kappa SCORES_PARQUET [--bootstrap N] [--floor F] [--delta-gate PATH]
```

Compute Cohen's and Fleiss' kappa with bootstrapped 95% CI; exit 66 if any axis falls below `--floor` or the delta-gate excludes zero.
`src/claude_sql/app/cli.py:2624`.

Flags:

- `--bootstrap N` — Bootstrap resamples for the CI (default 1000). `src/claude_sql/app/cli.py:2629`.
- `--floor F` — Fleiss kappa floor; below it trips exit 66 (default 0.6). `src/claude_sql/app/cli.py:2630`.
- `--delta-gate PATH` — Prior scores parquet; pre-registered stopping rule that trips exit 66 if delta-kappa CI excludes zero on any axis. `src/claude_sql/app/cli.py:2631`.

## bind

```
claude-sql bind [--repo PATH] [--commit-msg PATH] [--dry-run|--no-dry-run]
```

Attach the transcript-PR binding (trailers + `refs/notes/transcripts` JSON) to a commit (RFC 0001 entry point).
`src/claude_sql/app/cli.py:2699`.

Flags:

- `--repo PATH` — Repository root (defaults to the resolved repo from cwd). `src/claude_sql/app/cli.py:2702`.
- `--commit-msg PATH` — Explicit path to the commit-message file (overrides `CLAUDE_SQL_BIND_COMMIT_MSG` and `.git/COMMIT_EDITMSG`). `src/claude_sql/app/cli.py:2703`.
- `--dry-run` / `--no-dry-run` — Default `False`; with `--dry-run` prints the planned binding as JSON without touching git. `src/claude_sql/app/cli.py:2704`.

## resolve

```
claude-sql resolve COMMIT_SHA [--repo PATH] [--all-sources]
```

Resolve a commit's bound transcript per RFC 0001 resolution precedence (trailer first, note fallback, loud failure on disagreement).
`src/claude_sql/app/cli.py:2832`.

Flags:

- `--repo PATH` — Repository root (defaults to `git rev-parse --show-toplevel`). `src/claude_sql/app/cli.py:2837`.
- `--all-sources` — Return `{"trailer": ..., "note": ...}` instead of merging; never raises on disagreement. `src/claude_sql/app/cli.py:2838`.

## review-sheet

```
claude-sql review-sheet COMMIT_SHA [--repo PATH] [--no-thinking] [--dry-run|--no-dry-run]
```

Render a compressed PR review sheet for a merged commit, resolving its bound transcript and asking Sonnet 4.6 to populate the `PRReviewSheet` schema.
`src/claude_sql/app/cli.py:2938`.

Flags:

- `--repo PATH` — Repository root (defaults to the resolved repo from cwd). `src/claude_sql/app/cli.py:2943`.
- `--no-thinking` — Disable Sonnet adaptive thinking. `src/claude_sql/app/cli.py:2944`.
- `--dry-run` / `--no-dry-run` — Default `True`; dry-run prints the plan dict and skips the Bedrock call. `src/claude_sql/app/cli.py:2945`.

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 2 shared source files
- [claude-sql · System overview](../architecture/system-overview.md) — 2 shared source files
