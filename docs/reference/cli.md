# claude-sql · CLI

The `claude-sql` console script (cyclopts `App`, entry point `claude_sql.interfaces.cli.app:main`) has the following subcommands.

## Shared flags

Every subcommand carries the flattened `Common` flags — flags attach to the subcommand, never to the binary (`claude-sql query --format json "SELECT 1"`, not `claude-sql --format json query ...`).

`src/claude_sql/interfaces/cli/app.py:179-196`

Flags:

- `--verbose` — force DEBUG logging. `src/claude_sql/interfaces/cli/app.py:192`.
- `--quiet` — force ERROR logging; keeps default stderr empty for read-only flows. `src/claude_sql/interfaces/cli/app.py:193`.
- `--glob` — narrow the transcript JSONL universe (at most one `**` segment). `src/claude_sql/interfaces/cli/app.py:194`.
- `--subagent-glob` — same, for subagent sidecar files. `src/claude_sql/interfaces/cli/app.py:195`.
- `--format {auto,table,json,ndjson,csv}` — output format; `auto` = table on TTY, JSON on pipe. `src/claude_sql/interfaces/cli/app.py:196`.

## shell

```
claude-sql shell
```

Launch the interactive DuckDB REPL with every view, macro, and the HNSW index pre-registered.

`src/claude_sql/interfaces/cli/app.py:451-452`

## query

```
claude-sql query SQL [--profile-json]
```

Run one SQL query against the claude-sql catalog and emit results.

`src/claude_sql/interfaces/cli/app.py:540-547`

Flags:

- `--profile-json` — set DuckDB profiling PRAGMAs and write a JSON timing tree under `~/.claude/profiling/`. `src/claude_sql/interfaces/cli/app.py:545`.

## explain

```
claude-sql explain SQL [--analyze] [--profile-json]
```

Show the DuckDB query plan and highlight pushdown / noteworthy operators.

`src/claude_sql/interfaces/cli/app.py:619-627`

Flags:

- `--analyze` — run `EXPLAIN ANALYZE` (executes the query and reports real timings); off by default so probing is free. `src/claude_sql/interfaces/cli/app.py:624`.
- `--profile-json` — write a JSON profiling tree under `~/.claude/profiling/`. `src/claude_sql/interfaces/cli/app.py:625`.

## schema

```
claude-sql schema
```

List every registered view (with columns) and every macro signature, plus a `cached` map of which parquet-backed analytics entries are populated.

`src/claude_sql/interfaces/cli/app.py:729-730`

## list-cache

```
claude-sql list-cache
```

Report each parquet cache's presence, size, freshness, and row count (plus the persistent checkpointer DB).

`src/claude_sql/interfaces/cli/app.py:792-793`

## peek

```
claude-sql peek SESSION_ID
```

One-shot summary of a session: lines, role mix, top tools, and message samples; reads the catalog only (no Bedrock, no parquet caches).

`src/claude_sql/interfaces/cli/app.py:877-878`

## cache compact

```
claude-sql cache compact [--name CACHE] [--dry-run]
```

Consolidate `part-*.parquet` shards in a worker-append cache directory into a single compacted part file.

`src/claude_sql/interfaces/cli/app.py:1015-1021`

Flags:

- `--name CACHE` — restrict to one cache of `embeddings`, `session_classifications`, `message_trajectory`, `session_conflicts`, `user_friction`; default is all five. `src/claude_sql/interfaces/cli/app.py:1018`.
- `--dry-run` — default True; pass `--no-dry-run` to actually rewrite. `src/claude_sql/interfaces/cli/app.py:1019`.

## cache migrate

```
claude-sql cache migrate [--dry-run]
```

Move legacy single-file caches into the sharded directory layout, preserving each file's original mtime.

`src/claude_sql/interfaces/cli/app.py:1113-1118`

Flags:

- `--dry-run` — default True; pass `--no-dry-run` to actually move files. `src/claude_sql/interfaces/cli/app.py:1116`.

## skills sync

```
claude-sql skills sync [--dry-run]
```

Walk `~/.claude/skills` and `~/.claude/plugins/cache` and write `skills_catalog.parquet` (zero-cost filesystem walk).

`src/claude_sql/interfaces/cli/app.py:1216-1221`

Flags:

- `--dry-run` — count rows without writing the parquet. `src/claude_sql/interfaces/cli/app.py:1219`.

## skills ls

```
claude-sql skills ls [--kind KIND] [--plugin PLUGIN]
```

List entries from the skills catalog parquet in the shared `--format` shape.

`src/claude_sql/interfaces/cli/app.py:1258-1264`

Flags:

- `--kind KIND` — filter by `source_kind` (`user-skill`, `plugin-skill`, `plugin-command`, `builtin`). `src/claude_sql/interfaces/cli/app.py:1261`.
- `--plugin PLUGIN` — filter by plugin name (exact match). `src/claude_sql/interfaces/cli/app.py:1262`.

## embed

```
claude-sql embed [--since-days N] [--limit N] [--dry-run] [--embedding-provider PROVIDER]
```

Embed new messages with the active embedding provider and append to LanceDB (`--dry-run` is OFF by default here).

`src/claude_sql/interfaces/cli/app.py:1306-1314`

Flags:

- `--since-days N` — only consider messages newer than N days. `src/claude_sql/interfaces/cli/app.py:1309`.
- `--limit N` — cap the number of messages embedded this run. `src/claude_sql/interfaces/cli/app.py:1310`.
- `--dry-run` — preview only; emit plan JSON, no embedding calls. `src/claude_sql/interfaces/cli/app.py:1311`.
- `--embedding-provider {cohere-bedrock,ollama,onnx-bge}` — override the embedding backend for this run. `src/claude_sql/interfaces/cli/app.py:1312`.

## search

```
claude-sql search QUERY_TEXT [--k N] [--embedding-provider PROVIDER]
```

Semantic top-k nearest-neighbor search over message embeddings via HNSW cosine lookup.

`src/claude_sql/interfaces/cli/app.py:1388-1396`

Flags:

- `--k N` — top-k result count (default 10). `src/claude_sql/interfaces/cli/app.py:1393`.
- `--embedding-provider {cohere-bedrock,ollama,onnx-bge}` — override the backend used to embed the query; must match the provider that wrote the store. `src/claude_sql/interfaces/cli/app.py:1394`.

## classify

```
claude-sql classify [--since-days N] [--limit N] [--dry-run] [--no-thinking]
```

Classify sessions with Sonnet 4.6: autonomy tier, work category, success, and goal.

`src/claude_sql/interfaces/cli/app.py:1491-1499`

Flags:

- `--since-days N` — only classify sessions newer than N days. `src/claude_sql/interfaces/cli/app.py:1494`.
- `--limit N` — cap at N sessions this run. `src/claude_sql/interfaces/cli/app.py:1495`.
- `--dry-run` — default True; emit plan JSON, no Bedrock calls. Pass `--no-dry-run` to spend. `src/claude_sql/interfaces/cli/app.py:1496`.
- `--no-thinking` — disable Sonnet adaptive thinking (cheaper, less precise). `src/claude_sql/interfaces/cli/app.py:1497`.

## trajectory

```
claude-sql trajectory [--since-days N] [--limit N] [--dry-run] [--no-thinking] [--llm-analytics-provider PROVIDER]
```

Per-session windowed sentiment and topic-transition classification via structured output.

`src/claude_sql/interfaces/cli/app.py:1560-1569`

Flags:

- `--since-days N` — only process turns newer than N days. `src/claude_sql/interfaces/cli/app.py:1563`.
- `--limit N` — cap the items this run. `src/claude_sql/interfaces/cli/app.py:1564`.
- `--dry-run` — default True; emit plan JSON, no Bedrock calls. `src/claude_sql/interfaces/cli/app.py:1565`.
- `--no-thinking` — disable Sonnet adaptive thinking. `src/claude_sql/interfaces/cli/app.py:1566`.
- `--llm-analytics-provider {sonnet-bedrock,strands-luna}` — structured-output backend; default `sonnet-bedrock`, `strands-luna` routes through GPT-5.6-Luna. `src/claude_sql/interfaces/cli/app.py:1567`.

## conflicts

```
claude-sql conflicts [--since-days N] [--limit N] [--dry-run] [--no-thinking]
```

Per-session stance-conflict detection via Sonnet 4.6.

`src/claude_sql/interfaces/cli/app.py:1620-1628`

Flags:

- `--since-days N` — only process sessions newer than N days. `src/claude_sql/interfaces/cli/app.py:1623`.
- `--limit N` — cap at N sessions this run. `src/claude_sql/interfaces/cli/app.py:1624`.
- `--dry-run` — default True; emit plan JSON, no Bedrock calls. `src/claude_sql/interfaces/cli/app.py:1625`.
- `--no-thinking` — disable Sonnet adaptive thinking. `src/claude_sql/interfaces/cli/app.py:1626`.

## friction

```
claude-sql friction [--since-days N] [--limit N] [--dry-run] [--no-thinking]
```

Classify short user messages (≤300 chars) for friction signals (status_ping, unmet_expectation, confusion, interruption, correction, frustration, none).

`src/claude_sql/interfaces/cli/app.py:1666-1674`

Flags:

- `--since-days N` — only process messages newer than N days. `src/claude_sql/interfaces/cli/app.py:1669`.
- `--limit N` — cap the rows this run. `src/claude_sql/interfaces/cli/app.py:1670`.
- `--dry-run` — default True; emit plan JSON, no Bedrock calls. `src/claude_sql/interfaces/cli/app.py:1671`.
- `--no-thinking` — disable Sonnet adaptive thinking. `src/claude_sql/interfaces/cli/app.py:1672`.

## ingest

```
claude-sql ingest [--since-days N] [--limit N] [--dry-run]
```

Stamp every message with `approx_tokens` / `simhash64` / canonical_uuid (CPU-only, zero Bedrock cost).

`src/claude_sql/interfaces/cli/app.py:1718-1725`

Flags:

- `--since-days N` — only stamp messages newer than N days. `src/claude_sql/interfaces/cli/app.py:1721`.
- `--limit N` — cap stamped rows this run. `src/claude_sql/interfaces/cli/app.py:1722`.
- `--dry-run` — default True; emit plan JSON, no parquet writes. Pass `--no-dry-run` to stamp + resolve. `src/claude_sql/interfaces/cli/app.py:1723`.

## cluster

```
claude-sql cluster [--force]
```

Cluster message embeddings with UMAP (8D) + HDBSCAN; writes clusters.parquet (CPU-only, zero cost).

`src/claude_sql/interfaces/cli/app.py:1792-1793`

Flags:

- `--force` — re-cluster even if clusters.parquet already exists. `src/claude_sql/interfaces/cli/app.py:1793`.

## terms

```
claude-sql terms [--force]
```

Compute c-TF-IDF per-cluster term labels; writes cluster_terms.parquet.

`src/claude_sql/interfaces/cli/app.py:1825-1826`

Flags:

- `--force` — recompute even if cluster_terms.parquet already exists. `src/claude_sql/interfaces/cli/app.py:1826`.

## community

```
claude-sql community [--force] [--gamma FLOAT] [--resolution {coarse,medium,fine}] [--neighbors-of SID] [--top-k N] [--dry-run]
```

Session-level Leiden+CPM community detection over a mutual-kNN cosine graph of session centroids.

`src/claude_sql/interfaces/cli/app.py:1859-1869`

Flags:

- `--force` — re-detect even if session_communities.parquet exists. `src/claude_sql/interfaces/cli/app.py:1862`.
- `--gamma FLOAT` — explicit CPM γ; skips the resolution profile + sidecar. Mutually exclusive with `--resolution` / `--force` / `--neighbors-of`. `src/claude_sql/interfaces/cli/app.py:1863`.
- `--resolution {coarse,medium,fine}` — pick a γ plateau without specifying a value (default `medium`). `src/claude_sql/interfaces/cli/app.py:1864`.
- `--neighbors-of SID` — early-return path: skip Leiden, return top-k cosine neighbors of SID. `src/claude_sql/interfaces/cli/app.py:1865`.
- `--top-k N` — used with `--neighbors-of` (default 15). `src/claude_sql/interfaces/cli/app.py:1866`.
- `--dry-run` — plan-only: count candidate sessions via SQL, do not run Leiden. `src/claude_sql/interfaces/cli/app.py:1867`.

## analyze

```
claude-sql analyze [--since-days N] [--limit N] [--dry-run] [--no-thinking] [--skip-* ...] [--force-cluster] [--force-community] [--embedding-provider PROVIDER] [--llm-analytics-provider PROVIDER]
```

Run the full analytics pipeline end-to-end: skills sync → ingest → embed → cluster → terms → community → classify → trajectory → conflicts → friction.

`src/claude_sql/interfaces/cli/app.py:1992-2013`

Flags:

- `--since-days N` — scope all stages to the last N days (default 30). `src/claude_sql/interfaces/cli/app.py:1995`.
- `--limit N` — cap each LLM stage at N items. `src/claude_sql/interfaces/cli/app.py:1996`.
- `--dry-run` — default True; each LLM-touching stage logs a plan. Pass `--no-dry-run` to execute. `src/claude_sql/interfaces/cli/app.py:1997`.
- `--no-thinking` — disable Sonnet adaptive thinking across all stages. `src/claude_sql/interfaces/cli/app.py:1998`.
- `--skip-ingest` — drop the ingest stage. `src/claude_sql/interfaces/cli/app.py:1999`.
- `--skip-embed` — drop the embed stage. `src/claude_sql/interfaces/cli/app.py:2000`.
- `--skip-classify` — drop the classify stage. `src/claude_sql/interfaces/cli/app.py:2001`.
- `--skip-trajectory` — drop the trajectory stage. `src/claude_sql/interfaces/cli/app.py:2002`.
- `--skip-conflicts` — drop the conflicts stage. `src/claude_sql/interfaces/cli/app.py:2003`.
- `--skip-friction` — drop the friction stage. `src/claude_sql/interfaces/cli/app.py:2004`.
- `--skip-cluster` — drop the cluster stage (terms is bound to cluster). `src/claude_sql/interfaces/cli/app.py:2005`.
- `--skip-community` — drop the community stage. `src/claude_sql/interfaces/cli/app.py:2006`.
- `--skip-skills-sync` — drop the skills-sync stage. `src/claude_sql/interfaces/cli/app.py:2007`.
- `--force-cluster` — rebuild clusters.parquet (+ terms) even if present. `src/claude_sql/interfaces/cli/app.py:2008`.
- `--force-community` — rebuild session_communities.parquet even if present. `src/claude_sql/interfaces/cli/app.py:2009`.
- `--embedding-provider {cohere-bedrock,ollama,onnx-bge}` — override the embedding backend for the embed stage. `src/claude_sql/interfaces/cli/app.py:2010`.
- `--llm-analytics-provider {sonnet-bedrock,strands-luna}` — structured-output backend for the trajectory stage (default `sonnet-bedrock`). `src/claude_sql/interfaces/cli/app.py:2011`.
