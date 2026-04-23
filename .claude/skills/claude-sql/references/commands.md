# claude-sql commands reference

Every subcommand, its flags, and the shape of its output. All commands share
the top-level flags `--verbose / -v`, `--quiet / -q`, `--glob`,
`--subagent-glob`, and `--format {auto,table,json,ndjson,csv}`. Paths default
to `~/.claude/projects/*/*.jsonl` and
`~/.claude/projects/*/*/subagents/agent-*.jsonl`.

## Output format (agent-friendly)

`--format auto` (the default) picks `table` on a TTY and `json` when stdout is
piped. Agents calling the CLI from a subprocess get JSON without setting a
flag. Explicit values:

- `table` — human polars pretty-print. Row-truncated at 100 rows; use SQL
  `LIMIT` for more.
- `json` — a single JSON array of row objects (tabular) or a JSON object
  (schema, list-cache).
- `ndjson` — one JSON object per line (streaming-friendly).
- `csv` — CSV with header row.

## Exit codes

Stable across versions. Agents can branch on them without parsing messages.

| Code | Meaning |
|---|---|
| `0` | success |
| `2` | `search` called before `embed` populated the parquet |
| `64` | SQL parse error |
| `65` | unknown view / column / macro (catalog error) |
| `70` | other DuckDB runtime error (cast, conversion, etc.) |
| `127` | `shell` couldn't find the `duckdb` binary on PATH |

On non-TTY stdout, errors arrive on stderr as
`{"error": {"kind": ..., "message": ..., "hint": ...}}`.

## `schema`

List every registered view with its columns, then every macro.

```bash
claude-sql schema                    # human table
claude-sql schema --format json      # {"views": {...}, "macros": [...]}
```

Use the JSON form to discover column names/types programmatically.

## `query <sql>`

Run a SQL statement and emit the result.

```bash
claude-sql query "SELECT count(*) FROM sessions"
claude-sql query "SELECT * FROM sessions LIMIT 3" --format json
claude-sql query "SELECT ..." --format ndjson > out.ndjson
```

On parse/catalog errors the process exits 64 / 65 with a structured payload
(on non-TTY) or a human line (on TTY).

## `explain <sql>`

Static `EXPLAIN` by default (does not execute the query). Pass `--analyze`
for `EXPLAIN ANALYZE` when you actually want timings.

```bash
claude-sql explain "SELECT * FROM messages WHERE session_id = '...'"
claude-sql explain "SELECT * FROM sessions" --analyze
claude-sql explain "SELECT 1" --format json      # {"plan": "<plan text>"}
```

Pushdown markers (`READ_JSON`, `Filter`, `HNSW_INDEX_SCAN`, ...) are
highlighted green on TTY and plain in JSON output.

## `shell`

Drop into the `duckdb` REPL with every view, macro, and the HNSW index
pre-registered. Exit with `.quit`.

```bash
claude-sql shell
```

## `list-cache`

Introspect parquet cache state. Every analytics parquet is reported with
`{name, path, exists, bytes, mtime, rows}`. Use this to decide whether a
prerequisite stage (`embed`, `classify`, `cluster`, `community`) needs to
run before a `search` or `query`.

```bash
claude-sql list-cache
claude-sql list-cache --format json
```

## `embed`

Backfill Cohere Embed v4 embeddings for every message that doesn't yet
have one. Writes to `~/.claude/embeddings.parquet` in chunked checkpoints.

```bash
claude-sql embed --since-days 30                             # dry-run default on most commands; embed is opt-in real-run
AWS_PROFILE=... claude-sql embed --since-days 30 --no-dry-run
```

Flags:
- `--since-days N` — only embed messages newer than N days
- `--dry-run / --no-dry-run` — default `--dry-run`
- `--concurrency N` (default 2) — parallel Bedrock calls
- `--batch-size N` (default 96) — Cohere batch size per call
- `--output-dimension D` (default 1024) — Matryoshka truncation

## `search <text>`

HNSW cosine top-k semantic search over embeddings. Prints `{uuid, session_id,
role, sim, snippet}` per hit. Requires `embed` to have run; otherwise exits
with code `2` and a clear hint.

```bash
claude-sql search "temporal workflow determinism" --k 10
claude-sql search "..." --format ndjson | jq -r '.session_id' | sort -u
```

Flags:
- `--k N` (default 10)

**When NOT to use.** Pinpointing a single known session in a corpus where
the topic is common (e.g. "the one claude-sql session where I ran over 30
days") — semantic similarity will rank dozens of near-ties by generic
boilerplate and the target often sits outside the top-k. Use
`claude-sql query` with `ILIKE` on a distinctive token instead:

```bash
claude-sql query "SELECT DISTINCT session_id FROM messages_text \
  WHERE text_content ILIKE '%--since-days 30%'"
```

Rule of thumb: if the first `search` call returns >3 plausible hits at
similar `sim`, switch to SQL rather than rephrasing the query.

## `classify`

Sonnet 4.6 classifies each session on autonomy_tier (1/2/3), work_category,
success (success/partial/failure), and a short free-text goal. Writes to
`~/.claude/session_classifications.parquet`.

```bash
claude-sql classify --since-days 30
AWS_PROFILE=... claude-sql classify --since-days 30 --no-dry-run
```

Dry-run uses a pure SQL count so it stays fast on big corpora.

## `trajectory`

Per-message sentiment delta + `is_transition` flag.

```bash
claude-sql trajectory --session-id <uuid> --no-dry-run
claude-sql trajectory --since-days 7 --no-dry-run
```

## `conflicts`

Per-session stance conflict detection. Outputs `stance_a`, `stance_b`, and
`resolution` (resolved/abandoned/unresolved).

```bash
claude-sql conflicts --since-days 30 --no-dry-run
```

## `cluster`

UMAP (cosine, 50d + 2d viz) → HDBSCAN (min_cluster_size=20) → c-TF-IDF
(ngram (1,2), min_df=2). Deterministic with `CLAUDE_SQL_SEED=42`.

```bash
claude-sql cluster
```

No Bedrock cost.

## `community`

Louvain community detection over session centroids.

```bash
claude-sql community
```

No Bedrock cost.

## `analyze`

Chains the whole pipeline: embed → cluster → classify → trajectory →
conflicts → community. Every step respects `--dry-run` individually.

```bash
claude-sql analyze --since-days 30                # dry-run, prints costs
AWS_PROFILE=... claude-sql analyze --since-days 30 --no-dry-run
```

## Environment

All overridable via `CLAUDE_SQL_*` env vars (see the root README). Common
ones:

- `CLAUDE_SQL_DEFAULT_GLOB` — main transcript glob
- `CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH` — embeddings cache path
- `CLAUDE_SQL_CONCURRENCY` — parallel Bedrock calls
- `CLAUDE_SQL_SEED` — determinism for UMAP / HDBSCAN / Louvain
