# claude-sql — CLI manifest (generated)

**Do not hand-edit.** Regenerate with `uv run python scripts/gen_cli_manifest_doc.py` (or `mise run docs:cli-manifest`) after adding, renaming, or reshaping a command — it is derived by introspection from `src/claude_sql/app/cli.py` via `claude_sql.core.manifest.build_manifest`, so it can never describe a flag that doesn't exist. Machine-readable equivalent: `claude-sql manifest --format json`.

Version at generation time: `claude-sql 1.2.1`.

Zero-copy SQL + Cohere Embed v4 semantic search + Sonnet 4.6 analytics over
~/.claude/ JSONL transcripts (and their subagent sidecars).

## Conventions

- Every command accepts --format {auto,table,json,ndjson,csv}; auto emits a human table on a TTY and JSON on a pipe.
- Flags attach to the subcommand, not the binary (claude-sql query --format json 'SELECT 1').
- Errors carry a stable exit code (see exit_codes); non-TTY stderr carries a JSON {"error": {"kind", "message", "hint"}} payload.
- Cost-incurring commands (embed, classify, trajectory, conflicts, friction, analyze) default to --dry-run; pass --no-dry-run to spend.

## Global flags

Every command below accepts these in addition to its own parameters.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--verbose` | bool | no | — | `False` | — |
| `--quiet` | bool | no | — | `False` | — |
| `--glob` | str | no | — | — | — |
| `--subagent-glob` | str | no | — | — | — |
| `--format` | OutputFormat | no | auto, table, json, ndjson, csv | `auto` | — |

## Exit codes

| name | code |
|---|---|
| `ok` | `0` |
| `no_embeddings` | `2` |
| `invalid_input` | `64` |
| `parse_error` | `64` |
| `catalog_error` | `65` |
| `runtime_error` | `70` |
| `duckdb_missing` | `127` |

## Commands

### `claude-sql shell`

Launch the interactive duckdb REPL with every view, macro, and the HNSW index pre-registered.

_No parameters._

### `claude-sql query`

Run one SQL query against the claude-sql catalog and emit results.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `SQL` | str | yes | — | — | — |
| `--profile-json` | bool | no | — | `False` | — |

### `claude-sql explain`

Show the DuckDB query plan and highlight pushdown / noteworthy operators.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `SQL` | str | yes | — | — | — |
| `--analyze` | bool | no | — | `False` | — |
| `--profile-json` | bool | no | — | `False` | — |

### `claude-sql schema`

List every registered view (with columns) and every macro signature.

_No parameters._

### `claude-sql list-cache`

Report each parquet cache's presence, size, freshness, and row count.

_No parameters._

### `claude-sql peek`

One-shot summary of a session: lines, role mix, top tools, samples.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `SESSION_ID` | str | yes | — | — | — |

### `claude-sql cache compact`

Consolidate ``part-*.parquet`` shards into a single compacted part file.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--name` | str | no | — | — | — |
| `--dry-run` | bool | no | — | `True` | — |

### `claude-sql cache migrate`

Move legacy single-file caches into the sharded directory layout.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--dry-run` | bool | no | — | `True` | — |

### `claude-sql skills sync`

Walk ``~/.claude/skills`` and ``~/.claude/plugins/cache`` → skills_catalog.parquet.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--dry-run` | bool | no | — | `False` | — |

### `claude-sql skills ls`

List entries from the skills catalog parquet.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--kind` | str | no | — | — | — |
| `--plugin` | str | no | — | — | — |

### `claude-sql embed`

Embed new messages with Cohere Embed v4 and append to the embeddings parquet.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--since-days` | int | no | — | — | — |
| `--limit` | int | no | — | — | — |
| `--dry-run` | bool | no | — | `False` | — |

### `claude-sql search`

Semantic top-k nearest-neighbor search over message embeddings via HNSW.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `QUERY_TEXT` | str | yes | — | — | — |
| `--k` | int | no | — | `10` | — |

### `claude-sql classify`

Classify sessions with Sonnet 4.6: autonomy tier, work category, success, goal.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--since-days` | int | no | — | — | — |
| `--limit` | int | no | — | — | — |
| `--dry-run` | bool | no | — | `True` | — |
| `--no-thinking` | bool | no | — | `False` | — |

### `claude-sql trajectory`

Per-message sentiment + topic-transition classification (regex prefilter → Sonnet 4.6).

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--since-days` | int | no | — | — | — |
| `--limit` | int | no | — | — | — |
| `--dry-run` | bool | no | — | `True` | — |
| `--no-thinking` | bool | no | — | `False` | — |

### `claude-sql conflicts`

Per-session stance-conflict detection via Sonnet 4.6.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--since-days` | int | no | — | — | — |
| `--limit` | int | no | — | — | — |
| `--dry-run` | bool | no | — | `True` | — |
| `--no-thinking` | bool | no | — | `False` | — |

### `claude-sql friction`

Classify short user messages (≤300 chars) for friction signals.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--since-days` | int | no | — | — | — |
| `--limit` | int | no | — | — | — |
| `--dry-run` | bool | no | — | `True` | — |
| `--no-thinking` | bool | no | — | `False` | — |

### `claude-sql ingest`

Stamp every message with ``approx_tokens`` / ``simhash64`` / canonical_uuid.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--since-days` | int | no | — | — | — |
| `--limit` | int | no | — | — | — |
| `--dry-run` | bool | no | — | `True` | — |

### `claude-sql cluster`

Cluster message embeddings with UMAP (8D) + HDBSCAN. Writes clusters.parquet.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--force` | bool | no | — | `False` | — |

### `claude-sql terms`

Compute c-TF-IDF per-cluster term labels; writes cluster_terms.parquet.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--force` | bool | no | — | `False` | — |

### `claude-sql community`

Session-level Leiden+CPM community detection over a mutual-kNN cosine graph.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--force` | bool | no | — | `False` | — |
| `--gamma` | float | no | — | — | — |
| `--resolution` | Literal | no | coarse, medium, fine | `medium` | — |
| `--neighbors-of` | str | no | — | — | — |
| `--top-k` | int | no | — | `15` | — |
| `--dry-run` | bool | no | — | `False` | — |

### `claude-sql analyze`

Run the full analytics pipeline end-to-end: embed → structure → LLM analytics.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--since-days` | int | no | — | `30` | — |
| `--limit` | int | no | — | — | — |
| `--dry-run` | bool | no | — | `True` | — |
| `--no-thinking` | bool | no | — | `False` | — |
| `--skip-ingest` | bool | no | — | `False` | — |
| `--skip-embed` | bool | no | — | `False` | — |
| `--skip-classify` | bool | no | — | `False` | — |
| `--skip-trajectory` | bool | no | — | `False` | — |
| `--skip-conflicts` | bool | no | — | `False` | — |
| `--skip-friction` | bool | no | — | `False` | — |
| `--skip-cluster` | bool | no | — | `False` | — |
| `--skip-community` | bool | no | — | `False` | — |
| `--skip-skills-sync` | bool | no | — | `False` | — |
| `--force-cluster` | bool | no | — | `False` | — |
| `--force-community` | bool | no | — | `False` | — |

### `claude-sql judges`

List the cross-provider Bedrock judge catalog (shortname, model ID, family, notes).

_No parameters._

### `claude-sql manifest`

Emit a machine-readable manifest of every command, flag, and exit code.

_No parameters._

### `claude-sql freeze`

Pre-register a study: write an immutable manifest under ~/.claude/studies/<sha>/.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `RUBRIC` | Path | yes | — | — | — |
| `--panel` | str | yes | — | — | — |
| `--embed-model` | str | no | — | `global.cohere.embed-v4:0` | — |
| `--seed` | int | no | — | `42` | — |
| `--min-turns` | int | no | — | `10` | — |
| `--max-turns` | int | no | — | `40` | — |

### `claude-sql replay`

Load and echo a frozen study manifest by SHA.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `MANIFEST_SHA` | str | yes | — | — | — |

### `claude-sql blind-handover`

Strip identity markers from a parquet of sessions for grader-safe handover.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `INPUT_PATH` | Path | yes | — | — | — |
| `--output-path` | Path | yes | — | — | — |

### `claude-sql judge`

Dispatch a frozen study's judge panel over a sessions parquet.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `MANIFEST_SHA` | str | yes | — | — | — |
| `--sessions-parquet` | Path | yes | — | — | — |
| `--output-parquet` | Path | yes | — | — | — |
| `--dry-run` | bool | no | — | `True` | — |
| `--concurrency` | int | no | — | `4` | — |
| `--region` | str | no | — | `us-east-1` | — |

### `claude-sql ungrounded-claim`

Run the ungrounded-claim detector over a turns parquet.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `MANIFEST_SHA` | str | yes | — | — | — |
| `--turns-parquet` | Path | yes | — | — | — |
| `--output-parquet` | Path | yes | — | — | — |

### `claude-sql kappa`

Compute Cohen's + Fleiss' kappa with bootstrapped 95% CI.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `SCORES_PARQUET` | Path | yes | — | — | — |
| `--bootstrap` | int | no | — | `1000` | — |
| `--floor` | float | no | — | `0.6` | — |
| `--delta-gate` | Path | no | — | — | — |

### `claude-sql bind`

Attach the transcript-PR binding (trailers + git-notes JSON) to a commit.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `--repo` | Path | no | — | — | — |
| `--commit-msg` | Path | no | — | — | — |
| `--dry-run` | bool | no | — | `False` | — |

### `claude-sql resolve`

Resolve a commit's bound transcript per RFC 0001 §Resolution precedence.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `COMMIT_SHA` | str | yes | — | — | — |
| `--repo` | Path | no | — | — | — |
| `--all-sources` | bool | no | — | `False` | — |

### `claude-sql review-sheet`

Render a compressed PR review sheet for a merged commit.

| parameter | type | required | choices | default | help |
|---|---|---|---|---|---|
| `COMMIT_SHA` | str | yes | — | — | — |
| `--repo` | Path | no | — | — | — |
| `--no-thinking` | bool | no | — | `False` | — |
| `--dry-run` | bool | no | — | `True` | — |
