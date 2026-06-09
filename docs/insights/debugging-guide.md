# claude-sql · Debugging guide

This file answers the question *"something is broken — where do I look
first?"* for the claude-sql CLI and its analytics pipelines. Every row
traces back to a source citation against the current single-package layout
(`src/claude_sql/<layer>/...`); nothing here is invented.

claude-sql has two complementary error-handling spines. CLI surface
errors classify into stable exit codes via `classify_duckdb_error` and
`run_or_die` (`src/claude_sql/core/output.py:180-255`).
Worker pipelines (`classify`, `trajectory`, `conflicts`, `friction`,
`embed`) catch transient failures, enqueue the failing unit on a durable
SQLite/WAL retry queue
(`src/claude_sql/core/retry_queue.py:1-19`), and continue
draining other units so a single bad input never wedges a backfill. The
only logger is loguru — stdlib `logging` is a banned import in
`pyproject.toml:120-123`.

## Failure-mode index

| Symptom | Likely surface | First check | Citation |
| --- | --- | --- | --- |
| `claude-sql search` exits 2 with a `no_embeddings` error on stderr | Lance namespace empty; `register_vss` created an empty placeholder table so the macro still binds | `claude-sql list-cache --format json` and confirm the embeddings row reports `rows=0`; if so run `claude-sql embed` | `src/claude_sql/app/cli.py:1685-1706`, `src/claude_sql/core/sql_views.py:1740-1758` |
| Stderr carries `[catalog_error]` (exit 65) for an unknown view or column | Required analytics parquet not yet generated; `register_analytics` skipped that view at DEBUG | `claude-sql list-cache --format json` to see which parquet is missing, then run the matching generator | `src/claude_sql/core/output.py:195-201`, `src/claude_sql/core/sql_views.py:1931-1942` |
| `claude-sql query` exits 64 with `[parse_error]` on stderr | Malformed SQL in the user-supplied query | Read the `hint:` line — it points at `claude-sql schema --format json` for view/macro names | `src/claude_sql/core/output.py:188-194` |
| `claude-sql query` exits 64 with `--glob pattern '...' contains more than one '**' segment` | DuckDB `read_json` rejects multi-`**` globs; caught by the up-front `validate_glob` guard | Drop one `**`; use a single recursive segment plus a non-recursive tail | `src/claude_sql/core/output.py:156-177`, `src/claude_sql/app/cli.py:220-231` |
| `claude-sql shell` exits 127 with `` `duckdb` binary not found on PATH `` | System `duckdb` CLI missing | Install duckdb, or use `claude-sql query` which needs no system binary | `src/claude_sql/app/cli.py:684-692` |
| `register_analytics: <view> bound with fallback SELECT *` warning; downstream queries miss alias columns | On-disk parquet predates a v1.0 schema rewrite (trajectory `curr_uuid`, conflicts `turn_a_uuid`) | Re-run the matching worker (`trajectory` / `conflicts`); legacy shards are purged on first run | `src/claude_sql/core/sql_views.py:1958-1988` |
| Backfill leaves units unprocessed; logs say `failed (queued for retry)` | Worker enqueued the unit on the SQLite retry queue after a chunk failure | Inspect the queue, then re-run the same subcommand — `drain` runs at startup so retries fire automatically | `src/claude_sql/analytics/classify_worker.py:126-134`, `src/claude_sql/analytics/trajectory_worker.py:785-801`, `src/claude_sql/core/retry_queue.py:131-159` |
| `classify`/`trajectory`/`friction` rows appear as neutral placeholders | Bedrock returned `stop_reason=refusal`; `BedrockRefusalError` is terminal and a neutral row is stamped so it is never re-tried | None required — the placeholder is the contract; grep the log to confirm a refusal, not a bug | `src/claude_sql/core/llm_shared.py:490-518`, `src/claude_sql/analytics/trajectory_worker.py:777-782`, `src/claude_sql/analytics/friction_worker.py:573-587` |
| `claude-sql resolve <sha>` exits 70 with a runtime error mentioning trailer/note mismatch | Commit's `Claude-Transcript-*` trailer disagrees with the `refs/notes/transcripts` note | `claude-sql resolve <sha> --all-sources` shows both surfaces side-by-side (hint emitted on stderr) | `src/claude_sql/provenance/binding.py:90-109`, `src/claude_sql/app/cli.py:2878-2886` |
| `kappa` exits 66 even though it prints a report | A judge axis fell below `--floor`, or the `--delta-gate` CI excludes zero (pre-registered stopping rule) | Read the emitted JSON `fleiss[].below_floor` flags; the report is still on stdout | `src/claude_sql/app/cli.py:2634-2696` |

## Log and error surfaces

| Surface | Where it emits | What to grep for | Citation |
| --- | --- | --- | --- |
| Loguru stderr handler (the only logger) | stderr; level `INFO` by default, `DEBUG` with `--verbose`, `ERROR` with `--quiet`, overridable via `LOGURU_LEVEL` | `Retrying`, `failed (queued for retry)`, `bound with fallback SELECT *`, `purging cache` | `src/claude_sql/core/logging_setup.py:24-50`, `src/claude_sql/app/cli.py:191-196` |
| Classified-error JSON envelope | stderr, off-TTY only; shape `{"error": {"kind", "message", "hint"}}` | `"kind":"parse_error"`, `"kind":"catalog_error"`, `"kind":"runtime_error"`, `"kind":"invalid_input"` | `src/claude_sql/core/output.py:134-141,210-225` |
| Stable CLI exit codes | Process exit status; agents key off these without parsing tracebacks | `0` ok / `2` no_embeddings / `64` invalid_input or parse_error / `65` catalog_error / `66` kappa gate / `70` runtime_error / `127` duckdb_missing | `src/claude_sql/core/output.py:49-57`, `src/claude_sql/app/cli.py:2696` |
| Tenacity retry breadcrumbs | stderr via `loguru_before_sleep`; one line per retry naming function, sleep seconds, exception | `Retrying _invoke_classifier_sync`, `Retrying.*ThrottlingException`, `Retrying.*ServiceUnavailableException` | `src/claude_sql/core/logging_setup.py:53-95`, `src/claude_sql/core/llm_shared.py:395-401` |
| Bedrock prompt-cache trace (opt-in) | JSONL file at `$CLAUDE_SQL_BEDROCK_TRACE`; one row per classifier call with token mix, cache hits, elapsed ms | `cache_read_input_tokens`, `ephemeral_1h_input_tokens`, `stop_reason` | `src/claude_sql/core/llm_shared.py:78-83,285-325` |

## First-checks ladder

Run these in order; cheapest first, most invasive last.

1. `claude-sql list-cache --format json` — one read-only command that reports every parquet's `exists`, `bytes`, `mtime`, `rows`. Most `[catalog_error]` and `no_embeddings` failures resolve here. `src/claude_sql/app/cli.py:979-993`
2. Read the stderr envelope's `kind` field. `parse_error`/`catalog_error` point at the user SQL; `runtime_error` is anything else downstream of DuckDB; `invalid_input` is a malformed flag. `src/claude_sql/core/output.py:202-225`
3. `claude-sql schema --format json` — dumps the registered view + macro catalog so you can confirm the column the failing query referenced actually exists. `src/claude_sql/core/output.py:193-200`
4. Re-run with `--verbose`. View-registration paths log at DEBUG by default and surface skipped parquets, fallback `SELECT *` re-bindings, and Lance migration messages that the default INFO level hides. `src/claude_sql/core/logging_setup.py:38-43`, `src/claude_sql/core/sql_views.py:1931-1942`
5. Probe the Lance embeddings namespace. If the directory exists but the embeddings table does not, every search binds against the empty placeholder table and returns zero rows. `src/claude_sql/core/lance_store.py:48-57`, `src/claude_sql/core/sql_views.py:1740-1758`
6. Inspect the retry queue: `sqlite3 ~/.claude/state.db "SELECT pipeline, unit_id, attempts, error FROM retry_queue WHERE completed_at IS NULL"`. Non-empty rows survive across runs; the next subcommand drains them automatically. `src/claude_sql/core/retry_queue.py:131-159`
7. Confirm Bedrock model reachability. A `cris-profile-not-in-list` false alarm comes from `ListInferenceProfiles`; ping `invoke_model` directly before concluding the profile is unavailable, and confirm `region`/`CLAUDE_SQL_REGION`. `src/claude_sql/core/config.py:133-161`, `.erpaval/solutions/api-patterns/cris-profile-not-in-list.md:13-29`
8. Spend tokens to investigate slowness: `claude-sql query "..." --profile-json` writes a DuckDB profiling tree under `~/.claude/profiling/`. JSONL re-scan dominance is the usual culprit. `src/claude_sql/app/cli.py:695-716`

## Known incident patterns

The codebase carries no `INCIDENT:`/`POSTMORTEM:` comments and no `INCIDENTS.md` file. History is instead encoded in GH-issue references, named error classes, and the `.erpaval/solutions/` corpus — the patterns below are reconstructed from those tagged sources.

- **BedrockRefusalError (terminal, never retry):** when Bedrock returns `stop_reason=refusal` the parser raises this terminal class; workers stamp a neutral placeholder row and `mark_done` so the unit is not re-classified on every future run. Signal: log line `refused — neutral placeholders` (trajectory) or `refused by Bedrock — marking none` (friction). Mitigation: none required; the placeholder is the contract. `src/claude_sql/core/llm_shared.py:490-518`, `src/claude_sql/analytics/friction_worker.py:573-587`
- **Empty Lance namespace masquerading as a real dataset:** DuckDB's `ATTACH (TYPE LANCE)` succeeds on any directory, even one with no embeddings table; the catalog error fires later at view-bind time. `register_vss` probes via `lance_store._has_table` and creates an empty placeholder `message_embeddings` so `semantic_search` still binds. Signal: `No Lance embeddings table at <uri>`. Mitigation: `claude-sql embed`. `src/claude_sql/core/sql_views.py:1740-1758`, `.erpaval/solutions/api-patterns/duckdb-attach-lance-empty-namespace.md:13-35`
- **SQLite WAL cold-start PRAGMA race:** `PRAGMA journal_mode=WAL` and `CREATE TABLE IF NOT EXISTS` both grab the writer lock, so concurrent cold-start connections raced and the loser raised `database is locked` before `busy_timeout` could absorb it. Resolved by `_SCHEMA_BOOTSTRAP_LOCK` plus a per-path sentinel run once per process. Signal: historical only — the lock is silent. Mitigation: present in code; do not move the PRAGMAs back to per-connection setup. `src/claude_sql/core/checkpointer.py:243-257`, `src/claude_sql/core/retry_queue.py:62-76`, `.erpaval/solutions/best-practices/sqlite-wal-cold-start-pragma-race.md:13-48`
- **Stale glob entry between view registration and the materializing query (GH-class corpus rotation):** Claude Code rotates JSONLs, so a path visible at view-bind time can vanish before the JOIN runs. `session_text_corpus` catches `duckdb.IOException`, logs once, and returns whatever materialized; `session_bounds` retries the query exactly once. Signal: `session_text_corpus: IO error while materializing` or `session_bounds: stale glob entry — retrying once`. Mitigation: re-run the command; the glob has stabilized by the second invocation. `src/claude_sql/core/session_text.py:177-180,254-259`
- **Parquet schema rip-and-replace (v1.0 windowed trajectory / pair-keyed conflicts):** old per-message trajectory shards and old whole-session conflict shards are detected at worker startup and the whole cache is purged before new shards land. Signal: `trajectory: purged N legacy per-message shard(s)` or `conflicts: legacy shard ... — purging cache`. Mitigation: re-run the matching worker; the next `register_analytics` rebinds the curated projection. `src/claude_sql/analytics/trajectory_worker.py:356-376`, `src/claude_sql/analytics/conflicts_worker.py:88-122`
- **Curated projection vs legacy schema mismatch:** when an analytics parquet predates a v1.0 alias rewrite the projection's column reference fails; `register_analytics` falls back to bare `SELECT *` and emits a single warning so legacy shards stay queryable for read-only inspection until the next worker run regenerates them. Signal: `register_analytics: <view> bound with fallback SELECT * (legacy schema detected ...; run the matching worker to refresh)`. Mitigation: re-run the matching worker. `src/claude_sql/core/sql_views.py:1958-1988`
- **DuckDB read_json re-binds schema on every DESCRIBE (perf incident):** binding `v_raw_events` re-ran JSON schema inference per `DESCRIBE`/view bind (14.5 s cold on the live corpus), so `claude-sql schema` answers from a static `VIEW_SCHEMA` dict and the raw readers are materialized once as TEMP TABLEs with an explicit `columns={...}` projection. Signal: a cold `schema`/`SELECT count(*)` that takes seconds. Mitigation: keep the static schema dict and TEMP TABLE projection in sync (CI drift test guards this). `src/claude_sql/core/sql_views.py:95-115,401-414`, `.erpaval/solutions/api-patterns/duckdb-read-json-rebinds-schema-on-describe.md`

## See also

- [claude-sql · Contract map](contract-map.md) — 12 shared source files
- [claude-sql · Tech debt](tech-debt.md) — 11 shared source files
- [claude-sql · Public API](../reference/public-api.md) — 10 shared source files
- [claude-sql · Business logic](business-logic.md) — 9 shared source files
- [claude-sql · Module map](../architecture/module-map.md) — 9 shared source files
