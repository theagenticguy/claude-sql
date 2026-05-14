# claude-sql · Debugging guide

This file answers the question *"something is broken — where do I look
first?"* for the claude-sql CLI and its analytics pipelines. Every row
traces back to a source citation; nothing here is invented.

claude-sql has two complementary error-handling spines. CLI surface
errors classify into stable exit codes via
`classify_duckdb_error` and `run_or_die`
(`src/claude_sql/output.py:180-254`). Worker pipelines (`embed`,
`classify`, `trajectory`, `conflicts`, `friction`) catch transient
failures, enqueue the failing unit on a durable SQLite/WAL retry queue
(`src/claude_sql/retry_queue.py:1-19`), and continue draining other
sessions so a single bad input never wedges a backfill.

## Failure-mode index

| Symptom | Likely surface | First check | Citation |
| --- | --- | --- | --- |
| `claude-sql search` exits 2 with `No embeddings yet` on stderr | Lance namespace empty, `register_vss` created an empty placeholder table | Run `claude-sql list-cache` and confirm the `embeddings` row reports `rows=0`; if so run `claude-sql embed --no-dry-run` | `src/claude_sql/cli.py:1530-1534`, `src/claude_sql/sql_views.py:1763-1781` |
| Stderr carries `[catalog_error]` for an unknown view or column | Required analytics parquet not yet generated; `register_analytics` skipped at DEBUG | `claude-sql list-cache --format json` to see which parquet is missing, then run the matching generator | `src/claude_sql/output.py:195-201`, `src/claude_sql/sql_views.py:1937-1948` |
| `claude-sql query` exits 64 with `[parse_error]` on stderr | Malformed SQL in the user-supplied query | Stderr emits a `hint:` pointing at `claude-sql schema --format json` for view/macro names | `src/claude_sql/output.py:188-194` |
| `claude-sql query` exits 64 with `--glob pattern '...' contains more than one '**' segment` | DuckDB `read_json` rejects multi-`**` globs; surfaced by the up-front guard | Drop one `**` from the pattern; use a single recursive segment plus a non-recursive tail | `src/claude_sql/output.py:156-177` |
| `claude-sql shell` exits 127 with `` `duckdb` binary not found on PATH `` | System `duckdb` CLI missing | Install duckdb or use `claude-sql query` instead; the in-process API needs no system binary | `src/claude_sql/cli.py:684-692` |
| `register_analytics: <view> bound with fallback SELECT *` warning, downstream queries miss aliases | On-disk parquet predates a v1.0 schema rewrite (trajectory `curr_uuid`, conflicts `turn_a_uuid`) | Re-run the matching worker (`trajectory` / `conflicts`); legacy shards are detected and purged on first run | `src/claude_sql/sql_views.py:1961-1991`, `src/claude_sql/trajectory_worker.py:340-377`, `src/claude_sql/conflicts_worker.py:100-128` |
| Backfill leaves sessions unprocessed; logs say `failed (queued for retry)` | Worker enqueued the unit on the SQLite retry queue after a non-retryable error from Bedrock | Inspect the queue, then re-run the same subcommand — `drain` is called at startup so retries fire automatically | `src/claude_sql/trajectory_worker.py:862-877`, `src/claude_sql/friction_worker.py:589-597`, `src/claude_sql/retry_queue.py:131-194` |
| `claude-sql resolve <sha>` exits 70 with `BindingMismatchError` | Commit's `Claude-Transcript-*` trailer disagrees with the `refs/notes/transcripts` note | `claude-sql resolve <sha> --all-sources` shows both surfaces side-by-side, hint emitted on stderr | `src/claude_sql/binding.py:87-106`, `src/claude_sql/cli.py:2716-2724` |

## Log and error surfaces

| Surface | Where it emits | What to grep for | Citation |
| --- | --- | --- | --- |
| Loguru stderr handler | stderr; level `INFO` by default, `DEBUG` with `--verbose`, `ERROR` with `--quiet` | `Retrying` (tenacity), `failed (queued for retry)`, `bound with fallback SELECT *`, `purging cache` | `src/claude_sql/logging_setup.py:37-50` |
| Classified-error JSON envelope | stderr, off-TTY only; format `{"error": {"kind", "message", "hint"}}` | `\"kind\":\"parse_error\"`, `\"kind\":\"catalog_error\"`, `\"kind\":\"runtime_error\"`, `\"kind\":\"invalid_input\"` | `src/claude_sql/output.py:134-141,210-225` |
| Stable CLI exit codes | Process exit status; agents key off these without parsing tracebacks | `0` ok / `2` no_embeddings / `64` invalid_input or parse_error / `65` catalog_error / `66` kappa delta-gate tripped / `70` runtime_error / `127` duckdb_missing | `src/claude_sql/output.py:49-57`, `src/claude_sql/cli.py:2533-2534` |
| Tenacity retry breadcrumbs | stderr via `loguru_before_sleep`; one line per retry naming function, sleep seconds, exception | `Retrying _invoke_classifier_sync`, `Retrying.*ThrottlingException`, `Retrying.*ServiceUnavailableException` | `src/claude_sql/logging_setup.py:53-95`, `src/claude_sql/llm_shared.py:395-401` |
| Cache compact / shard purge log | INFO/WARN; emitted by the v1.0 schema-migration purge | `purged \d+ legacy per-message shard`, `legacy shard .* — purging cache`, `unreadable shard .* — purging cache` | `src/claude_sql/trajectory_worker.py:374-377`, `src/claude_sql/conflicts_worker.py:111-122` |
| LanceDB index-create fallback warning | WARN; index falls back to brute-force scan, search still works | `LanceDB create_index failed`, `Lance optimize failed` | `src/claude_sql/lance_store.py:140,155` |
| DuckDB `--profile-json` profiling tree | JSON file under `~/.claude/profiling/`; written by DuckDB itself when `--profile-json` is passed | filename pattern `<label>-<ts>.json`; inspect the timing tree to spot JSONL re-scan dominance | `src/claude_sql/cli.py:695-716` |

## First-checks ladder

Run these in order; cheapest first.

1. `claude-sql list-cache --format json` — single read-only command that
   reports every parquet's `exists`, `bytes`, `mtime`, `rows`. Most
   `[catalog_error]` and `no_embeddings` failures resolve here. `src/claude_sql/cli.py:979-1025`
2. Read the stderr envelope's `kind` field. `parse_error` and `catalog_error`
   point at the user-supplied SQL; `runtime_error` is anything else
   downstream of DuckDB; `invalid_input` is a malformed flag. `src/claude_sql/output.py:202-225`
3. `claude-sql schema --format json` — dumps the registered view + macro
   catalog. Confirms whether the column the failing query referenced
   actually exists. `src/claude_sql/output.py:193-200`
4. Re-run with `--verbose`. View-registration paths log at DEBUG by
   default and surface skipped parquets, fallback `SELECT *`
   re-bindings, and Lance migration messages that the default INFO
   level hides. `src/claude_sql/logging_setup.py:38-43`, `src/claude_sql/sql_views.py:1937-1948`
5. Probe the Lance embeddings namespace via `_has_table`. If the
   directory exists but the embeddings table doesn't, every search
   query binds against the empty placeholder table created by
   `register_vss` and returns zero rows. `src/claude_sql/lance_store.py:48-57`, `src/claude_sql/sql_views.py:1763-1781`
6. Inspect the retry queue: `sqlite3 ~/.claude/state.db "SELECT pipeline,
   unit_id, attempts, error FROM retry_queue WHERE completed_at IS NULL"`.
   Non-empty rows survive across runs; the next subcommand call will
   drain them automatically. `src/claude_sql/retry_queue.py:131-198`
7. Check the migration marker. If `~/.claude/.cache_migration_done`
   was stamped before legacy caches were restored, `list-cache` shows
   `legacy:<name>` entries that won't auto-migrate. Manually delete the
   marker or run `claude-sql cache migrate`. `src/claude_sql/cli.py:285-317,1031-1037`
8. Spend money to investigate: `claude-sql query "..." --profile-json`
   writes a DuckDB profiling tree under `~/.claude/profiling/`. Use it
   when a query is unexpectedly slow — JSONL re-scan dominance is the
   usual culprit. `src/claude_sql/cli.py:695-716`

## Known incident patterns

- **Parquet schema rip-and-replace (v1.0 windowed trajectory and pair-keyed conflicts):** old per-message trajectory shards (carrying `uuid` + `sentiment_delta`) and old whole-session conflicts shards (carrying `conflict_idx` / `empty`) are detected via parquet metadata at worker startup and the entire cache is purged before new shards land. Signal: `purged N legacy per-message shard(s)` or `legacy shard ... — purging cache`. Mitigation: re-run the matching worker; the next `register_analytics` rebinds the curated projection. `src/claude_sql/trajectory_worker.py:340-377`, `src/claude_sql/conflicts_worker.py:100-128`
- **Empty Lance namespace masquerading as a real dataset:** DuckDB's `ATTACH (TYPE LANCE)` succeeds on any directory, even one without an embeddings table. The catalog error fires later at `SELECT`/`CREATE VIEW` time. `register_vss` probes via `lance_store._has_table` and creates an empty placeholder `message_embeddings` table when absent so `semantic_search` still binds. Signal: `No Lance embeddings table at <uri>`. Mitigation: `claude-sql embed --no-dry-run`. `src/claude_sql/sql_views.py:1763-1781`, `src/claude_sql/lance_store.py:48-57`
- **SQLite WAL cold-start PRAGMA race in retry-queue bootstrap:** two threads racing the not-bootstrapped check could double-issue DDL and trip the writer lock. Resolved by `_SCHEMA_BOOTSTRAP_LOCK` and a per-path sentinel in the bootstrap path. Signal: historical only; no log line — the lock is silent. Mitigation: present in code; do not remove the lock. `src/claude_sql/retry_queue.py:62-76`
- **Stale glob entry between view registration and materializing query:** Claude Code rotates JSONLs; a path visible at view-bind time can vanish before the JOIN runs. `session_text_corpus` catches `duckdb.IOException`, logs once, and returns whatever materialized; `session_bounds` retries the query exactly once. Signal: `session_text_corpus: IO error while materializing` or `session_bounds: stale glob entry — retrying once`. Mitigation: re-run the command; the glob has stabilized by the second invocation. `src/claude_sql/session_text.py:177-180,253-260`
- **BedrockRefusalError is terminal — never retry:** when Bedrock returns `stop_reason=refusal`, callers stamp a neutral placeholder row and `mark_done` so the message is not re-classified on every future run. Signal: `refused — neutral placeholders` (trajectory) or `refused by Bedrock — marking none` (friction). Mitigation: none required; the placeholder is the contract. `src/claude_sql/llm_shared.py:490-518`, `src/claude_sql/trajectory_worker.py:775-783`, `src/claude_sql/friction_worker.py:572-588`
- **Curated projection vs legacy schema mismatch:** when an analytics parquet predates a v1.0 alias rewrite (`autonomy AS autonomy_tier`, `curr_sentiment AS sentiment`, etc.) the projection's column reference fails. `register_analytics` falls back to bare `SELECT *` and emits a single warning so legacy shards remain queryable for read-only inspection until the next worker run regenerates them. Signal: `register_analytics: <view> bound with fallback SELECT * (legacy schema detected at <path>; run the matching worker to refresh)`. Mitigation: re-run the matching worker. `src/claude_sql/sql_views.py:1961-1991`
