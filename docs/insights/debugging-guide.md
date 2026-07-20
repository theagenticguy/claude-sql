# claude-sql · Debugging guide

This file answers one question: *something is broken — where do I look first?*
It maps the symptoms a `claude-sql` operator actually sees onto the source that
produces them, gives the single cheapest confirming check for each, catalogs
every place the tool emits diagnostics, and records the incident patterns that
are already captured in source and in `.erpaval/solutions/`.

Two facts frame everything below. First, `claude-sql` has **one log sink**: a
loguru handler on stderr, no file log and no observability platform
(`src/claude_sql/infrastructure/logging_setup.py:44`). Second, it has a **stable
exit-code taxonomy** that agents rely on, so the exit code is usually the
fastest triage signal (`src/claude_sql/domain/errors.py:26`).

## Failure-mode index

| Symptom | Likely surface | First check | Citation |
|---|---|---|---|
| Command exits `64` on a query | Malformed SQL (`ParserException`) or a bad `--glob` flag, both mapped to `invalid_input`/`parse_error` | Read stderr JSON `error.kind`: `parse_error` means fix the SQL, `invalid_input` means fix the flag; run `claude-sql schema --format json` for valid names | `src/claude_sql/infrastructure/duckdb_errors.py:26`, `src/claude_sql/interfaces/cli/output.py:174` |
| Command exits `65` (`catalog_error`) | Unknown view/macro/column — a parquet cache the view depends on was never built, so the view no-ops and the name is absent | Run `claude-sql list-cache` to see which analytics parquets exist; the missing one explains the absent view | `src/claude_sql/infrastructure/duckdb_errors.py:33`, `src/claude_sql/interfaces/cli/app.py:922` |
| Command exits `70` (`runtime_error`) | Any other `duckdb.Error` at execute time (spill, OOM, corrupt shard) | Re-run with `claude-sql query "..." --profile-json` and inspect the timing tree under the profiling dir; check `CLAUDE_SQL_DUCKDB_MEMORY_LIMIT` | `src/claude_sql/infrastructure/duckdb_errors.py:40`, `src/claude_sql/interfaces/cli/output.py:183` |
| `search`/`peek` exits `2` (`no_embeddings`) | LanceDB store empty or absent; view bound over an empty schema so the macro binds but returns nothing | Confirm the store, then `claude-sql embed` to backfill; the gate fires before any query runs | `src/claude_sql/interfaces/cli/app.py:1475`, `src/claude_sql/infrastructure/duckdb_views.py:1912` |
| `shell` exits `127` (`duckdb_missing`) | System `duckdb` binary not on PATH (only `shell` shells out to it) | `which duckdb`; install it or use `claude-sql query '<sql>'` which needs no binary | `src/claude_sql/interfaces/cli/app.py:499` |
| Semantic search returns plausible-but-wrong neighbors after switching embedders | Provider/dim mismatch — vectors from a different model live in an incompatible space | The bind-time guard raises `EmbeddingProviderMismatch` naming both sides; if it did not fire, the store predates the guard — `rm -rf` the Lance dir and re-embed | `src/claude_sql/domain/embedding_guard.py:53`, `src/claude_sql/infrastructure/duckdb_views.py:1943` |
| An `analyze`/`classify`/`trajectory`/`friction` run stalls or a chunk never resolves | Bedrock throttling absorbed by the tenacity retry (up to 10 attempts, 2–60s backoff) | Watch stderr for `WARNING` retry lines from `loguru_before_sleep`; if sustained, drop `CLAUDE_SQL_EMBED_CONCURRENCY` / `CLAUDE_SQL_LLM_CONCURRENCY` | `src/claude_sql/infrastructure/bedrock/client.py:398`, `src/claude_sql/infrastructure/logging_setup.py:53` |
| A classification pipeline appears to skip messages, leaving neutral rows | Terminal `BedrockRefusalError` — the model refused under content policy; the pipeline stamps a neutral placeholder instead of cycling forever | Inspect the output parquet for placeholder rows; refused units are intentionally not retried | `src/claude_sql/application/use_cases/trajectory.py:554`, `src/claude_sql/domain/errors.py:63` |
| Opt-in Luna analytics silently produce nothing while the core run succeeds | `LlmAnalyticsUnavailable` fail-open — the Strands adapter logged a warning and degraded rather than crashing | Grep stderr for `strands-luna:` WARNING lines; the core SQL/embedding pipeline is unaffected by design | `src/claude_sql/infrastructure/llm_analytics/strands_luna.py:129`, `src/claude_sql/domain/errors.py:94` |
| A query over the live corpus errors with "Table ... does not exist" mid-run | A JSONL vanished/rotated between register and the JOIN; DuckDB raised `IOException` | The reader catches this and skips the stale file with a warning; if it surfaced, the glob matched zero files — check the path | `src/claude_sql/infrastructure/session_text_loader.py:134`, `src/claude_sql/infrastructure/transcript_reader.py:105` |
| First concurrent run against a fresh checkpointer DB raises "database is locked" | SQLite WAL cold-start race: `PRAGMA journal_mode=WAL` itself takes the writer lock before `busy_timeout` can absorb it | Retry once; steady-state runs do not hit it. See the incident pattern below | `src/claude_sql/infrastructure/sqlite_state/checkpointer.py:175`, `.erpaval/solutions/best-practices/sqlite-wal-cold-start-pragma-race.md` |
| View registration fails after a schema-changing release | Mixed legacy + new-schema parquet shards in one cache dir; DuckDB's glob can't unify the shapes | Detected via parquet metadata — stale shards are deleted on first run; if it surfaced, clear the cache dir and re-run | `.erpaval/solutions/best-practices/parquet-schema-migration-rip-and-replace.md`, `src/claude_sql/infrastructure/duckdb_views.py:558` |

## Log and error surfaces

There is exactly one log sink and no structured/observability backend — every
diagnostic is a loguru line on stderr, plus the classified-error JSON that
`emit_error` writes to stderr on non-TTY. Grep stderr; there is no log file to
tail.

| Surface | Where it emits | What to grep for | Citation |
|---|---|---|---|
| Primary log | stderr, single loguru handler; format `HH:mm:ss LEVEL {extra} message` | Level tokens `WARNING` / `ERROR`; `{extra}` binds pipeline context | `src/claude_sql/infrastructure/logging_setup.py:24`, `src/claude_sql/infrastructure/logging_setup.py:44` |
| Log level control | Set from `--verbose` (DEBUG) / `--quiet` (ERROR) / `LOGURU_LEVEL` env (default INFO) | If stderr is empty, you are at INFO and view-registration is DEBUG on purpose — pass `--verbose` | `src/claude_sql/infrastructure/logging_setup.py:39` |
| Classified error (agent path) | stderr as JSON `{"error":{"kind","message","hint"}}` when not a TTY | `"kind":` to distinguish parse/catalog/runtime/invalid_input | `src/claude_sql/interfaces/cli/output.py:155`, `src/claude_sql/domain/errors.py:118` |
| Classified error (human path) | stderr as `[kind] message` + `hint:` line on a TTY | `[parse_error]` / `[catalog_error]` / `[runtime_error]` prefixes | `src/claude_sql/interfaces/cli/output.py:150` |
| Retry backoff | stderr WARNING via `loguru_before_sleep` on every tenacity sleep | Retry-state lines during embed/classify; sustained means throttling | `src/claude_sql/infrastructure/logging_setup.py:53`, `src/claude_sql/infrastructure/bedrock/client.py:402` |
| Bedrock call trace (opt-in) | Appended as NDJSON to `_BEDROCK_TRACE_PATH` when set; failures to write are swallowed | One JSON row per call with `elapsed_ms`; tracing never breaks a run | `src/claude_sql/infrastructure/bedrock/client.py:322` |
| Query profile (opt-in) | JSON timing tree under the profiling dir via `--profile-json` | Node timings to find the slow scan | `src/claude_sql/interfaces/cli/app.py:508` |
| Cache inventory | `list-cache` reports `{exists, bytes, mtime, rows}` per parquet to stdout | Which analytics parquet is missing/stale | `src/claude_sql/interfaces/cli/app.py:399` |

## First-checks ladder

1. Read the **exit code**. It is the cheapest signal and it is stable: `0` ok,
   `2` no_embeddings, `64` invalid_input/parse, `65` catalog, `70` runtime,
   `127` duckdb binary missing. `src/claude_sql/domain/errors.py:26`
2. Read **stderr**. On a pipe/agent it carries `{"error":{"kind","message","hint"}}`;
   the `hint` usually names the fix. `src/claude_sql/interfaces/cli/output.py:142`
3. If nothing printed, **raise the log level**: re-run with `--verbose` — view
   registration and most lifecycle logs sit at DEBUG so the default stderr stays
   empty for read-only flows. `src/claude_sql/infrastructure/logging_setup.py:39`
4. On `65` (catalog) or empty results, run **`claude-sql list-cache`** — a view
   silently no-ops when its backing parquet is absent, so a missing cache entry
   explains an absent view or column. `src/claude_sql/interfaces/cli/app.py:399`
5. On `2` (no_embeddings) or empty semantic search, confirm the **LanceDB store**
   is populated; if empty the view binds over an empty schema and returns
   nothing — run `claude-sql embed`. `src/claude_sql/infrastructure/duckdb_views.py:1912`
6. On wrong-looking neighbors after any embedder change, expect
   **`EmbeddingProviderMismatch`** at bind time; recovery is `rm -rf` the Lance
   dir (default `~/.claude/embeddings_lance/`) then re-embed.
   `src/claude_sql/domain/embedding_guard.py:53`
7. On a stalled LLM run, **watch stderr for WARNING retry lines**; sustained
   throttling means lower `CLAUDE_SQL_LLM_CONCURRENCY` / `CLAUDE_SQL_EMBED_CONCURRENCY`
   to 2. `src/claude_sql/infrastructure/bedrock/client.py:398`
8. On `70` under load on a shared host, **bound DuckDB memory**: set
   `CLAUDE_SQL_DUCKDB_MEMORY_LIMIT='4GB'` (percentages resolve to MiB at apply
   time). `src/claude_sql/infrastructure/duckdb_connection.py:56`
9. On a slow-but-succeeding query, **profile it**: `claude-sql query "..."
   --profile-json` drops a JSON timing tree to disk without re-running blind.
   `src/claude_sql/interfaces/cli/app.py:508`
10. If the corpus mutated mid-query ("Table ... does not exist"), the reader
    already skips stale JSONLs with a warning — re-run; a persistent failure
    means the glob matches zero files, so verify the path.
    `src/claude_sql/infrastructure/session_text_loader.py:134`

## Known incident patterns

- **SQLite WAL cold-start race:** the first N concurrent connections to a fresh
  checkpointer DB each run `PRAGMA journal_mode=WAL`, which itself acquires the
  writer lock; the loser raises `database is locked` before `busy_timeout` can
  absorb it. Signal: `sqlite3.OperationalError: database is locked` only on cold
  start, never in steady state. Mitigation: serialize the WAL-mode PRAGMA on
  first open / retry once. `.erpaval/solutions/best-practices/sqlite-wal-cold-start-pragma-race.md`, `src/claude_sql/infrastructure/sqlite_state/checkpointer.py:175`
- **Parquet schema rip-and-replace:** a release that rekeys a cache schema
  (trajectory, conflicts, friction all changed keys in v1.0) leaves mixed legacy
  + new shards that DuckDB's glob cannot unify, failing view bind. Signal:
  bind-time column-shape mismatch across shards. Mitigation: detect stale shards
  via parquet metadata and delete the whole cache dir on first run before new
  shards land. `.erpaval/solutions/best-practices/parquet-schema-migration-rip-and-replace.md`, `src/claude_sql/infrastructure/duckdb_views.py:558`
- **DuckDB `ATTACH (TYPE LANCE)` is permissive:** ATTACH succeeds on any path
  (missing, empty, or metadata-only dir); the `Catalog Error: Table ...
  embeddings does not exist` fires later at SELECT/CREATE-VIEW time. Signal: a
  catalog error surfacing only when the view is queried, not when attached.
  Mitigation: gate on `lance_store._has_table(...)`, not filesystem heuristics,
  and create an empty table so the macro still binds. `.erpaval/solutions/api-patterns/duckdb-attach-lance-empty-namespace.md`, `src/claude_sql/infrastructure/duckdb_views.py:1912`
- **`except BaseException` in async worker bodies deadlocks shutdown:** catching
  `BaseException` in an anyio task swallows `CancelledError`, so the parent's
  `aclose()` waits forever. Signal: a `trajectory`/`conflicts` run that hangs on
  teardown after an error. Mitigation: catch `Exception` (recoverable errors are
  all subclasses; cancellation cascades) — used at the worker chunk boundary.
  `src/claude_sql/application/use_cases/trajectory.py:556`
- **Best-effort migration must drop its sentinel and come up clean:** the
  legacy-DuckDB→SQLite checkpointer migration wraps its read in a broad except;
  any failure logs, touches the sentinel, and returns so SQLite starts fresh
  rather than wedging. Signal: a one-time `Failed to read legacy DuckDB` log on
  upgrade. Mitigation: idempotent sentinel touch — no action needed.
  `src/claude_sql/infrastructure/sqlite_state/checkpointer.py:160`

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 10 shared source citations
- [claude-sql · Contract map](../insights/contract-map.md) — 8 shared source citations
- [claude-sql · Module map](../architecture/module-map.md) — 6 shared source citations
- [claude-sql · Tech debt](../insights/tech-debt.md) — 5 shared source citations
- [claude-sql · Processes](../behavior/processes.md) — 4 shared source citations
