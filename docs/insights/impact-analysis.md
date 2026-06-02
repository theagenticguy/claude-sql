# claude-sql · Impact analysis

This file answers one question: *if I change surface X, what else do I have to touch or carefully validate?*

**"High-impact surface" definition.** A surface is ranked by inbound reference count — the number of distinct source and test files that import a symbol from the module. The count was measured with `grep -rhoE 'from claude_sql\.<pkg>\.<module> import'` over `packages/*/src` and `tests/`. The raw ranking of `core.*` modules by inbound files is: `config` (41), `sql_views` (29), `parquet_shards` (16), `llm_shared` (13), `schemas` (12), `output` (12), `session_text` (7), `logging_setup` (4), `home` (3), `checkpointer` (3), `retry_queue` (1).

The top eight surfaces below deviate from pure count in three deliberate places: `cli` (the user-facing entry point), `checkpointer` (the incremental-rebuild cache contract), and `binding` (the git wire format that implements the transcript-provenance contract) displace `output` and `session_text`. Those three are the load-bearing *contracts* a changer reasons about, even though `output` outranks two of them on raw fan-in. `output` and `session_text` are mechanical helpers and are listed under `## Other notable surfaces`.

`Type` vocabulary: `direct import` (file imports the symbol), `indirect` (file consumes a value the surface produces without importing it directly), `runtime dispatch` (reached through CLI subcommand orchestration or SQL string, not a Python import), `test`, `config`. `Touch on change`: `yes` (signature change forces an edit), `likely` (review needed even without a signature change), `no` (only a behavioral change reaches it).

## core.config — Settings, DEFAULT_PRICING

Defined at: `packages/core/src/claude_sql/core/config.py:126` (`class Settings(BaseSettings)`); `DEFAULT_PRICING` at `packages/core/src/claude_sql/core/config.py:117`.

Highest fan-in surface in the repo (41 inbound files). Every analytics worker, the CLI, and the SQL-view registrar pull configuration from `Settings`.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `app/cli.py` | direct import | yes | `packages/app/src/claude_sql/app/cli.py` |
| analytics workers (`classify_worker`, `conflicts_worker`, `friction_worker`, `trajectory_worker`, `embed_worker`, `cluster_worker`, `community_worker`, `terms_worker`, `skills_catalog`, `ingest`) | direct import | yes | `packages/analytics/src/claude_sql/analytics/classify_worker.py` |
| `provenance/review_sheet_worker.py` | direct import | yes | `packages/provenance/src/claude_sql/provenance/review_sheet_worker.py` |
| `core/llm_shared.py`, `core/session_text.py`, `core/sql_views.py` | direct import (TYPE_CHECKING in `llm_shared`/`session_text`) | likely | `packages/core/src/claude_sql/core/sql_views.py` |
| `tests/test_config.py`, `tests/test_team_corpus.py`, `tests/conftest.py`, + 15 more test files | test | likely | `tests/test_config.py` |

Blast-radius notes:
- Env-var binding uses `env_prefix="CLAUDE_SQL_"` (`packages/core/src/claude_sql/core/config.py:134`); renaming a field silently changes the public environment-variable contract operators rely on, with no compile-time signal.
- The `_derive_team_corpus_globs` validator runs `mode="after"` (`packages/core/src/claude_sql/core/config.py:339`), so team-corpus glob fields are computed, not user-set — code that writes those fields directly is overwritten on the next model construction.
- `DEFAULT_PRICING` is a per-model `(input, output)` rate map (`packages/core/src/claude_sql/core/config.py:117`); cost-reporting consumers assume a key exists for every model id they pass, so adding a model without a pricing entry raises `KeyError` at report time, not at config load.

## core.sql_views — VIEW_NAMES, VIEW_SCHEMA, MACRO_NAMES, MACRO_SIGNATURES, register_all

Defined at: `packages/core/src/claude_sql/core/sql_views.py:62` (`VIEW_NAMES`), `:116` (`VIEW_SCHEMA`), `:314` (`MACRO_NAMES`), `:355` (`MACRO_SIGNATURES`), `:2078` (`register_all`).

Second-highest fan-in (29 inbound files), almost entirely from the test suite, which asserts the view/macro contract.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `app/cli.py` | direct import | yes | `packages/app/src/claude_sql/app/cli.py` |
| analytics workers | runtime dispatch | likely | `packages/analytics/src/claude_sql/analytics/conflicts_worker.py` |
| `tests/test_sql_views.py` (schema/macro drift) | test | yes | `tests/test_sql_views.py` |
| `tests/test_skill_views.py`, `tests/test_turn_window.py`, `tests/test_session_bounds.py`, `tests/test_v2_analytics.py`, + 15 more | test | likely | `tests/test_v2_analytics.py` |

Blast-radius notes:
- `VIEW_SCHEMA` and `MACRO_SIGNATURES` are the source of truth for drift tests (`packages/core/src/claude_sql/core/sql_views.py:116`, `:355`); changing a view's column set or a macro's argument list without updating these dicts fails `tests/test_sql_views.py` rather than runtime.
- Analytics workers consume the registered views through SQL strings and CLI orchestration, not Python imports — they will not appear in an import grep, but renaming a view in `VIEW_NAMES` breaks every query string referencing the old name (`packages/core/src/claude_sql/core/sql_views.py:62`).
- `register_all` registers the full view/macro set in one call (`packages/core/src/claude_sql/core/sql_views.py:2078`); any consumer that opens a fresh DuckDB connection must call it before querying, so a new code path that skips it sees missing-view errors.

## core.parquet_shards — write_part, read_all, replace_sessions, iter_part_files, count_rows, is_sharded_dir

Defined at: `packages/core/src/claude_sql/core/parquet_shards.py:87` (`write_part`), `:131` (`read_all`), `:167` (`replace_sessions`), `:72` (`iter_part_files`), `:149` (`count_rows`), `:54` (`is_sharded_dir`).

The on-disk sharded-parquet storage layer. Every worker that persists rows and the CLI that reads them depend on the directory layout these functions enforce.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `app/cli.py` | direct import | yes | `packages/app/src/claude_sql/app/cli.py` |
| `analytics/ingest.py`, `classify_worker.py`, `conflicts_worker.py`, `friction_worker.py`, `trajectory_worker.py` | direct import | yes | `packages/analytics/src/claude_sql/analytics/ingest.py` |
| `core/sql_views.py` | direct import | likely | `packages/core/src/claude_sql/core/sql_views.py` |
| `tests/test_parquet_shards.py`, `tests/test_ingest.py`, `tests/test_conflicts_storage_v2.py`, + 6 more | test | likely | `tests/test_parquet_shards.py` |

Blast-radius notes:
- `is_sharded_dir` (`packages/core/src/claude_sql/core/parquet_shards.py:54`) and `iter_part_files` (`:72`) encode the part-file naming convention; changing the filename pattern silently strands existing shards, since `read_all` discovers parts by globbing rather than from a manifest.
- `replace_sessions` (`packages/core/src/claude_sql/core/parquet_shards.py:167`) is the in-place update primitive workers call on re-run; it assumes session-id is the partition key, so a worker that writes rows without a stable session id duplicates rather than replaces on the next run.
- `read_all` returns `pl.DataFrame | None` (`packages/core/src/claude_sql/core/parquet_shards.py:131`) — `None` for an empty or absent shard dir; every caller must handle the `None` branch or it raises on first attribute access.

## core.llm_shared — classify_one, _invoke_classifier_sync, _build_bedrock_client, _parse_structured_payload, BedrockRefusalError

Defined at: `packages/core/src/claude_sql/core/llm_shared.py:563` (`async def classify_one`), `:402` (`_invoke_classifier_sync`), `:346` (`_build_bedrock_client`), `:500` (`_parse_structured_payload`), `:490` (`class BedrockRefusalError`).

The shared Bedrock-invocation layer behind every LLM-backed analytics pipeline.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `analytics/classify_worker.py`, `conflicts_worker.py`, `friction_worker.py`, `trajectory_worker.py` | direct import | yes | `packages/analytics/src/claude_sql/analytics/classify_worker.py` |
| `analytics/embed_worker.py` (`_build_bedrock_client` only) | direct import | likely | `packages/analytics/src/claude_sql/analytics/embed_worker.py` |
| `provenance/review_sheet_worker.py` | direct import | yes | `packages/provenance/src/claude_sql/provenance/review_sheet_worker.py` |
| `tests/test_llm_worker.py`, `tests/test_llm_worker_parser.py`, `tests/test_llm_worker_cache.py`, `tests/test_llm_worker_pipelines.py`, + 2 more | test | yes | `tests/test_llm_worker.py` |

Blast-radius notes:
- `_parse_structured_payload` (`packages/core/src/claude_sql/core/llm_shared.py:500`) is the single point that turns a Bedrock response into a dict; every worker's schema-validation step depends on its output shape, and `tests/test_llm_worker_parser.py` pins that shape directly.
- `BedrockRefusalError` (`packages/core/src/claude_sql/core/llm_shared.py:490`) is a typed control-flow signal, not a generic failure; callers branch on it to skip-and-continue, so widening it to a plain `Exception` would silently swallow refusals into the retry path.
- `classify_one` is `async` (`packages/core/src/claude_sql/core/llm_shared.py:563`) while `_invoke_classifier_sync` is the sync core (`:402`); a new synchronous call site needs the sync helper, not `classify_one`, or it must run inside an event loop.

## core.schemas — pydantic models + Bedrock JSON-schema dicts

Defined at: `packages/core/src/claude_sql/core/schemas.py:99` (`SessionClassification`), `:173` (`TrajectoryWindow`), `:266` (`TrajectoryArrayResult`), `:292` (`ConflictPair`), `:379` (`ConflictsResult`), `:406` (`UserFrictionSignal`), `:477` (`Correction`), `:509` (`PRReviewSheet`); `__all__` at `:583`.

The structured-output contract shared between workers and the LLM. Each model has a paired `*_SCHEMA` dict generated by `_bedrock_schema` for the Bedrock tool-use call.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `analytics/classify_worker.py`, `conflicts_worker.py`, `friction_worker.py`, `trajectory_worker.py` | direct import | yes | `packages/analytics/src/claude_sql/analytics/classify_worker.py` |
| `provenance/review_sheet_worker.py` | direct import | yes | `packages/provenance/src/claude_sql/provenance/review_sheet_worker.py` |
| `tests/test_schemas.py`, `tests/test_v2_analytics.py`, `tests/test_review_sheet_worker.py`, + 4 more | test | yes | `tests/test_schemas.py` |

Blast-radius notes:
- Each `*_SCHEMA` dict is derived from its model via `_bedrock_schema` (`packages/core/src/claude_sql/core/schemas.py:170`, `:289`, `:403`, `:474`, `:580`); editing a model field changes the JSON schema sent to Bedrock, so the LLM prompt contract moves even when no Python signature changes.
- The stored parquet column set mirrors these model fields; renaming a field changes both the LLM contract and the downstream `VIEW_SCHEMA` expectation, coupling this surface to `core.sql_views`.
- `__all__` (`packages/core/src/claude_sql/core/schemas.py:583`) is the curated public export list; adding a model without listing it leaves it importable but undocumented, and the drift check in `tests/test_schemas.py` reads the export set.

## app.cli — App, subcommands, main

Defined at: `packages/app/src/claude_sql/app/cli.py:164` (`app = App(...)`, cyclopts); `main` at `:3073`. Entry point `claude-sql = "claude_sql.app.cli:main"` (`packages/app/pyproject.toml:37`).

The single user-facing entry point and the integration seam that wires every package together. Sub-apps `cache_app` (`:1239`) and `skills_app` (`:1457`) mount under it.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `claude-sql` console script | config | yes | `packages/app/pyproject.toml:37` |
| `tests/test_cli.py`, `tests/test_cli_coverage.py` | test | yes | `tests/test_cli.py` |
| `tests/test_analyze_chain.py`, `tests/test_ingest.py`, `tests/test_peek.py`, `tests/test_profile_json.py`, `tests/test_pr3_perf.py`, + 2 more | test | likely | `tests/test_analyze_chain.py` |

Blast-radius notes:
- The console-script entry point binds `main` by string path (`packages/app/pyproject.toml:37`); renaming `main` or moving `cli.py` breaks the installed `claude-sql` command without any import-time error in the library.
- cyclopts maps subcommand and parameter names to the user-facing CLI surface (`packages/app/src/claude_sql/app/cli.py:164`); renaming a subcommand function or its arguments is a breaking change to operator usage and to the many tests that invoke commands by name.
- `cli.py` is the only file that imports `sql_views.register_all`, `parquet_shards`, and `binding` together; it is the de facto orchestration seam, so a change to any of those surfaces most often lands here first.

## core.checkpointer — PIPELINE_NAMES, filter_unchanged, mark_completed, load_as_map, _connect

Defined at: `packages/core/src/claude_sql/core/checkpointer.py:41` (`PIPELINE_NAMES`), `:285` (`filter_unchanged`), `:329` (`mark_completed`), `:264` (`load_as_map`), `:226` (`_connect`).

Low fan-in (3 inbound files) but a write-side cache contract: it decides which sessions a re-run skips. Included for its contract weight, not its count.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `core/retry_queue.py` | direct import | yes | `packages/core/src/claude_sql/core/retry_queue.py:28` |
| `tests/test_checkpointer.py` | test | yes | `tests/test_checkpointer.py` |
| `tests/test_retry_queue.py` | test | likely | `tests/test_retry_queue.py` |

Blast-radius notes:
- `retry_queue` shares the same SQLite file and reuses `_connect` (`packages/core/src/claude_sql/core/retry_queue.py:56`); both tables live in one DB, so a schema or connection-handling change in the checkpointer affects the retry queue even though the two are nominally separate features.
- `PIPELINE_NAMES` is the closed set of valid pipeline keys (`packages/core/src/claude_sql/core/checkpointer.py:41`); `retry_queue` validates against it (`packages/core/src/claude_sql/core/retry_queue.py:97`), so adding a pipeline requires updating this tuple or the new pipeline is rejected.
- `filter_unchanged` is the skip decision (`packages/core/src/claude_sql/core/checkpointer.py:285`) and `mark_completed` is its write-back counterpart (`:329`); they form an ordered pair — a worker that calls `filter_unchanged` but fails to call `mark_completed` reprocesses the same sessions on every run.

## provenance.binding — TranscriptBinding, git trailer/note wire format, resolvers

Defined at: `packages/provenance/src/claude_sql/provenance/binding.py:143` (`class TranscriptBinding`); wire constants `DIGEST_PREFIX` (`:61`), `NOTES_REF` (`:65`), `TRAILER_DIGEST` (`:69`), `TRAILER_URI` (`:72`), `TRAILER_RUNTIME` (`:81`); `resolve_all_sources` (`:653`), `resolve_commit_to_transcript` (`:674`); writers `write_trailer` (`:404`), `write_note` (`:445`); readers `read_trailer` (`:515`), `read_note` (`:599`).

Implements the transcript-to-commit provenance contract (RFC 0001). Low fan-in, but the wire format is durable on-disk state in git.

| Downstream | Type | Touch on change | Citation |
|---|---|---|---|
| `app/cli.py` (`bind` / `resolve` commands) | direct import | yes | `packages/app/src/claude_sql/app/cli.py` |
| `provenance/review_sheet_worker.py` (`resolve_commit_to_transcript`) | direct import | yes | `packages/provenance/src/claude_sql/provenance/review_sheet_worker.py` |
| `tests/test_binding.py`, `tests/test_binding_extras.py` | test | yes | `tests/test_binding.py` |
| `tests/test_cli.py`, `tests/test_pr3_perf.py` | test | likely | `tests/test_cli.py` |

Blast-radius notes:
- The `TRAILER_*` constants (`packages/provenance/src/claude_sql/provenance/binding.py:69`, `:72`, `:81`) are the literal commit-message trailer keys written to git history; renaming one breaks `read_trailer` against any commit written by an older version, since the parser keys on the exact string (`:561`).
- `NOTES_REF = "transcripts"` (`packages/provenance/src/claude_sql/provenance/binding.py:65`) is the git-notes ref the binding is stored under; changing it orphans every previously written note, because `read_note` looks only under the current ref.
- `write_trailer`/`read_trailer` and `write_note`/`read_note` are matched writer/reader pairs (`packages/provenance/src/claude_sql/provenance/binding.py:404`/`:515`, `:445`/`:599`); a change to one side's serialization must land on the other in the same commit or round-trip resolution breaks.

## Other notable surfaces

- `core.output` — table/JSON rendering helpers (`packages/core/src/claude_sql/core/output.py`); 12 inbound files but a mechanical formatting layer. Touch on change: `likely` for callers that pin column order.
- `core.session_text` — transcript-to-text flattening (`packages/core/src/claude_sql/core/session_text.py`); 7 inbound files, feeds the LLM workers. Touch on change: `likely`, since prompt content shifts if the flattening changes.
- `core.lance_store` — vector store wrapper (`packages/core/src/claude_sql/core/lance_store.py`); does not appear in the module-path import grep (reached internally and through `embed_worker`), exercised by `tests/test_lance_store.py` and `tests/test_hnsw_persistence.py`.
- `core.retry_queue` — single inbound (`cli.py`); a thin layer over `checkpointer`'s SQLite file (`packages/core/src/claude_sql/core/retry_queue.py`).
- `core.logging_setup` (4 inbound) and `core.home` (3 inbound) — cross-cutting setup helpers with shallow blast radius.

## See also

- [claude-sql · Module map](../architecture/module-map.md) — 8 shared source files
- [claude-sql · Contract map](contract-map.md) — 7 shared source files
- [claude-sql · Debugging guide](debugging-guide.md) — 7 shared source files
- [claude-sql · Business logic](business-logic.md) — 6 shared source files
- [claude-sql · Public API](../reference/public-api.md) — 6 shared source files
