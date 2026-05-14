# claude-sql · Tech debt

This register answers: *where is the rot, and what would I pay to fix it?*

The list combines four sources, in the order the reviewer scanned them:
explicit comment markers (`TODO` / `FIXME` / `HACK` / `XXX` / `DEPRECATED`
in `src/` and `tests/`), deprecation decorators and `DeprecationWarning`
sites, manifest version pins to known-old or known-narrow ranges in
`pyproject.toml`, and pattern-level smells the reviewer chose to flag
after reading the cited sites. Categories are drawn from a closed
vocabulary (`marker`, `wrong abstraction`, `error handling`,
`dead code adjacent`, `deprecated pattern`, `version pin`,
`duplicated logic`, `missing tests`); cost is `S` / `M` / `L` where
`S = ≤ 1 hour`, `M = a few hours`, `L = a day or more`.

The marker count is small by design. The repo's `CLAUDE.md` and
`.erpaval/` solutions library treat unbacked TODOs as a code smell —
debt is named in `docs/BACKLOG.md` and ADRs instead of left in source
comments. Most of the debt below is *named* somewhere; the value of
this file is collecting it in one ranked view.

## Ranked register

| Rank | Debt item | Category | Cost to fix | Citation |
| --- | --- | --- | --- | --- |
| 1 | Two divergent `_build_bedrock_client` implementations: `embed_worker`'s lacks the cached `(region, pool_size)` pool, uses `read_timeout=60` (vs `600`), and `retries.mode='standard'` (vs `'adaptive'`). Embed traffic does not benefit from the pool guard the LLM workers get. | duplicated logic | M | `src/claude_sql/embed_worker.py:182`, `src/claude_sql/llm_shared.py:346` |
| 2 | `task_spawns` view documented "Removed in the next minor release" — v1.0 already shipped, so the deferral window is closed. | deprecated pattern | S | `src/claude_sql/sql_views.py:976` |
| 3 | `Settings.concurrency` field documented "Removed in the next release" — same closed window. The deprecation warning fires only when set explicitly. | deprecated pattern | S | `src/claude_sql/config.py:182`, `src/claude_sql/config.py:386` |
| 4 | `describe_all` introspection function carries a `DeprecationWarning` and is "Kept for one release as a fallback"; only the drift test still consumes it. | deprecated pattern | M | `src/claude_sql/sql_views.py:2151` |
| 5 | `cli.py` is a 2 917-LOC single module hosting 30 command decorators (`grep -cE '^@.*\.command' src/claude_sql/cli.py` → 30). Hard to navigate, hard to test in isolation, no domain seams between subcommands. | wrong abstraction | L | `src/claude_sql/cli.py:164` |
| 6 | `sql_views.py` is a 2 228-LOC module mixing raw views, derived views, macros, analytics registration, and introspection in one namespace. Two top-level `try / except Exception:` blocks (no `noqa`) wrap each registration. | wrong abstraction | L | `src/claude_sql/sql_views.py:612`, `src/claude_sql/sql_views.py:1110` |
| 7 | Three broad `except Exception:` blocks lack a `noqa: BLE001` justification; the rest of the codebase consistently pairs broad excepts with one-line rationale. These are inconsistent with the documented pattern in `CLAUDE.md`'s "CodeQL hygiene" section. | error handling | S | `src/claude_sql/sql_views.py:612`, `src/claude_sql/sql_views.py:1110`, `src/claude_sql/checkpointer.py:209` |
| 8 | Python floor pinned at `3.13` because `hdbscan` lacks cp314 wheels — the ADR explicitly tracks this as a one-line PR deferred until upstream ships. Cost is the wait; the bump itself is small. | version pin | S | `pyproject.toml:10`, `docs/adr/0015-stack-modernization.md:19` |
| 9 | `lancedb` pinned to a single minor (`>=0.30,<0.31`) — narrowest range in the manifest. Each lancedb minor requires a manual relax-and-retest cycle. | version pin | S | `pyproject.toml:38` |
| 10 | Legacy parquet-shard migration code in `lance_store.migrate_from_parquet_shards` and the call site in `register_vss` exists only to absorb pre-LanceDB installs. Idempotent + sentinel-guarded, but pure carry-cost once users have migrated. | dead code adjacent | M | `src/claude_sql/lance_store.py:165`, `src/claude_sql/sql_views.py:1741` |
| 11 | Legacy DuckDB-checkpointer migration in `_migrate_from_duckdb_if_present` runs on every connect against a sentinel; pure carry-cost once users have moved to the SQLite WAL checkpointer. | dead code adjacent | M | `src/claude_sql/checkpointer.py:91` |
| 12 | `tests/test_logging_setup.py` does not exist; `logging_setup.py` has no direct test file even though it's the single sanctioned escape hatch from the stdlib-`logging` ban (`loguru_before_sleep` for tenacity). | missing tests | S | `src/claude_sql/logging_setup.py:1` |
| 13 | Embedded "DEPRECATED" comment marker on `task_spawns`. *judgment-call:* listed as a marker in addition to its higher-rank deprecated-pattern row because it is the only `\bDEPRECATED\b` source-comment in the corpus. | marker | S | `src/claude_sql/sql_views.py:976` |

## Explicit markers

The full result of `grep -rnE '\b(TODO\|FIXME\|HACK\|XXX)\b' src/ tests/`
is empty. The `\bDEPRECATED\b` marker fires twice. The wider
case-insensitive scan adds eight comments referring to "legacy" or
"deprecated" subsystems; those are not debt markers, they are
documentation of one-time-migration paths and back-compat aliases — they
appear in the smells section (Legacy-migration carry cost) and the
ranked register (Deprecated patterns).

- `# DEPRECATED: ``task_spawns`` predates the Task→Agent rename (v2.1.63)` — `src/claude_sql/sql_views.py:976`
- `#: DEPRECATED: use ``embed_concurrency`` / ``llm_concurrency``. Kept for` — `src/claude_sql/config.py:182`

## Pattern-level smells

### Divergent `_build_bedrock_client` copies

`embed_worker` and `llm_shared` each define a private builder for the
boto3 `bedrock-runtime` client. The two have drifted: `llm_shared`'s
caches by `(region, pool_size)` and uses `max_pool_connections =
max(32, max(embed_concurrency, llm_concurrency) * 2)`,
`read_timeout=600`, and `retries.mode='adaptive'`; `embed_worker`'s does
not cache, sets `read_timeout=60`, and uses `retries.mode='standard'`.
Both are imported by name across six call sites — five workers reach for
the `llm_shared` version, embed alone keeps its own. The fix is to make
embed re-export `llm_shared._build_bedrock_client` (the test suite
already mocks both by patching the module-local name, so the contract
holds). The smell is duplication that has actively forked behaviour, not
two identical functions.

Shows up in:
- `src/claude_sql/embed_worker.py:182` — the older builder, no cache, `read_timeout=60`.
- `src/claude_sql/llm_shared.py:346` — the cached pool-aware builder.
- `src/claude_sql/embed_worker.py:317` — call site that gets the un-pooled client.
- `src/claude_sql/embed_worker.py:379` — second call site, same module.
- `src/claude_sql/classify_worker.py:89` — sibling worker that imports from `llm_shared`.

Cost: M.

### Single-module CLI and views surfaces

Two files hold ~30% of the package source. `cli.py` (2 917 LOC) is the
single entry point for every subcommand — `analyze`, `embed`, `classify`,
`trajectory`, `community`, `cache`, `skills`, `query`, etc. — with 30
`@app.command` decorators counted directly. `sql_views.py` (2 228 LOC)
mixes `register_raw`, `register_views`, `register_macros`,
`register_analytics`, `register_vss`, plus `describe_all`, `list_macros`,
and a static `VIEW_SCHEMA` block. Both modules predate v1.0; both
register cleanly under ruff/ty; neither is broken. The smell is *load*:
adding a new view, a new macro, or a new subcommand has to thread
through these monoliths. A natural seam is: `cli/` package with one
module per subcommand group (each `@app.command` block already lives in
its own ~80-LOC zone), and `views/` package split into `raw.py`,
`derived.py`, `macros.py`, `analytics.py`, `vss.py`.

Shows up in:
- `src/claude_sql/cli.py:164` — the `App(...)` declaration.
- `src/claude_sql/cli.py:638` — first `@app.command` block (~30 follow).
- `src/claude_sql/sql_views.py:622` — `register_views` start.
- `src/claude_sql/sql_views.py:1115` — macros block start.
- `src/claude_sql/sql_views.py:1696` — `register_vss` lives 500 LOC into the same file.

Cost: L.

### Three deprecated paths past their stated removal window

The codebase shipped v1.0 (CHANGELOG.md `1.0.0`), but three deprecation
notices written under v0.x still live in source with the comment "Kept
for one release" or "Removed in the next release / next minor release".
Each is a small cleanup: drop the alias view, drop the field + validator,
move the drift test off `describe_all`. The smell is that the comment
text now lies about its own lifecycle.

Shows up in:
- `src/claude_sql/sql_views.py:976` — `task_spawns` view, "Removed in the next minor release".
- `src/claude_sql/config.py:182` — `Settings.concurrency`, "Removed once downstream callers migrate".
- `src/claude_sql/config.py:386` — the `_resolve_concurrency_alias` validator that materializes the back-compat shape.
- `src/claude_sql/sql_views.py:2151` — `describe_all`, "Kept for one release as a fallback".

Cost: M.

### Legacy-migration code paths permanently in the boot path

Three legitimate one-time migrations sit in modules that run on every
connect: parquet-shards → LanceDB (`lance_store.migrate_from_parquet_shards`
called from `register_vss`), DuckDB checkpointer → SQLite (`checkpointer
._migrate_from_duckdb_if_present`), and `~/.claude/` → `~/.claude-sql/`
cache home (`home.recognized_legacy_caches` plus the consumer in
`config.py`'s default-factories). Each is sentinel-guarded and idempotent
— no functional bug — but they're pure carry cost: 200+ LOC and a few
hundred microseconds per startup spent confirming the migration already
ran. *judgment-call:* this is debt because the team has already paid the
upgrade-comms cost and the BACKLOG entry that proposed the home
migration is materially complete (`home.py` shipped). One release of
overlap is reasonable; two starts to look like sediment.

Shows up in:
- `src/claude_sql/lance_store.py:165` — parquet → Lance migration body.
- `src/claude_sql/sql_views.py:1741` — call site inside `register_vss`.
- `src/claude_sql/checkpointer.py:91` — DuckDB → SQLite checkpointer migration.
- `src/claude_sql/home.py:75` — legacy-cache discovery used by `config.py` factories.

Cost: M.

### Broad excepts without a one-line `noqa` rationale

`CLAUDE.md`'s "CodeQL hygiene" section makes the rule explicit: every
`except Exception:` either gets a narrowed exception class or a
`noqa: BLE001 — <reason>` comment. The codebase mostly honors this — six
sites carry the rationale comment (`trajectory_worker.py:647 / :862`,
`judge_worker.py:339`, `sql_views.py:1751`, `checkpointer.py:159`). Three
sites do not. They function correctly today (each is the documented
"register-or-fail-loud" wrapper around a multi-step setup) but they
violate the project's own pattern.

Shows up in:
- `src/claude_sql/sql_views.py:612` — `register_raw` failure wrapper.
- `src/claude_sql/sql_views.py:1110` — `register_views` failure wrapper.
- `src/claude_sql/checkpointer.py:209` — bulk-INSERT rollback wrapper.

Cost: S.

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 6 shared citations
- [claude-sql · System overview](../architecture/system-overview.md) — 2 shared citations
- [claude-sql · Processes](../behavior/processes.md) — 2 shared citations
