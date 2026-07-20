# 04 — CLI + Interfaces Surface

> Perspective: the `app/` package — `cli.py` (3175 LOC, cyclopts), `install_source.py`, `core/home.py`, `core/output.py`, and the `.claude/skills/claude-sql/` skill. Written for the v2 hexagonal refactor (thin `interfaces/cli` over `application/use_cases`, drop the `evals/` plane, add pluggable embedding providers, keep retrieval + clustering).

All citations are `path:line` against the current tree. Command signatures are quoted verbatim.

---

## 0. TL;DR command taxonomy

Total: **31 invokable commands** (27 top-level `@app.command` + 4 sub-app commands) plus one `@app.default` hint. There is **no application/use-case layer** — `cli.py` *is* the application layer.

| Group | Commands | v2 |
|---|---|---|
| Find / read (mission-critical) | `query`, `explain`, `shell`, `schema`, `list-cache`, `peek` | KEEP ×6 |
| Semantic search | `search` | CHANGE (provider) |
| Embed / ingest | `embed`, `ingest` | `embed` CHANGE, `ingest` KEEP |
| LLM analytics | `classify`, `trajectory`, `conflicts`, `friction` | KEEP ×4 |
| Structure | `cluster`, `terms`, `community` | KEEP ×3 |
| Composite pipeline | `analyze` | CHANGE (provider + re-orchestrate) |
| Provenance | `bind`, `resolve`, `review-sheet` | KEEP ×3 |
| Eval plane | `judges`, `freeze`, `replay`, `blind-handover`, `judge`, `ungrounded-claim`, `kappa` | **DROP ×7** |
| Admin / cache | `cache compact`, `cache migrate`, `skills sync`, `skills ls` | KEEP ×4 |

**Counts: KEEP 21 · CHANGE 3 · DROP 7.**

---

## 1. Full command inventory

The app tree is built at `cli.py:168-172`:

```python
app = App(
    name="claude-sql",
    version=format_version,
    help=_APP_HELP,
)
```

Shared flags live on a `@Parameter(name="*")`-flattened dataclass `Common` (`cli.py:175-192`) so every subcommand accepts `--verbose`/`--quiet`, `--glob`, `--subagent-glob`, `--format` *after the subcommand name*. Two sub-apps are registered: `cache_app` (`cli.py:1284-1293`) and `skills_app` (`cli.py:1502-1512`).

Layer wiring note: every command body drives `core/` (DuckDB connection + view registration + output) and, for the expensive paths, a worker module in `analytics/` / `evals/` / `provenance/`. Worker imports are **deferred into the command body** to keep the fast read path lean (`cli.py:45-102`).

### 1a. Read / introspection — the free path (KEEP)

| Command | Signature (`cli.py`) | Purpose | Drives |
|---|---|---|---|
| `schema` | `def schema(*, common=None)` @ `920-921` | Print all 18 views (+ columns) and 14 macro signatures from static dicts `VIEW_SCHEMA` / `MACRO_SIGNATURES`; no DuckDB connection. Also emits a `cached` map of which analytics parquets are populated (`_compute_cached_map`, `871-917`). | `core.sql_views` (static dicts), `core.output` |
| `list-cache` | `def list_cache(*, common=None)` @ `983-984` | Per-cache `{name, path, exists, bytes, mtime, rows}` for every parquet + LanceDB store + SQLite checkpointer (`_describe_*_entry`, `519-634`). | `core.parquet_shards`, `core.lance_store`, `core.checkpointer`, `core.home` |
| `explain` | `def explain(sql, /, *, analyze=False, profile_json=False, common=None)` @ `810-818` | `EXPLAIN` (static by default) or `EXPLAIN ANALYZE` (`--analyze`); TTY highlights pushdown operators in `_EXPLAIN_MARKERS` (`508-516`). | `core.sql_views`, DuckDB |
| `peek` | `def peek(session_id, /, *, common=None)` @ `1074-1075` | One-shot session summary — see §2. | Hand-written SQL over `messages` |
| `shell` | `def shell(*, common=None)` @ `642-643` | `register_all` into a temp on-disk DuckDB, then `subprocess.run(["duckdb", db_path])`; exit 127 if the binary is missing (`642-696`). | `register_all`, system `duckdb` |

### 1b. SQL query (KEEP)

- `query` — `def query(sql, /, *, profile_json=False, common=None)` @ `731-738`. See §2.

### 1c. Semantic search + embed / ingest

| Command | Signature | v2 | Drives |
|---|---|---|---|
| `search` | `def search(query_text, /, *, k=10, common=None)` @ `1682-1689` | **CHANGE** | `analytics.embed_worker.embed_query` + hand-written HNSW SQL — see §2 |
| `embed` | `def embed(*, since_days=None, limit=None, dry_run=False, common=None)` @ `1605-1612` | **CHANGE** | `analytics.embed_worker.run_backfill` (async) |
| `ingest` | `def ingest(*, since_days=None, limit=None, dry_run=True, common=None)` @ `1999-2006` | KEEP | `analytics.ingest.{count_pending, stamp_messages, resolve_canonicals}` — CPU-only tiktoken + SimHash + canonical-uuid resolve |

`embed` is the one Bedrock command that defaults to `--dry-run=False` (`1610`) — all LLM analytics default to `True`. Its body opens a bare `:memory:` connection, optionally wires S3 (`settings_need_s3` / `configure_s3`), `register_raw` + `register_views`, then `asyncio.run(run_backfill(...))` (`1650-1679`).

### 1d. LLM analytics — Sonnet 4.6, default `--dry-run=True` (KEEP)

All four share an identical body shape: `_open_connection_full` → call one worker → `_emit_worker_result`. Signatures:

- `classify` — `def classify(*, since_days=None, limit=None, dry_run=True, no_thinking=False, common=None)` @ `1782-1790` → `classify_worker.classify_sessions` (`1831-1848`).
- `trajectory` — same kwargs @ `1851-1859` → `trajectory_worker.trajectory_messages` (`1881-1898`).
- `conflicts` — same kwargs @ `1901-1909` → `conflicts_worker.detect_conflicts` (`1927-1944`).
- `friction` — same kwargs @ `1947-1955` → `friction_worker.detect_user_friction` (`1979-1996`).

### 1e. Structure — CPU-only (KEEP)

- `cluster` — `def cluster(*, force=False, common=None)` @ `2073-2074` → `cluster_worker.run_clustering(settings, force=force)`. Note: this is the **only** analytics command that takes `settings` directly and *not* a shared `con` (`analyze` calls it out at `2381-2382`).
- `terms` — `def terms(*, force=False, common=None)` @ `2106-2107` → `terms_worker.run_terms(con, settings, force)`.
- `community` — `def community(*, force=False, gamma=None, resolution="medium", neighbors_of_session=None, top_k=15, dry_run=False, common=None)` @ `2140-2150` → `community_worker.{neighbors_of, run_communities}`. Richest analytics surface: mutual-exclusion guard on `--neighbors-of` (`2205-2216`), a `--dry-run` SQL count path (`2225-2253`), and the `--resolution {coarse,medium,fine}` / `--gamma` plateau selectors. `ResolutionLevel` is redefined locally at `cli.py:165` (pinned byte-identical to the worker enum) to avoid dragging `igraph`/`leidenalg` onto the module-load path.

### 1f. Composite pipeline (CHANGE)

- `analyze` — `def analyze(*, since_days=30, limit=None, dry_run=True, no_thinking=False, skip_ingest=False, skip_embed=False, skip_classify=False, skip_trajectory=False, skip_conflicts=False, skip_friction=False, skip_cluster=False, skip_community=False, skip_skills_sync=False, force_cluster=False, force_community=False, common=None)` @ `2280-2299`.

This is the **fattest** command. Its 230-line body (`2345-2517`) is a full pipeline orchestrator: it imports **nine** workers, opens one shared connection, then runs stages 0–9 (skills → ingest → embed → cluster → terms → community → classify → trajectory → conflicts → friction), interleaving `_rebind_vss` and `_refresh_analytics_views` calls between stages to fix the RFC §9.6 "stale connection" bug (`2422-2423`, `2451-2452`). All orchestration logic — stage ordering, the connection-sharing optimization, the VSS re-bind dance — lives in the CLI, not in a use-case.

### 1g. Provenance (KEEP)

| Command | Signature | Drives |
|---|---|---|
| `bind` | `def bind_cmd(*, repo=None, commit_msg=None, dry_run=False, common=None)` @ `2793-2800` | `provenance.binding` — attaches transcript↔PR trailers + git-notes; `prepare-commit-msg` hook entry point (RFC 0001) |
| `resolve` | `def resolve_cmd(commit_sha, /, *, repo=None, all_sources=False, common=None)` @ `2926-2934` | `provenance.binding.{resolve_commit_to_transcript, resolve_all_sources}` — trailer→note precedence, loud mismatch (exit 70) |
| `review-sheet` | `def review_sheet_cmd(commit_sha, /, *, repo=None, no_thinking=False, dry_run=True, common=None)` @ `3032-3041` | `provenance.review_sheet_worker.generate_review_sheet` (Sonnet 4.6) + `render_markdown`; owns a private `RenderFormat` markdown/json enum (`3003-3029`) |

`review-sheet` is the only command that emits human prose, so markdown rendering is deliberately kept out of the global `OutputFormat` (`output.py:33-37`, `cli.py:3003-3013`).

### 1h. Eval plane — **DROP all 7 in v2**

These are the `evals/` CLI surface that goes away when the eval plane is dropped. They form a self-contained study→judge→kappa pipeline that does **not** touch the transcript corpus (they read/write user-supplied parquets):

| Command | Signature | Drives |
|---|---|---|
| `judges` | `def judges_cmd(*, common=None)` @ `2519-2520` | `evals.judges.catalog()` — lists the cross-provider judge panel |
| `freeze` | `def freeze_cmd(rubric, /, *, panel, embed_model=..., seed=42, min_turns=10, max_turns=40, common=None)` @ `2539-2550` | `evals.freeze.freeze` — writes an immutable study manifest |
| `replay` | `def replay_cmd(manifest_sha, /, *, common=None)` @ `2581-2582` | `evals.freeze.replay` |
| `blind-handover` | `def blind_handover_cmd(input_path, /, output_path, *, common=None)` @ `2590-2597` | `evals.blind_handover.{strip_text, original_hash}` |
| `judge` | `def judge_cmd(manifest_sha, /, *, sessions_parquet, output_parquet, dry_run=True, concurrency=4, region="us-east-1", common=None)` @ `2621-2632` | `evals.judge_worker.run` |
| `ungrounded-claim` | `def ungrounded_cmd(manifest_sha, /, *, turns_parquet, output_parquet, common=None)` @ `2677-2685` | `evals.ungrounded_worker.{detect, to_parquet, summarize}` |
| `kappa` | `def kappa_cmd(scores_parquet, /, *, bootstrap=1000, floor=0.6, delta_gate=None, common=None)` @ `2716-2725` | `evals.kappa_worker.*`; exits `66` when a Fleiss-kappa floor or delta-gate trips (`2789-2790`) |

Because the eval commands cite `evals.*` at both module top (`freeze`, `judges`, `blind_handover` — `cli.py:91-95`) and inside bodies (`judge_worker`, `ungrounded_worker`, `kappa_worker`), dropping the plane also removes those imports. The three cheap module-top eval imports are the only eval coupling on the fast path.

### 1i. Admin / cache / skills (KEEP)

- `cache compact` — `def cache_compact(*, name=None, dry_run=True, common=None)` @ `1314-1320`; consolidates `part-*.parquet` shards (`1354-1409`).
- `cache migrate` — `def cache_migrate(*, dry_run=True, common=None)` @ `1412-1417`; moves legacy single-file caches into the sharded dir layout (`1438-1495`).
- `skills sync` — `def skills_sync(*, dry_run=False, common=None)` @ `1515-1520` → `analytics.skills_catalog.sync`.
- `skills ls` — `def skills_ls(*, kind=None, plugin=None, common=None)` @ `1557-1563`; reads `skills_catalog.parquet` directly with polars.

### 1j. Default

- `_default` — `def _default(*, common=None)` @ `3152-3153`; prints a subcommand hint when invoked bare. Its printed list (`3157-3161`) still enumerates the eval commands, so it must be rewritten in v2.

---

## 2. The find & read commands (mission-critical)

These are the surface the v2 direction calls "best-in-class at finding & reading Claude Code transcripts." All are read-only and cost nothing.

### `query` (`cli.py:731-807`)

```python
@app.command
def query(
    sql: str,
    /,
    *,
    profile_json: bool = False,
    common: Common | None = None,
) -> None:
```

Behavior: chooses the connection by **substring-scanning the SQL** — `_open_connection_full` if `_sql_uses_catalog(sql)` else `_open_connection_introspect` (`793-797`). This lets `SELECT 1` skip the ~25 s `register_all` chain entirely (`368-379`). Optionally sets DuckDB profiling PRAGMAs (`_capture_profile`, `713-728`), then `run_or_die(lambda: con.execute(sql).pl())` → `emit_dataframe`. Errors are classified (parse=64 / catalog=65 / runtime=70) by `run_or_die` (`output.py:228-254`). Output: polars table on TTY, JSON array of row dicts on pipe.

### `explain` (`cli.py:810-868`)

Same connection selection; prepends `EXPLAIN ` or `EXPLAIN ANALYZE ` and joins the plan rows. TTY colorizes lines containing pushdown markers; pipe emits `{"plan": "<text>"}`.

### `peek` (`cli.py:1074-1235`)

```python
@app.command
def peek(session_id: str, /, *, common: Common | None = None) -> None:
```

The one read command with **hand-written multi-statement SQL** rather than a view. It materializes the session's messages into a `_peek_msgs` TEMP table once (`1118-1127`) — an explicit ~5× optimization documented at `1104-1117` because DuckDB won't push the `session_id` predicate below the `content_blocks` UNNEST — then derives roles, top-10 tools, and first/last samples from a second `_peek_blocks` TEMP table. Emits `{session_id, source_file, total_lines, first_ts, last_ts, roles{}, top_tools[], samples{first_user,last_user,first_assistant_text}}`. Missing session → exit 65 (`not_found`, `1138-1148`). TTY rendering is a bespoke text layout (`_peek_render_table`, `1238-1264`).

### `search` (`cli.py:1682-1779`) — CHANGE

```python
@app.command
def search(
    query_text: str,
    /,
    *,
    k: int = 10,
    common: Common | None = None,
) -> None:
```

Pipeline: guard on `SELECT count(*) FROM message_embeddings == 0` → exit 2 `no_embeddings` (`1745-1749`); `embed_query(query_text, settings=settings)` embeds the query with Cohere `search_query` mode; then hand-written HNSW SQL — `ORDER BY array_cosine_distance(...) ASC` (which triggers the index) while selecting `array_cosine_similarity(...)` as `sim` (`1759-1774`). Output columns: `uuid, session_id, role, sim, snippet` (200-char). The docstring (`1719-1735`) explicitly steers agents to `query` + `ILIKE` when pinpointing one known session — this bias against over-recall is load-bearing product behavior that must survive the refactor.

**Session read**: there is no dedicated "read session N" command — the skill composes it from `peek <id>` (summary) + `query "SELECT ... FROM messages_text WHERE session_id = ..."` (full text). v2 could add a first-class `session read` command, but today it's a query recipe.

---

## 3. How the CLI wires to the engine today — it is a **fat CLI**

**There is no thin layer.** No `application/`, no `use_cases/`, no service objects. The command body *is* the use case. Each body: (1) `_configure(common)` installs logging; (2) `_resolve_settings(common)` builds `Settings` + validates globs; (3) opens a DuckDB connection and registers views; (4) calls one or more worker functions and/or hand-writes SQL; (5) renders via `core.output`. Connection lifecycle, PRAGMA tuning, legacy-cache migration, VSS re-binding, and pipeline orchestration all live in `cli.py`.

Entry point (`cli.py:3169-3172`):

```python
def main() -> None:
    """Entry point wired into ``[project.scripts]`` in ``pyproject.toml``."""
    app()
```

Wired via `pyproject.toml:52-53`: `claude-sql = "claude_sql.app.cli:main"`. `app/__init__.py` is a one-line docstring — the package exposes no importable API today; consumers must reach into `cli` or the workers directly.

Representative body — the LLM analytics shape (`classify`, `cli.py:1831-1848`):

```python
    from claude_sql.analytics.classify_worker import classify_sessions

    _configure(common)
    settings = _resolve_settings(common)
    con = _open_connection_full(settings)
    try:
        result = classify_sessions(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            dry_run=dry_run,
            no_thinking=no_thinking,
        )
        logger.info("classify: {} sessions processed (dry_run={})", result, dry_run)
        _emit_worker_result(result, common, pipeline="classify")
    finally:
        con.close()
```

The infrastructure concerns the CLI owns directly, which a v2 `application` layer would absorb:

- **Connection factory + PRAGMA tuning** — `_open_connection_full` / `_open_connection_introspect` / `_apply_duckdb_pragmas` (`cli.py:324-379`). These set threads/memory/temp-dir and gate VSS registration by SQL substring (`_sql_uses_vss`, `460-470`).
- **View re-binding mid-pipeline** — `_refresh_analytics_views` (`382-399`) and `_rebind_vss` (`402-444`). Pure infrastructure glue for the shared-connection optimization.
- **Legacy cache auto-migration** — `_maybe_migrate_legacy_caches` (called at connection open, `360`/`376`).
- **Pipeline orchestration** — the whole `analyze` stage machine (`2345-2517`).
- **Dry-run/plan emission** — `_emit_worker_result` (`492-504`), and inline plan dicts in `community` (`2238-2252`) and `ingest` (`2050-2060`).

Workers already take `(con, settings, ...)` and return `int | dict` (rows or a plan) — a clean-ish seam. But the **connection is created by the CLI and threaded in**, so a hexagonal v2 must invert this: a `DuckDBSession`/repository port owns the connection, and use-cases receive it. The orchestration in `analyze` is the single biggest extraction target.

---

## 4. Output rendering

All rendering lives in `core/output.py` (254 LOC) and is imported by `cli.py:62-73`. This is already a reasonably clean adapter, but it mixes two concerns that v2 should split.

- **`OutputFormat`** StrEnum (`output.py:26-45`): `auto | table | json | ndjson | csv`. `AUTO` resolves to `TABLE` on a TTY and `JSON` off it (`resolve_format`, `60-65`) — the agent-friendly default.
- **`emit_dataframe`** (`68-108`): polars → table (capped 100 rows / 120-char cells) or `write_json` / `write_ndjson` / `write_csv`.
- **`emit_json`** (`111-122`): non-tabular payloads (`schema`, `list-cache`, errors, plans).
- **Error model**: `ClassifiedError` (`125-141`), `InputValidationError` (`144-153`), `classify_duckdb_error` (`180-207`), `emit_error` (`210-225`), `run_or_die` (`228-254`). Stable exit codes in `EXIT_CODES` (`49-57`): `ok 0 · no_embeddings 2 · invalid_input/parse 64 · catalog 65 · runtime 70 · duckdb_missing 127`. `kappa` adds an ad-hoc `66` (`cli.py:2790`); `peek`/`resolve`/`review-sheet` reuse `65` for a `not_found` kind.
- **`validate_glob`** (`156-177`): rejects multi-`**` globs before DuckDB sees them.

**v2 placement.** `OutputFormat` resolution, table/json/ndjson/csv serialization, and error→exit-code mapping are **presentation** concerns → they belong in `interfaces/cli`. The **`ClassifiedError` taxonomy and `EXIT_CODES` are part of the public wire contract** (agents match on them per the skill, SKILL.md:113-114) → keep the error *types* in a shared/domain module and let the CLI adapter own the exit-code mapping and stderr formatting. Several commands bypass `emit_dataframe` and hand-serialize NDJSON/CSV inline (`list-cache` `1053-1061`, `cache_compact` `1401-1409`, `skills_ls` `1594-1601`) — that duplication should collapse into `emit_dataframe` during the move. Note `output.py:33-37` deliberately keeps markdown out of `OutputFormat`; the `review-sheet` `RenderFormat` (`cli.py:3003-3029`) is a local presentation enum and should live in the CLI adapter next to that command.

---

## 5. `install_source.py` + `core/home.py`

**`app/install_source.py`** (78 LOC) — install-source discovery for `--version`. `claude-sql` is not on PyPI; users `uv tool install --from . claude-sql`, and uv records the source in `$UV_TOOL_DIR/<tool>/uv-receipt.toml`. `read_install_source` (`34-63`) parses that receipt (all reads wrapped in try/except — uv's schema is not a public contract, `10-13`); `format_version` (`66-78`) returns `"claude-sql X.Y.Z"` plus a source line, and is passed as `App(version=...)` at `cli.py:170`. `__version__` comes from `importlib.metadata.version("claude-sql")` (`22`). v2: this stays an interface concern; keep it beside the CLI adapter.

**`core/home.py`** (93 LOC) — resolves where derived caches live. `claude_sql_home()` (`51-72`) resolution order: `$CLAUDE_SQL_HOME` → macOS `~/Library/Application Support/claude-sql` → `$XDG_DATA_HOME/claude-sql` → `~/.claude-sql`. It reads `os.environ` on every call (for test monkeypatching) and `mkdir(parents=True, exist_ok=True)`. `recognized_legacy_caches()` (`75-93`) returns the migration manifest of caches that lived under `~/.claude/` before RFC 0002 (`_LEGACY_CACHE_NAMES`, `33-48`) — consumed by `list-cache` (`cli.py:1035`) and the auto-migrator. Note this is `home` (cache root), distinct from `~/.claude/` **discovery** — the transcript glob is a `Settings` field (`default_glob`, `config.py`), and `~/.claude/skills` + `~/.claude/plugins/cache` discovery lives in `analytics.skills_catalog`. v2: `home.py` is a `core`/infrastructure adapter (filesystem port); it stays below the use-case layer.

---

## 6. The skill (`.claude/skills/claude-sql/`)

Files: `SKILL.md` (195 lines) + `references/commands.md` (226), `recipes.md` (234), `jobs.md` (183). The skill is what makes an agent reach for the tool when the user asks about their own Claude history (the `description` front-matter, SKILL.md:3, is a long trigger list).

What it documents today:
- **Only the read + analytics surface.** The "CLI at a glance" table (SKILL.md:79-97) and `commands.md` headings cover exactly: `schema`, `query`, `explain`, `shell`, `list-cache`, `search`, `embed`, `classify`, `trajectory`, `conflicts`, `cluster`, `community`, `analyze`, plus an Environment section. **The eval commands and provenance commands are never mentioned** — a `grep` for `judge|freeze|kappa|bind|resolve|review-sheet` finds only incidental hits (SQL `resolution` column, "unresolved conflicts"). So dropping the eval plane needs **no SKILL edits** for removed commands.
- Agent-friendly invocation rules (SKILL.md:99-117): always `--format json`, run `schema` then `list-cache` first, match on exit codes 64/65/70.
- The data shape (views/macros, SKILL.md:119-160) and the read-only-first / always-dry-run / trust-the-caches golden rules (179-194).

What will need rewriting in v2:
1. **Provider switch.** SKILL.md hardcodes Cohere Embed v4 + Sonnet 4.6 throughout — Prerequisites (32-35, IAM on `global.cohere.embed-v4:0`), the `embed` row (89), "Backfill Cohere v4 embeddings", and the golden rule "Don't touch cost estimation constants." Adding `--embedding-provider` (Cohere/Bedrock, Ollama, ONNX bge) means the Prerequisites/IAM section becomes provider-conditional (Ollama/ONNX need no AWS creds), and `embed`/`search`/`analyze` docs must show the flag. `references/commands.md` `## embed` / `## search` / `## analyze` sections need the flag + examples.
2. **Rebrand risk.** Every command example is literally `claude-sql <cmd>`. If the binary is renamed, all four files change — but a single find/replace since the invocation is uniform.
3. **No eval/provenance additions needed** unless v2 chooses to surface provenance in the skill (currently absent).
4. **`analyze` chain description** (SKILL.md:75-77, 95) lists "embed → cluster → classify → trajectory → conflicts" — already slightly stale vs. the real 10-stage order in code (`cli.py:2302-2314` adds skills/ingest/terms/community/friction); worth reconciling in the rewrite.

---

## 7. v2 interfaces target

### Layering

```
interfaces/cli/           # thin cyclopts adapter — arg parsing, --format, exit codes, --version
  └─ calls ──────────────►
application/use_cases/     # orchestration; owns the DuckDB session lifecycle + pipeline sequencing
  └─ calls ──────────────►
domain / ports             # EmbeddingProvider port, repositories
adapters/ (infra)          # DuckDB, LanceDB, Bedrock/Cohere, Ollama, ONNX-bge, filesystem (home.py), git (binding)
```

The CLI adapter keeps: `Common` flags, `OutputFormat` + all `emit_*` rendering, `ClassifiedError`→exit-code mapping, `--version`/`install_source`, the `RenderFormat` markdown enum. It **loses**: connection creation, `register_all`/`_rebind_vss`/`_refresh_analytics_views`, `_maybe_migrate_legacy_caches`, and the `analyze` stage machine — those move into `application/use_cases` (e.g. `RunAnalyzePipeline`, `EmbedMessages`, `SearchTranscripts`, `ClassifySessions`, `DetectCommunities`).

### Surviving command tree (post-refactor)

```
claude-sql
├── query <sql>                 KEEP   → QueryCatalog use-case
├── explain <sql> [--analyze]   KEEP   → QueryCatalog (explain mode)
├── shell                       KEEP   → OpenRepl use-case
├── schema                      KEEP   → DescribeCatalog
├── list-cache                  KEEP   → DescribeCaches
├── peek <session_id>           KEEP   → PeekSession
├── search <text> [--k]         CHANGE → SearchTranscripts (+ --embedding-provider)
├── embed [--since-days] ...    CHANGE → EmbedMessages     (+ --embedding-provider)
├── ingest                      KEEP   → IngestStamps
├── classify | trajectory |     KEEP   → Classify* / DetectConflicts / DetectFriction
│   conflicts | friction
├── cluster | terms | community KEEP   → Cluster / Terms / DetectCommunities
├── analyze [--skip-*] ...      CHANGE → RunAnalyzePipeline (+ --embedding-provider; orchestration moves out of CLI)
├── cache {compact, migrate}    KEEP   → CompactCaches / MigrateCaches
├── skills {sync, ls}           KEEP   → SyncSkills / ListSkills
└── provenance / prov           KEEP   → bind | resolve | review-sheet  (candidate to nest under one group)
   (DROPPED: judges freeze replay blind-handover judge ungrounded-claim kappa)
```

### The new `--embedding-provider` surface

Today embedding config is Cohere-only: `Settings.model_id = "global.cohere.embed-v4:0"` (`config.py:187`), `output_dimension` (189), `embedding_type` (190), `region` (183), and `active_model_id` (a property returning `model_id`, `409-411`). `embed_query` (`embed_worker.py:338-366`) and `run_backfill` build a Bedrock client from these directly. v2 introduces an **`EmbeddingProvider` port** with implementations for Cohere/Bedrock (current), Ollama, and ONNX bge. The CLI adds a `--embedding-provider {bedrock-cohere,ollama,onnx-bge}` flag (plus provider-specific knobs like `--embed-model`, `--embed-dim`), resolved into a provider instance the `EmbedMessages` / `SearchTranscripts` use-cases consume. Constraint to preserve: the query-side embedding in `search` **must use the same provider/dim** that populated the store, and the LanceDB index dim (`hnsw_metric`, dim 1024) must match — so provider selection needs a compatibility check against the existing store (surface a clear error, not a silent dim mismatch). This flag belongs on `Common` **only if** it must apply to `analyze` too; otherwise scope it to `embed`/`search`/`analyze`.

### Importable public API to sit alongside the binary

The v2 goal is "importable library AND standalone binary." Today nothing is exported — `app/__init__.py` is empty and `cli.py` is a wall of decorated functions. The public API should be the **use-case layer**, e.g.:

```python
from claude_sql import ClaudeSql              # facade over a resolved Settings + connection factory
from claude_sql.application import SearchTranscripts, RunAnalyzePipeline
from claude_sql.providers import BedrockCohereEmbedder, OllamaEmbedder, OnnxBgeEmbedder

engine = ClaudeSql(embedder=OllamaEmbedder(...))
hits = engine.search("temporal workflows", k=10)     # returns typed rows, not stdout
```

The cyclopts `main()` then becomes a thin adapter that constructs the same objects and routes results through `emit_dataframe`/`emit_json`. The import-linter contract (`pyproject.toml:264-273`) already enforces `app < {analytics|evals|provenance} < core`; v2 restructures that into `interfaces < application < domain < adapters` and drops the `evals` node.

---

## Appendix — layer & coupling facts

- Console script: `pyproject.toml:52-53` → `claude_sql.app.cli:main`.
- Import-linter layers today: `pyproject.toml:264-282` (`app` on top, `analytics|evals|provenance` independent siblings, `core` at the bottom).
- Deferred worker imports (fast-path protection): `cli.py:45-102`; pinned by `test_cli_import_is_lean`.
- Cheap eval imports at module top that DROP removes: `cli.py:91-95` (`freeze`, `judges`, `blind_handover`).
- `_default` hint still lists eval commands: `cli.py:3157-3161` (must be rewritten).
