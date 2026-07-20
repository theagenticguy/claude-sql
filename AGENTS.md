# claude-sql — agent & contributor instructions

This file is read by Claude Code (and any agent running in this repo). It tells
you how to work on and with `claude-sql` without re-deriving the design each
time. `CLAUDE.md` is a symlink to this file.

## What this project is

A standalone Python CLI (**v2.0.0**, shipped) that makes the user's own Claude
Code transcripts — everything under `~/.claude/projects/**/*.jsonl` plus subagent
sidecar files — queryable in place. Four analytics layers stack on top of those
JSONLs:

1. **SQL** — DuckDB reads the JSONLs with zero copy; views + macros normalize
   messages, tool calls, todos, subagents, costs.
2. **Semantic search** — an embedding provider (Cohere Embed v4 on Bedrock by
   default) embeds every message; a LanceDB `IVF_HNSW_SQ` cosine index, read
   back through DuckDB's `lance` extension, serves top-k in milliseconds. The
   embedder is pluggable behind an `EmbeddingProvider` port (Cohere-on-Bedrock,
   Ollama, local ONNX BGE — the latter two under optional extras).
3. **LLM analytics** — Sonnet 4.6 (global CRIS, `output_config.format`
   structured output) classifies sessions (autonomy tier, work category,
   success, goal), scores per-message sentiment, detects stance conflicts, and
   classifies short user messages for friction signals.
4. **Structure** — UMAP + HDBSCAN cluster message embeddings, c-TF-IDF names the
   clusters, Leiden + CPM groups sessions into communities over a mutual-kNN
   cosine graph of session centroids. The `community` subcommand is agent-first:
   it emits *signals* (medoid sessions, coherence, resolution-profile sidecar,
   `--neighbors-of` lookup) so a calling LLM can do progressive disclosure into
   the corpus before reading specific transcripts.

Every expensive output (embeddings, classifications, clusters, communities,
friction) is cached in parquet under `~/.claude/`. Views register only the
parquets that exist — missing ones warn and no-op, never crash.

## Structure — hexagonal layers

One distribution, one namespace root `src/claude_sql/`, in strict hexagonal
layers. The `import-linter` contract in `pyproject.toml` (`[tool.importlinter]`,
checked by the `lint:imports` gate) is a **single layers DAG**:

    interfaces > application > infrastructure > domain

`interfaces` highest (driving adapters / composition root), `domain` lowest
(pure). Dependencies point inward.

- **`domain/`** — pure logic + value-objects, no third-party adapters:
  classification/window math (`trajectory`, `structure/{cluster,community,terms}`),
  `dedup`, `friction` regex bank, `transcript`, `costs`, the pydantic
  classification `models`, the provider-port Protocols (`ports`: `EmbeddingProvider`
  / `LlmAnalyticsProvider` / `SchemaT`), `errors`, the embedding dimension guard
  (`embedding_guard`), `retrieval.SearchHit`, `skills`, per-pipeline config
  (`config`).
- **`application/`** — use-cases (`use_cases/{embed,classify,trajectory,conflicts,
  friction,cluster,community,terms,ingest,skills,peek}`), the `analyze` chain, the
  port surface (`ports` — re-exports the domain Protocols + `SearchHit`, defines
  the storage ports), task-framing `prompts`, the dry-run counter (`_shared`).
- **`infrastructure/`** — the ONLY place with boto3/duckdb/lancedb/onnx:
  `settings`, `duckdb_{views,connection,s3,errors}`, `lance_store`,
  `parquet_cache`, `sqlite_state/*`, `bedrock/{client,structured_output}`, the
  provider factories + adapters `embedding/{cohere_bedrock,ollama,onnx_bge}` and
  `llm_analytics/{sonnet_bedrock,strands_luna}`, `session_text_loader`,
  `session_search`, `transcript_reader`, `skills_fs`, `home`, `logging_setup`.
- **`interfaces/`** — `cli/{app,output,install_source}` — the cyclopts CLI, the
  driving adapter + composition root. `main` + `python -m
  claude_sql.interfaces.cli.app` live here.

`composition.py` (`ClaudeSql` facade) and the top-level `__init__` re-export sit
at the root; `composition` wires the layers and imports infrastructure lazily.
The provider-port Protocols live in `domain.ports` (**not** `application.ports`)
so the concrete infra adapters can be typed against them without importing UP
into `application` — `import-linter` counts `TYPE_CHECKING` imports as real
edges, so an `application`-home would break the layers contract. The `SearchHit`
re-export pattern is the same idiom. Build is `uv_build` namespace package to one
self-contained wheel.

For the design rationale, the ordered cut history, and the per-plane grounding,
see `docs/v2/DESIGN.md`, `docs/v2/MIGRATION.md`, and
`docs/v2/understanding/01-architecture.md` .. `07-tests-ci-build-docs.md`
(shipped in v2.0.0).

## How to work on it

- **Package manager: `uv` only.** Never hand-edit `[dependencies]` in
  `pyproject.toml` — use `uv add` / `uv remove` so `uv.lock` stays in sync.
- **Task runner: `mise` only.** Every developer command is a mise task. Run
  `mise tasks` to see the full list. Do not invent bare `uv run` / `ruff`
  invocations for things that already have a task.
- **Quality gate: `mise run check` must pass before any commit.** That is
  **six** gates (`[tasks.check].depends` in `mise.toml`):

      lint + fmt + typecheck + lint:imports + test + proofs

  - `lint:imports` runs `lint-imports` against the single hexagonal layers DAG.
  - `test` runs the full pytest suite (count with
    `uv run pytest --collect-only -q | tail -1`).
  - `proofs` is `lake build` under `proofs/` — core Lean 4, no mathlib,
    sorry-free; each `ClaudeSql/*.lean` module machine-checks one pure domain
    invariant against its cited Python surface.

  Treat every non-zero exit as a blocker to fix now, even if it looks
  pre-existing.
- **Security gate: `mise run security`** runs all four SAST/SCA/secrets scanners
  in parallel — `bandit + semgrep + osv + leaks` — and writes SARIF under
  `.sarif/` (gitignored), mirroring what CI uploads to GitHub code scanning. Each
  scanner uses `--exit-zero` / `--exit-code=0`
  (`.erpaval/solutions/best-practices/sarif-scanner-report-vs-gate.md`); gating
  happens in the code-scanning UI, not the scanner exit code. Run before opening
  a PR if you touched anything load-bearing (ingestion/CLI surface).
- **Python: 3.14 toolchain, `>=3.13` floor.** The dev toolchain runs 3.14
  (`.python-version` = `3.14`, `mise.toml [tools].python` = `3.14`), while
  `pyproject.toml requires-python` stays `>=3.13` so the wheel still installs on
  3.13. `hdbscan` is the sole `cp314` blocker for the clustering stack; the base
  planes (SQL + semantic search + LLM analytics) are cp314-ready. See
  `docs/adr/0015-stack-modernization.md`.
- **Formatting:** `mise run fmt:write` applies ruff formatting. Line length 100.
  Ruff runs a 32-family strict selector set (E, W, F, I, N, UP, B, SIM, ANN,
  ASYNC, BLE, C4, DTZ, ERA, FBT, G, ICN, ISC, LOG, PERF, PIE, PL, PT, PTH, RET,
  RSE, S, T20, TID, TRY, PGH, RUF) with principled ignores documented inline in
  `pyproject.toml`.
- **Type checker: `ty`** in strict mode (`[tool.ty.rules] all = "error"`) over
  `src/ tests/`. Editor Pyright warnings about unresolved imports or
  `Optional[...]` subscript are false positives from a global install that can't
  see the project's `.venv`; trust `ty`.
- **Reproducibility:** `[tool.uv]` pins `required-version >= 0.11.7`,
  `python-preference = only-managed`, `compile-bytecode = true`,
  `link-mode = clone`. `mise run lock:check` is the CI freshness gate.
- **Tests:** `pytest` under `tests/`. `mise run test`. Tests must not hit Bedrock
  — mock the client. Use a small fixture directory, not the live corpus under
  `~/.claude/projects/`.

### `returns` discipline (the port surface)

`returns` is used for `Result[T, DomainError]` only — the uniform value that
crosses a port boundary (`application/ports.py`). Do NOT use `.bind()`, `@safe`,
`flow`, `pipe`, or any HKT feature: they require the `returns` mypy plugin and
hard-error under `ty` (strict mode). Use only `Success` / `Failure`,
`is_successful` (from `returns.pipeline`, not `returns.result`), `.unwrap()`,
`.map()`, `.alt()`, and `match` narrowing.

## Logging: loguru only, no stdlib `logging`

Every module in `claude_sql` logs through `from loguru import logger`. Stdlib
`logging` is **banned** — `[tool.ruff.lint.flake8-tidy-imports.banned-api]` in
`pyproject.toml` rejects `import logging` / `from logging import …` with a
`TID251` error. Mixing the two leaves messages stuck in the stdlib root logger
that the user's loguru sink never sees.

For tenacity `@retry` callbacks that want a `before_sleep` log, use
`claude_sql.infrastructure.logging_setup.loguru_before_sleep("LEVEL")`, which
emits the retry-state line via loguru instead of a stdlib logger:

```python
from claude_sql.infrastructure.logging_setup import loguru_before_sleep
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    before_sleep=loguru_before_sleep("WARNING"),
    reraise=True,
)
def call_bedrock(...): ...
```

If a third-party API demands a stdlib logger, write a small loguru-backed adapter
under `logging_setup.py` rather than punching a hole in the ban.

## Bedrock

- **Global CRIS only.** Embeds + Sonnet classification go through the **global**
  CRIS profiles — `global.cohere.embed-v4:0` and
  `global.anthropic.claude-sonnet-4-6` (region `us-east-1`). Direct model IDs and
  US-only CRIS throttle under load; global is the only path that sustains
  throughput. Sonnet classification stays Bedrock-only; the embedding path is
  pluggable (Ollama / ONNX BGE as local alternatives under optional extras).
- **Cost guard:** every command that spends real money defaults to `--dry-run`.
  `analyze` chains embed → cluster → classify → trajectory → conflicts →
  friction and respects `--dry-run` on every step. The dry-run path uses a
  pure-SQL count (`application/use_cases/_shared.py`), not a full
  materialization, so it stays fast on large corpora.
- **Structured output:** Bedrock's GA `output_config.format` field (not
  tool_use/tool_choice). Schemas are pydantic v2 models flattened in
  `infrastructure/bedrock/structured_output.py` (inline `$ref`, inject
  `additionalProperties: false`, strip constraints the validator rejects).
  Adaptive thinking stays on; `citations` is the only feature incompatible with
  structured output — do not add it.

## The embedding dimension contract (migration hazard)

`output_dimension` threads into the Lance fixed-size vector schema (frozen on the
first write to a fresh dir), the DuckDB `FLOAT[dim]` view cast, and the
query-time cast. Switching providers requires a full re-embed (`rm -rf
~/.claude/embeddings_lance/` then `embed`), and even same-dim vectors from
different models live in incompatible spaces. Every Lance row carries a `model`
column; `domain/embedding_guard.ensure_store_matches` reads it back and fails
loud on a provider/dimension mismatch instead of silently corrupting the index.
`provider.dimension` is adapter-reported, not a free `Settings` field.

## Analytics determinism

`CLAUDE_SQL_SEED=42` (default) seeds UMAP, HDBSCAN, and Leiden so cluster IDs and
community IDs are stable across reruns. The Leiden seed flows into both
`leidenalg.find_partition(seed=...)` and `Optimiser.set_rng_seed(...)` for the
resolution-profile bisection, so same-seed reruns produce byte-equal parquets.
Don't reach for different seeds without a reason.

The clustering/community/terms internals — Leiden+CPM over the mutual-kNN graph,
the auto-γ resolution profile, the CountVectorizer + in-house c-TF-IDF math, and
the per-classifier (trajectory / conflicts / friction) pipeline schemas — are
documented in `docs/rfc/0002-vision-and-roadmap.md` (§3.4, §4.1–4.3),
`docs/v2/understanding/02-retrieval-clustering.md`, and `docs/behavior/processes.md`.
Do not reintroduce networkx Louvain / `python-louvain`, and do not pull in
`bertopic` — CPM (resolution-limit-free on cosine-weighted graphs) and the
visible c-TF-IDF math are deliberate.

## Agent-friendly CLI surface (load-bearing — do not regress)

- `--format {auto,table,json,ndjson,csv}` on every subcommand. `auto` resolves to
  `table` on TTY, `json` on pipe — agents calling via subprocess get JSON for
  free.
- Structured DuckDB errors: parse → exit **64**, catalog → exit **65**, runtime →
  exit **70**. Non-TTY stderr carries `{"error": {"kind","message","hint"}}`.
  The classifier is `classify_duckdb_error` in `infrastructure/duckdb_errors.py`
  (surfaced through `interfaces/cli/output.py`); keep it in sync if new DuckDB
  exception subclasses land.
- `list-cache` reports every parquet's `{exists, bytes, mtime, rows}`. When
  adding a new analytics parquet, extend the cache-describe calls in
  `interfaces/cli/app.py`.
- `explain` default is static (`EXPLAIN`, no execution). `--analyze` opts into
  `EXPLAIN ANALYZE`. Do NOT flip this back — the prior default silently executed
  slow queries when agents probed plans.
- `--quiet` is honored by every subcommand. View-registration logs are at DEBUG
  on purpose so default stderr stays empty for read-only flows.

## Resilience patterns to preserve

Re-cited to the current module paths — verify before editing.

- **DuckDB JSONL read flags** (`infrastructure/duckdb_views.py`,
  `infrastructure/transcript_reader.py`): `read_json(..., filename=true,
  union_by_name=true, sample_size=-1, ignore_errors=true,
  maximum_object_size=67108864)`. The corpus always has a few truncated or
  growing files; these flags keep them from aborting the query.
- **Stale-JSONL skip** (`infrastructure/session_text_loader.py`): the DuckDB
  query is wrapped in `try/except duckdb.IOException` and skips stale JSONLs with
  a one-shot retry + warning.
- **Dry-run count** (`application/use_cases/_shared.py`): a pure SQL
  `COUNT(DISTINCT session_id)` — do not replace with a full materialization.
- **Tenacity retries** (`infrastructure/bedrock/client.py`,
  `infrastructure/embedding/cohere_bedrock.py`): catch `SSLError`,
  `ConnectionError`, and `ThrottlingException` with exponential backoff.
  Botocore's own retry is disabled so tenacity owns the policy.
- **Embeddings parquet schema** (`application/use_cases/embed.py`,
  `infrastructure/lance_store.py`): write with explicit `pl.Array(pl.Float32,
  dim)`, otherwise polars infers `Object` and the roundtrip breaks.
- **Model-ID cost lookup** (`infrastructure/duckdb_views.py`, `cost_estimate`
  macro): the pricing join strips the dated suffix in SQL —
  `regexp_replace(m.model, '-\d{8}$', '')` joined on equality — so dated model
  IDs like `claude-sonnet-4-6-20260315` resolve to the base entry
  `claude-sonnet-4-6` in `DEFAULT_PRICING`. (SQL now, not a Python `re.sub`; the
  pure-Python `estimate_cost` in `domain/costs.py` deliberately does not do the
  strip.)
- **`BedrockRefusalError` is terminal** (`domain/errors.py`, consumed in
  `application/use_cases/{trajectory,friction}.py`): the LLM pipelines stamp a
  neutral placeholder row and clear the retry queue so refused messages don't
  cycle forever.
- **LanceDB empty-namespace gate** (`infrastructure/lance_store.py`,
  `infrastructure/duckdb_views.py`): DuckDB's `ATTACH (TYPE LANCE)` succeeds on
  any path — the catalog error fires later at SELECT/CREATE-VIEW time. Always
  probe via `lance_store._has_table(...)` before binding the view, NOT via
  filesystem heuristics. Pinned by `tests/test_lance_store.py`.
- **Sharded parquet caches** (`infrastructure/parquet_cache.py`): workers write
  each chunk as a fresh `<cache>/part-<ts_ns>.parquet` instead of
  read-concat-rewrite; readers glob via `iter_part_files`. Consolidate with
  `claude-sql cache compact`.
- **DuckDB tuning PRAGMAs** (`infrastructure/duckdb_connection.py`): set from
  `Settings.duckdb_threads / duckdb_memory_limit / duckdb_temp_dir`.
  `memory_limit` accepts percentage strings (`'70%'`), resolved to MiB at apply
  time (DuckDB rejects `%` as a unit). `temp_directory` is `~/.claude/duckdb_tmp`
  instead of `/tmp` — Amazon devboxes ship `/tmp` as a 4 GB tmpfs and a
  clustering spill there thrashes the host.
- **Cluster mtime sidecar** (`domain/structure/cluster.py`,
  `application/use_cases/cluster.py`): skip the ~40 s UMAP+HDBSCAN refit when the
  embeddings haven't moved, comparing the latest part-file mtime against the
  sidecar. `force=True` always rebuilds.

## Environment variables

All prefixed `CLAUDE_SQL_`. Defaults live in `infrastructure/settings.py`; the
README has the full table. Common overrides: `CLAUDE_SQL_DEFAULT_GLOB`,
`CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH`, `CLAUDE_SQL_USER_FRICTION_PARQUET_PATH`,
`CLAUDE_SQL_FRICTION_MAX_CHARS` (default 300), `CLAUDE_SQL_EMBED_CONCURRENCY`
(default 8), `CLAUDE_SQL_LLM_CONCURRENCY` (default 2),
`CLAUDE_SQL_DUCKDB_THREADS`, `CLAUDE_SQL_DUCKDB_MEMORY_LIMIT` (`'70%'` default),
`CLAUDE_SQL_DUCKDB_TEMP_DIR`, `CLAUDE_SQL_LANCE_URI`.

## How to run it

- **As a uv tool** (preferred end-user path): `mise run tool:install` →
  `claude-sql --version`. Reinstall after pulling with `mise run tool:upgrade`.
  Remove with `mise run tool:uninstall`.
- **In the project venv** (development): `mise run install` → `mise run cli --
  <subcommand>`.
- **REPL:** `claude-sql shell` drops into DuckDB with every view + macro
  registered.

## Commits, releases & CI hygiene

- **Gitflow + Conventional Commits.** Work on a typed branch (`feature/`, `fix/`,
  `chore/`, `docs/`, `refactor/`), one logical change per commit. The
  `commit-msg` lefthook enforces the conventional schema (commitizen);
  `git commit --no-verify` is an escape hatch, not a policy. First-time hook
  setup: `mise run install` (runs `hooks:install`). See `lefthook.yml`.
- `mise run check` must pass before every commit.
- **`origin/main` is protected** by a repository ruleset (pull-request required,
  required status checks, code scanning, linear history — no bypass actors).
  Direct pushes are rejected with `GH013`. Every change lands via a PR.
- **Releasing to PyPI, version bumping with `cz bump`, and the
  branch-protection branch → PR → squash-merge → re-tag → cut-release dance** are
  in **`docs/RELEASING.md`**.
- **The four CodeQL rules ruff misses** (empty-except, unused-local-in-test,
  catch-base-exception, file-not-closed-in-mkstemp-fakes) are in
  **`docs/contributing/codeql-hygiene.md`** — treat that discipline as part of
  writing the test in the first place.

## Deferred decisions (named so they don't get re-derived)

- **Snapshot tier / `CacheNode` DAG.** Deferred — the corpus doesn't justify the
  abstraction yet. Reopen when `EXPLAIN ANALYZE` shows JSONL re-scan dominating
  wall clock, or the corpus crosses ~10 GB.
- **`CreateModelInvocationJob` batch path / vector quantization.** Async at
  concurrency 8 saturates global CRIS without throttling; VSS doesn't natively
  support FLOAT16/INT8. Reopen when the embeddings store crosses ~10 GB.

## CodeGraph: code intelligence

This repository is indexed by CodeGraph (a `.codegraph/` directory exists at the
repo root). When working in this codebase, reach for CodeGraph BEFORE grep/find
or reading whole files to understand or locate code. It returns the relevant
symbols' verbatim source plus the call paths between them, including
dynamic-dispatch hops that text search cannot follow.

- **MCP tool:** `codegraph_explore` answers most code questions in one call. Name
  a file or symbol in the query to read its current line-numbered source.
  `codegraph_node` returns one symbol's source plus its caller/callee trail.
- **Shell (always works):**
  - `codegraph explore "<symbols or question>"`: same output as the MCP tool.
  - `codegraph query "<term>"`: search for symbols.
  - `codegraph node <name>`: one symbol's source plus its caller/callee trail.
  - `codegraph callers <symbol>` / `codegraph callees <symbol>`: call graph.
  - `codegraph impact <symbol>`: what a change to a symbol affects.
  - `codegraph affected <files...>`: test files affected by changed sources.
  - `codegraph files`: project file structure from the index.

Run `codegraph sync` after pulling new commits so the index stays aligned with
the working tree. `codegraph status` reports index staleness.
