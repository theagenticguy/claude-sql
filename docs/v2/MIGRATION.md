# claude-sql v2: Migration Plan

**Status:** proposed (2026-07-19). The ordered plan to move v1.2.1 to the v2
architecture in `docs/v2/DESIGN.md`. Every mechanical step here is grounded in
the understanding pass under `docs/v2/understanding/`, and each cited line is
against the v1.2.1 tree at HEAD `7f9e5a2`.

The plan runs in three phases: **A. Drop** (remove evals + provenance),
**B. Pluggable embeddings** (the one seam with new behavior), **C. Hexagonal
reshape** (lift adapters, then the CLI). Phases A and B are independent and can
land in either order; phase C depends on both.

---

## Definition of done

`mise run check` is the gate. It runs **five** checks, all blockers:

1. `lint`: `ruff check`
2. `fmt`: `ruff format --check`
3. `typecheck`: `ty check` (strict, `all = error`)
4. `lint:imports`: the import-linter DAG contract
5. `test`: pytest (826 tests today; the number drops as eval/provenance tests
   are deleted)

Two facts that make `lint:imports` load-bearing here: the import-linter
contracts in `pyproject.toml` referenced `claude_sql.evals` and
`claude_sql.provenance` by name, so they would **hard-fail the instant those
packages are deleted** unless the contract is edited in the same commit. Never
split the package deletion from the contract edit. (Phase A landed both together
in a0ec803: the contract block is now `pyproject.toml:271-281`, layered-only —
`app < analytics < core` — and the independence contract was removed.)

---

## Phase A: Drop evals and provenance — DONE (a0ec803)

**Status: landed in commit a0ec803.** Verified against HEAD: `src/claude_sql/evals/`
and `src/claude_sql/provenance/` are gone (A.1); all 10 eval/provenance commands
dropped from `cli.py` (A.2); the import-linter contract is layered-only with the
independence block removed (A.3, `pyproject.toml:271-281`); `anthropic` and
`scipy` are no longer direct deps (A.4); all 13 dead test files deleted (A.5);
`docs/reference/cli.md` carries no eval/provenance sections (A.7). Sub-item
line references below are historical (v1.2.1 tree at `7f9e5a2`).

Both planes are cleanly severable: the only cross-boundary importer of either
is `app/cli.py` (`docs/v2/understanding/05-drop-analysis.md`; CodeGraph
`impact`/`callers` evidence log in that doc's appendix). Nothing in `core/` or
`analytics/` depends on either. No shared runtime dependency is lost.

### A.1 Delete the package source

```
rm -rf src/claude_sql/evals/
rm -rf src/claude_sql/provenance/
```

### A.2 Remove the CLI commands

In `app/cli.py`:
- Delete the module-top imports of `claude_sql.evals.{blind_handover, freeze,
  judges}` and `claude_sql.provenance.{binding, review_sheet_render}`
  (`cli.py:91-97`).
- Delete the 7 eval commands: `judges`, `freeze`, `replay`, `blind-handover`,
  `judge`, `ungrounded-claim`, `kappa`.
- Delete the 3 provenance commands: `bind`, `resolve`, `review-sheet`.
- Remove eval/provenance command names from the `@app.default` hint text.

### A.3 Rewrite the import-linter contracts (same commit as A.1)

In `pyproject.toml` `[tool.importlinter]`:
- **Layered contract:** change the L1 layer from
  `"claude_sql.analytics | claude_sql.evals | claude_sql.provenance"` to
  `"claude_sql.analytics"`.
- **Independence contract:** with one L1 sibling left it is vacuous: delete the
  whole contract block.
- Update the explanatory comment above `[tool.importlinter]` and the
  package-layout comment under `[tool.uv.build-backend]` (five sub-packages →
  three: core / analytics / app).
- Update the `lint-imports` job comment in `lefthook.yml`.

### A.4 Remove dead dependencies

```
uv remove anthropic   # zero direct imports; the code uses Bedrock via boto3
uv remove scipy       # dead direct pin; transitive via sklearn/umap/hdbscan
```
No evals/provenance-exclusive dependency exists to remove.

### A.5 Delete the dead tests (13 files)

```
tests/test_judges.py  test_judges_extras.py  test_freeze.py
tests/test_judge_worker.py  test_judge_worker_extras.py  test_kappa_worker.py
tests/test_ungrounded_worker.py  test_blind_handover.py
tests/test_binding.py  test_binding_extras.py
tests/test_review_sheet_render.py  test_review_sheet_worker.py
tests/test_review_sheet_worker_extras.py
```

### A.6 Surgical test edits (shared files KEEP code also uses)

- `tests/test_cli.py`: delete the eval/provenance command tests; keep the
  `_resolve_settings` / `_resolve_memory_limit` core-helper tests.
- `core/schemas.py`: the `PRReviewSheet` / `Correction` /
  `PR_REVIEW_SHEET_SCHEMA` surface and their `__all__` entries are pruned
  (DONE). `tests/test_schemas.py` never imported them — no test edit was
  needed (the stale assumption here was that it did; verified via grep +
  codegraph, zero importers outside the deleted source).
- `tests/test_pr3_perf.py`: remove the four now-deleted entries from the
  `test_cli_import_is_lean` forbidden-eager-import guard
  (`claude_sql.evals.judge_worker`, `.evals.kappa_worker`,
  `.evals.ungrounded_worker`, `.provenance.review_sheet_worker`).
- `tests/test_pr3_perf.py` also carries 11 tests that die with both planes ,
  confirm which remain relevant post-drop.

### A.7 Docs

- `docs/reference/cli.md`: delete the eval/provenance command sections
  (everything from `## judges` to EOF; the keep surface ends at `## analyze`).
- `README.md`: the parallel README rewrite handles the eval/provenance
  annotations and removals; coordinate so this is not done twice.
- `docs/rfc/0001-transcript-pr-binding.md`: this RFC *is* the provenance
  design. Move it to the eval/provenance project, or mark it superseded.
- `docs/rfc/0002-vision-and-roadmap.md`: already carries a superseded banner
  (its "no model substitution" constraint is reversed by v2).
- The generated analysis packets (`docs/analysis/*`, `docs/architecture/*`,
  `docs/insights/*`, `docs/diagrams/*`) reference evals/provenance throughout.
  They are regenerated artifacts, so regenerate after the drop rather than
  hand-editing.

### A.8 Gate

Run the full `mise run check`. `lint:imports` must pass with the reduced
contract; `test_cli_import_is_lean` must stay green with the trimmed forbidden
list; `uv lock --check` after the `uv remove`.

---

## Phase B: Pluggable embeddings — DONE (077f362)

**Status: landed in commit 077f362.** The port, guard, three adapters, and CLI
flag all shipped. Line references below are historical (v1.2.1 tree); the seam
has since moved into `core/embedding/`. Post-landing locations:
`_invoke_bedrock_sync` → `core/embedding/cohere_bedrock.py:80`; `embed_query`
→ `analytics/embed_worker.py:194` (shim over `build_embedder`);
`embed_documents_async` → `embed_worker.py:165`; `run_backfill` →
`embed_worker.py:217`.

The seam was two functions in `analytics/embed_worker.py`: `_invoke_bedrock_sync`
(`:195`) and `embed_query` (`:338`), with `embed_documents_async` (`:268`) and
`run_backfill` (`:369`) as orchestrators. See `docs/v2/DESIGN.md` §4 and
`docs/v2/understanding/03-embedding-seam.md`.

### B.1 Introduce the port and lift the current behavior — DONE (077f362)

Landed: `EmbeddingProvider` Protocol in `core/embedding/base.py:42`;
`CohereBedrockEmbedder` in `core/embedding/cohere_bedrock.py:153` as a pure lift;
`embed_query`/`embed_documents_async`/`run_backfill` route through
`build_embedder` (`base.py:86`) instead of reading `Settings` directly.

1. Define `EmbeddingProvider` (Protocol, §4.2 of DESIGN).
2. Implement `CohereBedrockEmbedder` as a pure lift of the current path: no
   behavior change. `dimension` reports 1024; `model_id` reports
   `global.cohere.embed-v4:0`. Keep the document `int8` / query `float`
   asymmetry inside the adapter.
3. Route `embed_query`, `embed_documents_async`, and `run_backfill` through the
   port instead of reading `Settings` directly.

### B.2 The dimension guard (do this before adding new providers) — DONE (077f362)

Landed as a **single-table fail-loud guard** rather than `(provider, model, dim)`
namespace keying: `table_identity` (`lance_store.py:245`) reads the stamped
`(model, dim)` from the store's first row; `assert_provider_match`
(`core/embedding/base.py:117`) raises `EmbeddingProviderMismatch` (`base.py:30`)
on a mismatch. Enforced at both `run_backfill` open (`embed_worker.py:217`, check
at `:306`) and `register_vss` (`sql_views.py:1824`, check at `:1939`), driven by
`Settings.expected_embedding_identity()`. A provider switch still requires a full
re-embed (`rm -rf` the store, then `embed`); the guard refuses to read/write a
store stamped by a different provider instead of silently corrupting kNN.

Today `output_dimension` threads unvalidated into the Lance schema (frozen on
first write), the DuckDB `FLOAT[dim]` view cast, and the query cast; the `model`
column is write-only. Before any second provider can be safe:
- Key the Lance table/namespace by `(provider, model, dim)`.
- Flip the `model` column into a **read-and-enforce fail-loud guard** at
  `run_backfill` open and at `register_vss`: refuse to write or read a store
  whose stamped `(provider, model, dim)` does not match the active provider.

### B.3 Add the two adapters behind extras — DONE (077f362)

Landed: `OllamaEmbedder` (`core/embedding/ollama.py:25`) and `OnnxBgeEmbedder`
(`core/embedding/onnx_bge.py:25`), both dispatched by `build_embedder`
(`base.py:86`). Optional extras `[ollama]` (`pyproject.toml:61`) and `[onnx]`
(`pyproject.toml:64`) keep the base wheel lean.

- `OllamaEmbedder` (`[ollama]` extra, `httpx`): POST `/api/embed`, batch,
  L2-normalize.
- `OnnxBgeEmbedder` (`[onnx]` extra, `onnxruntime` + `tokenizers`, no torch):
  BAAI `bge` v1.5, **CLS pooling then L2-normalize**, query-instruction prefix
  on queries only.

### B.4 CLI surface — DONE (077f362)

Landed: `_apply_embedding_provider` (`cli.py:231`) applies the
`--embedding-provider` flag on `search`/`embed`/`analyze`;
`CLAUDE_SQL_EMBEDDING_PROVIDER` binds `Settings.embedding_provider`
(`config.py:200`).

Add `--embedding-provider` to `search`, `embed`, and `analyze`. Provider
selection threads through pydantic-settings
(`CLAUDE_SQL_EMBEDDING_PROVIDER` + per-provider settings).

### B.5 Re-embed on provider switch

A provider switch invalidates existing vectors (different dim, or incompatible
space at matching dim). The path is: `rm -rf` the provider's store dir and
re-run `claude-sql embed` (empty store triggers a full backfill; the cluster
mtime sidecar auto-refits downstream). Document this in the README embed
section.

### B.6 Gate — DONE (077f362)

Landed: adapter-parity coverage in `tests/test_embedding_providers.py`.

`mise run check`, plus a new adapter-parity test: each adapter round-trips
`embed_query`/`embed_documents`, reports a stable `dimension`, and trips the
fail-loud guard when pointed at a store stamped by a different provider.

---

## Phase C: Hexagonal reshape — DONE

Cut in dependency order, lowest risk first. Full derivation in
`docs/v2/understanding/01-architecture.md` §5.2. All eight steps landed this
session; the port Protocols live at `application/ports.py` (not the
initially-sketched `domain/ports.py` — ports are an application-layer concern
here, with pure math in `domain/`).

1. **Define the port Protocols — DONE.** Landed at `application/ports.py`:
   `TranscriptReaderPort`, `SessionSearchPort`, `VectorStorePort`,
   `CheckpointPort`, `RetryQueuePort`, `CachePort`, `ReaderPort`. Pure additive.
2. **Wrap the already-adapter-shaped modules behind their ports — DONE.**
   `lance_store.py` → `VectorStorePort`, `duckdb_s3.py` + `parquet_cache.py` →
   infra, checkpoint/retry under `infrastructure/sqlite_state/`.
3. **Split `llm_shared.py` — DONE.** Transport + retry + prompt-cache accounting
   under `infrastructure/bedrock/` (`client.py`, `structured_output.py`);
   task-framing prompts under `application/prompts.py`. `BedrockRefusalError`
   generalized to the terminal domain `RefusalError` (`domain/errors.py`).
4. **Lift the domain math into `domain/` — DONE.** `costs.py`, `dedup.py`,
   `friction.py`, `retrieval.py`, `skills.py`, `trajectory.py`, `transcript.py`,
   and `domain/structure/` (community/cluster/terms math).
5. **Decompose the god-`Settings` — DONE.** Per-adapter config objects; the
   domain sees value-objects like `TranscriptCaps`, never a Bedrock model ID.
6. **Wrap DuckDB as `ReaderPort` — DONE.** `analyze` orchestration moved to
   `application/analyze.py`; the register → write → rebind cycle is modeled
   explicitly so mid-run writes stay visible (RFC §9.6 stale-connection bug
   stays fixed). Retrieval seam split into pure `domain/transcript.py` +
   `infrastructure/transcript_reader.py` + `infrastructure/session_text_loader.py`.
7. **Move CLI to `interfaces/cli/` — DONE.** `app.py` + `output.py` +
   `install_source.py` under `interfaces/cli/`; each subcommand builds and
   injects its adapters. Lazy-import discipline preserved (adapters constructed
   inside command bodies).
8. **Publish the importable facade — DONE.** `ClaudeSql(...)` in
   `composition.py`; downstream consumers can `uv add` claude-sql and drop their
   reader reimplementations. The retrieval seam is a byte-parity drop-in for
   the consumer's `_collapse` — proven by `tests/test_collapse_parity.py` (T-6-1).

### Preserve through the whole reshape

- Exit-code contract (parse=64, catalog=65, runtime=70): agents match on it.
- Lazy-import discipline (sub-second `schema`/`query`/`explain`).
- The empty-namespace LanceDB gate.
- `BedrockRefusalError`/`RefusalError` is terminal and non-retryable.
- The parquet-existence gate on analytics view registration.

---

## Sequencing summary

| Phase | Depends on | Risk | New behavior? | Status |
|---|---|---|---|---|
| A: Drop evals + provenance | nothing | low | no (pure removal) | DONE (a0ec803) |
| B: Pluggable embeddings | nothing | medium | yes (2 new adapters + guard) | DONE (077f362) |
| C: Hexagonal reshape | A + B | high (step 6 only) | no (structure only) | DONE |

Land A first (it shrinks the surface every later step touches). B is
independent and delivers the headline user-facing capability. C is the largest
and should follow both, with the DuckDB `ReaderPort` cut sequenced last within
it. All three phases have landed on `feat/v2-hexagonal`.
