# claude-sql · Module map

The codebase is one package (`claude-sql`) with five layer sub-packages under `src/claude_sql/` (`core`, `analytics`, `evals`, `provenance`, `app`), all declared by the single root `pyproject.toml`. The internal dependency graph is a fan: `core` has no internal dependencies and is imported by every other layer, while `app` depends on all four others and owns the console script (`pyproject.toml:53`). Modules below are ordered by that graph — orchestrator first (`app`), then the foundation (`core`), then the three worker layers (`analytics`, `evals`, `provenance`) that each depend only on `core`. Each layer's 1-LOC `__init__.py` is a namespace marker omitted from bullet lists except where a layer would otherwise have fewer than three files.

## app

The entry-point layer `app`, whose only substantial module is the Cyclopts CLI that wires the `claude-sql` console script to thirteen subcommands and is the single place importing every other layer (`src/claude_sql/app/cli.py:1`). The CLI keeps the fast read-only path (`schema`, `query`, `explain`) cheap by lazy-importing asyncio, subprocess, and worker modules inside the commands that need them (`src/claude_sql/app/cli.py:17`). It classifies DuckDB errors into parse/catalog/runtime and maps them to stable exit codes 64/65/70 with a JSON error payload off-TTY (`src/claude_sql/app/cli.py:11`). The layer depends on `core`, `analytics`, `evals`, and `provenance` (root `pyproject.toml` dependencies block at `pyproject.toml:27`), and its only other module reads the `uv` install receipt so `--version` can report whether the binary on PATH came from a checkout or a git URL (`src/claude_sql/app/install_source.py:1`).

- `src/claude_sql/app/cli.py` (3079 LOC)
- `src/claude_sql/app/install_source.py` (78 LOC)
- `src/claude_sql/app/__init__.py` (1 LOC)

## core

The shared foundation `core`, imported by every other layer and depending on no internal layer; its `config` module alone is referenced 15 times across the package (`src/claude_sql/core/config.py:1`). The SQL backbone `sql_views.py` (2182 LOC) wires a DuckDB connection to the on-disk `~/.claude/` JSONL corpus and exposes it as zero-copy views, analytical macros, an HNSW-indexed embeddings table, and parquet-backed views for the v2 analytics outputs (`src/claude_sql/core/sql_views.py:1`). The LLM hub `llm_shared.py` (1341 LOC) owns Bedrock client construction, the retryable `invoke_model` wrapper, the `classify_one` structured-output dispatcher, the per-pipeline cache-stat accumulator, and the four task-framing system prompts — every stage worker imports from here and none import each other (`src/claude_sql/core/llm_shared.py:1`). Supporting modules cover the Pydantic v2 Bedrock-compatible schemas (`src/claude_sql/core/schemas.py:1`), per-session text-window assembly clipped to Sonnet 4.6's context (`src/claude_sql/core/session_text.py:1`), a SQLite-WAL checkpointer that lets re-runs skip unchanged sessions (`src/claude_sql/core/checkpointer.py:1`), the LanceDB embeddings store (`src/claude_sql/core/lance_store.py:1`), and sharded parquet append I/O for the worker caches (`src/claude_sql/core/parquet_shards.py:1`).

- `src/claude_sql/core/sql_views.py` (2182 LOC)
- `src/claude_sql/core/llm_shared.py` (1341 LOC)
- `src/claude_sql/core/schemas.py` (597 LOC)
- `src/claude_sql/core/session_text.py` (387 LOC)
- `src/claude_sql/core/config.py` (382 LOC)
- `src/claude_sql/core/checkpointer.py` (378 LOC)
- `src/claude_sql/core/lance_store.py` (261 LOC)
- `src/claude_sql/core/parquet_shards.py` (253 LOC)

## analytics

The `analytics` layer holds the per-stage workers that stream sessions to Bedrock and append parquet caches, each importing only from `core` and never from each other (enforced by the import-linter layers contract at `pyproject.toml:264`). Its largest module runs the windowed sentiment-trajectory pipeline, pairing each text turn with its predecessor and batching windows into chunks of <=16 per Sonnet 4.6 request (`src/claude_sql/analytics/trajectory_worker.py:1`). The friction worker classifies short user-role messages into signals such as `status_ping`, `unmet_expectation`, and `correction` (`src/claude_sql/analytics/friction_worker.py:1`), the community worker builds session-centroid mutual-kNN cosine graphs and runs Leiden+CPM partitioning with auto-picked resolution (`src/claude_sql/analytics/community_worker.py:1`), and the embed worker backfills Cohere Embed v4 vectors in batches of up to 96 texts per Bedrock call (`src/claude_sql/analytics/embed_worker.py:1`).

- `src/claude_sql/analytics/trajectory_worker.py` (1005 LOC)
- `src/claude_sql/analytics/friction_worker.py` (741 LOC)
- `src/claude_sql/analytics/community_worker.py` (667 LOC)
- `src/claude_sql/analytics/ingest.py` (526 LOC)
- `src/claude_sql/analytics/embed_worker.py` (507 LOC)
- `src/claude_sql/analytics/skills_catalog.py` (354 LOC)
- `src/claude_sql/analytics/conflicts_worker.py` (341 LOC)
- `src/claude_sql/analytics/classify_worker.py` (254 LOC)

## evals

The `evals` layer implements the cross-provider judge harness and its reliability statistics, depending only on `core` (enforced by the import-linter layers contract at `pyproject.toml:264`). Its judge worker runs a panel of Bedrock models over sessions through the Converse API — the path that works uniformly across Anthropic, Moonshot, DeepSeek, Mistral, Qwen, and other lineages — writing one parquet row per (session, axis, judge) (`src/claude_sql/evals/judge_worker.py:1`). The kappa worker consumes those score parquets and computes Cohen's and Fleiss' kappa with bootstrapped 95% CIs over 1000 resamples (`src/claude_sql/evals/kappa_worker.py:1`). Pre-registration is handled by `freeze.py`, which hashes the full study spec into a deterministic manifest SHA and rebuilds the `Study` on replay (`src/claude_sql/evals/freeze.py:1`), while `blind_handover.py` strips identity markers from a transcript so an external grader cannot use authorship as a cue (`src/claude_sql/evals/blind_handover.py:1`).

- `src/claude_sql/evals/judge_worker.py` (462 LOC)
- `src/claude_sql/evals/kappa_worker.py` (257 LOC)
- `src/claude_sql/evals/judges.py` (239 LOC)
- `src/claude_sql/evals/ungrounded_worker.py` (190 LOC)
- `src/claude_sql/evals/freeze.py` (189 LOC)
- `src/claude_sql/evals/blind_handover.py` (155 LOC)

## provenance

The `provenance` layer binds merged commits back to the transcripts that produced them and renders PR review sheets, depending only on `core` (enforced by the import-linter layers contract at `pyproject.toml:264`). Its anchor module implements RFC 0001's transcript-to-PR binding through a three-trailer plus JSON git-note convention using pure-stdlib helpers (`src/claude_sql/provenance/binding.py:1`). The review-sheet worker compresses a bound transcript into a ~1K-token digest via Sonnet 4.6 structured output, so a reviewer gets "what was the agent trying to do, and what did it verify" without scrolling the raw JSONL (`src/claude_sql/provenance/review_sheet_worker.py:1`). A separate pure-formatting renderer turns that structured dict into Markdown, kept distinct from the worker so the CLI can choose JSON or Markdown output (`src/claude_sql/provenance/review_sheet_render.py:1`).

- `src/claude_sql/provenance/binding.py` (743 LOC)
- `src/claude_sql/provenance/review_sheet_worker.py` (465 LOC)
- `src/claude_sql/provenance/review_sheet_render.py` (167 LOC)

## See also

- [claude-sql · Public API](../reference/public-api.md) — 13 shared source files
- [claude-sql · Contract map](../insights/contract-map.md) — 12 shared source files
- [claude-sql · Processes](../behavior/processes.md) — 12 shared source files
- [claude-sql · Tech debt](../insights/tech-debt.md) — 12 shared source files
- [claude-sql · Business logic](../insights/business-logic.md) — 10 shared source files
