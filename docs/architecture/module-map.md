# claude-sql · Module map

claude-sql is a flat single-package Python project: every source file lives directly under `src/claude_sql/` with no nested subpackages (`src/claude_sql/__init__.py:1`). This map partitions the package by concern and follows the same module ordering as the Mermaid flow in `architecture/system-overview.md` (`docs/architecture/system-overview.md:75`) — CLI surface first, then SQL backbone, then LLM and vector pipelines, then the IRR study and PR-review surfaces. Inbound-import counts (used to break ties) come from a `grep` over `src/` and `tests/`; LOC counts come from `wc -l`.

## CLI surface

`cli.py` is the Cyclopts entry point that wires the `claude-sql` console script to its subcommands and is the only module that imports every other concern in the package (`src/claude_sql/cli.py:1`). It carries 2917 LOC because each subcommand parses its own flags, opens its own DuckDB connection, and lazy-imports the worker it dispatches to so the fast read-only path (`schema`, `query`, `explain`) does not drag asyncio or boto3 into startup (`src/claude_sql/cli.py:17`). `output.py` is the cross-cutting formatter every subcommand calls: `OutputFormat.AUTO` resolves to TABLE on a TTY and JSON on a pipe, and DuckDB exceptions classify into parse / catalog / runtime exit codes 64 / 65 / 70 (`src/claude_sql/output.py:1`). `logging_setup.py` removes the default loguru handler and re-installs a stderr sink that honors `--verbose` / `--quiet` and `LOGURU_LEVEL`, and exposes `loguru_before_sleep` so workers pass tenacity a loguru-native callback instead of stdlib `logging.getLogger` (`src/claude_sql/logging_setup.py:1`). `install_source.py` reads `$UV_TOOL_DIR/<tool>/uv-receipt.toml` so `claude-sql --version` can tell the user whether the binary on PATH came from a directory checkout, a git URL, or the project venv (`src/claude_sql/install_source.py:1`).

- `src/claude_sql/cli.py` (2917 LOC)
- `src/claude_sql/output.py` (254 LOC)
- `src/claude_sql/logging_setup.py` (95 LOC)
- `src/claude_sql/install_source.py` (77 LOC)

## DuckDB views and storage

`sql_views.py` is the SQL backbone — it registers 18 zero-copy views and 14 analytical macros over the on-disk JSONL corpus and the parquet-backed analytics caches, with per-cache existence probes so missing parquets warn and no-op instead of crashing (`src/claude_sql/sql_views.py:1`). `session_text.py` materializes `messages_text` / `tool_calls` / `tool_results` into in-memory arrow once per pipeline run so per-session slicing avoids quadratic JSONL re-scans (`src/claude_sql/session_text.py:8`). `lance_store.py` owns the LanceDB embeddings dataset and exposes it back to DuckDB through the `lance` core extension; `parquet_shards.py` replaces the prior read-concat-rewrite pattern for the four append caches with `<dir>/part-<ts_ns>.parquet` shards (`src/claude_sql/lance_store.py:1`, `src/claude_sql/parquet_shards.py:1`). `config.py` is the pydantic-settings root for every `CLAUDE_SQL_*` knob, `home.py` resolves the `CLAUDE_SQL_HOME` parent for derived caches (XDG on Linux, Application Support on macOS), `ingest.py` stamps each message with `approx_tokens` / `simhash64` / `canonical_uuid`, and `skills_catalog.py` walks `~/.claude/skills/` plus plugin caches into a SKILL.md catalog parquet (`src/claude_sql/config.py:1`, `src/claude_sql/home.py:10`, `src/claude_sql/ingest.py:1`, `src/claude_sql/skills_catalog.py:1`).

- `src/claude_sql/sql_views.py` (2228 LOC)
- `src/claude_sql/config.py` (413 LOC)
- `src/claude_sql/session_text.py` (387 LOC)
- `src/claude_sql/skills_catalog.py` (354 LOC)
- `src/claude_sql/lance_store.py` (261 LOC)
- `src/claude_sql/parquet_shards.py` (253 LOC)
- `src/claude_sql/ingest.py` (526 LOC)
- `src/claude_sql/home.py` (93 LOC)

## LLM analytics pipelines

`llm_shared.py` hosts the Bedrock plumbing every classifier needs: client construction, the retryable `invoke_model` wrapper, the async `classify_one` dispatcher under a concurrency limiter, the per-pipeline cache-stat accumulator, and the four task-framing system prompts; the per-stage workers import from here and never from each other (`src/claude_sql/llm_shared.py:1`). `trajectory_worker.py` is the v1.0 windowed sentiment-trajectory pipeline (one row per `(prev_uuid, curr_uuid)` text-turn pair, batched ≤16 windows per Sonnet call) (`src/claude_sql/trajectory_worker.py:1`). `friction_worker.py` detects seven user-friction labels in user-role messages ≤ `friction_max_chars`, with a regex fast-path for unambiguous patterns and Sonnet 4.6 for the rest (`src/claude_sql/friction_worker.py:1`). `conflicts_worker.py` is the pair-keyed v1.0 stance-conflict detector (one row per `(turn_a_uuid, turn_b_uuid)`, no `empty=True` sentinel) and `classify_worker.py` is the per-session autonomy-tier / work-category / success classifier with a session-level checkpointer + per-uuid anti-join (`src/claude_sql/conflicts_worker.py:1`, `src/claude_sql/classify_worker.py:1`). `schemas.py` flattens every pydantic v2 schema into the Bedrock-compatible JSON-Schema-Draft-2020-12 subset that `output_config.format` accepts (`src/claude_sql/schemas.py:1`).

- `src/claude_sql/llm_shared.py` (1341 LOC)
- `src/claude_sql/trajectory_worker.py` (1005 LOC)
- `src/claude_sql/friction_worker.py` (741 LOC)
- `src/claude_sql/schemas.py` (597 LOC)
- `src/claude_sql/conflicts_worker.py` (341 LOC)
- `src/claude_sql/classify_worker.py` (254 LOC)

## Embeddings, clustering, communities

`embed_worker.py` discovers messages without an embedding, calls `cohere.embed-v4:0` on Bedrock in parallel batches (up to 96 texts per call), and appends FLOAT[1024] vectors to the LanceDB dataset; the int8 response is converted to float because the VSS index requires `FLOAT[]` columns (`src/claude_sql/embed_worker.py:1`). `community_worker.py` builds a mutual-kNN cosine graph (k=15 by default) over session centroids, picks γ via `leidenalg.Optimiser.resolution_profile` (longest-plateau auto-pick), then runs `find_partition(g, CPMVertexPartition, ...)` with `seed=settings.seed` for byte-stable reruns (`src/claude_sql/community_worker.py:1`). `cluster_worker.py` reduces message embeddings via UMAP to 50d (HDBSCAN) plus 2d (viz) and writes `clusters.parquet` with `(uuid, cluster_id, x, y, is_noise)`; `terms_worker.py` builds one pseudo-document per cluster and computes c-TF-IDF in-house with `sklearn.CountVectorizer` rather than pulling in BERTopic (`src/claude_sql/cluster_worker.py:1`, `src/claude_sql/terms_worker.py:1`).

- `src/claude_sql/community_worker.py` (667 LOC)
- `src/claude_sql/embed_worker.py` (533 LOC)
- `src/claude_sql/cluster_worker.py` (215 LOC)
- `src/claude_sql/terms_worker.py` (145 LOC)

## IRR study tooling

`judge_worker.py` runs a panel of cross-provider Bedrock judges over sessions through the Converse API — the only path that works uniformly across Anthropic, Moonshot, DeepSeek, MiniMax, Mistral, Z.AI, Qwen, Writer, and Nemotron — emitting `score=<int>\nrationale=<text>` parquet rows without provider-specific tool_use machinery (`src/claude_sql/judge_worker.py:1`). `judges.py` is the catalog of judge identifiers, families, and shortname aliases validated against `aws bedrock list-foundation-models` in `us-east-1` (`src/claude_sql/judges.py:1`). `kappa_worker.py` consumes those judge-score parquets and computes Cohen's pairwise kappa, Fleiss' kappa across all judges, bootstrapped 95% CIs over 1000 resamples, and a stopping-rule gate that exits non-zero when the delta-kappa CI excludes zero (`src/claude_sql/kappa_worker.py:1`). `freeze.py` is the pre-registration audit trail — it hashes a study spec into a deterministic manifest SHA under `~/.claude/studies/<sha>/` so every parquet the workers emit carries a `freeze_sha` column (`src/claude_sql/freeze.py:1`). `blind_handover.py` strips identity markers (Slack IDs, agent personas, MCP tool names, OTel trace IDs) so a session bundle is grader-safe, and `ungrounded_worker.py` flags assistant claims about entities (paths, function names, env vars, CLI subcommands) that never appear in the same session's tool outputs (`src/claude_sql/blind_handover.py:1`, `src/claude_sql/ungrounded_worker.py:1`).

- `src/claude_sql/judge_worker.py` (462 LOC)
- `src/claude_sql/kappa_worker.py` (257 LOC)
- `src/claude_sql/judges.py` (239 LOC)
- `src/claude_sql/ungrounded_worker.py` (190 LOC)
- `src/claude_sql/freeze.py` (189 LOC)
- `src/claude_sql/blind_handover.py` (155 LOC)

## PR review and transcript binding

`binding.py` implements the RFC 0001 transcript-to-PR binding convention — three commit trailers plus a JSON git-note pointing each merged commit at the AI-agent transcript that produced it, with `subprocess.run([...], check=False)` and zero new dependencies (`src/claude_sql/binding.py:1`). `review_sheet_worker.py` resolves a `commit_sha` through `binding.resolve_commit_to_transcript`, compresses the bound transcript, and produces a 1K-token PR review sheet via Sonnet 4.6 with `output_config.format` structured output (`src/claude_sql/review_sheet_worker.py:1`). `review_sheet_render.py` is a pure-formatting Markdown renderer split out from the worker so the JSON path does not drag Bedrock imports through (`src/claude_sql/review_sheet_render.py:1`).

- `src/claude_sql/review_sheet_worker.py` (463 LOC)
- `src/claude_sql/binding.py` (740 LOC)
- `src/claude_sql/review_sheet_render.py` (167 LOC)

## Supporting code

- `src/claude_sql/checkpointer.py` (376 LOC)
- `src/claude_sql/retry_queue.py` (210 LOC)
- `src/claude_sql/__init__.py` (5 LOC)

## See also

- [claude-sql · Risk hotspots](../analysis/risk-hotspots.md) — 10 shared citations
- [claude-sql · Processes](../behavior/processes.md) — 7 shared citations
- [claude-sql · System overview](../architecture/system-overview.md) — 3 shared citations
- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 2 shared citations
