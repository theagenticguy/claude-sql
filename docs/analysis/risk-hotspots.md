# claude-sql · Risk hotspots

"Risk" here is a composite per-file score over the last 30 days of git history. With every static-analysis gate clean — `bandit`, `semgrep`, `osv-scanner`, and `betterleaks` all report `0` results in `.sarif/*.sarif` — the default formula `2 × error_count + 0.5 × warn_count + rising` collapses to a single trend term and produces no useful ranking. Per packet section 4 ("Adapt if signals are missing; document the composition in the file's intro") the score this report uses is `commits_30d + (loc / 100) + 5 × (trend == "rising")`. `loc / 100` proxies blast radius: a finding-free 2917-line CLI is still riskier to refactor than a finding-free 200-line module. Trend buckets are computed against the median + 1σ of the per-file commit-count distribution (`rising` = count > median + σ).

Two signal limitations are worth naming up front. First, `Open findings` reads `0 warn, 0 error` for every row — that is the genuine state of the SARIF gates as of `.sarif/bandit.sarif`, not a gap in instrumentation; the project's `pyproject.toml` `[tool.bandit] skips` list (`B101 / B404 / B603 / B607 / B608`, `src/claude_sql/cli.py:1` upstream) intentionally yields ruff-S those classes, and ruff itself returns empty too. Second, the standard last-resort marker fallback (TODO / FIXME / HACK / XXX) was attempted and produced zero strict-word-boundary hits across `src/`; the loose-grep matches resolved to docstring prose mentions, not real code markers, so that signal does not break ranking ties.

| File | Trend | Open findings | Top owner | Citation |
| - | - | - | - | - |
| `src/claude_sql/cli.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/cli.py:1` (2917 LOC) |
| `src/claude_sql/sql_views.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/sql_views.py:1` (2228 LOC) |
| `src/claude_sql/llm_shared.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon (93%) | `src/claude_sql/llm_shared.py:1` (1341 LOC) |
| `src/claude_sql/config.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/config.py:1` (413 LOC) |
| `src/claude_sql/embed_worker.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon (90%) | `src/claude_sql/embed_worker.py:1` (533 LOC) |
| `src/claude_sql/friction_worker.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/friction_worker.py:1` (741 LOC) |
| `src/claude_sql/community_worker.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/community_worker.py:1` (667 LOC) |
| `src/claude_sql/trajectory_worker.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/trajectory_worker.py:1` (1005 LOC) |
| `src/claude_sql/schemas.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/schemas.py:1` (597 LOC) |
| `src/claude_sql/judge_worker.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon (83%) | `src/claude_sql/judge_worker.py:1` (462 LOC) |
| `src/claude_sql/session_text.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/session_text.py:1` (387 LOC) |
| `src/claude_sql/checkpointer.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon (100%) | `src/claude_sql/checkpointer.py:1` (376 LOC) |

## Per-file drill-down

### `src/claude_sql/cli.py`

**What's there.** The cyclopts CLI entry point — wires the `claude-sql` console script to its subcommands and centralizes the `--verbose` / `--quiet` / `--glob` / `--format` shared flags on a flattened `Common` dataclass (`src/claude_sql/cli.py:1`, `src/claude_sql/cli.py:173`). The single 2917-line module also owns DuckDB connection plumbing (`_open_connection_full` at `src/claude_sql/cli.py:338`, `_apply_duckdb_pragmas` at `src/claude_sql/cli.py:320`), error classification (parse → 64, catalog → 65, runtime → 70), and every subcommand registration (`@app.command` at `src/claude_sql/cli.py:638`, `:727`, `:806`, `:916`, `:979`).

**Recent activity.** 25 commits in the last 30 days — the highest in the codebase. Trend bucket: `↑ rising` (count > median 5 + σ 6.8). Recent commits include the v1.0 windowed-pipelines rewrite (`19293da`, `4eccbae`), the lance-backed embeddings switch (`8b14fe3`), and the two-tier-connections perf change (`1d067e3`).

**Owners.** Laith Al-Saadoon — 25/25 commits = 100% top-owner share over the window. No co-authors.

**Findings.** `0 warn, 0 error` — `.sarif/bandit.sarif`, `.sarif/semgrep.sarif`, `.sarif/osv.sarif`, `.sarif/betterleaks.sarif` all return empty `results` arrays at the canonical project config (`pyproject.toml [tool.bandit] skips`). Ruff and ty are also clean. Risk on this file is structural — module size + change velocity — not finding-driven.

### `src/claude_sql/sql_views.py`

**What's there.** The DuckDB view, macro, and VSS registry — wires a connection to `~/.claude/projects/**/*.jsonl` and exposes the corpus as zero-copy SQL views, analytical macros, and an HNSW-indexed embeddings table (`src/claude_sql/sql_views.py:1`). The 2228-line module owns the `register_raw` + `register_views` + `register_macros` + `register_vss` + `register_analytics` chain (`src/claude_sql/sql_views.py:487`, `:622`, `:1216`, `:1691`, `:1832`) and declares the canonical `VIEW_NAMES` tuple at `src/claude_sql/sql_views.py:62`. v2 analytics views (`session_classifications`, `message_trajectory`, `session_conflicts`, `message_clusters`, `cluster_terms`, `session_communities`) are parquet-gated and skipped with a warning when their parquets are missing.

**Recent activity.** 18 commits in the last 30 days, trend `↑ rising`. Most-recent touches: the v1.0 windowed-pipelines turn-window view (`4eccbae`), the lance-backed embeddings store (`8b14fe3`), the two-tier-connections perf change (`1d067e3`), the static-catalog + `ago()` macro (`aa6d98c`), and the Leiden+CPM swap (`3285082`).

**Owners.** Laith Al-Saadoon — 18/18 = 100%. No co-authors.

**Findings.** `0 warn, 0 error` across all four SARIF gates. The project's `[tool.bandit] skips` includes `B608` because every SQL view DDL is f-string-built (DuckDB rejects prepared parameters as table-function arguments), so the canonical risk surface here is correctness-of-SQL-rendering rather than injection.

### `src/claude_sql/llm_shared.py`

**What's there.** The Bedrock plumbing every Sonnet-classifier pipeline shares — client construction, the retryable `invoke_model` wrapper, the `classify_one` async helper under a concurrency limiter, the per-pipeline `pipeline_cache_stats` accumulator, and the four task-framing system prompts (classify / trajectory / conflicts / friction) (`src/claude_sql/llm_shared.py:1`). Key surfaces: `cacheable_text_block` / `build_system_content_block` (`src/claude_sql/llm_shared.py:86`, `:104`), the tenacity-decorated `_invoke_classifier_sync` (`src/claude_sql/llm_shared.py:395`, `:402`), `BedrockRefusalError` (`src/claude_sql/llm_shared.py:490`), the structured-payload parser (`src/claude_sql/llm_shared.py:500`), and the async `classify_one` dispatcher (`src/claude_sql/llm_shared.py:563`).

**Recent activity.** 15 commits in the last 30 days (following the `llm_worker.py` → `llm_shared.py` rename at `4eccbae`), trend `↑ rising`. Recent commits cover the v1.0 windowed-pipelines rewrite, the cache-floor padding fix (`b53848d`), prompt-caching + filter-Claude-Code-system-markers (`c1b4eab`), and the shared-boto3-client + anyio limiter perf change (`e540b94`).

**Owners.** Laith Al-Saadoon — 14/15 = 93%. One commit by `bonk-ai[bot]` (6%).

**Findings.** `0 warn, 0 error`. The CLAUDE.md banned-API ban on stdlib `logging` is enforced by ruff TID251, so this module routes every retry-state log line through the loguru-native `loguru_before_sleep` helper (`src/claude_sql/llm_shared.py:53`).

### `src/claude_sql/config.py`

**What's there.** Pydantic v2 `BaseSettings` populated from `CLAUDE_SQL_`-prefixed env vars; defaults are picked for a single-user devbox install pointing at `~/.claude/projects/**/*.jsonl` (`src/claude_sql/config.py:1`). The `Settings` class (`src/claude_sql/config.py:127`) owns every cache path, every glob, every concurrency knob, and a model-validator-driven team-corpus override (`src/claude_sql/config.py:150`). Default-factory functions for embeddings, lance, classifications, trajectory, conflicts, clusters, communities, friction, skills-catalog, checkpoint-db, and DuckDB temp-dir paths cover lines `src/claude_sql/config.py:21` through `src/claude_sql/config.py:113`.

**Recent activity.** 21 commits in the last 30 days, trend `↑ rising`. Recent commits include the v1.0 `CLAUDE_SQL_HOME` rewrite (`4eccbae`), the lance-backed embeddings store (`8b14fe3`), the two-tier-connections perf change (`1d067e3`), the Leiden+CPM swap (`3285082`), and the team-corpus root field (`9a19121`).

**Owners.** Laith Al-Saadoon — 21/21 = 100%. No co-authors.

**Findings.** `0 warn, 0 error`. Risk on this file is contract-shape: every public env var that lands here becomes a documented user-facing knob in CLAUDE.md, so PRs that touch `Settings` ripple into operator-facing rollback paths and the README.

### `src/claude_sql/embed_worker.py`

**What's there.** The Cohere Embed v4 backfill worker — discovers messages with no embedding yet, invokes `cohere.embed-v4:0` on Amazon Bedrock in parallel batches (up to 96 texts per call), and appends the resulting vectors to the lance-backed embeddings store (`src/claude_sql/embed_worker.py:1`). Key surfaces: `discover_unembedded` (`src/claude_sql/embed_worker.py:101`), `_invoke_bedrock_sync` under tenacity retry (`src/claude_sql/embed_worker.py:207`, `:217`), the async batch helper `_embed_one_batch` (`src/claude_sql/embed_worker.py:267`), the `embed_documents_async` driver (`src/claude_sql/embed_worker.py:290`), the synchronous `embed_query` for ad-hoc lookups (`src/claude_sql/embed_worker.py:360`), and the public `run_backfill` entry called from `cli.py` (`src/claude_sql/embed_worker.py:391`).

**Recent activity.** 11 commits in the last 30 days, trend `→ flat` (count below median + σ floor of 11.8 — exactly on the boundary). Recent commits include the v1.0 windowed-pipelines rewrite (`4eccbae`), the lance-backed embeddings store rebuild (`8b14fe3`), the test-coverage lift (`c78f7a0`), and the parquet-shard switch (`ca87271`).

**Owners.** Laith Al-Saadoon — 10/11 = 90%. One commit by `bonk-ai[bot]` (9%).

**Findings.** `0 warn, 0 error`. The conservative `MAX_CHARS_PER_TEXT = 50_000` cap (`src/claude_sql/embed_worker.py:53`) keeps payloads below Bedrock's 20 MB body ceiling on a full 96-text batch — the kind of limit a future change could quietly break, which is why the file ranks ahead of larger but lower-velocity workers.

## See also

- [claude-sql · Module map](../architecture/module-map.md) — 10 shared citations
- [claude-sql · Processes](../behavior/processes.md) — 9 shared citations
- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 7 shared citations
- [claude-sql · Public API](../reference/public-api.md) — 5 shared citations
- [claude-sql · System overview](../architecture/system-overview.md) — 4 shared citations
