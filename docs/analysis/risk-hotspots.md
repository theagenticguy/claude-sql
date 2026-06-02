# claude-sql · Risk hotspots

Risk here is composed from two signals the environment supplies: a 30-day code-activity trend and static-analysis finding severity. Activity trend is the follow-aware commit count per file over the window 2026-04-14 to present (the entire 121-commit history of the repo fits inside this ~30-day window), classified as `↑ rising` when a file's commit count exceeds the median+1σ of the per-file distribution (median 3.0, σ 4.25, so the rising cutoff is 8 commits), `→ flat` within one σ, and `↓ falling` below median−1σ. Finding severity is read from the four SARIF reports under `.sarif/` (`bandit.sarif`, `betterleaks.sarif`, `osv.sarif`, `semgrep.sarif`) produced by `mise run security`. The default composite score `2×error + 0.5×warn + 1×rising` is computed per file, with ties broken by raw commit count.

Two signal limitations shape this report. First, all four scanners return zero results — `semgrep.sarif:1` loaded 597 rules and matched nothing, `betterleaks.sarif:1` loaded 269 rules and matched nothing, and `bandit.sarif:1` / `osv.sarif:1` are empty — so the finding-severity term contributes `0` to every file and the ranking is driven entirely by the activity trend. The Open findings column is retained because the signal exists and is verifiable, but it does not discriminate between files. Second, because median−1σ is negative, no file can fall below it, so `↓ falling` is unreachable in this window and no file carries that arrow. The HEAD commit `982595b` relocated every source module from `src/claude_sql/` into five PEP 420 namespace packages under `packages/`, so all activity counts use `git log --follow` to thread history across that rename boundary.

| File | Trend | Open findings | Top owner | Citation |
|---|---|---|---|---|
| `packages/app/src/claude_sql/app/cli.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon 100% | `packages/app/src/claude_sql/app/cli.py:1` (3079 LOC) |
| `packages/core/src/claude_sql/core/config.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon 100% | `packages/core/src/claude_sql/core/config.py:1` (382 LOC) |
| `packages/core/src/claude_sql/core/sql_views.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon 100% | `packages/core/src/claude_sql/core/sql_views.py:1` (2182 LOC) |
| `packages/core/src/claude_sql/core/llm_shared.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon 94% | `packages/core/src/claude_sql/core/llm_shared.py:1` (1341 LOC) |
| `packages/analytics/src/claude_sql/analytics/embed_worker.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon 92% | `packages/analytics/src/claude_sql/analytics/embed_worker.py:1` (507 LOC) |
| `tests/test_sql_views.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon 100% | `tests/test_sql_views.py:1` (743 LOC) |
| `packages/analytics/src/claude_sql/analytics/friction_worker.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon 100% | `packages/analytics/src/claude_sql/analytics/friction_worker.py:1` (741 LOC) |
| `packages/analytics/src/claude_sql/analytics/community_worker.py` | ↑ rising | 0 warn, 0 error | Laith Al-Saadoon 100% | `packages/analytics/src/claude_sql/analytics/community_worker.py:1` (667 LOC) |
| `packages/evals/src/claude_sql/evals/judge_worker.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon 86% | `packages/evals/src/claude_sql/evals/judge_worker.py:1` (462 LOC) |
| `packages/core/src/claude_sql/core/checkpointer.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon 100% | `packages/core/src/claude_sql/core/checkpointer.py:1` (378 LOC) |
| `tests/test_v2_analytics.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon 100% | `tests/test_v2_analytics.py:1` (607 LOC) |
| `packages/core/src/claude_sql/core/schemas.py` | → flat | 0 warn, 0 error | Laith Al-Saadoon 100% | `packages/core/src/claude_sql/core/schemas.py:1` (597 LOC) |

## Per-file drill-down

### `packages/app/src/claude_sql/app/cli.py`

What's there: the Cyclopts CLI entry point that wires the `claude-sql` console script to its thirteen subcommands, with shared flags (`--verbose`/`--quiet`, `--glob`, `--subagent-glob`, `--format`) on a flattened `Common` dataclass (`packages/app/src/claude_sql/app/cli.py:1`). It owns agent-friendly behavior including `--format auto` (human table on a TTY, JSON when piped) and DuckDB error classification into stable exit codes 64/65/70, and imports nearly every worker module across the analytics, core, evals, and app packages (`packages/app/src/claude_sql/app/cli.py:45`). Recent activity: 27 commits over the window, the most of any file, with the latest changes being the namespace-package split (#60), the new `peek` subcommand (#51/#59), and the v1.0 windowed-pipelines rewrite (#42) (`packages/app/src/claude_sql/app/cli.py:1`). Owners: Laith Al-Saadoon at 100% commit share. Findings: 0 warn, 0 error — no SARIF run reported a result for this path (`semgrep.sarif:1`, `bandit.sarif:1`).

### `packages/core/src/claude_sql/core/config.py`

What's there: runtime configuration via a Pydantic v2 `BaseSettings` populated from env vars prefixed `CLAUDE_SQL_`, with devbox defaults pointing at `~/.claude/projects/**/*.jsonl` (`packages/core/src/claude_sql/core/config.py:1`). It centralizes default-path factories for the embeddings store (now LanceDB, with the legacy parquet shards kept only for one-time migration) and for the classifications, trajectory, and conflicts parquet outputs (`packages/core/src/claude_sql/core/config.py:34`, `packages/core/src/claude_sql/core/config.py:42`). Recent activity: 23 commits over the window, touched by the namespace split (#60), the 1.0.0→1.0.1 release bump (#55), and the windowed-pipelines change that introduced `CLAUDE_SQL_HOME` (#42) (`packages/core/src/claude_sql/core/config.py:1`). Owners: Laith Al-Saadoon at 100% commit share. Findings: 0 warn, 0 error (`betterleaks.sarif:1`, `semgrep.sarif:1`). The small LOC (382) against high churn marks this as a high-touch coordination point rather than a complexity hotspot.

### `packages/core/src/claude_sql/core/sql_views.py`

What's there: the DuckDB view, macro, and VSS registry that wires a DuckDB connection to the on-disk `~/.claude/` JSONL corpus and exposes it as zero-copy SQL views, analytical macros, and an HNSW-indexed embeddings table (`packages/core/src/claude_sql/core/sql_views.py:1`). Reads are zero-copy via `read_json(..., filename=true)` with no parquet ingestion, nested `message.content` is flattened at query time, and v2 analytics outputs (classifications, trajectory, conflicts, clusters, communities) are surfaced as parquet-backed views via `register_analytics` (`packages/core/src/claude_sql/core/sql_views.py:9`, `packages/core/src/claude_sql/core/sql_views.py:21`). Recent activity: 22 commits over the window, with the two most recent functional changes both bug fixes — casting `parent_uuid` to VARCHAR so recursive CTEs work (#47/#58) and read-side dedup for `message_trajectory` (#46/#57) (`packages/core/src/claude_sql/core/sql_views.py:1`). Owners: Laith Al-Saadoon at 100% commit share. Findings: 0 warn, 0 error (`semgrep.sarif:1`). The concentration of recent fixes plus 2182 LOC makes this the structurally riskiest core module.

### `packages/core/src/claude_sql/core/llm_shared.py`

What's there: the shared Bedrock plumbing for all per-stage LLM workers — Bedrock client construction, the retryable `invoke_model` wrapper, the `classify_one` async helper under a concurrency limiter, per-pipeline cache-stat accumulation, and the four task-framing system prompts for classify / trajectory / conflicts / user-friction (`packages/core/src/claude_sql/core/llm_shared.py:1`). It is the single hub that `classify_worker`, `trajectory_worker`, `conflicts_worker`, `friction_worker`, and `review_sheet_worker` all import from, with no cross-imports between the stage workers themselves (`packages/core/src/claude_sql/core/llm_shared.py:15`). Recent activity: 16 commits over the window (this file follows from the historical `llm_worker.py` across the rename), with recent work in the windowed-pipelines rewrite (#42) and a coverage lift to 94% across 28 modules (#22) (`packages/core/src/claude_sql/core/llm_shared.py:1`). Owners: Laith Al-Saadoon at 94%, with `bonk-ai[bot]` at 6%. Findings: 0 warn, 0 error (`bandit.sarif:1`, `semgrep.sarif:1`). As the fan-in dependency for five workers, a regression here propagates widely.

### `packages/analytics/src/claude_sql/analytics/embed_worker.py`

What's there: the Cohere Embed v4 backfill worker that discovers messages without embeddings, invokes `cohere.embed-v4:0` on Amazon Bedrock in parallel batches of up to 96 texts, and appends the resulting vectors keyed by message `uuid` (`packages/analytics/src/claude_sql/analytics/embed_worker.py:1`). It converts the int8 Bedrock response to float on insert because DuckDB's VSS HNSW index requires `FLOAT[]` columns, and retries transient Bedrock errors through the shared `loguru_before_sleep` tenacity helper, depending on `lance_store` and `llm_shared._build_bedrock_client` (`packages/analytics/src/claude_sql/analytics/embed_worker.py:6`, `packages/analytics/src/claude_sql/analytics/embed_worker.py:41`). Recent activity: 13 commits over the window, touched by the namespace split (#60), the 1.0.1 release (#55), the windowed-pipelines change (#42), and the Lance-backed embeddings migration (#37) (`packages/analytics/src/claude_sql/analytics/embed_worker.py:1`). Owners: Laith Al-Saadoon at 92%, with `bonk-ai[bot]` at 8%. Findings: 0 warn, 0 error (`osv.sarif:1`, `semgrep.sarif:1`).

## See also

- [claude-sql · Module map](../architecture/module-map.md) — 10 shared source files
- [claude-sql · Contract map](../insights/contract-map.md) — 8 shared source files
- [claude-sql · Tech debt](../insights/tech-debt.md) — 8 shared source files
- [claude-sql · Business logic](../insights/business-logic.md) — 7 shared source files
- [claude-sql · Debugging guide](../insights/debugging-guide.md) — 6 shared source files
