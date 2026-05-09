# ERPAVal lessons index

Lessons learned from prior ERPAVal sessions. Claude reads this at
session start and greps `.erpaval/solutions/**` for relevant
lessons before starting work.

## By category

### api-patterns/

- [DuckDB ATTACH catalog check](solutions/api-patterns/duckdb-attach-catalog-check.md) — ATTACH on a fresh path leaves a 12 KB header-only file; gate rebuild on `duckdb_tables()`, not file size.
- [DuckDB memory_limit % rejection](solutions/api-patterns/duckdb-memory-limit-percent.md) — `'70%'` raises Parser Error; resolve to MiB at apply time via `os.sysconf`.
- [DuckDB list_avg is per-row](solutions/api-patterns/duckdb-list-avg-aggregate.md) — for centroid math use `unnest+generate_subscripts+groupby+list ORDER BY pos`.
- [Bedrock shared client + pool sizing](solutions/api-patterns/bedrock-shared-client-and-pool.md) — one shared boto3 client with `max_pool_connections=max(32, 2×conc)`; default of 10 silently caps concurrency.
- [Anthropic tokenizer gap + cache minimums](solutions/api-patterns/anthropic-prompt-cache-tokenizer-gap.md) — cl100k_base overcounts Anthropic tokens ~22%; Haiku 4.5 cache min is 4096 vs Sonnet's 1024.
- [Global CRIS not in ListInferenceProfiles](solutions/api-patterns/cris-profile-not-in-list.md) — `global.<vendor>.<model>` may invoke fine even when absent from the profile list; verify by invoke, not listing.
- [ty strict mode is `rules.all = "error"`](solutions/api-patterns/ty-strict-mode-all-error.md) — ty has no `--strict` flag; the canonical strict knob is `[tool.ty.rules] all = "error"` + `error-on-warning`, minus `division-by-zero`.
- [hdbscan cp314 wheel gap](solutions/api-patterns/hdbscan-cp314-wheel-gap.md) — 0.8.42 is the lone Python 3.14 blocker in the scientific-python closure; sdist fallback is unacceptable for `uv tool install` end-users.
- [Lefthook pre-push `HEAD~` fallback](solutions/api-patterns/lefthook-push-gate-fallback.md) — `@{push}` fails with exit 128 on first-push-of-branch; keep the `|| git diff --name-only HEAD~` fallback.
- [Semgrep in CI: container, not deprecated action](solutions/api-patterns/semgrep-ci-container-not-action.md) — `returntocorp/semgrep-action` is deprecated; `semgrep ci` rejects `--config`. Run `semgrep scan` inside the `semgrep/semgrep` container.
- [CycloneDX SBOM for uv projects](solutions/api-patterns/cyclonedx-python-uv-environment.md) — cdxgen + `uv export` don't speak uv.lock; `cyclonedx-py environment .venv` is the working path.
- [Bedrock structured output: `output_config.format` ≠ Converse](solutions/api-patterns/bedrock-output-config-vs-converse.md) — two API shapes with different field paths and refusal vocabularies; stay on InvokeModel + `output_config.format` to match the existing pipeline (refusal stop_reason, prompt caching, adaptive thinking).
- [PR↔transcript binding: trailers + git notes from `prepare-commit-msg`](solutions/api-patterns/git-trailer-and-notes-binding.md) — commit-trailers + `refs/notes/transcripts` is the survivable, host-agnostic primitive; run from `prepare-commit-msg` (not `commit-msg` or `pre-commit`).
- [bandit `-f sarif` needs the `[sarif]` extra](solutions/api-patterns/bandit-sarif-formatter-extras.md) — without `bandit[sarif]`, `-f sarif` raises `ModuleNotFoundError` at runtime, not install time. Pair with `[tool.bandit] skips` aligned 1:1 with ruff's S-ignores.
- [betterleaks isn't on PyPI — Go binary, install via mise `aqua:`](solutions/api-patterns/betterleaks-not-on-pypi.md) — `uv add betterleaks` fails (not a Python package); use `aqua:betterleaks/betterleaks` locally + raw curl + `tar -xzf` in CI. Filename pattern is `betterleaks_<version>_linux_x64.tar.gz` — note the embedded version.
- [Claude Code tool taxonomy migrated 2026 (v2.1.16/v2.1.63)](solutions/api-patterns/claude-code-tool-taxonomy-2026.md) — `Task`→`Agent` rename and `TodoWrite`→`TaskCreate` family split for interactive sessions; split DuckDB views by semantic (launcher vs tracker), keep `TodoWrite` for `--print`/SDK, COALESCE `taskId`/`id` across native + mcp variants.

### best-practices/

- [Sharded cache field-name stability](solutions/best-practices/sharded-cache-field-name-stability.md) — when a path's storage shape shifts (file→dir), reinterpret semantics in a helper, don't rename the Settings field.
- [Anthropic canonical XML tags](solutions/best-practices/anthropic-xml-canonical-tags.md) — use `<instructions>`, `<context>`, `<examples><example>`, `<anti_patterns>`; avoid invented tags like `<task>`.
- [anyio for blocking-IO async pipelines](solutions/best-practices/anyio-structured-concurrency-for-blocking-io.md) — CapacityLimiter + `to_thread.run_sync` propagate cancellation to the thread pool, unlike `asyncio.to_thread`.
- [SARIF scanners: split report vs gate](solutions/best-practices/sarif-scanner-report-vs-gate.md) — run scanner with `|| true` + SARIF → upload → re-run to gate. Decouples code-scanning visibility from CI blocking.
- [Settings derivation via model_validator — never break user pins](solutions/best-practices/team-corpus-glob-via-model-validator.md) — when one Settings field re-derives others, compare against factory output (not literal strings) to detect "user pinned"; mutate via `object.__setattr__`.
- [Stacked PRs need a synchronize event after retarget](solutions/best-practices/branch-protection-pr-targeting-ci-trigger.md) — workflows filter on `branches: [main]`; `gh pr edit --base main` alone doesn't trigger CI; push an empty `ci: trigger` commit.
- [SARIF uploads need a unique `category:` per scanner since 2025-07-22](solutions/best-practices/sarif-category-uniqueness-since-2025-07.md) — GitHub stopped merging same-tool SARIF runs; collisions silently overwrite findings without an error. One unique `category:` per scanner per matrix leg.

## Recent additions

- 2026-05-09 — Claude Code tool-taxonomy audit (session-38d7e6 → chore/tool-taxonomy-2026): 1 new lesson capturing the v2.1.16 `TodoWrite`→`TaskCreate`-family split for interactive sessions and the v2.1.63 `Task`→`Agent` rename. Pairs with ADR 0017 + new DuckDB views (`subagent_spawns` / `task_creations` / `task_updates` / `tasks_state_current`).
- 2026-05-09 — Security hardening (PR #15 → chore/security-hardening): 3 new lessons on bandit's `[sarif]` extra requirement, betterleaks not being on PyPI (Go binary install pattern), and the post-2025-07 SARIF category-uniqueness rule.
- 2026-05-09 — Strategy actions 1+2+3 implementation (session-2293a5 → 4 PRs: #10 strategy memo, #11 team-corpus, #12 binding, #13 review-sheet): 4 new lessons on git-trailer + notes binding, Bedrock InvokeModel-vs-Converse, Settings derivation via model_validator, and stacked-PR CI triggers.
- 2026-05-08 — CI-hardening session (session-c4635d → chore/ci-hardening): 3 new lessons on Semgrep-in-CI (container not deprecated action), CycloneDX SBOM for uv projects, and SARIF scanner report-vs-gate split.
- 2026-05-08 — Stack-modernization session (session-638df1 → chore/stack-modernization): 3 new lessons on ty strict mode, hdbscan as the lone 3.14 blocker, and the lefthook first-push `@{push}` gotcha.
- 2026-05-08 — DuckDB perf + LLM prompt-quality session (sessions session-f988cf → fix/llm-prompt-quality): 5 new lessons on Bedrock concurrency, tokenizer gotchas, CRIS resolution, XML prompting, and anyio patterns.
- 2026-05-07 — Initial perf optimization: 4 lessons on DuckDB ATTACH, memory_limit, list_avg semantics, and sharded-cache migration strategy.
