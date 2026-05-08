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

### best-practices/

- [Sharded cache field-name stability](solutions/best-practices/sharded-cache-field-name-stability.md) — when a path's storage shape shifts (file→dir), reinterpret semantics in a helper, don't rename the Settings field.
- [Anthropic canonical XML tags](solutions/best-practices/anthropic-xml-canonical-tags.md) — use `<instructions>`, `<context>`, `<examples><example>`, `<anti_patterns>`; avoid invented tags like `<task>`.
- [anyio for blocking-IO async pipelines](solutions/best-practices/anyio-structured-concurrency-for-blocking-io.md) — CapacityLimiter + `to_thread.run_sync` propagate cancellation to the thread pool, unlike `asyncio.to_thread`.

## Recent additions

- 2026-05-08 — DuckDB perf + LLM prompt-quality session (sessions session-f988cf → fix/llm-prompt-quality): 5 new lessons on Bedrock concurrency, tokenizer gotchas, CRIS resolution, XML prompting, and anyio patterns.
- 2026-05-07 — Initial perf optimization: 4 lessons on DuckDB ATTACH, memory_limit, list_avg semantics, and sharded-cache migration strategy.
