# claude-sql · Tech debt

This register is assembled from four sources, in decreasing order of objectivity: (1) explicit comment markers grepped from source (`TODO`/`FIXME`/`HACK`/`XXX`/`REFACTOR` and case-insensitive variants); (2) stale references left by the hexagonal reshape (commits `4038edb..7670b4c`) — docstrings and manifest strings pointing at packages that no longer exist; (3) manifest version pins and layering-contract observations; (4) pattern-level smells the reviewer chose to flag, each backed by 2–5 representative sites.

Two facts shape this file. First, **the codebase carries zero action markers** — a `grep` for `\bTODO\b`/`\bFIXME\b`/`\bHACK\b`/`\bXXX\b`/`\bREFACTOR\b` across `src/`, `tests/`, and `proofs/` returns nothing; the only hits for broader terms ("bug location", "deprecated kwargs") are descriptive prose, not work markers. The Explicit markers section is therefore short by design, and it reflects a genuine "no TODO in code" discipline rather than an incomplete scan. Second, most of the real debt is **reshape residue**: the move to `domain/application/infrastructure/interfaces` landed cleanly at the module level, but a layer of docstrings, comments, and one manifest string still name the deleted `core`/`analytics`/`app` packages, and several use-cases kept direct infrastructure calls as pragmatic monkeypatch seams rather than routing through the new ports.

Category vocabulary is closed: `marker`, `wrong abstraction`, `error handling`, `dead code adjacent`, `deprecated pattern`, `version pin`, `duplicated logic`, `missing tests`. Cost is `S`/`M`/`L`. Rank is the reviewer's product of cost-to-fix and consequence-of-leaving.

## Ranked register

| Rank | Debt item | Category | Cost to fix | Citation |
|---|---|---|---|---|
| 1 | Banned-api message points at nonexistent `claude_sql.core.logging_setup` — a developer who trips the `logging` ban is told to import from a deleted module | deprecated pattern | `S` | `pyproject.toml:175` |
| 2 | `domain/costs.py` docstring claims it "moved out of `core.llm_shared` (which is now a back-compat shim)" and that `_estimate_cost` is "re-exported from `core.llm_shared`" — the `core` package is deleted, no such shim exists | deprecated pattern | `S` | `src/claude_sql/domain/costs.py:9` |
| 3 | `domain/friction.py` docstring claims re-export "from `analytics.friction_worker` (a back-compat shim)" — the `analytics` package is deleted | deprecated pattern | `S` | `src/claude_sql/domain/friction.py:15` |
| 4 | pyproject build/importlinter comments still describe a "transitional `core`" package and "transitional `core` package is gone (dissolved in T-8-2)" side by side — mixed present/past tense about the same removed package | dead code adjacent | `S` | `pyproject.toml:77`, `pyproject.toml:281` |
| 5 | `application` layer imports `infrastructure` directly in 9 of 16 use-case modules — the layers contract permits top-down so `lint:imports` stays green, but the ports abstraction is bypassed for concrete infra symbols | wrong abstraction | `L` | `src/claude_sql/application/use_cases/trajectory.py:69`, `src/claude_sql/application/use_cases/embed.py:31`, `src/claude_sql/application/use_cases/classify.py:38` |
| 6 | `terms`/`community`/`skills` use-cases take a raw `duckdb` connection and import `infrastructure.settings`/`skills_fs` directly — no port fit; three analytics stages sit outside the hexagon's dependency-inversion story | wrong abstraction | `M` | `src/claude_sql/application/use_cases/terms.py:31`, `src/claude_sql/application/use_cases/community.py:228`, `src/claude_sql/application/use_cases/skills.py:31` |
| 7 | `session_bounds`/`filter_unchanged` called as module-level functions (not through `TranscriptReaderPort`) to preserve `monkeypatch.setattr(worker, "session_bounds", ...)` seams — the port method exists but is opt-in, so tests pin the pre-hexagon call shape | wrong abstraction | `M` | `src/claude_sql/application/use_cases/classify.py:72`, `src/claude_sql/application/use_cases/trajectory.py:590`, `src/claude_sql/application/use_cases/friction.py:360` |
| 8 | CI runs no proof gate — `mise run check` depends on `proofs` (`lake build`), but no GitHub workflow invokes lean/lake; the Lean 4 invariants only gate on a developer's local machine | missing tests | `M` | `mise.toml:126`, `mise.toml:140` |
| 9 | `duckdb_views.py` is 2394 LOC of view/macro DDL in one module — the largest file in the tree; any view change means navigating a 2.4K-line wall | wrong abstraction | `L` | `src/claude_sql/infrastructure/duckdb_views.py:1` |
| 10 | `interfaces/cli/app.py` is 2121 LOC with 58 command/function definitions, including `cache compact`/`cache migrate` business logic that belongs in a use-case, not the CLI shell | wrong abstraction | `M` | `src/claude_sql/interfaces/cli/app.py:1015`, `src/claude_sql/interfaces/cli/app.py:57` |
| 11 | `Settings` is a 66-field `BaseSettings` god-object — every config concern (DuckDB tuning, Bedrock, Lance, friction, legacy paths) in one flat namespace | wrong abstraction | `L` | `src/claude_sql/infrastructure/settings.py:1` |
| 12 | `strands_luna` except block is commented "Fail open" but re-raises `LlmAnalyticsUnavailable` — the comment contradicts the control flow it annotates | error handling | `S` | `src/claude_sql/infrastructure/llm_analytics/strands_luna.py:126` |
| 13 | Legacy single-file parquet back-compat path threaded through settings, ports, and cache — a permanent maintenance tax for a one-time migration that new installs never hit | dead code adjacent | `M` | `src/claude_sql/infrastructure/parquet_cache.py:23`, `src/claude_sql/infrastructure/settings.py:61`, `src/claude_sql/application/ports.py:223` |
| 14 | Legacy `embeddings/part-*.parquet` → Lance migration kept "in place for rollback" — dead on every fresh install, live only for pre-Lance upgraders | dead code adjacent | `M` | `src/claude_sql/infrastructure/lance_store.py:169`, `src/claude_sql/infrastructure/lance_store.py:45` |
| 15 | Rebrand pending — every CLI doc example is literally `claude-sql <cmd>`, and `SECURITY.md` is still the stock GitHub template with placeholder version table; a rename touches all four CLI-doc files | deprecated pattern | `S` | `docs/v2/understanding/04-cli-interfaces.md:262`, `docs/v2/understanding/07-tests-ci-build-docs.md:240` |
| 16 | `logging_setup.loguru_before_sleep` has no dedicated test — the loguru-backed tenacity retry adapter (referenced 6× in src) is exercised only incidentally | missing tests | `S` | `src/claude_sql/infrastructure/logging_setup.py:1` |
| 17 | `numpy` pinned `>=2.4.4,<2.5` with an inline comment that the ceiling exists solely to stop uv from backsolving `numba` down to 0.53.1 — a transitive-resolver workaround baked into a direct pin | version pin | `S` | `pyproject.toml:38` |

## Explicit markers

The action-marker vocabulary (`TODO`/`FIXME`/`HACK`/`XXX`/`REFACTOR`) returns **zero hits** across `src/`, `tests/`, and `proofs/`. The bullets below are the only comment-embedded references that name a defect or a deprecation; each is descriptive prose the team left in place, not an outstanding work item.

- `` # RFC §9.6: this is THE explicitly-named bug location -- community `` — `src/claude_sql/application/analyze.py:246`
- `` # ``metric=/vector_column_name=/index_type=`` kwargs are deprecated (they `` — `src/claude_sql/infrastructure/lance_store.py:138`
- `` # answer-less payload (see the conflicts retry-queue silent-drop bug, `` — `src/claude_sql/infrastructure/settings.py:293`
- `` # prior corpus state or a future write-path bug ever lands duplicate `` — `src/claude_sql/infrastructure/duckdb_views.py:2116`
- `` deprecated ``/api/embeddings``): ``/api/embed`` is batch-capable (``input`` `` — `src/claude_sql/infrastructure/embedding/ollama.py:4`

## Pattern-level smells

### Layer-crossing shortcuts that defeat the ports abstraction

The reshape declared four layers with `application` depending on `domain` ports, and adapters in `infrastructure` implementing them. In practice 9 of 16 use-case modules import concrete `infrastructure` symbols at runtime, and three analytics use-cases (`terms`/`community`/`skills`) take a raw `duckdb` connection plus a direct `infrastructure.settings` import — no port sits between them and the database. The import-linter `layers` contract is top-down, so these calls are legal and `lint:imports` stays green; the debt is that the dependency-inversion the hexagon was supposed to buy is only realized for the embedding and LLM-analytics providers, not for the SQL/analytics plane. This is the load-bearing architectural gap, not a cosmetic one.

Shows up in:
- `src/claude_sql/application/use_cases/trajectory.py:69` (7 infra imports)
- `src/claude_sql/application/use_cases/terms.py:31` (raw `con` + domain-math call, no port)
- `src/claude_sql/application/use_cases/community.py:228` (`run_communities(con, settings)`)
- `src/claude_sql/application/use_cases/skills.py:31` (direct `infrastructure.skills_fs` import)
- `pyproject.toml:298` (the layers contract that permits it)

Cost: `L`

### Monkeypatch-preserving direct calls instead of injected ports

`classify`, `trajectory`, and `friction` each define an optional `reader` port parameter but fall back to calling the module-level `session_bounds(con, ...)` / `checkpointer.filter_unchanged(...)` functions directly when it's absent. The code comments are explicit that this is to keep existing `monkeypatch.setattr(worker, "session_bounds", ...)` tests biting (`classify.py:72-73`). The port exists (`application/ports.py:100`) but the default path never uses it, so the test suite is pinning the pre-hexagon call shape and the injection seam is dead weight until the tests are rewritten.

Shows up in:
- `src/claude_sql/application/use_cases/classify.py:76` (`reader.session_bounds` vs direct)
- `src/claude_sql/application/use_cases/trajectory.py:593`
- `src/claude_sql/application/use_cases/friction.py:362`
- `src/claude_sql/application/ports.py:100` (the port method that goes unused by default)

Cost: `M`

### Stale references to deleted packages

The reshape deleted `core/`, `analytics/`, and `app/`, but a scatter of docstrings, comments, and one manifest string still name them as if they were live. The most dangerous is `pyproject.toml:175`: the `logging`-ban error message instructs developers to import `claude_sql.core.logging_setup.loguru_before_sleep` — a module that no longer exists at that path (it now lives under `infrastructure/`). The domain docstrings claiming re-export "from `core.llm_shared` (a back-compat shim)" and "from `analytics.friction_worker` (a back-compat shim)" describe shims that were removed, so a reader trusting the docstring will look for code that isn't there.

Shows up in:
- `pyproject.toml:175` (banned-api hint → deleted module path)
- `src/claude_sql/domain/costs.py:9` (`core.llm_shared` shim, gone)
- `src/claude_sql/domain/friction.py:15` (`analytics.friction_worker` shim, gone)
- `pyproject.toml:77` and `pyproject.toml:281` (present-tense "transitional core" beside past-tense "core is gone")

Cost: `S`

### Permanent back-compat tax for one-time migrations

Two migration paths — legacy single-file parquet caches, and the pre-Lance `embeddings/part-*.parquet` directory — are threaded through settings fields, port docstrings, cache logic, and a dedicated `migrate_legacy_*` function. Both are "left in place for rollback" and fire only for users upgrading from a prior layout; a fresh install exercises neither. The code is correct, but it is a standing maintenance surface (extra Settings fields, branchy `is_file()` vs directory logic, a CLI `cache migrate` command) that will never retire on its own. *Judgment-call*: flagged because the rollback window for these migrations has no stated expiry, so the "temporary" branches are effectively permanent.

Shows up in:
- `src/claude_sql/infrastructure/parquet_cache.py:23` (file-vs-dir legacy branch)
- `src/claude_sql/infrastructure/lance_store.py:169` (`migrate_legacy` copy)
- `src/claude_sql/infrastructure/settings.py:61` (legacy shard-dir field kept "for one-time migration only")
- `src/claude_sql/interfaces/cli/app.py:57` (`_maybe_migrate_legacy_caches` re-exported only for tests)

Cost: `M`

### Oversized modules concentrating change risk

Three modules dominate the LOC distribution: `duckdb_views.py` (2394 LOC, all view/macro DDL), `app.py` (2121 LOC, 58 command/function defs), and `trajectory.py` (903 LOC). The CLI file additionally carries `cache compact`/`cache migrate` business logic that belongs in a use-case rather than the interfaces shell. Large single-purpose DDL files are somewhat inherent to a SQL-heavy tool, but the concentration means any view or command change forces navigation of a multi-thousand-line file, and the CLI's embedded compaction logic is a layering leak (business logic in the interface layer).

Shows up in:
- `src/claude_sql/infrastructure/duckdb_views.py:1` (2394 LOC)
- `src/claude_sql/interfaces/cli/app.py:1015` (`cache_compact` logic in the CLI)
- `src/claude_sql/application/use_cases/trajectory.py:1` (903 LOC)
- `src/claude_sql/infrastructure/settings.py:1` (66 fields in one flat `BaseSettings`)

Cost: `L`

## See also

- [claude-sql · Processes](../behavior/processes.md) — 10 shared source citations
- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 9 shared source citations
- [claude-sql · Contract map](../insights/contract-map.md) — 7 shared source citations
- [claude-sql · Module map](../architecture/module-map.md) — 5 shared source citations
- [claude-sql · Sequences](../diagrams/behavioral/sequences.md) — 5 shared source citations
