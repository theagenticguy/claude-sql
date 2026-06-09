# claude-sql · Tech debt

This register answers: *where is the rot, and what would I pay to fix it?*

The list is assembled from four sources, in the order the reviewer scanned
them: (1) explicit comment markers (`TODO` / `FIXME` / `HACK` / `XXX` /
`REFACTOR` / `DEPRECATED`) across `src/claude_sql/` and `tests/`; (2) the
documented `# noqa:` suppressions and what each one waives; (3) manifest
version pins to narrow or upstream-blocked ranges in the root
`pyproject.toml`; and (4) pattern-level smells
the reviewer flagged after reading the cited sites. Categories are drawn
from a closed vocabulary (`marker`, `wrong abstraction`, `error handling`,
`dead code adjacent`, `deprecated pattern`, `version pin`,
`duplicated logic`, `missing tests`); cost is `S` / `M` / `L` where
`S = ≤ 1 hour`, `M = a few hours`, `L = a day or more`. Rank is
`cost-to-fix × consequence-of-leaving` — the reviewer's judgment.

The classic-marker count is zero by design. A `grep` for
`\b(TODO|FIXME|HACK|XXX|REFACTOR|DEPRECATED)\b` over `src/claude_sql/` and
`tests/` returns no matches. The repo's `CLAUDE.md` and its
`.erpaval/solutions/` lessons library treat unbacked TODOs as a smell —
debt is named in `docs/BACKLOG.md`, ADRs, and "Deferred decisions"
sections instead of left in source comments. The value of this file is
collecting that named-elsewhere debt, plus the structural smells no comment
names, into one ranked view. The repo is genuinely well-disciplined: there
is no dead commented-out code (ruff `ERA` is enabled), no bare `except:`,
and every broad `except Exception:` either carries a `noqa: BLE001 —
<reason>` or re-raises behind a plain rationale comment.

## Ranked register

| Rank | Debt item | Category | Cost to fix | Citation |
| --- | --- | --- | --- | --- |
| 1 | `_is_retryable` + `_RETRY_CODES` are copy-pasted three times across the `claude_sql.*` layers; `llm_shared`'s docstring even says "Same policy as `embed_worker._is_retryable`". `analytics.embed_worker` could import `core.llm_shared`'s copy (analytics may import core) but keeps its own; `evals.judge_worker` is forced to duplicate by the import-linter independence contract. Drift here silently changes which Bedrock errors retry. | duplicated logic | M | `src/claude_sql/core/llm_shared.py:328`, `src/claude_sql/analytics/embed_worker.py:63`, `src/claude_sql/evals/judge_worker.py:61` |
| 2 | Two divergent Bedrock client builders. `llm_shared._build_bedrock_client` caches on `(region, pool_size)`, uses `read_timeout=600`, `retries.mode='adaptive'`, and `max_pool_connections>=32`. `judge_worker._bedrock_client` is uncached, `read_timeout=120`, `retries.mode='standard'`, no pool sizing — judge traffic gets none of the pool/timeout guards the other five workers share. | duplicated logic | M | `src/claude_sql/core/llm_shared.py:346`, `src/claude_sql/evals/judge_worker.py:221` |
| 3 | `cli.py` is a 3 079-LOC single module hosting 33 command decorators (`grep -cE '\.command' → 33`). No domain seams between subcommand groups; adding or testing a subcommand threads through the whole monolith. | wrong abstraction | L | `src/claude_sql/app/cli.py:164`, `src/claude_sql/app/cli.py:638` |
| 4 | `sql_views.py` is a 2 182-LOC module mixing raw-view registration, derived views, macros, analytics registration, VSS binding, and introspection in one namespace. A natural seam (`raw.py` / `derived.py` / `macros.py` / `analytics.py` / `vss.py`) is unrealized. | wrong abstraction | L | `src/claude_sql/core/sql_views.py:476`, `src/claude_sql/core/sql_views.py:1668` |
| 5 | Python floor pinned at `3.13` solely because `hdbscan` ships no cp314 wheel; ADR 0015 tracks the bump as a one-line PR deferred until `hdbscan 0.8.43+` publishes cp314. Cost is the wait, not the change. | version pin | S | `pyproject.toml:8`, `docs/adr/0015-stack-modernization.md:19` |
| 6 | `lancedb` pinned to a single minor (`>=0.30,<0.31`) — the narrowest range in the dependency closure. Each lancedb minor forces a manual relax-and-retest cycle; the `.tables` / `_has_table` workaround already exists because `db.table_names()` deprecated in 0.30. | version pin | S | `pyproject.toml:35`, `src/claude_sql/core/lance_store.py:54` |
| 7 | Legacy parquet-shard → LanceDB migration runs on every connect against a row-count gate. Idempotent and sentinel-light, but pure carry-cost once users have migrated; it exists only to absorb pre-LanceDB installs. | dead code adjacent | M | `src/claude_sql/core/lance_store.py:158`, `src/claude_sql/core/sql_views.py:1722` |
| 8 | Legacy DuckDB-checkpointer → SQLite migration (`_migrate_from_duckdb_if_present`) runs on every checkpointer open against a sentinel; pure carry-cost once users moved to the SQLite WAL checkpointer. | dead code adjacent | M | `src/claude_sql/core/checkpointer.py:91` |
| 9 | Legacy `~/.claude/` single-file cache discovery + auto-migration fires on every CLI invocation via `_maybe_migrate_legacy_caches`, walking `recognized_legacy_caches()` until a migration marker is dropped. Carry-cost for the parquet-shards-directory transition. | dead code adjacent | M | `src/claude_sql/core/home.py:75`, `src/claude_sql/app/cli.py:273` |
| 10 | Conflicts v1.0 still runs a whole-session LLM prompt; the RFC §4.2 pair-scanner that emits one row per adjacent turn pair "is v1.1 work" — a named deferred feature carrying a more expensive interim implementation. | deprecated pattern | M | `src/claude_sql/analytics/conflicts_worker.py:8` |
| 11 | `binding.py` declares three URI schemes (`file://`, `s3://`, `git-notes://`) but the reference implementation "only emits `file://`; the other two are spec-only entry points for future emitters". Two of three documented schemes are unbuilt. | deprecated pattern | M | `src/claude_sql/provenance/binding.py:72` |
| 12 | `logging_setup.py` (95 LOC) has no colocated `tests/test_logging_setup.py`, yet `loguru_before_sleep` is the single sanctioned escape from the repo's stdlib-`logging` ban — every tenacity `@retry` callback routes through it. A regression here is silent (retry logs vanish). | missing tests | S | `src/claude_sql/core/logging_setup.py:1` |
| 13 | `CLAUDE.md` states `major_version_zero = true` (twice) but `pyproject.toml` has `false` and the package already shipped `1.0.1`. The doc lies about the bump policy; a contributor following it would mispredict the next version. *judgment-call:* a documentation-drift smell, flagged because the misstatement governs release mechanics. | deprecated pattern | S | `CLAUDE.md:283`, `pyproject.toml:226` |

## Explicit markers

The full result of `grep -rnE '\b(TODO|FIXME|HACK|XXX|REFACTOR|DEPRECATED)\b'`
over `src/claude_sql/` and `tests/` is **empty** — zero classic debt
markers. This is a deliberate cultural property, not an oversight (see the
intro). The only marker-shaped artifacts in source are the documented
`# noqa:` suppressions, each of which waives a ruff rule with an inline
reason. They are not debt in the "rot" sense, but they are the closest
verbatim markers the corpus carries, quoted here in full per the format
contract:

- `except Exception as exc:  # noqa: BLE001 — log + skip; the study has 10+ judges` — `src/claude_sql/evals/judge_worker.py:339`
- `except Exception as exc:  # noqa: BLE001 — propagate to caller's retry/skip logic; CancelledError still cancels the task group` — `src/claude_sql/analytics/trajectory_worker.py:647`
- `except Exception as exc:  # noqa: BLE001 — non-cancel exceptions go to retry; CancelledError still tears down the task group` — `src/claude_sql/analytics/trajectory_worker.py:862`
- `except Exception as exc:  # noqa: BLE001 — migration is best-effort` — `src/claude_sql/core/sql_views.py:1728`
- `except Exception:  # noqa: BLE001 — migration is best-effort; any failure must drop the sentinel and let SQLite come up clean` — `src/claude_sql/core/checkpointer.py:159`
- `assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"  # noqa: S101 — input invariant` — `src/claude_sql/evals/kappa_worker.py:63`
- `assert note is not None  # noqa: S101  type-narrow for the type checker` — `src/claude_sql/provenance/binding.py:742`
- `rows.append(  # noqa: PERF401` — `src/claude_sql/evals/ungrounded_worker.py:141`
- `# Planning (dry-run)  # noqa: ERA001 — section header, not commented-out code` — `src/claude_sql/evals/judge_worker.py:265`
- `# v2: TF-IDF  # noqa: ERA001 — section header, not commented-out code` — `src/claude_sql/core/config.py:316`
- `del embeddings_parquet_path  # legacy kwarg — view is the source of truth now` — `src/claude_sql/analytics/community_worker.py:97`

## Pattern-level smells

### Triplicated retry-classification logic

Three of the `claude_sql.*` layers each define their own `_is_retryable(exc)`
predicate plus a `_RETRY_CODES` set that names which Bedrock error codes are
worth retrying.
`llm_shared`'s docstring openly acknowledges the copy ("Same policy as
`embed_worker._is_retryable`"). The structural cause is the import-linter
independence contract — `analytics`, `evals`, and `provenance` are declared
mutually independent siblings, so `evals.judge_worker` *cannot* import the
`analytics` copy and is forced to duplicate. But `core` sits below all
three, so `analytics.embed_worker` could import `core.llm_shared._is_retryable`
and chooses not to. The debt is that the canonical home (`core.llm_shared`)
exists and is under-used: a fix would move the predicate + codes into `core`
and have every worker import it, collapsing three sites to one. Drift is the
live risk — adding a retryable error code to one copy and not the others
changes retry behaviour silently per pipeline.

Shows up in:
- `src/claude_sql/core/llm_shared.py:328` — the `core` copy (with `_RETRY_CODES` at `:61`).
- `src/claude_sql/analytics/embed_worker.py:63` — avoidable duplicate (analytics may import core; `_RETRY_CODES` at `:55`).
- `src/claude_sql/evals/judge_worker.py:61` — contract-forced duplicate (`_RETRY_CODES` at `:52`).

Cost: M.

### Divergent Bedrock client builders

Two private functions build the boto3 `bedrock-runtime` client and have
forked behaviour. `llm_shared._build_bedrock_client` is process-cached on
`(region, pool_size)`, sizes `max_pool_connections` to at least
`2 × max(embed_concurrency, llm_concurrency)` with a floor of 32, sets
`read_timeout=600` (Sonnet 4.6 with adaptive thinking holds the connection),
and uses `retries.mode='adaptive'`. The evals package's
`judge_worker._bedrock_client` is uncached, sets `read_timeout=120`,
`retries.mode='standard'`, and never sizes the connection pool — so a
high-fanout judge study ("10+ judges", per the worker's own comment) runs
against botocore's default 10-connection pool and a shorter timeout. The
two have drifted on every tuning axis that matters under load. The smell is
duplication that has actively forked, not two identical functions; the fix
is to lift the builder into `core` and have evals consume it.

Shows up in:
- `src/claude_sql/core/llm_shared.py:346` — cached, pool-aware, 600 s timeout, adaptive retries.
- `src/claude_sql/evals/judge_worker.py:221` — uncached, 120 s timeout, standard retries, no pool sizing.
- `src/claude_sql/evals/judge_worker.py:382` — the judge call site that gets the un-pooled client.

Cost: M.

### Single-module CLI and views surfaces

Two files hold a large share of the package source. `cli.py` (3 079 LOC) is
the single entry point for every subcommand — `analyze`, `embed`,
`classify`, `trajectory`, `community`, `cache`, `skills`, `query`, etc. —
with 33 command decorators counted directly. `sql_views.py` (2 182 LOC)
mixes `register_raw`, derived views, `register_macros`, analytics
registration, `register_vss`, and introspection in one namespace. Both
predate v1.0; both register cleanly under ruff/ty; neither is broken. The
smell is *load*: adding a view, a macro, or a subcommand has to thread
through these monoliths, and the lack of seams makes unit-testing one
subcommand in isolation awkward. A natural seam is a `cli/` package with one
module per subcommand group and a `views/` package split into `raw.py` /
`derived.py` / `macros.py` / `analytics.py` / `vss.py`.

Shows up in:
- `src/claude_sql/app/cli.py:164` — the `App(...)` declaration.
- `src/claude_sql/app/cli.py:638` — first `@app.command` block (~32 more follow).
- `src/claude_sql/core/sql_views.py:476` — `register_raw` start.
- `src/claude_sql/core/sql_views.py:1668` — `register_vss` lives ~1 200 LOC into the same file.

Cost: L.

### Legacy-migration code paths permanently in the boot path

Three legitimate one-time migrations sit in modules that run on every
connect or invocation: parquet-shards → LanceDB
(`lance_store.migrate_from_parquet_shards`, called from `register_vss`),
DuckDB checkpointer → SQLite (`checkpointer._migrate_from_duckdb_if_present`),
and legacy `~/.claude/` single-file caches → sharded directories
(`home.recognized_legacy_caches` consumed by `cli._maybe_migrate_legacy_caches`).
Each is sentinel-guarded or row-count-gated and idempotent — no functional
bug — but each is pure carry-cost: combined a few hundred LOC plus a probe
per startup confirming the migration already ran. *judgment-call:* this is
debt because the upgrade-comms cost is already paid and the target shapes
have shipped. One release of overlap is reasonable; carrying all three
indefinitely is sediment. The fix is to gate each behind an explicit
`claude-sql cache migrate` opt-in and delete the on-boot probes after a
deprecation window.

Shows up in:
- `src/claude_sql/core/lance_store.py:158` — parquet → Lance migration body.
- `src/claude_sql/core/sql_views.py:1722` — on-connect call site inside `register_vss`.
- `src/claude_sql/core/checkpointer.py:91` — DuckDB → SQLite checkpointer migration.
- `src/claude_sql/app/cli.py:273` — `_maybe_migrate_legacy_caches`, runs on CLI startup.

Cost: M.

### Named-but-unbuilt feature surfaces

The codebase carries documented entry points for features that are
specified but not implemented, each a small interface-surface liability
(callers may assume they work). The conflicts pipeline still runs an
expensive whole-session LLM prompt because the cheaper RFC §4.2 pair-scanner
"is v1.1 work". `binding.py` declares three transcript-URI schemes but the
reference implementation only emits `file://` — `s3://` and `git-notes://`
are "spec-only entry points for future emitters". None is a bug; the smell is
that the spec surface is wider than the built surface, so the gap has to be
tracked out-of-band or it gets re-derived.

Shows up in:
- `src/claude_sql/analytics/conflicts_worker.py:8` — pair-scanner deferred to v1.1.
- `src/claude_sql/provenance/binding.py:72` — `s3://` / `git-notes://` URI schemes spec-only.

Cost: M.

## See also

- [claude-sql · Module map](../architecture/module-map.md) — 12 shared source files
- [claude-sql · Debugging guide](debugging-guide.md) — 11 shared source files
- [claude-sql · Contract map](contract-map.md) — 10 shared source files
- [claude-sql · Processes](../behavior/processes.md) — 10 shared source files
- [claude-sql · Public API](../reference/public-api.md) — 10 shared source files
