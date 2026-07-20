# claude-sql · Business logic

This file indexes the domain rules baked into `claude-sql` — the validations, invariants, calculations, and policy gates that shape behavior beyond what interface shapes alone reveal.

**Scope.** After the v2 hexagonal reshape (`feat/v2-hexagonal`), the rules concentrate in the innermost hexagon: `src/claude_sql/domain/`. This document covers those pure domain rules plus the application-layer enforcement sites that invoke them (`src/claude_sql/application/`) and the thin adapter/interface guards that translate them to wire contracts (`src/claude_sql/infrastructure/`, `src/claude_sql/interfaces/`). Four invariants are additionally **machine-checked by Lean 4 proofs** under `proofs/` (core Lean, no mathlib, wired into `mise run check`); those are cited as proven rules, not just asserted ones.

In scope: pydantic v2 classification schemas (the LLM-analytics contract), the dedup/dim/friction/cost domain math, the transcript-collapse ordering rules, the retry-queue backoff, the CLI error taxonomy, and the two SQL sites that enforce a domain rule (near-dup self-join, embedding-store guard). Out of scope: the DuckDB view / macro definitions in `infrastructure/duckdb_views.py` (that surface belongs to the contract/reference docs), and database-level DDL except the one `retry_queue` primary key that shapes application dedup behavior. There is no relational database with migrations — persistence is parquet shards, a LanceDB dataset, and a SQLite state file. The old `core/`, `analytics/`, `evals/`, and `provenance/` packages are gone; they are not documented here.

## Validations

The LLM-analytics plane is validated entirely through pydantic v2 models (`domain/models.py`) — six structured-output schemas, each `extra="forbid"`, whose field-level `Literal` / range / length constraints ARE the validation layer. Bedrock's `output_config.format` enforces the flattened JSON-schema on generation; pydantic re-validates on parse. User-input validation is a thin CLI guard layer.

| Rule | Domain | Citation | Failure mode |
|---|---|---|---|
| Session `autonomy_tier` ∈ {manual, assisted, autonomous} | Classification | `domain/models.py:34` | Reject (schema/parse) |
| Session `work_category` ∈ {sde, admin, strategy_business, events, thought_leadership, other} | Classification | `domain/models.py:44` | Reject |
| Session `success` ∈ {success, partial, failure, unknown} | Classification | `domain/models.py:65` | Reject |
| Session `goal` length 1–280 chars | Classification | `domain/models.py:75` | Reject |
| Session `confidence` in [0.0, 1.0] | Classification | `domain/models.py:85` | Reject |
| Trajectory `curr_uuid` non-empty; must match a supplied `<window>` uuid | Trajectory | `domain/models.py:119` | Reject; host echo-verify re-runs missing windows |
| Trajectory `prev_sentiment` / `curr_sentiment` ∈ {negative, neutral, positive} | Trajectory | `domain/models.py:128`, `:137` | Reject |
| Trajectory `transition_kind` ∈ 6-value enum (frustration_spike, resolution, reset, drift, clarification, none) | Trajectory | `domain/models.py:158` | Reject at schema; coerce to `none` at row-build if out-of-set |
| Trajectory `confidence` in [0.0, 1.0] | Trajectory | `domain/models.py:177` | Reject |
| Conflict `turn_a_uuid` / `turn_b_uuid` length 1–64, copied verbatim, must differ | Conflicts | `domain/models.py:223`, `:234` | Reject |
| Conflict `conflict_kind` ∈ {disagreement, correction, reversal, impasse} | Conflicts | `domain/models.py:244` | Reject |
| Conflict `severity` ∈ {low, medium, high} | Conflicts | `domain/models.py:257` | Reject |
| Conflict `agent_position` / `user_position` length 1–280 | Conflicts | `domain/models.py:268`, `:278` | Reject |
| Friction `label` ∈ 7-value enum (status_ping, unmet_expectation, confusion, interruption, correction, frustration, none) | Friction | `domain/models.py:340` | Reject |
| Friction `rationale` length 1–200 | Friction | `domain/models.py:370` | Reject |
| Friction / conflict `confidence` in [0.0, 1.0] | Friction / Conflicts | `domain/models.py:380`, `:287` | Reject |
| `--glob` may contain at most one `**` recursive segment | CLI input | `interfaces/cli/output.py:130` | `InputValidationError` → exit 64 with hint |
| Team-corpus globs rewritten from `team_corpus_root` unless a per-glob user pin differs from factory default | Config | `infrastructure/settings.py:430` | User pin wins; no rewrite |
| Retry-queue `pipeline` must be a known `PIPELINE_NAMES` value | Retry queue | `infrastructure/sqlite_state/retry_queue.py:98` | `raise ValueError` |
| Embedding provider selector must be a known provider | Embedding | `infrastructure/embedding/__init__.py:57` | `raise ValueError` |
| Lance vector-store metric must be supported | Embedding | `infrastructure/duckdb_views.py:1885` | `raise ValueError` |

**Friction pre-filter.** Only user-role messages of length ≤ `friction_max_chars` (default 300) are classified at all; longer messages are treated as genuine task turns and skipped. This is a SQL `WHERE length(mt.text_content) <= {max_chars}` guard (`application/use_cases/friction.py:282`), keeping Bedrock cost linear in the interesting slice.

## Invariants

Four invariants are formally proven in Lean 4 under `proofs/` and gated by `mise run check`; the rest are enforced in Python or SQL. The proven set is marked "Lean-proven".

| Invariant | Where enforced | Citation |
|---|---|---|
| Hamming distance is reflexive (`d(x,x)=0`), symmetric (`d(x,y)=d(y,x)`), and bounded by width 64 — so the `< 3` near-dup test is always meaningful | Lean-proven; Python impl at `domain/dedup.py:179` | `proofs/ClaudeSql/Hamming.lean:38`, `:46`, `:66` |
| Retry backoff is capped at 60 min for any attempt count, is monotone non-decreasing in attempts, and saturates at exactly 60 once `2^a ≥ 60` — a runaway counter can never schedule a retry years out | Lean-proven; Python impl at `retry_queue.py:79` | `proofs/ClaudeSql/Backoff.lean:26`, `:30`, `:36` |
| The transcript sort key `(ts, kind_rank, uuid)` is a total order on distinct rows (uuid tiebreak is load-bearing); the `(ts, kind_rank)`-only key is NOT connex, so dropping uuid would let DuckDB scan order leak in and break determinism | Lean-proven; Python impl at `domain/transcript.py:327` | `proofs/ClaudeSql/TurnSort.lean:54`, `:60`, `:68` |
| Lean toolchain builds and discharges a trivial goal — a red `proofs` gate always means a real regression, never a broken toolchain | Lean-proven | `proofs/ClaudeSql/Basic.lean:9` |
| `render_turn_text` is pure and deterministic: same input renders byte-identical bytes (rests on the TurnSort total-order proof above) | Application code | `domain/transcript.py:316` |
| `SessionTextCorpus.assemble` output is byte-stable — the four LLM pipelines checkpoint on it, so string literals and tie-break ordering must not drift | Application code | `domain/transcript.py:126` |
| Embedding store's stamped `(model, dim)` must match the active embedder; different models produce incompatible vector spaces even at matching width | Domain guard, invoked on both write and read/bind paths | `domain/embedding_guard.py:22`; write `application/use_cases/embed.py:321`; read `infrastructure/duckdb_views.py:1943` |
| Near-dup canonical is the *earliest-seen* row within ≤3 Hamming bits; `MIN(b.uuid)` breaks ties so canonical assignment is deterministic | SQL self-join over `ingest_stamps` | `application/use_cases/ingest.py:319` |
| SimHash signature coerced to signed 64-bit BIGINT range so it round-trips parquet → DuckDB without high-bit loss | Application code | `domain/dedup.py:176` |
| Config value-objects (`ClusteringConfig`, `CommunityConfig`, `TermsConfig`, `TranscriptCaps`) are frozen — a config handed to a worker can't mutate mid-run, keeping `seed` determinism honest | Frozen dataclass | `domain/config.py:27` |
| Analytics determinism: `seed` threads into both UMAP `random_state` calls, `leidenalg.find_partition(seed)`, and the resolution-profile bisection RNG — same seed + same input ⇒ byte-identical output | Application code | `domain/structure/cluster.py:64`, `:81`; `domain/structure/community.py:114`, `:186` |
| Community relabel is stable: communities sorted by descending size, ties broken by smallest node index | Application code | `domain/structure/community.py:283` |
| igraph edge attribute MUST be named `"weight"` — `leidenalg` looks it up by string | Application code | `domain/structure/community.py:92` |
| `RefusalError` / `BedrockRefusalError` is terminal: pipelines stamp a neutral placeholder row and clear the retry queue rather than cycling forever | Domain error + worker handling | `domain/errors.py:48`; trajectory `application/use_cases/trajectory.py:554`; friction `:500` |
| Retry queue enforces one row per `(pipeline, unit_id)` (PRIMARY KEY); repeat failures increment `attempts` in place | SQLite DDL + application | `infrastructure/sqlite_state/retry_queue.py:50` |

## Calculations

| Calculation | Inputs | Output | Citation |
|---|---|---|---|
| SimHash 64-bit signature | text | signed 64-bit int | `domain/dedup.py:119` |
| Hamming distance | two 64-bit ints | bit-difference count | `domain/dedup.py:179` |
| Approx token count (Anthropic-billing scale) | list of texts | per-text int count | `domain/dedup.py:89` |
| Token-budget bucket | approx token count | xs / sm / md / lg / xl | `domain/dedup.py:191` |
| Dollar cost estimate (dry-run) | n_items, avg_in/out tokens, `(in_rate, out_rate)` $/MTok | dollars | `domain/costs.py:21` |
| Sentiment delta | prev/curr sentiment labels | float in {-2..2} or None | `domain/trajectory.py:131` |
| Exponential retry backoff | attempt count | timedelta in minutes | `infrastructure/sqlite_state/retry_queue.py:79` |
| c-TF-IDF term weights | per-cluster pseudo-docs, `TermsConfig` | top-N `(cluster, term, weight, rank)` rows | `domain/structure/terms.py:27` |
| UMAP + HDBSCAN cluster labels | `(N,dim)` float32 matrix, `ClusteringConfig` | per-row cluster labels + 2D viz coords | `domain/structure/cluster.py:30` |
| Mutual-kNN edge list | symmetric similarity matrix, k, floor | `(edges, weights)` | `domain/structure/community.py:41` |
| Medoid + coherence per community | similarity matrix, labels | medoid node set + `{cid: coherence}` | `domain/structure/community.py:225` |
| Resolution-profile γ pick | Leiden profile, level | selected γ | `domain/structure/community.py:144` |

**SimHash (`domain/dedup.py:119`).** Lower-case, split on `\w+`, take the set of word 3-grams (1-gram fallback for < 3 tokens). blake2b(digest_size=8) each gram into a uint64. Vote across all 64 bit positions vectorized: bit `b` of the signature is set iff a majority of grams have bit `b` set (`2 * set_count > n_grams`) — byte-identical to the scalar ±1 tally. Empty/degenerate input returns 0.

**Approx tokens (`domain/dedup.py:89`).** Encode each text with cl100k_base (OpenAI's public tokenizer), then multiply by the empirical scaling factor `_ANTHROPIC_RATIO = 0.78` (`dedup.py:61`) to match Anthropic billing within ~5%. Anthropic's tokenizer is closed-source; 0.78 is the measured gap, recomputed against fresh cache receipts twice per minor release.

**Cost estimate (`domain/costs.py:21`).** `n * (in_tokens * in_rate + out_tokens * out_rate) / 1e6` — a flat linear projection, no minimums, tiers, or cache accounting. Powers every `--dry-run` price. Model-ID dated-suffix normalization (`-YYYYMMDD` strip) is a separate concern living in the `cost_estimate` DuckDB macro, not this estimator (`domain/costs.py:14`).

**Sentiment delta (`domain/trajectory.py:131`).** Labels map to `{negative: -1, neutral: 0, positive: 1}` (`trajectory.py:37`); delta = `curr - prev`, or None when either side is None. At row-build the model's emitted delta is trusted when it parses to a number, otherwise recomputed from labels as an audit trail (`trajectory.py:159`).

**c-TF-IDF (`domain/structure/terms.py:27`).** Per-class term frequency (L1-normalized per cluster) times `idf = log(1 + avg_docs_per_term / col_sum)`, over `CountVectorizer` with configurable `min_df` / `max_df` / ngram bounds. Hand-rolled deliberately (no bertopic) to keep the weighting visible and patchable. Terms with non-positive weight are dropped; top-N kept per cluster, ranks 1-based.

**Backoff (`retry_queue.py:79`).** `min(2^attempts, 60)` minutes — 2, 4, 8, 16, 32, then pinned at 60. Cap and monotonicity are the Lean-proven invariants above.

## Policy and gates

- **Cost dry-run default:** every command that spends real money defaults to `--dry-run`; the dry-run path uses a pure-SQL `COUNT(DISTINCT session_id)` rather than a full materialization, so it stays fast on large corpora. Honored in classify/trajectory/conflicts/friction/embed use-cases and chained through the 10-stage `analyze` pipeline. `application/analyze.py:104`, `application/use_cases/classify.py:234`, `.../trajectory.py:829`, `.../conflicts.py:392`.
- **Regex fast-path bypass:** unambiguous friction shapes (`status_ping`, `interruption`, `correction`) are caught by a frozen regex bank at flat 0.9 confidence, never paying Bedrock; anything ambiguous (e.g. `screenshot?`) deliberately falls through to the LLM so a single mis-tuned pattern can't poison the corpus. `domain/friction.py:102`, invoked at `application/use_cases/friction.py:405`.
- **Fail-loud embedding-provider guard:** appending vectors into a store written by a different provider/model is refused (raises `EmbeddingProviderMismatch` naming both sides and the `rm -rf` recovery command); an empty store is a no-op that any provider may claim. `domain/embedding_guard.py:22`.
- **Fail-open LLM-analytics:** the opt-in analytics provider surfacing `LlmAnalyticsUnavailable` NEVER crashes the core SQL/embedding pipeline — the worker treats it like any recoverable per-chunk failure (enqueue, stamp neutral placeholders, or skip). `domain/errors.py:94`, trajectory handling at `application/use_cases/trajectory.py:534`.
- **Terminal-refusal gate:** a Bedrock content-policy refusal (`stop_reason == "refusal"`, no content) is terminal and non-retryable — pipelines stamp a neutral placeholder and clear the retry queue. `domain/errors.py:63`.
- **CLI error taxonomy → exit codes:** DuckDB parse errors → 64, catalog errors → 65, everything else runtime → 70; missing embeddings → 2; malformed input → 64; missing `duckdb` binary → 127. Stable wire contract for agents. `domain/errors.py:26`, classified at `infrastructure/duckdb_errors.py:19`.
- **Bounded windowed retry:** a trajectory chunk missing windows triggers exactly ONE retry of just the missing `(prev_uuid, curr_uuid)` keys; persistent misses become neutral placeholder rows so one refusing chunk never wedges the pipeline. `domain/trajectory.py:199`, `domain/trajectory.py:138`.
- **Team-corpus glob rewrite (opt-in):** setting `team_corpus_root` replaces (does not union) the three personal-corpus globs — but only when the user hasn't pinned any glob away from its factory default. `infrastructure/settings.py:430`.
- **Disconnected-community warn-only:** a Leiden community whose induced subgraph is weakly-disconnected logs a warning but is NOT split (Park et al. Connectivity Modifier deferred until the warning fires regularly on the live corpus). `domain/structure/community.py:196`.
- **Small-community collapse:** communities below `leiden_min_community_size` are relabeled to the noise sentinel `-1` with `is_medoid=False`, `coherence=0.0`. `domain/structure/community.py:293`.

## See also

- [claude-sql · System overview](../architecture/system-overview.md) — 6 shared source citations
