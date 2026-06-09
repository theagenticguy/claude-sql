# claude-sql · Business logic

This file indexes the domain rules baked into claude-sql: the validations, invariants, derivations, and gate logic that govern how Claude Code transcripts are ingested, classified, scored, and bound to commits.

**Scope.** "Business logic" here means application-layer rules: pydantic schema constraints on LLM outputs, the deterministic guard clauses that protect parquet caches from poisoned data, the numeric derivations (sentiment delta, kappa, cosine community detection), and the policy gates (parquet-existence gating, refusal handling, the kappa stopping rule, blind-handover stripping). Out of scope: the DuckDB view DDL itself (covered in `architecture/data-flow.md`), Bedrock prompt *content* beyond the rules it encodes, and the JSONL-on-disk format. There is no relational database with UNIQUE/FK constraints in this system — the durable store is parquet shards plus a sqlite checkpoint DB — so the Invariants section captures parquet-shape and idempotence invariants rather than DB constraints.

Domains used below mirror the worker decomposition: **Classification** (session-level), **Trajectory** (sentiment arc), **Conflicts** (stance clashes), **Friction** (short-message signals), **Community** (session clustering), **Evals** (judge agreement / blind handover), **Provenance** (commit-to-transcript binding), **Output** (CLI error/exit contract), **Config**.

## Validations

The classifier outputs are validated structurally twice: pydantic v2 models (`model_config = ConfigDict(extra="forbid")`) reject unknown fields, and a flattened JSON Schema is enforced server-side by Bedrock's `output_config.format`. `_bedrock_schema` strips constraints Bedrock's Draft 2020-12 subset rejects (numeric `ge`/`le`, string `minLength`/`pattern`, array `minItems`) but pydantic still enforces them at response-parse time `src/claude_sql/core/schemas.py:82-92`.

| Rule | Domain | Citation | Failure mode |
|---|---|---|---|
| `autonomy_tier` ∈ {manual, assisted, autonomous} | Classification | `src/claude_sql/core/schemas.py:108` | Bedrock schema rejects; pydantic raises on parse |
| `work_category` ∈ {sde, admin, strategy_business, events, thought_leadership, other} | Classification | `src/claude_sql/core/schemas.py:118-125` | Schema reject / pydantic raise |
| `success` ∈ {success, partial, failure, unknown} | Classification | `src/claude_sql/core/schemas.py:139` | Schema reject / pydantic raise |
| `goal` is 1–280 chars | Classification | `src/claude_sql/core/schemas.py:149-158` | pydantic raise at parse (length stripped from Bedrock schema) |
| `confidence` ∈ [0.0, 1.0] | Classification, Trajectory, Conflicts, Friction | `src/claude_sql/core/schemas.py:159-167` | pydantic raise at parse |
| `delta` ∈ {-2,-1,0,1,2,null} (integer-encoded curr−prev) | Trajectory | `src/claude_sql/core/schemas.py:218-226` | Out-of-set value fails JSON Schema validation |
| `transition_kind` ∈ {frustration_spike, resolution, reset, drift, clarification, none} | Trajectory | `src/claude_sql/core/schemas.py:235-242` | Schema reject; worker coerces unknown to `none` `src/claude_sql/analytics/trajectory_worker.py:559` |
| `prev_sentiment`/`curr_sentiment` ∈ {negative, neutral, positive} (prev nullable) | Trajectory | `src/claude_sql/core/schemas.py:205-217` | Schema reject |
| `turn_a_uuid` / `turn_b_uuid` are 1–64 chars, copied verbatim from `[uuid=...]` headers | Conflicts | `src/claude_sql/core/schemas.py:303-323` | pydantic raise; worker skips degenerate pairs |
| Conflict pair must have distinct, non-empty UUIDs | Conflicts | `src/claude_sql/analytics/conflicts_worker.py:227-238` | Defensive guard drops the row with a WARNING, keeps the rest |
| `conflict_kind` ∈ {disagreement, correction, reversal, impasse} | Conflicts | `src/claude_sql/core/schemas.py:324` | Schema reject |
| `severity` ∈ {low, medium, high} | Conflicts | `src/claude_sql/core/schemas.py:337` | Schema reject |
| Friction `label` ∈ {status_ping, unmet_expectation, confusion, interruption, correction, frustration, none} | Friction | `src/claude_sql/core/schemas.py:423-431` | Schema reject |
| Friction `rationale` is 1–200 chars | Friction | `src/claude_sql/core/schemas.py:453-462` | pydantic raise at parse |
| Friction classifier only runs on user-role messages ≤ `friction_max_chars` (default 300) | Friction | `src/claude_sql/analytics/friction_worker.py:361`, `src/claude_sql/core/config.py:255` | Longer messages silently dropped from candidate SQL (treated as genuine task turns) |
| Claude Code bookkeeping strings ("Continue from where you left off.", "[Request interrupted by user for tool use]") excluded from friction candidates | Friction | `src/claude_sql/analytics/friction_worker.py:343-346,363` | Filtered at the SQL boundary; never reach Bedrock |
| `--glob` may contain at most one `**` recursive segment | Output/CLI | `src/claude_sql/core/output.py:156-177` | `InputValidationError` → exit 64 with hint |
| Judge-scores parquet must carry columns {session_id, axis, judge_shortname, score} | Evals | `src/claude_sql/evals/kappa_worker.py:149-152,253-256` | `ValueError` naming the missing columns |
| Trailer/note binding requires all three wire fields (digest, uri, agent_runtime) | Provenance | `src/claude_sql/provenance/binding.py:561-565,586-587` | Returns `None` (treated as "no binding") if any are missing |
| If `team_corpus_root` is set, the three transcript globs are rewritten — unless the user pinned any glob, in which case none are rewritten | Config | `src/claude_sql/core/config.py:340-377` | All-or-nothing: a single user pin blocks the rewrite |

## Invariants

| Invariant | Where enforced | Citation |
|---|---|---|
| Pydantic schemas forbid extra fields (`extra="forbid"`), and every emitted JSON Schema sets `additionalProperties: false` at each object level | Application (pydantic + Bedrock schema) | `src/claude_sql/core/schemas.py:77-78,106` |
| Every text turn in a session yields exactly one trajectory row; the session-first window has `prev_uuid IS NULL` and a synthetic `prev_sentiment="neutral"` | Application (trajectory_worker) | `src/claude_sql/analytics/trajectory_worker.py:5-8,527-545` |
| Classifier output is idempotent: same input must produce same output across runs (no randomness, no invented detail) | Application (system-prompt quality bar) | `src/claude_sql/core/llm_shared.py:1258-1266` |
| Community detection is deterministic: `settings.seed` flows into `find_partition(seed=...)` and `Optimiser.set_rng_seed`, and cluster IDs are relabeled by descending size, so same seed + same input ⇒ byte-identical parquet | Application (community_worker) | `src/claude_sql/analytics/community_worker.py:46-51,215,367-418` |
| Kappa pipeline does zero Bedrock calls; pure stats, safe to run unlimited times | Application (kappa_worker) | `src/claude_sql/evals/kappa_worker.py:13` |
| Cohen's / Fleiss' kappa return 0.0 (never NaN) when `pe >= 1.0`, keeping downstream stats valid | Application (kappa_worker) | `src/claude_sql/evals/kappa_worker.py:73-75,96-98` |
| Cohen's kappa requires equal-shape rater arrays (`assert a.shape == b.shape`) | Application (assert) | `src/claude_sql/evals/kappa_worker.py:63` |
| Re-running a worker on an unchanged session is a no-op: the checkpointer gates on advancing `(latest_ts, message_count)` bounds | Application (checkpointer) | `src/claude_sql/analytics/trajectory_worker.py:693-700`; `src/claude_sql/core/checkpointer.py:294-313` |
| Conflicts is rekeyed on `(turn_a_uuid, turn_b_uuid)`; sessions with no conflicts produce ZERO rows (the legacy `empty=True` sentinel is gone) | Application (conflicts_worker + schema) | `src/claude_sql/core/schemas.py:379-400`; `src/claude_sql/analytics/conflicts_worker.py:5-6,220-224` |
| Parquet column types are fixed module constants because the analytics view binding fails if types drift across reruns | Application (worker schema constants) | `src/claude_sql/analytics/trajectory_worker.py:588-601`; `src/claude_sql/analytics/conflicts_worker.py:70-80` |
| Stale-shape parquet shards are purged on first run (trajectory v0 per-message shards; conflicts v0 `conflict_idx`/`empty` shards) so a mixed-schema directory never reaches view registration | Application (worker purge step) | `src/claude_sql/analytics/trajectory_worker.py:329-377`; `src/claude_sql/analytics/conflicts_worker.py:85-128` |
| A growing active session is de-duplicated on rerun via `replace_sessions` before `write_part`, since the checkpointer does not touch the parquet cache (GH #45) | Application (trajectory_worker) | `src/claude_sql/analytics/trajectory_worker.py:890-905` |
| Blind handover: trailer and note must agree on digest; disagreement is a loud failure, not a silent merge | Provenance (resolve) | `src/claude_sql/provenance/binding.py:718-727` |
| Every transcript digest carries the `sha256:` prefix and is the SHA-256 of the JSONL's raw bytes (recomputable in one line of any language) | Provenance | `src/claude_sql/provenance/binding.py:61-63,232-246` |
| Trailer writes are idempotent under `git commit --amend` via `--if-exists replace`; note writes overwrite via `-f` | Provenance | `src/claude_sql/provenance/binding.py:404-442,445-477` |
| The cache-stats accumulator is protected by a `threading.Lock` and never breaks a real run (failures swallowed) | Trajectory/Conflicts/Friction shared | `src/claude_sql/core/llm_shared.py:147-148,193-214` |

## Calculations

| Calculation | Inputs | Output | Citation |
|---|---|---|---|
| Sentiment delta | prev sentiment, curr sentiment (each mapped negative=−1, neutral=0, positive=+1) | float `curr − prev` ∈ {−2…2}, or null when prev is null | `src/claude_sql/analytics/trajectory_worker.py:76,520-524` |
| Cohen's kappa | two equal-length rater arrays | (po − pe) / (1 − pe), where pe is summed category-proportion products | `src/claude_sql/evals/kappa_worker.py:57-75` |
| Fleiss' kappa | (n_items × n_categories) judge-count matrix | (P̄ − P̄ₑ) / (1 − P̄ₑ) | `src/claude_sql/evals/kappa_worker.py:78-98` |
| Bootstrapped 95% CI on kappa | rater arrays, 1000 resamples, seed=42 | (2.5th, 97.5th) quantiles of resampled kappas | `src/claude_sql/evals/kappa_worker.py:24,101-139` |
| Delta-kappa CI (stopping-rule input) | current + prior `FleissKappa` (kappa + CI bounds) | does the 95% CI on (current − prior) exclude zero? | `src/claude_sql/evals/kappa_worker.py:225-247` |
| Session centroid embedding | per-message embeddings grouped by `session_id` | mean over rows, then L2-normalized | `src/claude_sql/analytics/community_worker.py:99-144` |
| Mutual-kNN cosine graph | session centroids, k (default 15), edge floor | symmetric edge list where i,j are mutually top-k and `sim ≥ floor` | `src/claude_sql/analytics/community_worker.py:147-186` |
| Community medoid + coherence | per-community similarity submatrix | medoid = node with max mean cosine to peers; coherence = mean pairwise off-diagonal cosine (singletons → coherence 1.0) | `src/claude_sql/analytics/community_worker.py:331-364` |
| LLM cost estimate (`--dry-run`) | n_items, avg in/out tokens, (in_rate, out_rate) pricing | `(n·in·in_rate + n·out·out_rate) / 1e6` dollars | `src/claude_sql/core/llm_shared.py:1291-1299` |
| Cache discount ratio (log line) | accumulated cache_read + fresh input tokens | `(cache_read + fresh) / fresh` as `Nx` | `src/claude_sql/core/llm_shared.py:238-241` |
| Bedrock client pool size | `embed_concurrency`, `llm_concurrency` | `max(32, max(embed, llm) × 2)` | `src/claude_sql/core/llm_shared.py:375-378` |

**Leiden+CPM community pipeline (multi-step).** `run_communities` (`src/claude_sql/analytics/community_worker.py:472-667`) composes a derived clustering rather than a single value: (1) load + L2-normalize session centroids; (2) build the mutual-kNN cosine graph and symmetrize with `max(w_ij, w_ji)` (a no-op since the matrix is symmetric); (3) when no explicit γ is passed, run `Optimiser.resolution_profile` over `[range_lo, range_hi]` and pick γ via `_pick_zoom` — `medium` = midpoint of the longest plateau, `coarse` = lowest n_communities partition (n ≥ 2), `fine` = highest n_communities partition (`src/claude_sql/analytics/community_worker.py:250-278`); (4) run `find_partition(CPMVertexPartition, …, seed, n_iterations)`; (5) warn (not split) on disconnected induced subgraphs; (6) compute medoid + coherence; (7) relabel by descending size and collapse communities below `leiden_min_community_size` to `NOISE_COMMUNITY_ID = -1` (`src/claude_sql/analytics/community_worker.py:75,367-418`). The plain-prose delta encoding table the trajectory model is held to lives at `src/claude_sql/analytics/trajectory_worker.py:149-162`.

## Policy and gates

- **Parquet-existence gating:** an analytics view or macro is registered only when its backing parquet is populated (at least one part file with `st_size > 16`); a missing parquet warns at DEBUG and no-ops rather than crashing the query path. `src/claude_sql/core/sql_views.py:1794-1806,1930-1942,1152-1162`.
- **Defense-in-depth macro bind:** `_safe_macro` demotes a `duckdb.Error` (e.g. a parquet vanishing between gate-check and DDL-bind) to DEBUG so read-only commands never flood stderr. `src/claude_sql/core/sql_views.py:1165-1190`.
- **Bedrock refusal handling:** a `stop_reason == "refusal"` response is terminal and non-retryable; the worker stamps a neutral placeholder row (trajectory) or a `label="none", source="refused"` row (friction) so the unit is not re-tried every run. `src/claude_sql/core/llm_shared.py:490-497,517-518`; `src/claude_sql/analytics/trajectory_worker.py:775-783`; `src/claude_sql/analytics/friction_worker.py:572-588`.
- **Retryable-error policy:** only the throttle/transient Bedrock codes plus SSL/connection/read-timeout exceptions are retried (tenacity, 10 attempts, exponential backoff); everything else propagates to the per-unit retry queue. `src/claude_sql/core/llm_shared.py:61-75,328-339,395-401`.
- **Trajectory missing-window gate:** when the model omits requested `(prev_uuid, curr_uuid)` windows, one bounded retry re-requests only the missing windows; anything still missing becomes a neutral placeholder so a single refusing chunk never wedges the pipeline. `src/claude_sql/analytics/trajectory_worker.py:805-842`.
- **Friction source-priority gate:** classification of a short user message is decided by regex fast-path first (flat confidence 0.9), then three deterministic SQL stamps, then Sonnet — regex > sql > llm, so the LLM only sees what the cheap paths could not resolve. `src/claude_sql/analytics/friction_worker.py:148-162,171-173,271-312,475-516`.
- **Friction SQL stamps (RFC §4.3, §9.4):** Rule 1 — a user message whose normalized text repeats an earlier user message within 10 turns → `unmet_expectation` (0.85); Rule 2 — ≤30-char message whose first token ∈ {stop, redo, revert, rollback, undo, restart} → `correction` (0.9); Rule 3 — a trailing-`?` user message immediately after an error tool_result → `confusion` (0.85). Rule 3 degrades gracefully (skips) when the `messages` view is absent. `src/claude_sql/analytics/friction_worker.py:195-312`.
- **Kappa stopping-rule gate:** with `--floor 0.6 --delta-gate <prior.parquet>`, the run returns non-zero exit when the delta-kappa CI excludes zero, matching the pre-registered rebaseline policy. `src/claude_sql/evals/kappa_worker.py:8-12,225-247`.
- **Disconnected-community policy:** Leiden communities whose induced subgraph is weakly disconnected are warned about but not split (delete-in-30s reversible); the splitter is deferred until the warning fires on the live corpus. `src/claude_sql/analytics/community_worker.py:16-21,302-328`.
- **Community recompute gate:** if the communities parquet exists and `st_size > 16` and `force` is false, recomputation is skipped and cached stats are returned. `src/claude_sql/analytics/community_worker.py:509-525`.
- **Blind-handover policy:** before a transcript is handed to a cross-provider judge, every identity marker (Slack IDs, agent persona tokens, protocol tokens, MCP tool names, OTel/UUID/work-item/thread-ts system IDs, mrkdwn refs) is stripped; the original session_id is replaced by a SHA256[:16] hash so bundles stay re-linkable without leaking identity. `src/claude_sql/evals/blind_handover.py:1-21,81-156`.
- **Binding resolution precedence:** resolve commit→transcript by trailer first, note fallback, and loud failure (`BindingMismatchError`) on digest disagreement; neither present raises `LookupError` (CLI exit 2). `src/claude_sql/provenance/binding.py:674-743`.
- **Trailer-duplication tolerance:** on read, the first occurrence of a duplicated trailer key wins, tolerating rebase/fixup-squash duplication. `src/claude_sql/provenance/binding.py:506-512`.
- **Agent-runtime detection:** the binding reads `CLAUDE_AGENT_RUNTIME` from the environment, falling back to `claude-code/unknown`. `src/claude_sql/provenance/binding.py:249-262`.
- **CLI exit-code contract:** stable codes agents can rely on — ok=0, no_embeddings=2, invalid_input/parse_error=64, catalog_error=65, runtime_error=70, duckdb_missing=127; `run_or_die` maps `InputValidationError` and `duckdb.Error` to these. `src/claude_sql/core/output.py:49-57,180-207,228-254`.
- **AUTO output format:** `--format auto` resolves to TABLE on a TTY and JSON otherwise, so pipes and agent subprocesses get machine-readable output without a flag. `src/claude_sql/core/output.py:60-65`.
- **`tasks_state_current` status contract:** a TaskCreate row defaults to status `pending` (`COALESCE(ls.status, 'pending')`) and reflects the latest `TaskUpdate.status` (e.g. `in_progress` → `completed`) keyed per `(session_id, task_id)`, with task_id recovered from the tool_result or per-session creation order. `src/claude_sql/core/sql_views.py:921-971`.

## See also

- [claude-sql · Module map](../architecture/module-map.md) — 10 shared source files
- [claude-sql · Contract map](contract-map.md) — 9 shared source files
- [claude-sql · Debugging guide](debugging-guide.md) — 9 shared source files
- [claude-sql · Public API](../reference/public-api.md) — 9 shared source files
- [claude-sql · Tech debt](tech-debt.md) — 9 shared source files
