# RFC 0002 — Vision, data model, and production roadmap

**Status**: draft
**Filed**: 2026-05-12
**Owner**: Laith
**Hard constraints (binding for every change in this RFC)**

1. **No signal lost.** The new pipelines must capture every signal the
   old ones aimed for. No truncation that drops content. No coarser
   label shapes. No skipped messages.
2. **Perf only goes up.** Wall-clock and dollar-cost both improve.
3. **No model substitution.** Sonnet 4.6 (`global.anthropic.claude-
   sonnet-4-6` via Bedrock) for classification; Cohere Embed v4
   (`global.cohere.embed-v4:0`) for embeddings. The signal-quality bar
   does not move.

**Operating model:** single-user repo, single owner, no installed user
base to migrate. Rip and replace is the rule. No env flags for opt-in.
No parity-test ratchets to flip defaults. No deprecation periods. New
worker lands → old worker is deleted in the same PR. Derived parquets
are regenerable from JSONLs in minutes; throw them away when schemas
change. The only thing that's actually immutable is `~/.claude/projects/`
(the JSONL transcripts).

## Why this RFC exists

`claude-sql` v0.7.0 is a working analytics tool over one developer's
`~/.claude/projects/` corpus. It hits a small wall as soon as the
ambition expands: the data model is per-message and per-session because
that's what the JSONL files are. The right unit of analysis for the
*questions transcript mining wants to answer* is often something else
— turn-windows, bursts, decisions, threads — and the right ingest unit
for cost-effective LLM analytics is rarely the same as the right unit
for retrieval.

This RFC commits to:

1. A **vision** for what claude-sql becomes at v3+.
2. A **data model** that supports that vision without breaking v0.7.x
   consumers.
3. A **roadmap** with four versioned milestones (v1, v2, v3, v4) and
   the single highest-leverage change at each.
4. A **first deliverable PR scope** small enough to ship next week.

## 1. Vision

> **Transcript mining as the highest-fidelity record of how work gets
> done.**

Every Claude Code session — every prompt, every tool call, every
correction, every refusal, every cache hit, every plan, every
decision-reversed, every todo opened-and-not-closed — is a primary
source. JSONL is the storage format. claude-sql is the system of
record over the JSONL.

What v3 looks like in concrete terms:

- **One-developer mode**: `~/.claude/projects/` (today's mode). 11K
  sessions, 165K messages. `claude-sql analyze` keeps it indexed.
  Queries answer in seconds.
- **Team mode**: an S3-backed corpus shared across N developers, with
  per-author attribution and team-level aggregates. Same code path,
  different `CLAUDE_SQL_HOME`. Query "what bug did *anyone* on the
  platform team hit twice this week".
- **Org mode**: million-session corpus, all engineering, all sales
  engineers, all support, all product designers. Same code path,
  different LanceDB/parquet store. Query "every reversal of decision
  on the auth migration since the kickoff doc landed".
- **Live mode**: a daemon watches `~/.claude/projects/**/*.jsonl` for
  growth and incrementally re-indexes affected sessions. Cookbook
  queries always reflect the last 60 seconds of activity.
- **API + dashboard**: SQL is the primitive but a small HTTP service
  (`claude-sql serve`) exposes the same surface as REST + a single-
  page dashboard. Block Kit / Slack notifications when a friction
  pattern crosses threshold or a stance conflict is unresolved for
  >24h.
- **Continuous improvement loop**: the analytics surface that detects
  patterns of agent failure feeds back into the agent's own context.
  When you start a session about topic X, the system surfaces
  "you've hit corruption-on-WAL-rotation 3 times in this codebase;
  here are the 3 prior resolutions and the lesson file from
  `.erpaval/`". This is the closed loop — corpus mining produces
  context that improves the next session, which produces corpus.

The product is the loop. The loop only works if every step is
*production-grade*: deterministic, reproducible, cost-bounded, signal-
preserving.

## 2. Where v0.7.x leaves signal and perf on the table

The full audit (corpus shape inventory + data-model critique +
chunking strategy review) is captured in the body of this RFC.
Headline findings:

- **Embed filters out 78% of role-bearing turns** (everything that's
  pure tool I/O). `semantic_search` cannot answer "when did I last
  see this stack trace" or "find sessions that ran `pytest` with this
  flag combo". Recall gap by design — fixable additively.
- **Anthropic's runtime achieves 11× cache_read / cache_create ratio
  on the same transcripts that we now ship to Bedrock with ~5%
  effective caching.** Our system block is cached; their per-turn
  user prefix is. The mechanism is reachable; we don't reach.
- **Per-message trajectory is a schema/semantics mismatch.** A
  "transition" is a property of an adjacent pair. The model is
  silently inferring the pair, then projecting back. Pay tokens for
  the projection.
- **Whole-session conflicts is the worst input/output ratio in the
  suite.** 30K-token haystack to find ~3 needle-pairs. The signal is
  local; the input is global.
- **800K-char total clamp binds for 2-3% of sessions** — wildly
  oversized for the 80% under 200K. Tool_result-per-event clamp
  (50K) is generous; the binding constraint on long sessions is
  *count* of events, not text-per-event.
- **Compact-summary rows (`isCompactSummary:true`) are synthetic** but
  treated as ordinary user turns by trajectory and conflicts. Bug.
- **Subagent JSONLs (`subagents/agent-*.jsonl`) may not be in the
  glob.** If not, the entire research-subagent workstream is invisible.
  Confirmation needed.
- **Stale-connection bug in `analyze`**: when run end-to-end, the
  community stage sees an empty `message_embeddings` view because
  the connection was created before embed populated Lance. Real bug
  found during the e2e run that produced this RFC; workaround used
  was `analyze --skip-embed --skip-cluster` after embed/cluster
  landed.
- **Lancedb 0.30 function-name drift** breaks scalar-subquery shapes
  (`WHERE me.embedding ... (SELECT v FROM seed)`) with
  `Catalog Error: Table Function with name __lance_table_scan does
  not exist!`. Same query as CROSS JOIN works.
- **Cookbook recipes use literal placeholders** (`<known-message-uuid>`,
  `<community_id>`, `<session-id>`) that produce confusing
  prepared-statement-parameter errors when run as-is.

## 3. Data model

### 3.1 Core principles

- **JSONL is the source of truth.** Never copied, never mutated.
- **Every derived artifact is a parquet (or LanceDB) under
  `CLAUDE_SQL_HOME`** (default `~/.claude-sql/`, see backlog item
  in `docs/BACKLOG.md`). Reproducible by re-running the producing
  worker.
- **Schemas change freely.** When a parquet's schema needs to
  change, change it. Delete the old parquet. Re-run the worker. No
  migration code, no compatibility shims. The whole derived cache
  is rebuildable in under 90 minutes from JSONLs.
- **Cookbook recipes are updated in the same commit as the schema
  change.** They're documentation, not a compatibility contract.
- **Every chunking decision honors the hard constraints.** No
  truncation strategy that drops a message; no coarser label
  shape; no per-message → per-N-message substitution. Window-based
  reorganizations that preserve every message's classification are
  fine.

### 3.2 New core unit: turn-window

A **turn-window** is an adjacent pair of text-bearing turns within a
session, ordered by timestamp.

```sql
CREATE OR REPLACE VIEW turn_window AS
SELECT
  m1.session_id,
  m1.uuid               AS prev_uuid,
  m1.role               AS prev_role,
  m1.ts                 AS prev_ts,
  m2.uuid               AS curr_uuid,
  m2.role               AS curr_role,
  m2.ts                 AS curr_ts,
  date_diff('millisecond', m1.ts, m2.ts) AS gap_ms,
  row_number() OVER (PARTITION BY m1.session_id ORDER BY m2.ts) AS window_idx
FROM messages_text m1
JOIN messages_text m2
  ON m1.session_id = m2.session_id
 AND m2.ts = (
       SELECT min(m3.ts) FROM messages_text m3
        WHERE m3.session_id = m1.session_id
          AND m3.ts > m1.ts
     )
WHERE m1.is_compact_summary = false
  AND m2.is_compact_summary = false;
```

Why turn-window:

- Matches the actual semantic unit of trajectory and conflicts.
- A session with N text-messages produces N-1 windows. Live corpus
  (~26K text messages, ~10.8K sessions) yields ~15K windows — 45%
  fewer units than per-message because every session loses its
  first message from window-eligibility.
- Adjacent windows share a turn (window N+1's `prev` is window N's
  `curr`). This is the cache anchor for prompt caching: walking
  forward, the prefix of window N+1 is the suffix of window N.

### 3.3 New embedding kinds (additive)

`messages_text` today is the only embed source. We add three more
kinds to the same LanceDB table, distinguished by a `kind` column.
Every kind preserves what's already there — none replace.

| `kind`            | Body                                                       | Why |
|-------------------|------------------------------------------------------------|-----|
| `message`         | text content of an assistant or user turn (today's input)  | base recall |
| `tool_io`         | `tool_name` + 400-char input preview + `→` + 8K-char result | "find when I saw this error" / "find sessions that ran X" |
| `turn_pair`       | `(assistant_turn, following_user_turn)` concatenated        | dialogue exchanges as cause/effect |
| `burst`           | concatenated turns within a >5-min-gap-bounded burst        | topic chunks larger than a single turn |
| `session_summary` | the goal/outcome string from classify (already produced)    | one vector per session for community/dashboard surfaces |

Storage: 4-5× more vectors at 1024-dim int8 ≈ ~52-65 MB additional
LanceDB. Trivial.

### 3.4 New labeled artifacts

- **`message_trajectory.parquet` rewritten** keyed on
  `(prev_uuid, curr_uuid)` with `prev_sentiment`, `curr_sentiment`,
  `delta`, `is_transition`, `transition_kind` (enum:
  `frustration_spike | resolution | reset | drift | clarification |
  none`), `confidence`. First message of each session has
  `prev_uuid = NULL` and a synthetic `(neutral, curr_sentiment)`
  pair. Cookbook recipes 5.1/5.2/5.3 are updated in the same PR.
- **`session_conflicts.parquet` rewritten** keyed on
  `(turn_a_uuid, turn_b_uuid)` with `conflict_kind` (enum:
  `disagreement | correction | reversal | impasse`), `severity`
  (enum: `low | medium | high`), `agent_position`, `user_position`,
  `confidence`. Empty-sentinel scheme deleted: sessions with no
  conflicts produce zero rows; `conflicts_summary` view rebuilds
  counts via `count(*) GROUP BY session_id`.

### 3.5 New ingest-time stamps

- **`approx_tokens`** (int) per text row. `tiktoken cl100k_base ×
  0.78` per the `.erpaval` Anthropic correction. Replaces every
  character-clamp in the codebase with token-clamps — what Bedrock
  actually bills on. **One-time CPU cost ~5 min for 165K-message
  corpus, parallel-trivial.**
- **`simhash64`** (uint64) per text row. SimHash over normalized
  whitespace + lowercased word 3-grams. Identifies near-duplicates
  via Hamming ≤ 3. Tool_result outputs are massively duplicate;
  empirically 30-50% are near-dups of an existing canonical.
- **`canonical_uuid`** (UUID?) populated by a periodic dedup pass
  that joins on `simhash64`. Embed pipeline reads this and skips
  rows where `canonical_uuid IS NOT NULL` — a query against a
  near-dup's content still hits the canonical's vector.
- **`token_budget_bucket`** (small enum) — convenience: `xs` (≤256),
  `sm` (≤2048), `md` (≤8192), `lg` (≤32768), `xl` (>32768). Lets
  packers filter by bucket without re-scanning `approx_tokens`.

## 4. Pipeline redesigns (windowed, signal-preserving)

### 4.1 Trajectory v2 (windowed, per-session batching)

**Per-session call** carrying all that session's windows as a small
array. Median session: 42 turns → ~20 windows → 1 call. p95: 134 turns
→ ~67 windows → split into ≤16-window chunks with shared anchor.
Honest cap is **16 windows per array** (Sonnet 4.6 structured-output
reliability falls off sharply past ~10-16 items in 2026).

Prompt shape:

```
system block (cached, ttl=1h):
  TRAJECTORY_SYSTEM_PROMPT (~1500 tokens)

user message as content blocks:
  block 1 (cached, ttl=1h):
    SCHEMA_REMINDER + SESSION_HEADER (session_id, model, started_at)
  block 2 (uncached):
    <window idx=0>
      <prev role=user uuid=A>...</prev>
      <curr role=assistant uuid=B>...</curr>
    </window>
    <window idx=1>
      <prev role=assistant uuid=B>...</prev>     ← shared anchor
      <curr role=user uuid=C>...</curr>
    </window>
    ...

output_config.format:
  array<TrajectoryWindow>  (UUIDs echoed back)
```

**Verification**: each `TrajectoryWindow` carries `(prev_uuid,
curr_uuid)`. If N go in, N-k come back, retry only the missing k by
their UUIDs. Bounded cost. **No window left unclassified — same
guarantee as today.**

**Expected impact** (signal-preserving):

- Calls: 26,551 → ~15,000 windows packed into ~13,000 calls
  (sessions with ≤16 windows = 1 call each; tail sessions split into
  multiple calls).
- Wall: dominated by per-call latency, not call count. Cache anchors
  make TTFT lower on 2nd+ window in each call. Net ~2-3× faster
  end-to-end.
- Cost: cache_read at 0.1× input rate on the system + block 1.
  Roughly 5-10× cheaper per session-equivalent.

### 4.2 Conflicts v2 (pair-scanner)

Replace whole-session prompts with a sliding window over `(user,
assistant, user)` triples filtered by:

- `curr_role = 'user'` AND `length(curr_text) ≤ 500`
- `curr_text` matches `\b(no|not|actually|wrong|instead|but|stop|
  revert|undo)\b`

Typical session yields 3-8 candidate triples. **Signal preserved**
because the structural pre-filter is permissive — it admits every
candidate humans would label, plus some false positives that the LLM
disambiguates. Sessions with no candidate triples produce zero
conflicts (today they produce one sentinel row — same coverage,
different storage).

Per-call shape mirrors trajectory v2: small array of triples, system
+ schema cached.

**Expected impact**: 5-8× cost cut, dramatic wall-clock cut on
sessions where the only signal was negative.

### 4.3 Friction v2 (SQL-stamped + LLM-fallback)

Three deterministic SQL heuristics stamped *before* the LLM call:

1. Repeated user message body within 10 turns → `unmet_expectation`,
   confidence 0.85, source `sql`.
2. ≤30 chars + first-token ∈ `{stop, redo, revert, rollback, undo,
   restart}` → `correction`, 0.9, source `sql`.
3. Trailing `?` after error tool_result → `confusion`, 0.85, source
   `sql`.

LLM still runs on every candidate not deterministically labeled —
**no signal lost, just shifted to a faster path for the obvious
shapes**.

### 4.4 Embed v2 (additive kinds)

`embed_worker._gather_pending_texts` extended to UNION:

- existing `messages_text` (kind=message)
- new `tool_io_text` view (kind=tool_io)
- new `turn_pair_text` view (kind=turn_pair)
- new `burst_text` view (kind=burst)
- new `session_summary_text` view, populated by classify (kind=session_summary)

LanceDB schema gains `kind` column. `semantic_search` macro gets a
`--kind` filter; cookbook recipes are unchanged because kind defaults
to `message` for compatibility.

**SimHash dedup runs first** (Q1) — 30-50% reduction in net embed
calls. **No signal lost** because near-dups point to canonicals.

### 4.5 Classify v2 (multi-session adaptive packing)

Pack 5-8 sessions per Sonnet call, capped by `approx_tokens` budget
of 600K. Each session's input retains the full session_text (no
truncation). Output is array of SessionLabel keyed by session_id.
Verification + retry-on-missing as in trajectory v2.

**Honest call reduction**: 10,763 → ~1,500 calls (~7×). Not the
fantasy 100×. Wall-clock improvement comes from: (a) call-count
reduction, (b) cache anchors on the system + schema reminder, (c) 1h
cache_control TTL during pipeline runs.

**Signal preserved**: every session gets full-text input and full
SessionLabel output. No truncation. No head/tail split. The honest
constraint is "fewer calls with bigger but bounded payloads", not
"fewer calls by clipping payloads".

### 4.6 Caching strategy across all v2 workers

- **System block**: `cache_control: ephemeral` always; `ttl: "1h"`
  during pipeline runs (>30 min). 1h ttl costs 2× input rate up
  front but stays valid for an hour. Amortizes after 5-10 calls.
  One-line change in `llm_worker.py:262`.
- **Schema reminder block** (block 1 of user message): cached per
  pipeline. Bit-stable across calls within a run.
- **Session header block** (when present): cached per session. Same
  session reclassified later hits this.
- **Anchor turns within multi-window arrays**: not formal cache
  hits (same call, not subsequent), but the model's attention reuses
  prefix-stable byte sequences efficiently.

## 5. Production-grade infrastructure (v2-v3)

### 5.1 `~/.claude-sql/` parent dir

Move every derived cache to `CLAUDE_SQL_HOME` (default
`~/.claude-sql/`, with `XDG_DATA_HOME` and macOS Application Support
variants). See `docs/BACKLOG.md` for the full proposal. Eliminates
the rule-conflict blast radius around `~/.claude/projects/` (which
is hard-rule immutable).

Idempotent one-time auto-migration on first connect: if
`~/.claude-sql/` is empty AND we find recognized caches under
`~/.claude/`, log a notice and `mv`. Per-setting overrides
(`CLAUDE_SQL_*_PATH`) keep working.

### 5.2 `claude-sql watch` (live indexing daemon)

A small daemon (anyio + watchdog) that:

1. Watches `~/.claude/projects/**/*.jsonl` for size growth.
2. On growth: re-runs ingest stamps (approx_tokens, simhash) on the
   *new tail* only (incremental).
3. Schedules embed/classify/trajectory/conflicts/friction for any
   touched session at next quiet idle (debounce 30s).
4. Updates the LanceDB store + parquet shards atomically.

Resource cap: single-process; if the user invokes `claude-sql analyze`
manually, the daemon yields. Implements the "cookbook queries always
reflect the last 60 seconds of activity" promise.

### 5.3 `claude-sql serve` (HTTP API + dashboard)

Read-only HTTP service on a local port. Same surface as the CLI:
`/query`, `/search`, `/schema`, `/list-cache`, `/explain`. JSON
in/out. Local-only by default; reverse-proxy + auth for multi-user
deployments.

A Vite single-page dashboard ships in the wheel (or a separate
`claude-sql-ui` package): top sessions by cost, friction trends,
community map, session arc viewer. The dashboard is a thin client
over `/query` — every panel is a SQL recipe.

### 5.4 `claude-sql team` (S3-backed shared corpus)

Same code path, different store. `CLAUDE_SQL_CORPUS=s3://bucket/path`
points the JSONL glob at S3. LanceDB store can also live in S3 (Lance
supports S3 natively in 0.30+). Per-author attribution joins on
`message.author` (an existing JSONL field per Claude Code v2.1.x).

Auth: AWS SSO + IAM. No claude-sql-specific access layer; defer to
S3 + IAM as the policy surface.

### 5.5 `claude-sql alert` (threshold notifications)

Define rules over the analytics surface:

```yaml
rule: "friction-spike-platform-team"
query: "SELECT count(*) FROM friction_recent('platform') WHERE label = 'frustration'"
threshold: ">5"
schedule: "@hourly"
notify: "slack://channel/platform-engineering"
```

Implementation: cron-like scheduler + Slack/email/webhook sinks.
Stored as YAML under `~/.claude-sql/alerts/`. The closed loop's
output side.

### 5.6 Continuous improvement loop

The endgame. When you start a session:

1. `claude-sql` reads the working directory + first user prompt.
2. Runs `semantic_search` against your corpus on the prompt.
3. Surfaces top-K prior sessions with success/failure labels and
   `.erpaval/solutions/` lessons cross-referenced.
4. Injects a "context primer" into the session via Claude Code's
   `SessionStart` hook.

This is the v3 vision. The v0.7.0 surface (parquets, views, search)
is already 80% of what's needed; v3 is the wiring.

## 6. Roadmap by version

The repo has one user. Versioning is "what ships next", not "what
upgrades from". When a milestone lands, the prior shape is gone.

| Version | Theme | Headline change |
|---------|-------|-----------------|
| **v0.7.x** (current) | Working baseline | as-is |
| **v1.0** | Rip-and-replace to windowed pipelines | `turn_window` view + `approx_tokens` + `simhash64` ingest stamps; `trajectory` rewritten windowed (old worker deleted); `friction` SQL-stamped; `cache_control: ttl=1h` everywhere; `~/.claude-sql/` migration; analyze stale-connection bug fixed; cookbook recipes rewritten in same PR. **Cold `analyze` ≤ 90 min.** |
| **v1.1** | Conflicts + classify rewrites | `conflicts` pair-scanner replaces whole-session shape (old worker deleted); `classify` adaptive multi-session packing replaces 1-call-per-session shape. **Cold `analyze` ≤ 60 min.** |
| **v1.2** | Embed expansion | `tool_io_text`, `turn_pair_text`, `burst_text`, `session_summary_text` added to embed pipeline; SimHash near-dup dedup. LanceDB `kind` column. `semantic_search` filterable by kind. |
| **v2.0** | Live indexing daemon | `claude-sql watch` watches `~/.claude/projects/`, incremental ingest + debounced reclassify. **Incremental analyze ≤ 1 min.** |
| **v2.1** | HTTP + dashboard | `claude-sql serve` + Vite SPA shipping in wheel. Local-only. |
| **v3.0** | Closed loop | `SessionStart` hook injects context primer from `semantic_search` + `.erpaval/solutions/` cross-references. |
| **v3.x** | Optional team / org modes | S3-backed corpus, per-author attribution, alerts. Stays optional — solo mode stays the default. |

The targets in §8 are the gating condition — a version doesn't ship
until its cold-or-incremental analyze hits the listed wall-clock.

## 8. Cost & throughput targets (binding)

Measured on the live corpus (~10.8K sessions, 26.6K text messages,
26.5K embeddings, 1.45K communities, p95 session 134 turns).

| Stage | v0.7.x today | v1.x target | v2.x target |
|-------|--------------|-------------|-------------|
| embed (full corpus, cold) | 8 min | 4 min (SimHash dedup) | 2 min (incremental via `watch`) |
| cluster + terms + community | 7 min | 7 min | 7 min |
| classify (full corpus, cold) | 135 min | 25 min (multi-session pack + 1h cache) | 5 min (incremental) |
| trajectory (full corpus, cold) | 40 min | 12 min (windowed) | 2 min (incremental) |
| conflicts (full corpus, cold) | 135 min | 18 min (pair-scanner) | 3 min (incremental) |
| friction (full corpus, cold) | 30 min | 8 min (SQL stamps) | 1 min (incremental) |
| **total `analyze` cold** | **~5 hours** | **~75 min** | **~20 min** |
| **incremental analyze (last 1h activity)** | n/a | ~5 min | ~30s |

These numbers are the contract. v1.x doesn't ship until the cold-run
total hits ≤90 min. v2.x doesn't ship until incremental hits ≤1 min.

## 9. First deliverable PR (v1.0 — single PR, rip-and-replace)

1. **`sql_views.py`**:
   - `turn_window` view (~30 LOC), excluding `isCompactSummary`
   - new `messages_text` filter to drop `type=attachment` rows
2. **`ingest.py`** (new):
   - `approx_tokens` column on text rows (`tiktoken` × 0.78)
   - `simhash64` column on text rows
   - parallel polars compute
3. **`trajectory_worker.py`** (rewrite, not new file):
   - replaces per-message worker with per-session windowed worker
   - emits `(prev_uuid, curr_uuid)`-keyed rows + `transition_kind`
   - old per-message code deleted; old parquet shards deleted on
     first run
4. **`friction_worker.py`** (extend):
   - SQL stamps for repeated-message / imperative-verb / trailing-
     `?`-after-error before the LLM call
5. **`llm_worker.py`** (edit):
   - `cache_control: {"type": "ephemeral", "ttl": "1h"}` on system
     block
   - per-stage `cache_read_input_tokens` /
     `cache_creation_input_tokens` totals emitted to stderr at end
     of each pipeline run
6. **`cli.py:analyze`** (fix):
   - re-bind `register_vss` between cluster and community stages
     (the stale-connection bug found during the e2e run)
7. **`config.py` + `home.py`** (new):
   - `CLAUDE_SQL_HOME` default = `~/.claude-sql/`
   - one-time auto-move of recognized caches from `~/.claude/` on
     first connect (idempotent)
8. **Cookbook**:
   - rewrite recipes 5.1 and 5.2 (CROSS JOIN form for lancedb 0.30,
     `{{uuid}}` placeholder convention)
   - rewrite analytics_cookbook recipes that use literal angle-
     bracket placeholders
9. **Tests**:
   - `test_turn_window.py` — materializes correctly, compact-
     summary rows excluded
   - `test_simhash.py` — Hamming distance, canonical resolution
   - `test_trajectory_windowed.py` — windows produced match
     `messages_text` ordering, no message left without a window
     except session-first
10. **Run `analyze` end-to-end on the live corpus, validate cold
    wall-clock ≤ 90 min, attach the timing breakdown to the PR**

No env flags. No fallback workers. Old shapes deleted in the same
PR. Rollback path is `git revert`.

## 10. Non-goals

- **Not changing the embedding model.** Cohere Embed v4 stays.
- **Not changing the classifier model.** Sonnet 4.6 stays. Haiku 4.5
  is *not* a substitute even if it's cheaper.
- **Not introducing a separate vector DB.** LanceDB on disk handles
  the corpus and the v3 org-mode scale on a sharded LanceDB cluster.
- **Not building a UI before the data model is right.** v2.1 ships
  the dashboard; v1.x stays headless.
- **Not building an API gateway.** `claude-sql serve` is local-only
  in v2.1; auth is whoever's logged into the host. Real org-scale
  auth is v3.x.
- **Not building user-facing fine-tuning.** The "self-improving
  rubric" in v4.0 is internal — proposed prompt updates surface as
  diffs to humans before landing.

## 11. Open risks (resolve during v1.0 build)

1. **Anchor-turn cache hit rate within a session is theory.**
   Build, run, measure `cache_read_input_tokens` /
   `cache_creation_input_tokens` ratio on the live corpus. If the
   ratio is < 5×, the windowed shape's caching claim is wrong and
   the chunking strategy needs another iteration.
2. **Sonnet 4.6 array-output reliability past 16 items.** The
   honest cap is empirical. If reliability breaks at 8 instead of
   16, the per-call window count drops and the call savings shrink
   accordingly. Pick the actual cap during the build.
3. **TransitionKind enum.** Five values is a first-cut. Build,
   run, look at the distribution, retire empty buckets, add what
   showed up. One re-run.
4. **Subagent JSONL coverage.** Confirm `subagents/agent-*.jsonl`
   files are in the embed/classify glob. If not, surface them as
   additional rows under parent `session_id` (decision made — the
   parent-pointer approach is simpler than first-class sessions).
5. **DuckDB Lance scalar-subquery shape** triggers
   `__lance_table_scan` not-found. Pin lancedb minor version or
   rewrite the cookbook recipe to CROSS JOIN form. Pick one
   during the build.

## 12. References & lessons applied

- `.erpaval/INDEX.md` — every applicable lesson (DuckDB ATTACH,
  bedrock client pooling, anthropic tokenizer gap, ty strict mode,
  Lance pagination, SQLite WAL cold-start, etc.) is honored in this
  RFC.
- `docs/BACKLOG.md` — `~/.claude-sql/` parent dir is filed and
  promoted into the v1.3 milestone.
- `docs/cookbook.md`, `docs/analytics_cookbook.md` — every recipe
  is part of the v1.x compatibility contract.
- Anthropic prompt caching API:
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- LanceDB on S3:
  https://lancedb.github.io/lancedb/cloud/
- Bedrock Message Batches API:
  https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html

---

**Decision summary:**

> Build a turn-window data model that matches what trajectory and
> conflicts actually measure. Make every prompt cache-friendly via
> stable prefix blocks and 1h TTL. Make every Bedrock call as big as
> Sonnet reliably handles without truncating any input or dropping
> any message. Move every derived cache to `~/.claude-sql/`. Build
> the v2 daemon, server, and v3 closed loop on top of that data
> model, not before.
>
> Single user, no env flags, no fallbacks. New worker lands → old
> worker is deleted in the same PR. Schema changes → delete the
> parquet, re-run, ~90 minutes. **No signal lost. Perf only goes up.**
