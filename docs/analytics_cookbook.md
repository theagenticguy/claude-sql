# claude-sql Analytics Cookbook

Runnable SQL recipes for the **v2 analytics surface** — clusters,
communities, classifications, trajectory, conflicts, friction — against
`~/.claude/projects/**/*.jsonl` via the `claude-sql` CLI.

Run any recipe as:

```bash
claude-sql query "{{SQL}}"
```

First invocation takes roughly a minute because `register_all()`
force-reads the full corpus for schema inference (`sample_size=-1`),
attaches the LanceDB embeddings store, and materializes the analytics
views.

**v1 recipes** (sessions, messages, tool_calls, todo_events, subagent_*,
semantic search) live in [`cookbook.md`](cookbook.md). This file covers
only the v2 additions.

### Placeholder convention

Recipes that need a per-user value use **Mustache-style double braces** —
`{{uuid}}`, `{{session_id}}`, `{{community_id}}`. Substitute the literal
value (with surrounding quotes for strings) before running. Do not paste
`{{...}}` into `claude-sql query` raw — DuckDB will treat it as a
prepared-statement parameter and error out.

## Which sections run today

Caches live under `claude_sql_home()` — `$CLAUDE_SQL_HOME` overrides;
otherwise `$XDG_DATA_HOME/claude-sql/` (Linux) /
`~/Library/Application Support/claude-sql/` (macOS) / `~/.claude-sql/`
(universal fallback). The migration from `~/.claude/` is one-time
idempotent, so existing installs may still see the legacy parent on
first run.

| Section | Backing parquet | Prerequisite |
|---|---|---|
| 1. Clustering overview | `clusters.parquet` | `claude-sql cluster` |
| 2. Cluster topic labels | `cluster_terms.parquet` | `claude-sql cluster` (auto-builds terms) |
| 3. Community distribution | `session_communities.parquet` | `claude-sql community` |
| 4. Classification analytics | `session_classifications/` | `claude-sql classify --no-dry-run` |
| 5. Trajectory + sentiment | `message_trajectory/` | `claude-sql trajectory --no-dry-run` |
| 6. Conflicts | `session_conflicts/` | `claude-sql conflicts --no-dry-run` |
| 7. Cost-of-classification estimate | (stderr only) | none |
| 8. Full pipeline | n/a | none |

Missing parquets don't crash the CLI — the matching view registers empty,
warns once at connection open, and every query against it returns zero
rows. Run the listed prerequisite first.

All output shapes below are illustrative — they show column layout and
expected row counts, not data from any specific user's corpus.

## 1. Clustering overview

`message_clusters` is a parquet-backed view over assistant-message
embeddings. `cluster_id = -1` marks HDBSCAN noise points. At the default
config (UMAP 2d + HDBSCAN `min_cluster_size=20`), textual-embedding
HDBSCAN typically lands in a 25–45% noise band.

```sql
SELECT count(*) AS total,
       count(DISTINCT cluster_id) AS clusters,
       count(*) FILTER (WHERE is_noise) AS noise
FROM message_clusters;
```

The reported `clusters` count includes the `-1` noise bucket, so the
signal cluster count is `clusters - 1`.

Top 10 clusters by message count:

```sql
SELECT cluster_id, count(*) AS n
FROM message_clusters
WHERE NOT is_noise
GROUP BY 1
ORDER BY n DESC
LIMIT 10;
```

## 2. Cluster topic labels

`cluster_terms` holds c-TF-IDF weights (top 10 1–2gram terms per cluster)
derived in-house via `sklearn.feature_extraction.text.CountVectorizer`
over a per-cluster pseudo-document built from DuckDB `string_agg`. The
`cluster_top_terms(cid, n)` macro is the fast way to look at a single
cluster.

```sql
SELECT * FROM cluster_top_terms(5, 10);
```

Returns `term, weight, rank` ordered by rank. Treat the top 2–5 bigrams
as the cluster's rough topic label.

Eyeball the first 10 clusters in one roll-up:

```sql
SELECT cluster_id,
       string_agg(term, ', ' ORDER BY rank) AS terms
FROM cluster_terms
WHERE rank <= 5
GROUP BY 1
ORDER BY cluster_id
LIMIT 10;
```

"Find the cluster for a topic" — reverse-lookup by term. The vocabulary is
1- and 2-grams; prefer the bigram when one exists:

```sql
SELECT cluster_id, weight
FROM cluster_terms
WHERE term = 'hnsw index'
ORDER BY weight DESC
LIMIT 5;
```

## 3. Community distribution

`session_communities` is a parquet-backed view of session-centroid
embeddings grouped into Leiden+CPM communities via
`leidenalg.find_partition(... CPMVertexPartition ...)` over a
mutual-kNN cosine graph (k=15, edge floor 0.3) of session centroids.
Every session has exactly one row, plus `is_medoid` (best
representative), `coherence` (mean intra-community cosine), and the γ
used. The `community_profile` sidecar reports one row per γ tested by
`Optimiser.resolution_profile`.

```sql
SELECT count(*) AS total_sessions,
       count(DISTINCT community_id) AS communities
FROM session_communities;
```

Community sizes are heavy-tailed: a handful of dense communities capture
most of the corpus, and a long tail of singletons sits at size 1.

```sql
SELECT community_id, count(*) AS n
FROM session_communities
GROUP BY 1
ORDER BY n DESC
LIMIT 10;
```

To label a community, roll up the HDBSCAN clusters its sessions' messages
fall into and show the top c-TF-IDF terms per cluster. That's what
`community_top_topics(cid, n)` does:

```sql
SELECT * FROM community_top_topics({{community_id}}, 10);
```

Returns `cluster_id, n_msgs, top_terms` — read the `top_terms` column
across the first few rows to get a rough label for the community.

## 4. Classification analytics

> **Requires `claude-sql classify --no-dry-run` first.** Sessions are
> classified by Sonnet 4.6 via Bedrock `output_config.format` structured
> output, so the dry-run estimate (section 6) tells you the real cost
> before you commit.

Four standard cuts — autonomy tier, work mix, success rate by work
category, and weekly autonomy trend — are exposed as table macros so you
don't have to memorize the `GROUP BY`.

### 4.1 Autonomy-tier breakdown

```sql
SELECT autonomy_tier, count(*)
FROM session_classifications
GROUP BY 1
ORDER BY 2 DESC;
```

Returns counts across the three tiers: `autonomous`, `supervised`,
`manual`.

### 4.2 Work mix for last 30 days

```sql
SELECT * FROM work_mix(30);
```

Returns `work_category, n_sessions, share` (share as a fraction of the
window).

### 4.3 Success rate per work category

```sql
SELECT * FROM success_rate_by_work(30);
```

Returns `work_category, sessions, success_rate, failure_rate,
partial_rate`.

### 4.4 Autonomy trend per week (last 90 days)

```sql
SELECT * FROM autonomy_trend(90);
```

Returns one row per `(week, autonomy_tier)` with a count. Plot `week` on
the x-axis, count on y, colored by tier.

## 5. Trajectory + sentiment

> **Requires `claude-sql trajectory --no-dry-run` first.** v1.0 uses a
> **windowed** schema: each row is a `(prev_uuid, curr_uuid)` pair that
> reports the *transition* from the previous turn to the current turn.
> Schema: `(session_id, prev_uuid, curr_uuid, prev_sentiment,
> curr_sentiment, delta, is_transition, transition_kind, confidence,
> classified_at)`. `delta` is the integer `(curr - prev)` polarity diff
> in `{-2, -1, 0, 1, 2}` (`null` on the session-first window).
> `transition_kind ∈ {frustration_spike, resolution, reset, drift,
> clarification, none}`. The `sentiment_arc(sid)` macro joins
> `message_trajectory` to `messages` ordered by time for a single
> session — its output columns are `(ts, role, curr_sentiment, delta,
> transition_kind, is_transition, confidence)`.
>
> The `message_trajectory` view exposes `curr_sentiment AS sentiment`
> as a back-compat alias so older recipes still bind, but new recipes
> should target the canonical column names.

### 5.1 Per-session sentiment arc

```sql
SELECT * FROM sentiment_arc('{{session_id}}');
```

Returns `ts, role, curr_sentiment, delta, transition_kind, is_transition,
confidence` for the session timeline. Plot cumulative `delta` against
`ts` to see momentum; `transition_kind` colors the segments.

For finer control without the macro — e.g. you want the prev/curr UUIDs
and don't need the `messages` join — query the windowed shape directly:

```sql
SELECT prev_uuid,
       curr_uuid,
       prev_sentiment,
       curr_sentiment,
       delta,
       transition_kind,
       is_transition,
       confidence,
       classified_at
FROM message_trajectory
WHERE session_id = '{{session_id}}'
ORDER BY classified_at;
```

### 5.2 Sessions where sentiment dropped sharply

Sharp negative transitions: `delta < -0.5` only fires meaningfully on
the integer encoding when the magnitude is ≥1. Pair with
`is_transition` and the categorical `transition_kind` for an
agent-actionable cut.

```sql
SELECT session_id,
       count(*) FILTER (WHERE transition_kind = 'frustration_spike') AS spikes,
       count(*) FILTER (WHERE transition_kind = 'reset')              AS resets,
       min(delta)                                                     AS worst_delta
FROM message_trajectory
WHERE delta IS NOT NULL
  AND delta <= -1
  AND is_transition = false
  AND transition_kind IN ('frustration_spike', 'reset')
GROUP BY session_id
ORDER BY spikes DESC, worst_delta ASC
LIMIT 20;
```

### 5.3 Find frustration spikes across the corpus

`transition_kind = 'frustration_spike'` is the load-bearing label for
"things just got worse". The `is_transition = false` filter drops pure
filler turns so the result is substantive negative pivots.

```sql
SELECT session_id, prev_uuid, curr_uuid, delta, confidence, classified_at
FROM message_trajectory
WHERE transition_kind = 'frustration_spike'
  AND is_transition = false
ORDER BY classified_at DESC
LIMIT 25;
```

### 5.4 Filler fraction — share of turns that are transitional

`is_transition` flags the *current* turn as pure status / handoff filler
(e.g. "Now I'll check…", "Got it, moving on to…") regardless of the
sentiment shape. The ratio of transitions to substantive windows is a
crude "autonomy fluency" signal.

```sql
SELECT is_transition, count(*)
FROM message_trajectory
GROUP BY 1;
```

### 5.5 Global delta histogram

`delta` is the integer encoding of `(curr - prev)` sentiment. Group on
the integer for a clean five-bucket histogram (plus a NULL bucket for
session-first windows).

```sql
SELECT delta, count(*) AS n
FROM message_trajectory
GROUP BY 1
ORDER BY 1 NULLS FIRST;
```

## 6. Conflicts

> **Requires `claude-sql conflicts --no-dry-run` first.** v1.0 stores
> one row per conflicting *pair of turns*. Schema: `(session_id,
> turn_a_uuid, turn_b_uuid, conflict_kind, severity, agent_position,
> user_position, confidence, detected_at)`. The legacy
> `(conflict_idx, stance_a, stance_b, resolution, empty)` shape is
> gone — sessions with no conflicts simply have **zero rows** in
> `session_conflicts`. The `conflicts_summary` view (registered
> alongside `session_conflicts`) gives `(session_id, conflict_count)`
> for any session that has at least one conflict.
>
> `conflict_kind ∈ {disagreement, correction, reversal, impasse}`.
> `severity ∈ {low, medium, high}`.

### 6.1 Conflict counts per session

```sql
SELECT session_id, conflict_count
FROM conflicts_summary
ORDER BY conflict_count DESC
LIMIT 15;
```

### 6.2 No-conflict sessions

Use a `LEFT JOIN` against `conflicts_summary` and filter on the missing
row — the v1.0 surface no longer carries an `empty=true` sentinel.

```sql
SELECT s.session_id
FROM sessions s
LEFT JOIN conflicts_summary cs USING (session_id)
WHERE cs.conflict_count IS NULL OR cs.conflict_count = 0
ORDER BY s.started_at DESC
LIMIT 25;
```

### 6.3 High-severity reversals across the corpus

A `reversal` is the same party flipping their own earlier position
("actually let's NOT do X"). High-severity ones are the most
load-bearing rework signals.

```sql
SELECT session_id,
       turn_a_uuid,
       turn_b_uuid,
       agent_position,
       user_position,
       confidence,
       detected_at
FROM session_conflicts
WHERE conflict_kind = 'reversal'
  AND severity = 'high'
ORDER BY detected_at DESC
LIMIT 20;
```

### 6.4 Sessions where the user disagreed with the agent and won

`disagreement` (both sides hold a position) plus medium-or-high
severity surfaces the cases where pushback was substantive enough to
matter. Pair with `success` from `session_classifications` to see
whether the disagreement landed on a successful outcome.

```sql
SELECT sc.session_id,
       count(*)                AS disagreements,
       max(sc.severity)        AS top_severity,
       sk.success              AS outcome
FROM session_conflicts sc
LEFT JOIN session_classifications sk USING (session_id)
WHERE sc.conflict_kind = 'disagreement'
  AND sc.severity IN ('medium', 'high')
GROUP BY sc.session_id, sk.success
ORDER BY disagreements DESC
LIMIT 20;
```

### 6.5 Distribution of conflict kinds and severities

The two enums together summarize the corpus's conflict profile in one
roll-up — useful as a "how much rework is happening" baseline.

```sql
SELECT conflict_kind, severity, count(*) AS n
FROM session_conflicts
GROUP BY 1, 2
ORDER BY 1, 2;
```

## 7. Cost-of-classification estimate

Before running the real Sonnet 4.6 classifier, check the dry-run estimate
(default mode for `claude-sql classify`):

```bash
claude-sql classify --dry-run --since-days 30 --limit 10 2>&1 | tail -5
```

The dry-run uses a pure-SQL `COUNT(DISTINCT session_id)` to count pending
sessions — sessions not yet in `session_classifications` — and multiplies
by 8,000 input + 300 output tokens per session at the Sonnet 4.6 Bedrock
rate (`$3 / MTok` in, `$15 / MTok` out). Adaptive thinking adds a few
hundred output tokens on average; pass `--no-thinking` if the estimate is
uncomfortable.

At roughly `$0.028` per session with thinking on, a 10K-session backlog is
roughly `$280`. The per-session price is the knob, not the count.

The same dry-run flag exists on `trajectory`, `conflicts`, and
`friction`:

```bash
claude-sql trajectory --dry-run --since-days 30
claude-sql conflicts  --dry-run --since-days 30
claude-sql friction   --dry-run --since-days 30
```

Trajectory's estimate assumes 500 input / 50 output tokens per LLM
message; `conflicts` assumes 6K input / 400 output per session;
`friction` counts only user-role messages under
`CLAUDE_SQL_FRICTION_MAX_CHARS` (default 300).

## 8. Full pipeline

`claude-sql analyze` orchestrates every v2 stage in dependency order:
`embed → (cluster + community) → classify → trajectory → conflicts →
friction`. Default is `--dry-run`, so every LLM-touching stage prints
pending counts and cost estimates; pass `--no-dry-run` to execute. Use
`--skip-<stage>` to drop a stage, and `--force-cluster` /
`--force-community` to rebuild those (non-LLM) parquets even if they
already exist.

```text
$ claude-sql analyze --help
Usage: claude-sql analyze [OPTIONS]

Run the full v2 analytics pipeline.

Stages: embed → cluster + community → classify → trajectory → conflicts → friction.

Default is --dry-run — every LLM-touching stage just prints pending counts and
cost estimates. Pass --no-dry-run to execute for real. Use --skip-<stage> to
drop a stage entirely. --force-cluster / --force-community rebuild those
parquet outputs even if they already exist.
```

Typical first invocation:

```bash
# 1. Dry-run everything to see counts and cost estimates.
claude-sql analyze --dry-run --since-days 30

# 2. Commit to the non-LLM stages (free / cheap).
claude-sql analyze --no-dry-run --since-days 30 \
    --skip-classify --skip-trajectory --skip-conflicts --skip-friction

# 3. Once the estimates look acceptable, let the LLM stages run.
claude-sql analyze --no-dry-run --since-days 30
```

Stage-by-stage, you can also use the single-purpose subcommands
(`cluster`, `community`, `classify`, `trajectory`, `conflicts`,
`friction`) — `analyze` just chains them with shared defaults.
