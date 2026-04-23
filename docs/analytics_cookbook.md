# claude-sql Analytics Cookbook

Runnable SQL recipes for the **v2 analytics surface** — clusters,
communities, classifications, trajectory, conflicts, friction — against
`~/.claude/projects/**/*.jsonl` via the `claude-sql` CLI.

Run any recipe as:

```bash
claude-sql query "<SQL>"
```

First invocation takes roughly a minute because `register_all()`
force-reads the full corpus for schema inference (`sample_size=-1`),
rebuilds the HNSW index from `~/.claude/embeddings.parquet`, and
materializes the analytics views.

**v1 recipes** (sessions, messages, tool_calls, todo_events, subagent_*,
semantic search) live in [`cookbook.md`](cookbook.md). This file covers
only the v2 additions.

## Which sections run today

| Section | Backing parquet | Prerequisite |
|---|---|---|
| 1. Clustering overview | `~/.claude/clusters.parquet` | `claude-sql cluster` |
| 2. Cluster topic labels | `~/.claude/cluster_terms.parquet` | `claude-sql cluster` (auto-builds terms) |
| 3. Community distribution | `~/.claude/session_communities.parquet` | `claude-sql community` |
| 4. Classification analytics | `~/.claude/session_classifications.parquet` | `claude-sql classify --no-dry-run` |
| 5. Trajectory + sentiment | `~/.claude/message_trajectory.parquet` | `claude-sql trajectory --no-dry-run` |
| 6. Cost-of-classification estimate | (stderr only) | none |
| 7. Full pipeline | n/a | none |

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
embeddings grouped into Louvain communities via
`networkx.algorithms.community.louvain_communities` over a
cosine-similarity session-to-session graph. Every session has exactly
one row.

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
SELECT * FROM community_top_topics(<community_id>, 10);
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

> **Requires `claude-sql trajectory --no-dry-run` first.** Each assistant
> message gets a `sentiment_delta` (`positive` / `neutral` / `negative`)
> and an `is_transition` boolean. The `sentiment_arc(sid)` macro joins
> `message_trajectory` to `messages` ordered by time for a single
> session.

### 5.1 Per-session sentiment arc

```sql
SELECT * FROM sentiment_arc('<session-id>');
```

Returns `ts, role, sentiment_delta, is_transition, text` for the session
timeline. Plot cumulative `sentiment_delta` against `ts` to see momentum.

### 5.2 Global sentiment-delta histogram

```sql
SELECT sentiment_delta, count(*)
FROM message_trajectory
GROUP BY 1
ORDER BY 1;
```

### 5.3 Filler fraction — share of assistant turns that are transitional

`is_transition` flags assistant messages that are pure status / handoff
filler (e.g. "Now I'll check…", "Got it, moving on to…"). The ratio of
transitions to substantive turns is a crude "autonomy fluency" signal.

```sql
SELECT is_transition, count(*)
FROM message_trajectory
GROUP BY 1;
```

## 6. Cost-of-classification estimate

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

## 7. Full pipeline

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
