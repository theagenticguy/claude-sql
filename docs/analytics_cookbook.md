# claude-sql v2 Analytics Cookbook

Runnable SQL recipes for the v2 analytics surface (clusters, communities,
classifications, trajectory, conflicts) against `~/.claude/projects/**/*.jsonl`
via the `claude-sql` CLI. All outputs below were captured on 2026-04-19 against
the dev-host corpus (9,503 top-level sessions, 26,158 embeddings, 299 HDBSCAN
clusters, 6,329 session centroids, 1,512 Louvain communities).

Run any recipe as:

```bash
cd /efs/lalsaado/workplace/claude-sql && uv run claude-sql query "<SQL>"
```

First invocation takes ~60-90s because `register_all()` force-reads the full
corpus for schema inference (`sample_size=-1`), rebuilds the HNSW index from
`~/.claude/embeddings.parquet`, and materializes the analytics views.

**v1 recipes** (sessions, messages, tool_calls, todo_events, subagent_*,
semantic_search) live in [docs/cookbook.md](cookbook.md). This file covers only
the v2 additions.

Which sections run today (2026-04-19):

| Section | Backing parquet | Status |
|---|---|---|
| 1. Clustering overview | `~/.claude/clusters.parquet` | Populated (26,158 rows) |
| 2. Cluster topic labels | `~/.claude/cluster_terms.parquet` | Populated (2,990 rows) |
| 3. Community distribution | `~/.claude/session_communities.parquet` | Populated (6,329 rows) |
| 4. Classification analytics | `~/.claude/session_classifications.parquet` | **Requires `claude-sql classify --no-dry-run` first** |
| 5. Trajectory + sentiment | `~/.claude/message_trajectory.parquet` | **Requires `claude-sql trajectory --no-dry-run` first** |
| 6. Cost-of-classification estimate | (stderr only) | Runs any time |
| 7. Full pipeline | n/a | Runs any time |

## 1. Clustering overview

`message_clusters` is a parquet-backed view over 26,158 assistant-message
embeddings. `cluster_id = -1` marks HDBSCAN noise points. At the default config
(UMAP 2d + HDBSCAN `min_cluster_size=20`) the corpus falls into 299 clusters
with a 37.7% noise share.

```sql
SELECT count(*) AS total,
       count(DISTINCT cluster_id) AS clusters,
       count(*) FILTER (WHERE is_noise) AS noise
FROM message_clusters;
```

```
shape: (1, 3)
┌───────┬──────────┬───────┐
│ total ┆ clusters ┆ noise │
│ ---   ┆ ---      ┆ ---   │
│ i64   ┆ i64      ┆ i64   │
╞═══════╪══════════╪═══════╡
│ 26158 ┆ 300      ┆ 9859  │
└───────┴──────────┴───────┘
```

(300 includes the `-1` noise bucket, so the signal cluster count is 299.)

Top 10 clusters by message count:

```sql
SELECT cluster_id, count(*) AS n
FROM message_clusters
WHERE NOT is_noise
GROUP BY 1
ORDER BY n DESC
LIMIT 10;
```

```
shape: (10, 2)
┌────────────┬─────┐
│ cluster_id ┆ n   │
│ ---        ┆ --- │
│ i32        ┆ i64 │
╞════════════╪═════╡
│ 26         ┆ 403 │
│ 219        ┆ 289 │
│ 114        ┆ 260 │
│ 261        ┆ 244 │
│ 136        ┆ 240 │
│ 227        ┆ 207 │
│ 280        ┆ 194 │
│ 260        ┆ 182 │
│ 286        ┆ 164 │
│ 24         ┆ 153 │
└────────────┴─────┘
```

## 2. Cluster topic labels

`cluster_terms` holds c-TF-IDF weights (top 10 1-2gram terms per cluster)
derived in-house via CountVectorizer over a per-cluster pseudo-document
(messages joined with `string_agg`). The `cluster_top_terms(cid, n)` macro is
the fast way to look at a single cluster.

```sql
SELECT * FROM cluster_top_terms(5, 10);
```

```
shape: (10, 3)
┌──────────────────────┬──────────┬──────┐
│ term                 ┆ weight   ┆ rank │
│ ---                  ┆ ---      ┆ ---  │
│ str                  ┆ f32      ┆ i32  │
╞══════════════════════╪══════════╪══════╡
│ feedback need        ┆ 0.000802 ┆ 1    │
│ instructed by        ┆ 0.000384 ┆ 2    │
│ hook need            ┆ 0.000336 ┆ 3    │
│ structuredoutput as  ┆ 0.000219 ┆ 4    │
│ tool as              ┆ 0.000156 ┆ 5    │
│ the stop             ┆ 0.000125 ┆ 6    │
│ as instructed        ┆ 0.000122 ┆ 7    │
│ the structuredoutput ┆ 0.000121 ┆ 8    │
│ instructed           ┆ 0.000119 ┆ 9    │
│ required by          ┆ 0.000112 ┆ 10   │
└──────────────────────┴──────────┴──────┘
```

Cluster 5 is clearly the "tool-use-enforcement-hook feedback" topic.

Quick way to eyeball the first 10 clusters in one roll-up:

```sql
SELECT cluster_id,
       string_agg(term, ', ' ORDER BY rank) AS terms
FROM cluster_terms
WHERE rank <= 5
GROUP BY 1
ORDER BY cluster_id
LIMIT 10;
```

```
shape: (10, 2)
┌────────────┬───────────────────────────────────────────────────────────────────────────────┐
│ cluster_id ┆ terms                                                                         │
│ ---        ┆ ---                                                                           │
│ i32        ┆ str                                                                           │
╞════════════╪═══════════════════════════════════════════════════════════════════════════════╡
│ 0          ┆ starting assessment, begin starting, and begin, need and, tools we            │
│ 1          ┆ inspiring line, bigger too, halting, you left, continue from                  │
│ 2          ┆ overview of, an overview, get an, ll parse, session ll                        │
│ 3          ┆ use request, request interrupted, interrupted by, user for, by user           │
│ 4          ┆ request interrupted, interrupted by, use request, user for, by user           │
│ 5          ┆ feedback need, instructed by, hook need, structuredoutput as, tool as         │
│ 6          ┆ hook need, instructed by, request as, structuredoutput to, this request       │
│ 7          ┆ happened in, ll parse, session ll, the transcript, transcript to              │
│ 8          ┆ housekeeping starting, begin housekeeping, housekeeping running, housekeeping │
│            ┆ targets, starting the                                                         │
│ 9          ┆ error parsing, untrusted, bootstrap sequence, error error, worktree mise      │
└────────────┴───────────────────────────────────────────────────────────────────────────────┘
```

"Find the cluster for a topic": reverse-lookup by term. The vocabulary is
1- and 2-grams, so prefer the bigram when it exists.

```sql
SELECT cluster_id, weight
FROM cluster_terms
WHERE term = 'hnsw index'
ORDER BY weight DESC
LIMIT 5;
```

```
shape: (1, 2)
┌────────────┬──────────┐
│ cluster_id ┆ weight   │
│ ---        ┆ ---      │
│ i32        ┆ f32      │
╞════════════╪══════════╡
│ 211        ┆ 0.000745 │
└────────────┴──────────┘
```

So cluster 211 is the DuckDB-VSS/HNSW topic. Single-token hits work the same
way (`WHERE term = 'macros'` yields cluster 148, the DuckDB-macro topic).

## 3. Community distribution

`session_communities` is a parquet-backed view of 6,329 session-centroid
embeddings grouped into 1,512 Louvain communities (via
`networkx.algorithms.community.louvain_communities` over a cosine-similarity
session-to-session graph). Every session has exactly one row.

```sql
SELECT count(*) AS total_sessions,
       count(DISTINCT community_id) AS communities
FROM session_communities;
```

```
shape: (1, 2)
┌────────────────┬─────────────┐
│ total_sessions ┆ communities │
│ ---            ┆ ---         │
│ i64            ┆ i64         │
╞════════════════╪═════════════╡
│ 6329           ┆ 1512        │
└────────────────┴─────────────┘
```

Size distribution is heavy-tailed: a handful of dense communities capture
most of the corpus, and a long tail of singletons sits at size 1.

```sql
SELECT community_id, count(*) AS n
FROM session_communities
GROUP BY 1
ORDER BY n DESC
LIMIT 10;
```

```
shape: (10, 2)
┌──────────────┬─────┐
│ community_id ┆ n   │
│ ---          ┆ --- │
│ i32          ┆ i64 │
╞══════════════╪═════╡
│ 59           ┆ 759 │
│ 1439         ┆ 711 │
│ 304          ┆ 686 │
│ 159          ┆ 450 │
│ 212          ┆ 309 │
│ 854          ┆ 302 │
│ 276          ┆ 149 │
│ 434          ┆ 128 │
│ 1068         ┆ 51  │
│ 370          ┆ 46  │
└──────────────┴─────┘
```

To label a community, roll up the HDBSCAN clusters that its sessions' messages
fall into and show the top c-TF-IDF terms per cluster. That's what the
`community_top_topics(cid, n)` macro does:

```sql
SELECT * FROM community_top_topics(59, 10);
```

```
shape: (10, 3)
┌────────────┬────────┬────────────────────────────────────────────────────────────────────────────┐
│ cluster_id ┆ n_msgs ┆ top_terms                                                                  │
│ ---        ┆ ---    ┆ ---                                                                        │
│ i32        ┆ i64    ┆ str                                                                        │
╞════════════╪════════╪════════════════════════════════════════════════════════════════════════════╡
│ 219        ┆ 283    ┆ update only, pass status, marked complete, status acknowledgment,          │
│            ┆        ┆ acknowledgment                                                             │
│ 136        ┆ 238    ┆ mostly empty, appear empty, the preview, json structure, preview let       │
│ 261        ┆ 228    ┆ cross package, pyright errors, just pyright, pyright is, import resolution │
│ 227        ┆ 201    ┆ plan both, findings while, read its, both research, synthesize findings    │
│ 280        ┆ 186    ┆ fatal unexpected, already imported, the unused, test_dashboard_tools,      │
│            ┆        ┆ imported at                                                                │
│ 260        ┆ 181    ┆ 154, all 154, list id, f0arks7kxmz, create response                        │
│ 286        ┆ 162    ┆ required_signatures, the ruleset, the bypass, ruleset, by github           │
│ 164        ┆ 151    ┆ tiller_api routers, erpaval for, erpaval task, task structure, launch      │
│            ┆        ┆ explore                                                                    │
│ 237        ┆ 131    ┆ chat update, stopstream, 12k, appendstream, buffer_size                    │
│ 282        ┆ 128    ┆ push deploy, commit ahead, committed now, 34 passed, me rebase             │
└────────────┴────────┴────────────────────────────────────────────────────────────────────────────┘
```

Community 59 is the biggest community (759 sessions) and reads as the
"codebase engineering" blob: TodoWrite status hygiene, pyright import
resolution, ruleset/bypass merges, ERPAVal task scaffolding, commit-deploy
cycles.

## 4. Classification analytics

> **Requires `claude-sql classify --no-dry-run` first.** This section's
> queries run against `session_classifications`, which is materialized by the
> Sonnet 4.6 classifier. A full-corpus run costs roughly $455 at Sonnet
> pricing ($3/MTok input, $15/MTok output) -- always inspect the dry-run
> estimate (section 6) before committing.

The four standard cuts — autonomy tier, work mix, success rate by work
category, and weekly autonomy trend — are table macros so you don't have to
memorize the GROUP BY.

### 4.1 Autonomy-tier breakdown

```sql
SELECT autonomy_tier, count(*)
FROM session_classifications
GROUP BY 1
ORDER BY 2 DESC;
```

Expected shape once populated:

```
shape: (N, 2)
┌───────────────┬──────────────┐
│ autonomy_tier ┆ count_star() │
│ ---           ┆ ---          │
│ str           ┆ i64          │
╞═══════════════╪══════════════╡
│ autonomous    ┆ ...          │
│ supervised    ┆ ...          │
│ manual        ┆ ...          │
└───────────────┴──────────────┘
```

### 4.2 Work mix for last 30 days

```sql
SELECT * FROM work_mix(30);
```

### 4.3 Success rate per work category

```sql
SELECT * FROM success_rate_by_work(30);
```

Returns `work_category, sessions, success_rate, failure_rate, partial_rate`.

### 4.4 Autonomy trend per week (last 90 days)

```sql
SELECT * FROM autonomy_trend(90);
```

Returns one row per `(week, autonomy_tier)` with a count. Plot week on the
x-axis, count on y, colored by tier.

## 5. Trajectory + sentiment

> **Requires `claude-sql trajectory --no-dry-run` first.** This section's
> queries run against `message_trajectory`, which is materialized by a
> regex-prefilter + Sonnet 4.6 pipeline. Each assistant message gets a
> `sentiment_delta` (-2..+2) and an `is_transition` boolean. The
> `sentiment_arc(sid)` macro joins message_trajectory to messages ordered by
> time for a single session.

### 5.1 Per-session sentiment arc

```sql
SELECT * FROM sentiment_arc('<session-id>');
```

Returns `ts, role, sentiment_delta, is_transition, text` for the session
timeline. Plot `sentiment_delta` cumulative sum against `ts` to see momentum.

### 5.2 Global sentiment-delta histogram

```sql
SELECT sentiment_delta, count(*)
FROM message_trajectory
GROUP BY 1
ORDER BY 1;
```

### 5.3 Filler fraction — what share of assistant turns are transitional

`is_transition` flags assistant messages that are pure status/handoff filler
(e.g. "Now I'll check...", "Got it, moving on to..."). Ratio of transitions
to substantive turns is a crude "autonomy fluency" signal.

```sql
SELECT is_transition, count(*)
FROM message_trajectory
GROUP BY 1;
```

## 6. Cost-of-classification estimate

Before running the real Sonnet 4.6 classifier, check the dry-run estimate
(default mode for `claude-sql classify`):

```bash
uv run claude-sql classify --dry-run --since-days 30 --limit 10 2>&1 | tail -5
```

The dry-run loads the DuckDB views, runs `iter_session_texts` to count
pending sessions (those not yet in `session_classifications`), and
multiplies by 8,000 input + 300 output tokens per session at the Sonnet 4.6
Bedrock rate (`$3/MTok` in, `$15/MTok` out).

Real output on this host (`--limit 10` to keep the text-build phase short —
the loader walks every session's transcript, which is the slow part):

```
16:36:31 INFO    {} iter_session_texts: 10 sessions pending
16:39:43 INFO    {} classify --dry-run: 10 sessions pending.  Estimated cost ~$0.28 (thinking=adaptive, model=global.anthropic.claude-sonnet-4-6)
16:39:43 INFO    {} classify: 0 sessions processed (dry_run=True)
```

At $0.028/session, the full 6,325-session backlog on this host would be
roughly $177 (or ~$455 on a larger 16K-session corpus — the per-session
price is the knob, not the count).

Same dry-run flag exists on `trajectory` and `conflicts`:

```bash
uv run claude-sql trajectory --dry-run --since-days 30
uv run claude-sql conflicts  --dry-run --since-days 30
```

Trajectory's estimate assumes 500-token input / 50-token output per LLM
message; conflicts assumes 6K input / 400 output per session.

## 7. Full pipeline

`claude-sql analyze` is the orchestrator that runs every v2 stage in
dependency order: embed -> (cluster + community) -> classify -> trajectory ->
conflicts. Default is `--dry-run` so the LLM stages only print counts and
cost estimates; pass `--no-dry-run` to spend real money. Use `--skip-<stage>`
to drop a stage, and `--force-cluster` / `--force-community` to rebuild those
(non-LLM) parquets even if they already exist.

```
$ uv run claude-sql analyze --help
Usage: claude-sql analyze [OPTIONS]

Run the full v2 analytics pipeline.

Stages: embed -> classify + cluster + community -> trajectory -> conflicts.

Default is --dry-run -- every LLM-touching stage just prints pending counts and
cost estimates.  Pass --no-dry-run to execute for real. Use --skip-<stage> to
drop a stage entirely.  --force-cluster / --force-community rebuild those
parquet outputs even if they already exist.

╭─ Parameters ─────────────────────────────────────────────────────────────────╮
│ --since-days                  [default: 30]                                  │
│ --limit                                                                      │
│ --dry-run --no-dry-run        [default: True]                                │
│ --no-thinking                 [default: False]                               │
│ --skip-embed --no-skip-embed  [default: False]                               │
│ --skip-classify               [default: False]                               │
│ --skip-trajectory             [default: False]                               │
│ --skip-conflicts              [default: False]                               │
│ --skip-cluster                [default: False]                               │
│ --skip-community              [default: False]                               │
│ --force-cluster               [default: False]                               │
│ --force-community             [default: False]                               │
│ --verbose --quiet             [default: False]                               │
│ --glob                                                                       │
│ --subagent-glob                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Typical first invocation:

```bash
# 1. Dry-run everything to see counts and cost estimates.
uv run claude-sql analyze --dry-run --since-days 30

# 2. Commit to the non-LLM stages (cheap / free).
uv run claude-sql analyze --no-dry-run --since-days 30 \
    --skip-classify --skip-trajectory --skip-conflicts

# 3. Once the estimates look acceptable, let the LLM stages run.
uv run claude-sql analyze --no-dry-run --since-days 30
```

Stage-by-stage you can also just use the single-purpose subcommands
(`claude-sql cluster`, `community`, `terms`, `classify`, `trajectory`,
`conflicts`) — `analyze` just chains them with shared defaults.
