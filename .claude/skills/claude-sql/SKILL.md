---
name: claude-sql
description: Query the user's own Claude Code transcript history with SQL, semantic search, and LLM-driven analytics. Use this skill whenever the user asks about their past Claude sessions, their coding history with Claude, what they worked on last week/month/quarter, how much they spent on Opus or Sonnet, which sessions succeeded or failed, their autonomy patterns, the themes across their work, recurring topics, todos they never closed, or any analytical question about their own ~/.claude/ transcripts. Also triggers for "my claude history", "my claude sessions", "my transcripts", "analyze my sessions", "what did I do in claude", "search my conversations", "my opus cost", "my claude spend", "autonomy tier", "work categories", "group my sessions", "cluster my work", "find sessions about X", "sessions where I disagreed with myself", or any phrasing that implies analyzing the user's own Claude Code corpus.
---

# claude-sql

A zero-copy SQL + semantic search + LLM analytics engine over the user's
`~/.claude/projects/**/*.jsonl` transcripts. Every one of their past
Claude Code sessions is already on disk — this skill turns that corpus
into a live database without any export or pipeline step.

## When to use this skill

Trigger this skill the moment a question is about the **user's own
Claude Code history**, not code in the current repo. Examples:

- "What was that thing I did last Tuesday with DuckDB?"
- "Which Opus sessions cost me more than \$5 this month?"
- "Group my sessions by topic and tell me what's trending."
- "Find every conversation I've had about temporal workflows."
- "Which todos did I open and never close?"
- "Sessions where I took two opposing positions on the same decision."
- "How much have I spent on Claude this quarter?"
- "Which tools do I lean on most? Which ones fail the most?"

If the user asks about code in the working directory, a bug in their
project, or a file they are editing, this skill does **not** apply.

## Prerequisites

Before running anything that hits Bedrock, confirm `AWS_PROFILE` is set
in the environment and has `bedrock:InvokeModel` on
`inference-profile/global.cohere.embed-v4:0` and
`inference-profile/global.anthropic.claude-sonnet-4-6`.

If `claude-sql` is not on PATH, run `mise run tool:install` from the
repo or `uv tool install --from . claude-sql`.

## How to work through a request

1. **Read-only first.** If the question can be answered by a SQL query
   over the views that already exist, write that query — do not kick
   off an expensive embed/classify/cluster run. See
   `references/recipes.md` for 13 ready-to-adapt queries.

2. **Need semantic match?** (question is about *meaning*, not exact
   text — "conversations about X"): use `claude-sql search "<text>"`.
   If embeddings don't exist yet, run `claude-sql embed --since-days N`
   first.

3. **Need classifications or trajectory?** (autonomy tier, work
   category, success, sentiment arc, conflicts): these are LLM outputs.
   Run the relevant `claude-sql classify|trajectory|conflicts` command
   with `--dry-run` first to see the cost, then re-run with
   `--no-dry-run` once the user confirms. Results land in a parquet
   cache that the views register automatically.

4. **Need themes across the corpus?** (cluster / group / organize):
   `claude-sql cluster` for message clusters, `claude-sql community`
   for session communities. Both are deterministic — same input, same
   cluster/community IDs.

5. **Need everything at once?** `claude-sql analyze --since-days N`
   chains embed → cluster → classify → trajectory → conflicts. Always
   dry-run first.

## The CLI at a glance

| Command | Purpose | Spends money? |
|---|---|---|
| `schema` | List every view + macro | No |
| `query <sql>` | Run a SQL query | No |
| `explain <sql>` | EXPLAIN ANALYZE with pushdown markers | No |
| `shell` | DuckDB REPL with everything registered | No |
| `search <text>` | HNSW cosine top-k semantic search | No |
| `embed` | Backfill Cohere v4 embeddings | Yes (defaults to `--dry-run`) |
| `classify` | Sonnet 4.6 session classification | Yes (defaults to `--dry-run`) |
| `trajectory` | Per-message sentiment + transitions | Yes (defaults to `--dry-run`) |
| `conflicts` | Per-session stance conflicts | Yes (defaults to `--dry-run`) |
| `cluster` | UMAP → HDBSCAN → c-TF-IDF | No (CPU only) |
| `community` | Louvain over session centroids | No (CPU only) |
| `analyze` | Everything above, in order | Yes (defaults to `--dry-run`) |

Full per-command flags and examples are in `references/commands.md`.

## Shape of the data

The transcript layer:

- `sessions` — one row per `.jsonl` file
- `messages` — one row per chat message, with `role`, `model`, and
  token usage
- `content_blocks` — flattened `message.content[]` (text, tool_use,
  tool_result, thinking)
- `tool_calls` / `tool_results` — paired by `tool_use_id`
- `todo_events` / `todo_state_current` — TodoWrite snapshots and the
  latest status per `(session, subject)`
- `task_spawns` / `subagent_sessions` / `subagent_messages` — subagent
  launch sites and rolled-up subagent runs

The analytics layer (registered only if the parquet exists):

- `session_classifications` — autonomy_tier, work_category, success,
  goal per session
- `message_trajectory` — per-message sentiment_delta + is_transition
- `session_conflicts` — stance_a, stance_b, resolution per conflicted
  session
- `message_clusters` — cluster_id + 2D UMAP viz coords
- `cluster_terms` — c-TF-IDF top terms per cluster
- `session_communities` — Louvain community per session

The macro layer (call these like SQL functions):

- `model_used(session_id)` → latest model observed
- `cost_estimate(session_id)` → USD spend with dated-ID prefix match
- `tool_rank(last_n_days)` → tool leaderboard
- `todo_velocity(session_id)` → completed / distinct todos
- `subagent_fanout(session_id)` → subagent runs for a session
- `semantic_search(query_vec, k)` → HNSW top-k
- `autonomy_trend(window_days)` → weekly autonomy mix
- `work_mix(since_days)` → work-category distribution
- `success_rate_by_work(since_days)` → success/fail/partial per category
- `cluster_top_terms(cid, n)` / `community_top_topics(cid, n)` /
  `sentiment_arc(sid)` — the rest of the analytics pivots

Run `claude-sql schema` any time to get the full list with column types.

## Reference files

Load the relevant reference file when the task calls for it — they're
small, focused, and meant to be read top-to-bottom when you need them.

- `references/commands.md` — every subcommand, its flags, and
  copy-pasteable examples. Read this when the user asks you to run a
  specific `claude-sql` command or you need to remember a flag.
- `references/recipes.md` — 13 runnable recipes mapped to the kinds of
  questions users actually ask (recall, spend, patterns, themes,
  contradictions). Read this first when the user asks an analytical
  question over their corpus.
- `references/jobs.md` — 7 jobs-to-be-done describing the multi-step
  workflows that combine several commands. Read this when the user's
  ask spans more than one command (e.g., "group my sessions and show
  me what's trending" = embed + cluster + community + c-TF-IDF).

## Golden rules

- **Read-only first.** Favor a SQL query over `sessions` / `messages` /
  macros before spending Bedrock tokens.
- **Always dry-run the spenders.** Every command that calls Bedrock
  defaults to `--dry-run`. Show the cost estimate before re-running
  with `--no-dry-run`.
- **Trust the caches.** Embeddings, classifications, clusters, and
  communities are all parquet-backed. Don't rebuild unless the user
  explicitly asked or new transcripts were added.
- **Don't touch cost estimation constants.** They live in `config.py`
  and are kept in sync with Bedrock list prices.
- **Never replace Louvain with python-louvain or add bertopic.** We
  use `networkx.community.louvain_communities` and sklearn-based
  c-TF-IDF on purpose — so the logic stays visible and patchable.
