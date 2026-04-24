# claude-sql Cookbook

Runnable SQL recipes against `~/.claude/projects/**/*.jsonl` via the
`claude-sql` CLI. These cover the **v1 surface**: `sessions`, `messages`,
`content_blocks`, `tool_calls`, `tool_results`, `todo_events`,
`subagent_*`, and `semantic_search`. The **v2 analytics surface** (clusters,
communities, classifications, trajectory, conflicts, friction) lives in
[`analytics_cookbook.md`](analytics_cookbook.md).

Run any recipe as:

```bash
claude-sql query "<SQL>"
```

First invocation takes tens of seconds because `register_all()` force-reads
the full corpus for schema inference (`sample_size=-1`). Subsequent queries
in the same process are fast. Drop into `claude-sql shell` if you want an
interactive REPL with every view and macro pre-registered.

All output shapes below are illustrative — they show column layout and
expected row counts, not data from any specific user's corpus.

## 1. Cost and models

### 1.1 Opus sessions over $5 in the last 30 days

The headline cost query: session-level spend on Opus for the last month,
filtered to `> $5`.

```sql
SELECT session_id,
       model_used(session_id) AS model,
       cost_estimate(session_id) AS usd
FROM sessions
WHERE started_at >= current_timestamp - INTERVAL 30 DAY
  AND model_used(session_id) LIKE '%opus%'
  AND cost_estimate(session_id) > 5.0
ORDER BY usd DESC
LIMIT 15;
```

`cost_estimate(sid)` joins `messages.model` against an inline `VALUES`
pricing table built from `config.DEFAULT_PRICING`:
`(input_tokens + cache_write) * in_rate + output_tokens * out_rate`, scaled
by `1e6`. Cache reads are excluded because they are free under current
Anthropic billing.

### 1.2 Biggest cache-read tokens in the last 7 days

Cache reads are a proxy for "reuse of a big system prompt across many
turns" — high values usually indicate a long session with lots of tool
output.

```sql
SELECT session_id, sum(cache_read) AS cache_read_tokens
FROM messages
WHERE ts >= current_timestamp - INTERVAL 7 DAY
GROUP BY session_id
ORDER BY cache_read_tokens DESC NULLS LAST
LIMIT 10;
```

### 1.3 Total estimated spend by model, last 30 days

```sql
SELECT model_used(session_id) AS model,
       sum(cost_estimate(session_id)) AS usd,
       count(*) AS n_sessions
FROM sessions
WHERE started_at >= current_timestamp - INTERVAL 30 DAY
  AND model_used(session_id) IS NOT NULL
GROUP BY 1
ORDER BY usd DESC NULLS LAST;
```

Two result artifacts to expect:

- **Dated model IDs** (e.g. `claude-haiku-4-5-20251001`) appear verbatim
  from the transcripts. The pricing table keys on the undated base name;
  `cost_estimate(sid)` strips a trailing `-YYYYMMDD` via `regexp_replace`
  before joining, so dated snapshots resolve to their base rate
  automatically.
- **`<synthetic>`** shows up for sessions where Claude Code injects a
  system-only message with no model field, and **`null`** shows up in
  older transcripts that predate the `message.model` field. Both are
  expected; `cost_estimate` returns `NULL` for them.

### 1.4 Thread walker (from `docs/queries/thread_walk.sql`)

Reconstruct conversation forests by chasing `parent_uuid → uuid`. Pin
to a single session for a fast walk; the corpus-wide version in
`thread_walk.sql` expands to millions of rows and is better run against a
filtered subset or paired with `EXPLAIN`.

```sql
WITH RECURSIVE thread AS (
    SELECT uuid AS thread_root_uuid, uuid AS descendant_uuid,
           session_id, 0 AS depth
    FROM messages
    WHERE parent_uuid IS NULL
      AND session_id = '<your-session-id>'
    UNION ALL
    SELECT t.thread_root_uuid, m.uuid AS descendant_uuid,
           m.session_id, t.depth + 1 AS depth
    FROM thread t
    JOIN messages m ON m.parent_uuid = t.descendant_uuid
    WHERE m.session_id = '<your-session-id>'
)
SELECT depth, count(*) AS n
FROM thread
GROUP BY depth
ORDER BY depth
LIMIT 20;
```

Per-depth counts typically stay at 1 in linear conversations and fan out
where the agent launched parallel sub-tasks via `Task`. The full query is
in [`docs/queries/thread_walk.sql`](queries/thread_walk.sql).

## 2. Tool usage

### 2.1 `tool_rank(30)` — top tools in the last 30 days

`tool_rank(last_n_days)` is the fastest way to get a tool-usage
leaderboard.

```sql
SELECT * FROM tool_rank(30) LIMIT 20;
```

Typical hits across a Claude Code corpus: `Bash`, `Read`, `Edit`, `Grep`,
`Write`, `Glob`, `TodoWrite`, `Task`, `Agent`, `Skill`, plus whatever MCP
tools you have installed.

### 2.2 Slowest Bash commands (call → result latency)

Pair each `tool_use` with its matching `tool_result` via `tool_use_id` and
take the wall-clock delta.

```sql
WITH bash_calls AS (
    SELECT session_id, tool_use_id, ts AS call_ts
    FROM tool_calls WHERE tool_name = 'Bash'
),
bash_results AS (
    SELECT tool_use_id, ts AS result_ts FROM tool_results
)
SELECT bc.session_id, bc.tool_use_id,
       date_diff('second', bc.call_ts, br.result_ts) AS secs
FROM bash_calls bc
JOIN bash_results br USING (tool_use_id)
WHERE date_diff('second', bc.call_ts, br.result_ts) BETWEEN 0 AND 3600
ORDER BY secs DESC
LIMIT 10;
```

The `BETWEEN 0 AND 3600` filter drops call/result pairs written
out-of-order in the transcript (rare; happens in compacted sessions where
the result arrives in a later file). Clean 3,001-second hits are Bash
invocations that ran to the default hook timeout.

### 2.3 Most common tool per session

`ROW_NUMBER()` over `(session, count)` gives the modal tool for each
session in one pass.

```sql
WITH ranked AS (
    SELECT session_id, tool_name, count(*) AS n,
           row_number() OVER (
               PARTITION BY session_id ORDER BY count(*) DESC
           ) AS rk
    FROM tool_calls
    WHERE tool_name IS NOT NULL
    GROUP BY session_id, tool_name
)
SELECT session_id, tool_name, n
FROM ranked
WHERE rk = 1
ORDER BY n DESC
LIMIT 10;
```

On heavy sessions the modal tool is almost always `Bash` — matching the
corpus-wide `tool_rank(30)` leaderboard.

## 3. Task tracking (todos)

### 3.1 Sessions that opened todos but completed none

`HAVING` post-filters the `todo_state_current` aggregates in the same pass.

```sql
SELECT session_id,
       count(DISTINCT subject) AS total,
       count(DISTINCT subject) FILTER (WHERE status = 'completed') AS completed
FROM todo_state_current
GROUP BY session_id
HAVING completed = 0 AND total >= 3
ORDER BY total DESC
LIMIT 10;
```

These are sessions that crashed mid-work or got abandoned. Good candidates
to feed into a retry or resume flow.

### 3.2 Longest-running `in_progress` todos

Todos stuck in `in_progress` since their last `TodoWrite` snapshot — oldest
first.

```sql
SELECT session_id, subject, status, written_at,
       date_diff('minute', written_at, current_timestamp) AS age_minutes
FROM todo_state_current
WHERE status = 'in_progress'
ORDER BY written_at ASC
LIMIT 10;
```

A cluster of stale `in_progress` todos all sharing the same `session_id`
typically means one long-running session was interrupted before
`TodoWrite` could flip them to `completed`.

### 3.3 Todo velocity distribution

`todo_velocity(sid) = completed / distinct_subjects`. Bucketing across all
sessions shows whether todos tend to close out or rot.

```sql
WITH per_session AS (
    SELECT session_id, todo_velocity(session_id) AS v
    FROM (SELECT DISTINCT session_id FROM todo_state_current) s
),
bucketed AS (
    SELECT CASE
        WHEN v IS NULL              THEN 'null'
        WHEN v = 0.0                THEN '0.0'
        WHEN v < 0.25               THEN '(0, 0.25)'
        WHEN v < 0.5                THEN '[0.25, 0.5)'
        WHEN v < 0.75               THEN '[0.5, 0.75)'
        WHEN v < 1.0                THEN '[0.75, 1.0)'
        WHEN v = 1.0                THEN '1.0 (all done)'
        ELSE '>1.0'
    END AS bucket
    FROM per_session
)
SELECT bucket, count(*) AS sessions FROM bucketed GROUP BY 1 ORDER BY 1;
```

This distribution is typically bimodal: most sessions either finish
everything or finish nothing, with a smaller tail of partial completions.

## 4. Subagents

### 4.1 Subagent fan-out per session, last 30 days

`subagent_fanout(sid)` counts rows in `subagent_sessions` whose
`parent_session_id` matches.

```sql
SELECT parent_session_id,
       subagent_fanout(parent_session_id) AS fanout
FROM (
    SELECT DISTINCT parent_session_id
    FROM subagent_sessions
    WHERE started_at >= current_timestamp - INTERVAL 30 DAY
) s
ORDER BY fanout DESC
LIMIT 10;
```

### 4.2 Which `agent_type`s are used most often

```sql
SELECT agent_type, count(*) AS runs
FROM subagent_sessions
WHERE agent_type IS NOT NULL
GROUP BY agent_type
ORDER BY runs DESC
LIMIT 15;
```

Common values: `Explore`, `general-purpose`, `Plan`, plus whatever custom
plugin agents you have registered (names like
`<plugin>:<agent-name>`).

### 4.3 Which subagent types burn the most tokens

Join `subagent_messages` on `(parent_session_id, agent_hex)` to roll
messages up into the subagent run, then sum by `agent_type`.

```sql
SELECT s.agent_type,
       sum(coalesce(m.input_tokens, 0) + coalesce(m.output_tokens, 0)) AS total_tokens,
       count(DISTINCT s.agent_hex) AS runs
FROM subagent_sessions s
JOIN subagent_messages m
  ON m.parent_session_id = s.parent_session_id
 AND m.agent_hex = s.agent_hex
WHERE s.agent_type IS NOT NULL
GROUP BY s.agent_type
ORDER BY total_tokens DESC
LIMIT 10;
```

## 5. Semantic search (requires embeddings parquet)

Run the backfill first — `claude-sql search` exits `2` with a hint if the
embeddings parquet doesn't exist yet:

```bash
AWS_PROFILE=your-profile claude-sql embed --since-days 7
```

`register_vss()` builds the HNSW index at connection open from
`~/.claude/embeddings.parquet`. Until that parquet exists, the
`message_embeddings` table is empty. `--since-days 7` is a cheap starter;
ramp to `30` or `90` once you've confirmed the pipeline.

### 5.1 `claude-sql search "<query>" --k 5`

The CLI computes a query embedding via Cohere Embed v4 with
`input_type=search_query`, then runs the shape below (equivalent SQL after
the CLI binds the `FLOAT[1024]` query vector):

```sql
WITH qv AS (SELECT CAST(? AS FLOAT[1024]) AS v)
SELECT m.uuid,
       m.session_id,
       m.role,
       array_cosine_similarity(me.embedding, (SELECT v FROM qv)) AS sim,
       substr(mt.text_content, 1, 200) AS snippet
FROM message_embeddings me
JOIN messages m USING (uuid)
LEFT JOIN messages_text mt ON mt.uuid = m.uuid
ORDER BY array_distance(me.embedding, (SELECT v FROM qv))
LIMIT ?;
```

`ORDER BY array_distance LIMIT k` is the specific pattern DuckDB VSS
recognizes and rewrites into an HNSW index scan (look for
`HNSW_INDEX_SCAN` in `claude-sql explain`). `array_cosine_similarity` is
carried alongside so result rows surface a human-readable similarity
number.

CLI shape:

```bash
claude-sql search "temporal workflow determinism" --k 10
```

### 5.2 Messages semantically similar to a known uuid

Once embeddings exist, "find neighbors of message X" is a self-join on
`message_embeddings` (no query-side embedding needed):

```sql
WITH seed AS (
    SELECT embedding AS v
    FROM message_embeddings
    WHERE uuid = '<known-message-uuid>'
)
SELECT me.uuid,
       array_cosine_similarity(me.embedding, (SELECT v FROM seed)) AS sim
FROM message_embeddings me
WHERE me.uuid <> '<known-message-uuid>'
ORDER BY array_distance(me.embedding, (SELECT v FROM seed))
LIMIT 10;
```

Primary use case: given an interesting message, surface other times you
asked the same question or hit the same bug.

## 6. Skills

`skill_invocations` captures every way a user or the assistant reaches for a
named Skill in the transcripts:

- `source = 'tool'` — the assistant invoked the built-in `Skill` tool with
  `tool_input = {"skill": "<name>", "args": "..."}`.
- `source = 'slash_command'` — the user typed `/<name>` in chat, which
  Claude Code serializes as `<command-name>/<name></command-name>` inside
  the user-role text block.

`skills_catalog` is the seedable side — a parquet written by
`claude-sql skills sync` that walks `~/.claude/skills/` and
`~/.claude/plugins/cache/**` to bind a `skill_id` (bare *and* namespaced
`plugin:name`) to its human description, owning plugin, and version.
Built-in slash commands (`/clear`, `/compact`, `/plugin`, `/mcp`, …) are
also seeded and tagged with `source_kind = 'builtin'`. The
`skill_usage` view joins the two so every row is enriched with
`skill_name`, `plugin`, and `is_builtin`.

### 6.1 Seed the catalog

```bash
claude-sql skills sync
```

Run it whenever you install or upgrade a plugin. The output parquet is
cheap to rebuild (filesystem walk, no Bedrock). `claude-sql analyze`
runs it automatically as step 0; pass `--skip-skills-sync` to freeze the
catalog.

### 6.2 Top skills in the last 30 days

```sql
SELECT * FROM skill_rank(30) LIMIT 20;
```

Returns `skill_id, skill_name, plugin, is_builtin, n, sessions`. Filter
out the built-ins to focus on real skills:

```sql
SELECT skill_id, skill_name, plugin, n, sessions
FROM skill_rank(30)
WHERE NOT is_builtin
ORDER BY n DESC
LIMIT 20;
```

### 6.3 How is each skill invoked?

`/erpaval` from chat and an assistant-driven `Skill` call are two valid
shapes — `skill_source_mix` splits them.

```sql
SELECT * FROM skill_source_mix(30) LIMIT 15;
```

Returns `skill_id, skill_name, n_tool, n_slash, n_total`. Built-ins are
excluded — they'd drown the rest. A skill with `n_tool = 0, n_slash > 0`
is one you drive by hand; `n_tool > 0, n_slash = 0` is one you let the
agent reach for.

### 6.4 Skills you have but never use

```sql
SELECT * FROM unused_skills(30) LIMIT 20;
```

Returns every `user-skill` / `plugin-skill` / `plugin-command` in the
catalog that the window missed. Useful after installing a big plugin
pack — lets you spot the skills that never caught on.

### 6.5 Join through to see what a skill is *for*

Chain `skill_usage` back to `messages_text` to read the actual user
intent behind each invocation:

```sql
SELECT su.ts, su.source, su.skill_name, su.plugin,
       substr(mt.text_content, 1, 160) AS context
FROM skill_usage su
JOIN messages_text mt ON mt.uuid = su.message_uuid
WHERE su.skill_id = 'erpaval'
ORDER BY su.ts DESC
LIMIT 10;
```

## Explain: prove the pushdown

```bash
claude-sql explain "SELECT uuid, role FROM messages WHERE session_id = '<your-session-id>'"
```

Green-highlighted markers worth confirming in the plan:

- `READ_JSON` — zero-copy scan at the leaf.
- `Projections: type, sessionId, message.role, uuid` — only a handful of
  the inferred columns hit the scan.
- `Filters: (sessionId = ...)` — filter attached to the scan.

Session-pinned queries still scan every file because `sessionId` is a
*field* inside each JSONL, not a partition key. Add
`AND source_file LIKE '%<session_id>.jsonl'` to let the filename filter
prune the scan to a single file.
