# claude-sql Cookbook

Runnable SQL recipes against `~/.claude/projects/**/*.jsonl` via the
`claude-sql` CLI. All outputs below were captured on 2026-04-19 against the
dev-host corpus (9,503 top-level sessions, 3,145 subagent sessions, 10,077
`TodoWrite` events).

Run any recipe as:

```bash
uv run claude-sql query "<SQL>"
```

First invocation takes ~30-60s because `register_all()` force-reads the full
corpus for schema inference (`sample_size=-1`). Subsequent queries in the
same process are fast.

## 1. Cost and models

### 1.1 Opus sessions over $5 in the last 30 days (headline acceptance query)

The headline work-item acceptance check: session-level spend on Opus for the
last month, filtered to >$5.

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

```
shape: (15, 3)
┌──────────────────────────────────────┬─────────────────┬─────────────┐
│ session_id                           ┆ model           ┆ usd         │
╞══════════════════════════════════════╪═════════════════╪═════════════╡
│ 86f03e27-e763-4d1b-9399-5e2688a856bb ┆ claude-opus-4-6 ┆ 2903.690475 │
│ f2cb561a-cfba-4645-8e21-d5c04af56ae6 ┆ claude-opus-4-6 ┆ 2880.38313  │
│ a019490e-da7e-4f8b-a11f-af03b76d5cc1 ┆ claude-opus-4-6 ┆ 2050.905585 │
│ bc805709-55ca-4503-b0bd-456f9ffb3434 ┆ claude-opus-4-6 ┆ 1909.684695 │
│ 4920d6e1-93db-4bee-8ce6-5043dc40866b ┆ claude-opus-4-6 ┆ 1900.71993  │
│ 4a5594b5-172d-46b9-b670-3941823551bd ┆ claude-opus-4-6 ┆ 1887.793245 │
│ 0e466ed0-bdd6-4a34-9e8a-43892ee9bfdc ┆ claude-opus-4-6 ┆ 1861.24212  │
│ 6bfa0f3f-1baa-4a0f-939d-23942b5f3f1d ┆ claude-opus-4-6 ┆ 1725.268335 │
│ 5e33d13f-161d-407f-910c-ca91791d9286 ┆ claude-opus-4-6 ┆ 1510.131375 │
│ b67b7b89-a6b5-40fa-a5b5-f563bd19f18b ┆ claude-opus-4-6 ┆ 1485.66546  │
│ 028427a7-444e-4bf9-9ff8-b48746e844e2 ┆ claude-opus-4-6 ┆ 1377.343515 │
│ e96aaa54-6761-42b6-9918-ee313cfafd8c ┆ claude-opus-4-6 ┆ 1371.81171  │
│ ec27e4c8-77ee-43c8-8e49-70827ac73126 ┆ claude-opus-4-6 ┆ 1331.03439  │
│ 24ff2db2-f527-4a9b-a905-82028ec5fc66 ┆ claude-opus-4-6 ┆ 1321.391415 │
│ 20568f85-99ce-43af-8bda-08cd31aede37 ┆ claude-opus-4-6 ┆ 1271.3637   │
└──────────────────────────────────────┴─────────────────┴─────────────┘
```

`cost_estimate(sid)` joins `messages.model` against an inline `VALUES`
pricing table built from `config.DEFAULT_PRICING`: `(input_tokens +
cache_write) * in_rate + output_tokens * out_rate`, scaled by `1e6`. Cache
reads are free under current Anthropic billing, so they are excluded.

### 1.2 Biggest cache-read tokens in the last 7 days

Cache reads are a proxy for "reuse of a big system prompt across many turns"
-- high values usually mean a long session with extensive tool output.

```sql
SELECT session_id, sum(cache_read) AS cache_read_tokens
FROM messages
WHERE ts >= current_timestamp - INTERVAL 7 DAY
GROUP BY session_id
ORDER BY cache_read_tokens DESC NULLS LAST
LIMIT 10;
```

```
shape: (10, 2)
┌──────────────────────────────────────┬───────────────────┐
│ session_id                           ┆ cache_read_tokens │
╞══════════════════════════════════════╪═══════════════════╡
│ 23971597-ad55-4e82-8997-f062a688aabe ┆ 158221595         │
│ c8c6812d-ac80-4e0a-b0b2-5b145a4bd303 ┆ 127796982         │
│ 50186830-03b7-4629-b317-e7bb94f9272a ┆ 111643253         │
│ 15cfb101-5d74-4b22-a83c-061298ddaa4a ┆ 105413711         │
│ 19fcafdb-16aa-484f-b5ea-2f4098cbad20 ┆ 97279492          │
│ 79a7c8ab-02d7-468d-852e-94ce598a44cd ┆ 78029826          │
│ 2e6ec4cf-4c53-400a-8d24-bd742d542e44 ┆ 61516350          │
│ a552031d-7e4f-4bac-998d-958eee6d4bba ┆ 58652968          │
│ 26c5d23a-997b-418a-a0b3-0549566808c2 ┆ 47912055          │
│ af6caab1-665d-4f4c-b7da-bb26b9cdeaf2 ┆ 44759593          │
└──────────────────────────────────────┴───────────────────┘
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

```
shape: (6, 3)
┌───────────────────────────┬─────────────┬────────────┐
│ model                     ┆ usd         ┆ n_sessions │
╞═══════════════════════════╪═════════════╪════════════╡
│ claude-opus-4-6           ┆ 99933.96744 ┆ 1558       │
│ claude-opus-4-7           ┆ 1228.48539  ┆ 43         │
│ claude-sonnet-4-6         ┆ 643.74969   ┆ 851        │
│ null                      ┆ null        ┆ 43         │
│ <synthetic>               ┆ null        ┆ 9          │
│ claude-haiku-4-5-20251001 ┆ null        ┆ 6984       │
└───────────────────────────┴─────────────┴────────────┘
```

Two things jump out:
- `claude-haiku-4-5-20251001` is the dated model ID as it appears in the
  transcripts. The pricing table in `config.DEFAULT_PRICING` keys on
  `claude-haiku-4-5` (undated), so `cost_estimate` returns NULL for those
  sessions. Fix: either normalize the dated suffix inside the macro or add
  the dated key to the pricing dict.
- `<synthetic>` appears in sessions where Claude Code injects a system
  message with no model, and `null` is the older transcripts where
  `message.model` wasn't written.

### 1.4 Thread walker excerpt (from `docs/queries/thread_walk.sql`)

The recursive CTE below reconstructs conversation forests by chasing
`parent_uuid` -> `uuid`. Scoped here to one session (the corpus-wide version
in `thread_walk.sql` expands to millions of rows and is better run against
a pre-filtered subset or paired with `EXPLAIN`).

```sql
WITH RECURSIVE thread AS (
    SELECT uuid AS thread_root_uuid, uuid AS descendant_uuid,
           session_id, 0 AS depth
    FROM messages
    WHERE parent_uuid IS NULL
      AND session_id = 'f2cb561a-cfba-4645-8e21-d5c04af56ae6'
    UNION ALL
    SELECT t.thread_root_uuid, m.uuid AS descendant_uuid,
           m.session_id, t.depth + 1 AS depth
    FROM thread t
    JOIN messages m ON m.parent_uuid = t.descendant_uuid
    WHERE m.session_id = 'f2cb561a-cfba-4645-8e21-d5c04af56ae6'
)
SELECT depth, count(*) AS n
FROM thread
GROUP BY depth
ORDER BY depth
LIMIT 20;
```

```
shape: (20, 2)
┌───────┬─────┐
│ depth ┆ n   │
╞═══════╪═════╡
│   0   ┆  1  │
│   1   ┆  1  │
│   2   ┆  1  │
│   3   ┆  1  │
│   4   ┆  1  │
│   5   ┆  1  │
│   6   ┆  1  │
│   7   ┆  1  │
│   8   ┆  2  │
│   9   ┆  3  │
│  10   ┆  2  │
│  11   ┆  1  │
│  12   ┆  2  │
│  13   ┆  2  │
│  14   ┆  1  │
│  15   ┆  1  │
│  16   ┆  1  │
│  17   ┆  1  │
│  18   ┆  1  │
│  19   ┆  1  │
└───────┴─────┘
```

Thread fan-out at depth 8-13 is where this session spawned parallel sub-tasks
via `Task`. The full query is in `docs/queries/thread_walk.sql`.

## 2. Tool usage

### 2.1 `tool_rank(30)` -- top tools in the last 30 days

The `tool_rank(last_n_days)` table macro is the fastest way to get a
tool-usage leaderboard.

```sql
SELECT * FROM tool_rank(30) LIMIT 20;
```

```
shape: (20, 2)
┌──────────────────────────────────┬───────┐
│ tool_name                        ┆ n     │
╞══════════════════════════════════╪═══════╡
│ Bash                             ┆ 14167 │
│ Read                             ┆ 6520  │
│ ToolSearch                       ┆ 3503  │
│ Edit                             ┆ 3025  │
│ mcp__probe-tools__work_list ┆ 2268  │
│ Grep                             ┆ 2145  │
│ StructuredOutput                 ┆ 1779  │
│ TaskUpdate                       ┆ 1643  │
│ Agent                            ┆ 1591  │
│ Write                            ┆ 981   │
│ TodoWrite                        ┆ 837   │
│ Glob                             ┆ 818   │
│ TaskCreate                       ┆ 711   │
│ mcp__internal__slack_post_blocks     ┆ 682   │
│ mcp__tasks__task_update          ┆ 676   │
│ mcp__slack__slack_post_markdown  ┆ 554   │
│ mcp__tasks__task_create          ┆ 321   │
│ mcp__internal__slack_upload_file     ┆ 288   │
│ mcp__internal__otel_logs             ┆ 215   │
│ Skill                            ┆ 198   │
└──────────────────────────────────┴───────┘
```

### 2.2 Slowest Bash commands (call -> result latency)

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

```
shape: (10, 3)
┌──────────────────────────────────────┬─────────────────────────────────────┬──────┐
│ session_id                           ┆ tool_use_id                         ┆ secs │
╞══════════════════════════════════════╪═════════════════════════════════════╪══════╡
│ 4f352191-5e95-4121-b2d6-24f7f69ba16f ┆ toolu_bdrk_019uAhKWAExnBmSBovQYC81E ┆ 3001 │
│ af4d4f2f-e8fe-4712-ab96-15d0099f64bd ┆ toolu_bdrk_014uqP2TvzRnQiCjciqWT4iE ┆ 3001 │
│ 4f352191-5e95-4121-b2d6-24f7f69ba16f ┆ toolu_bdrk_01PPWUiYsQZv7yTtCWiM3aw1 ┆ 1804 │
│ 4f352191-5e95-4121-b2d6-24f7f69ba16f ┆ toolu_bdrk_01MrPjiZ5Ge5HDLUdvdvLpRP ┆ 1802 │
│ ad08e6e2-c97d-48dc-9a2f-4327e9911bcb ┆ toolu_bdrk_01FL8J51qAYqDzbLtmHqyrmK ┆ 1802 │
│ af4d4f2f-e8fe-4712-ab96-15d0099f64bd ┆ toolu_bdrk_01U5NjuPjT4cYhiZE24jx5iC ┆ 1802 │
│ af4d4f2f-e8fe-4712-ab96-15d0099f64bd ┆ toolu_bdrk_01L6ZUpz6hdtHnWv7Nciw8Kt ┆ 1801 │
│ 200fd5b4-92cf-4c4d-a6a9-9bedeee8dca1 ┆ toolu_bdrk_01LkhuZeafGexfZixAd92eiM ┆ 1801 │
│ 6d9a14b0-6bfb-478f-ace1-6a621dd96945 ┆ toolu_bdrk_01PgTtfTSQ6XAoUa6XYDQ6L3 ┆ 1333 │
│ 4f352191-5e95-4121-b2d6-24f7f69ba16f ┆ toolu_bdrk_01VXymkaZWDLTHhXtCLfPPsS ┆ 1202 │
└──────────────────────────────────────┴─────────────────────────────────────┴──────┘
```

Two clean 3,001-second hits are Bash invocations that ran to the default
hook timeout. The `BETWEEN 0 AND 3600` filter drops call/result pairs that
were written out-of-order in the transcript (rare; happens in compacted
sessions where the result arrives in a later file).

### 2.3 Most common tool per session

`ROW_NUMBER()` over `(session, count)` gives the modal tool for each session
in one pass.

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

```
shape: (10, 3)
┌──────────────────────────────────────┬───────────┬─────┐
│ session_id                           ┆ tool_name ┆ n   │
╞══════════════════════════════════════╪═══════════╪═════╡
│ a019490e-da7e-4f8b-a11f-af03b76d5cc1 ┆ Bash      ┆ 239 │
│ 23971597-ad55-4e82-8997-f062a688aabe ┆ Bash      ┆ 191 │
│ c8c6812d-ac80-4e0a-b0b2-5b145a4bd303 ┆ Bash      ┆ 188 │
│ 24ff2db2-f527-4a9b-a905-82028ec5fc66 ┆ Bash      ┆ 185 │
│ bc805709-55ca-4503-b0bd-456f9ffb3434 ┆ Bash      ┆ 178 │
│ f2cb561a-cfba-4645-8e21-d5c04af56ae6 ┆ Bash      ┆ 172 │
│ 19fcafdb-16aa-484f-b5ea-2f4098cbad20 ┆ Bash      ┆ 165 │
│ b67b7b89-a6b5-40fa-a5b5-f563bd19f18b ┆ Bash      ┆ 156 │
│ 0e466ed0-bdd6-4a34-9e8a-43892ee9bfdc ┆ Bash      ┆ 156 │
│ 4920d6e1-93db-4bee-8ce6-5043dc40866b ┆ Bash      ┆ 151 │
└──────────────────────────────────────┴───────────┴─────┘
```

The modal tool across the heaviest sessions is `Bash` in every case -- matches
the corpus-wide `tool_rank(30)` result above.

## 3. Task tracking (todos)

### 3.1 Sessions that opened todos but completed none

`HAVING` lets us post-filter the `todo_state_current` aggregates in the same
pass.

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

```
shape: (10, 3)
┌──────────────────────────────────────┬───────┬───────────┐
│ session_id                           ┆ total ┆ completed │
╞══════════════════════════════════════╪═══════╪═══════════╡
│ 1aaab549-bfd7-44de-bd6d-df5ca5af692a ┆ 10    ┆ 0         │
│ 0dc258f4-6584-45be-b63c-0dbafe18a1a3 ┆ 9     ┆ 0         │
│ 02dd6978-a9ef-4246-a16d-0a5bc240e006 ┆ 8     ┆ 0         │
│ b1f7e8ee-5b31-423c-a90e-166fa653ff48 ┆ 7     ┆ 0         │
│ 4ce56fc3-1afa-4e1a-897a-0991a802c8c3 ┆ 6     ┆ 0         │
│ b0069b58-b293-4ce9-848e-1fa0b1259172 ┆ 5     ┆ 0         │
│ 9374889f-a2f9-42ae-aaaa-4478034dbf49 ┆ 5     ┆ 0         │
│ a2eff17f-f4c1-4e3c-a474-f60b8dfc3fa6 ┆ 5     ┆ 0         │
│ 378d2790-d7b8-450f-a38b-55d632162e53 ┆ 5     ┆ 0         │
│ fcc41304-1d2c-47c3-9860-e202eb1a0132 ┆ 5     ┆ 0         │
└──────────────────────────────────────┴───────┴───────────┘
```

Typically sessions that crashed mid-work or got abandoned. Worth feeding the
top few into a retry/resume flow.

### 3.2 Longest-running `in_progress` todos

Todos stuck in `in_progress` since their last `TodoWrite` snapshot -- oldest
first.

```sql
SELECT session_id, subject, status, written_at,
       date_diff('minute', written_at, current_timestamp) AS age_minutes
FROM todo_state_current
WHERE status = 'in_progress'
ORDER BY written_at ASC
LIMIT 10;
```

```
shape: (10, 5)
┌──────────────────────────────────────┬─────────────────────────────────────┬─────────────┬─────────────────────────┬─────────────┐
│ session_id                           ┆ subject (truncated)                 ┆ status      ┆ written_at              ┆ age_minutes │
╞══════════════════════════════════════╪═════════════════════════════════════╪═════════════╪═════════════════════════╪═════════════╡
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 2C: Memory performance …       ┆ in_progress ┆ 2026-03-20 20:02:16.927 ┆ 42306       │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 2B: Access log decoupling …    ┆ in_progress ┆ 2026-03-20 20:02:16.927 ┆ 42306       │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 2A: Scheduling guardrails …    ┆ in_progress ┆ 2026-03-20 20:02:16.927 ┆ 42306       │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 2D: Slack rate limiting …      ┆ in_progress ┆ 2026-03-20 20:02:16.927 ┆ 42306       │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 3B: Externalize paths …        ┆ in_progress ┆ 2026-03-20 20:17:42.379 ┆ 42291       │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 3C: memory OTEL spans …   ┆ in_progress ┆ 2026-03-20 20:17:42.379 ┆ 42291       │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 3A: events.py extraction …     ┆ in_progress ┆ 2026-03-20 20:17:42.379 ┆ 42291       │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 3: Architecture (3 agents)     ┆ in_progress ┆ 2026-03-20 20:17:42.379 ┆ 42291       │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 4F: Memory search recall …     ┆ in_progress ┆ 2026-03-20 20:25:46.821 ┆ 42283       │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ Wave 4J: OTEL continuous profiling  ┆ in_progress ┆ 2026-03-20 20:25:46.821 ┆ 42283       │
```

Output truncated here for page width; run it yourself to get the full
`subject` column. Every stale one of these belongs to the same
refactoring session from 2026-03-20.

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

```
shape: (6, 2)
┌────────────────┬──────────┐
│ bucket         ┆ sessions │
╞════════════════╪══════════╡
│ 0.0            ┆ 19       │
│ 1.0 (all done) ┆ 64       │
│ [0.25, 0.5)    ┆ 7        │
│ [0.5, 0.75)    ┆ 18       │
│ [0.75, 1.0)    ┆ 19       │
│ null           ┆ 1        │
└────────────────┴──────────┘
```

Bimodal: most sessions either finish everything (64) or finish nothing (19).
The middle buckets are the partial-completion tail.

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

```
shape: (10, 2)
┌──────────────────────────────────────┬────────┐
│ parent_session_id                    ┆ fanout │
╞══════════════════════════════════════╪════════╡
│ f2cb561a-cfba-4645-8e21-d5c04af56ae6 ┆ 29     │
│ 4a5594b5-172d-46b9-b670-3941823551bd ┆ 28     │
│ ec27e4c8-77ee-43c8-8e49-70827ac73126 ┆ 25     │
│ 098eb22c-8936-4067-8987-5859b25df99a ┆ 24     │
│ 38933071-11fb-426e-92d8-c52c66d852e9 ┆ 22     │
│ 20568f85-99ce-43af-8bda-08cd31aede37 ┆ 22     │
│ 028427a7-444e-4bf9-9ff8-b48746e844e2 ┆ 19     │
│ 2e6ec4cf-4c53-400a-8d24-bd742d542e44 ┆ 18     │
│ 45d1bfc0-07c6-4c84-b1db-881c63f339fa ┆ 18     │
│ 23971597-ad55-4e82-8997-f062a688aabe ┆ 17     │
└──────────────────────────────────────┴────────┘
```

### 4.2 Which agent_types are used most often

```sql
SELECT agent_type, count(*) AS runs
FROM subagent_sessions
WHERE agent_type IS NOT NULL
GROUP BY agent_type
ORDER BY runs DESC
LIMIT 15;
```

```
shape: (15, 2)
┌─────────────────────────────────────────────┬──────┐
│ agent_type                                  ┆ runs │
╞═════════════════════════════════════════════╪══════╡
│ Explore                                     ┆ 629  │
│ general-purpose                             ┆ 591  │
│ relationship-mapper                         ┆ 93   │
│ arc-synthesizer                             ┆ 93   │
│ behavior-synthesizer                        ┆ 93   │
│ gap-analyst                                 ┆ 93   │
│ synthesizer                                 ┆ 93   │
│ personal-plugins:code-researcher            ┆ 88   │
│ contradiction-hunter                        ┆ 75   │
│ personal-plugins:researcher                 ┆ 64   │
│ skill-extractor                             ┆ 42   │
│ personal-plugins:web-browser                ┆ 23   │
│ Plan                                        ┆ 19   │
│ personal-plugins:deep-researcher            ┆ 14   │
│ personal-plugins:claude-agent-sdk-assistant ┆ 13   │
└─────────────────────────────────────────────┴──────┘
```

### 4.3 Which subagent types burn the most tokens

Join `subagent_messages` on `(parent_session_id, agent_hex)` to roll
messages up into the subagent run, then by `agent_type`.

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

```
shape: (10, 3)
┌──────────────────────────────────┬──────────────┬──────┐
│ agent_type                       ┆ total_tokens ┆ runs │
╞══════════════════════════════════╪══════════════╪══════╡
│ Explore                          ┆ 1615798528   ┆ 629  │
│ general-purpose                  ┆ 1389191131   ┆ 591  │
│ arc-synthesizer                  ┆ 231876798    ┆ 93   │
│ personal-plugins:code-researcher ┆ 199717177    ┆  88  │
│ gap-analyst                      ┆ 185171075    ┆  93  │
│ relationship-mapper              ┆ 150907242    ┆  93  │
│ Plan                             ┆  89989907    ┆  19  │
│ personal-plugins:researcher      ┆  74939536    ┆  64  │
│ personal-plugins:web-browser     ┆  59235887    ┆  23  │
│ synthesizer                      ┆  57608573    ┆  93  │
└──────────────────────────────────┴──────────────┴──────┘
```

## 5. Semantic search (requires embeddings parquet)

**This host has no embeddings yet.** Run the backfill first:

```bash
AWS_PROFILE=lalsaado-handson uv run claude-sql embed --since-days 7
```

`register_vss()` builds the HNSW index at connection open from
`~/.claude/embeddings.parquet` -- until that parquet exists, the
`message_embeddings` table is empty and `claude-sql search` exits 2 with the
hint above. `semantic_search --since-days 7` is a cheap starter; ramp to 30
or 90 once you've confirmed the pipeline.

### 5.1 `claude-sql search "<query>" --k 5`

Under the hood the CLI computes a query embedding via Cohere Embed v4 with
`input_type=search_query`, then runs this shape (equivalent SQL after the
CLI binds the FLOAT[1024] query vector):

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
recognizes to rewrite into an HNSW index scan (look for `HNSW_INDEX_SCAN` in
`claude-sql explain`). `array_cosine_similarity` is carried alongside so
result rows surface a human-readable similarity number.

Expected CLI shape once embeddings exist:

```bash
$ uv run claude-sql search "temporal workflow determinism" --k 10
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

Primary use case: given an interesting message, surface the other times you
asked the same question or hit the same bug.

## Explain: prove the pushdown

```bash
uv run claude-sql explain "SELECT uuid, role FROM messages WHERE session_id = '20fed4d7-6dd4-46bb-b1d0-8dbdeedaaded'"
```

Green-highlighted markers in the plan:
- `READ_JSON` -- zero-copy scan at the leaf.
- `Projections: type, sessionId, message.role, uuid` -- only 4 of 15 inferred
  columns hit the scan.
- `Filters: (sessionId = ...)` -- filter attached to the scan.

Session-pinned queries still scan every file because `sessionId` is a *field*
inside each JSONL, not a partition key. Add `AND source_file LIKE
'%<session_id>.jsonl'` to let the filename filter prune the scan to a single
file.
