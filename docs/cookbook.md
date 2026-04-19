# claude-sql Cookbook

Practical SQL queries against `~/.claude/projects/**/*.jsonl` using the
`claude-sql` CLI. All examples assume you've run `claude-sql schema` at least
once to verify views are registered.

Run any recipe as: `uv run claude-sql query "<SQL>"`.

Corpus at time of writing (2026-04-19): 9,503 top-level sessions,
3,145 subagent sessions, 10,077 TodoWrite events.

## 1. Opus sessions over $5 in the last 30 days

```sql
SELECT session_id, cost_estimate(session_id) AS usd
FROM sessions
WHERE started_at >= current_date - INTERVAL 30 DAY
  AND model_used(session_id) ILIKE '%opus%'
  AND cost_estimate(session_id) > 5.0
ORDER BY usd DESC
LIMIT 10;
```

Output (real corpus, 2026-04-19):

```
┌──────────────────────────────────────┬─────────────┐
│ session_id                           ┆ usd         │
├──────────────────────────────────────┼─────────────┤
│ 86f03e27-e763-4d1b-9399-5e2688a856bb ┆ 2903.690475 │
│ f2cb561a-cfba-4645-8e21-d5c04af56ae6 ┆ 2880.38313  │
│ a019490e-da7e-4f8b-a11f-af03b76d5cc1 ┆ 2050.905585 │
│ bc805709-55ca-4503-b0bd-456f9ffb3434 ┆ 1909.684695 │
│ 4920d6e1-93db-4bee-8ce6-5043dc40866b ┆ 1900.71993  │
│ ...                                  ┆             │
└──────────────────────────────────────┴─────────────┘
```

`cost_estimate` joins `messages.model` to `config.DEFAULT_PRICING`; uncached
input + cache-write tokens are billed at `in_rate`, outputs at `out_rate`,
scaled by 1e6. Cache reads are excluded.

## 2. Sessions where todos went stale

Sessions that logged >3 todos but never completed any.

```sql
SELECT session_id
FROM todo_state_current
GROUP BY session_id
HAVING sum(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) = 0
   AND count(*) > 3
LIMIT 10;
```

Run it to see results — expect a mix of aborted refactors, crashed sessions,
and exploratory threads the user walked away from.

## 3. Longest-running in-progress todos

The ones still marked `in_progress` with the oldest `written_at` — prime
candidates for "I forgot about that."

```sql
SELECT written_at, subject
FROM todo_state_current
WHERE status = 'in_progress'
ORDER BY written_at
LIMIT 10;
```

`todo_state_current` is the per-(session, subject) latest row from
`todo_events`, so `written_at` is the moment that todo was last written
(not when it started).

## 4. Subagent fan-out per session in the last 30 days

How many subagent runs each recent session kicked off.

```sql
SELECT session_id, subagent_fanout(session_id) AS n
FROM sessions
WHERE started_at >= current_date - INTERVAL 30 DAY
  AND subagent_fanout(session_id) > 0
ORDER BY n DESC
LIMIT 10;
```

`subagent_fanout` counts rows in `subagent_sessions` where
`parent_session_id` matches. Subagent transcripts live in
`~/.claude/projects/<project>/<session>/subagents/agent-*.jsonl` and surface
through the dedicated `subagent_sessions` / `subagent_messages` views.

## 5. Which subagent types consume the most events

Total message events per subagent type — a proxy for how heavily each type
gets used.

```sql
SELECT agent_type, sum(message_count) AS msgs, count(*) AS runs
FROM subagent_sessions
GROUP BY 1
ORDER BY msgs DESC;
```

`agent_type` comes from the sibling `agent-<hex>.meta.json` files, which
`register_raw` registers as `v_raw_subagent_meta` and `subagent_sessions`
joins in. Rows with NULL `agent_type` are subagents that ran before metadata
was written (or where the meta.json got pruned).

## 6. Tool-use leaderboard, last 7 days

```sql
SELECT * FROM tool_rank(7) LIMIT 20;
```

`tool_rank` is a table macro over `tool_calls` (content_blocks where
`block_type='tool_use'`) filtered to the last N days and grouped by
`tool_name`. Expect `Bash`, `Read`, `Edit`, and the most-active MCP tools
near the top.

## 7. Semantic search

Requires embeddings. Generate them once with:

```bash
uv run claude-sql embed --since-days 30
```

Then use the CLI:

```bash
uv run claude-sql search "temporal determinism" --k 5
```

Or the equivalent SQL, which is what the CLI wraps:

```sql
WITH qv AS (SELECT embed_query('temporal determinism') AS v)
SELECT m.uuid,
       m.session_id,
       m.role,
       s.sim,
       substr(mt.text_content, 1, 200) AS snippet
FROM semantic_search((SELECT v FROM qv), 5) s
JOIN messages m USING (uuid)
LEFT JOIN messages_text mt ON mt.uuid = m.uuid;
```

`semantic_search(query_vec, k)` is a table macro that issues
`ORDER BY array_distance(embedding, query_vec) LIMIT k` against
`message_embeddings`, which triggers the HNSW cosine-index rewrite.
`embed_query` is a Python helper exposed via the CLI — in raw SQL, build the
vector outside DuckDB and pass it as a `FLOAT[1024]` parameter.

## 8. Explain pushdown

Confirm `read_json` only reads the columns and files it needs:

```bash
uv run claude-sql explain "SELECT uuid, role FROM messages WHERE session_id = '20fed4d7-6dd4-46bb-b1d0-8dbdeedaaded'"
```

Look for these markers in the plan (the CLI highlights them in green):

- `READ_JSON` at the leaf — zero-copy scan, no intermediate materialization.
- `Projections: type / sessionId / message.role / uuid` — only 4 of 15 raw
  columns hit the scan. The projection pruner pushed the `SELECT uuid, role`
  plus the filter columns all the way down to `read_json`.
- `Total Files Read: 9505` — the glob `~/.claude/projects/*/*.jsonl`
  expanded to 9,505 files. On a warm cache the scan completes in ~32s
  end-to-end; cold, closer to a minute.

Session-pinned queries scan every file because `session_id` is a *field*
inside each JSONL, not a partition key. For a single-session query, add
`AND source_file LIKE '%<session_id>.jsonl'` to let the filename filter
prune down to one file.

## Tradeoffs

- **int8 embeddings stored as FLOAT[]**: Cohere Embed v4 returns native
  `int8` vectors (4× storage savings over float), but DuckDB VSS only
  supports `FLOAT` element type. `register_vss` casts on insert:
  `CAST(embedding AS FLOAT[1024])`. The int8 savings are lost at rest; we
  accept the 4× bloat for a single-user corpus because the parquet still
  fits in memory comfortably.
- **Lazy JSON typing for `message.content`**: Rather than eagerly shredding
  `content` into a `STRUCT` (which would lock the view to the block types
  known at ingest), `messages.content_json` stays as `JSON`. The
  `content_blocks` view flattens it via `UNNEST(json_extract(content_json,
  '$[*]'))` and `switch on block_type`. This is resilient to new block
  types (`tool_use`, `tool_result`, `thinking`, and whatever ships next)
  without schema migrations. Per-block scans pay a small `json_extract`
  cost; for hot queries, materialize a typed projection of
  `block_type='text'` only.
