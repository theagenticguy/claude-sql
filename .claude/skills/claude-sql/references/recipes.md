# claude-sql recipes

Ready-to-adapt queries mapped to the kinds of questions users actually
ask. Every recipe is copy-pasteable into `claude-sql query "..."` —
adjust windows and filters to taste.

## Recall: "remember what I worked on"

### 1. Sessions from last Tuesday

```sql
SELECT session_id, started_at, ended_at,
       (ended_at - started_at) AS duration
FROM sessions
WHERE started_at::DATE = DATE '2026-04-14'
ORDER BY started_at;
```

### 2. Every session that mentioned "temporal"

```sql
SELECT DISTINCT m.session_id, s.started_at
FROM messages_text m
JOIN sessions s USING (session_id)
WHERE m.text_content ILIKE '%temporal%'
ORDER BY s.started_at DESC;
```

For fuzzy/conceptual matches, prefer:

```bash
claude-sql search "temporal workflow determinism" --k 20
```

### 3. The longest session of the last week

```sql
SELECT session_id, started_at,
       (ended_at - started_at) AS duration
FROM sessions
WHERE started_at >= current_timestamp - INTERVAL 7 DAY
ORDER BY duration DESC
LIMIT 10;
```

## Spend: "where did my money go"

### 4. Opus sessions over \$5 this month

```sql
SELECT session_id,
       model_used(session_id) AS model,
       cost_estimate(session_id) AS usd,
       started_at
FROM sessions
WHERE started_at >= date_trunc('month', current_timestamp)
  AND model_used(session_id) LIKE '%opus%'
  AND cost_estimate(session_id) > 5.0
ORDER BY usd DESC;
```

### 5. Total spend by model, last 30 days

```sql
SELECT model_used(session_id) AS model,
       count(*) AS sessions,
       round(sum(cost_estimate(session_id)), 2) AS usd_total
FROM sessions
WHERE started_at >= current_timestamp - INTERVAL 30 DAY
GROUP BY 1
ORDER BY usd_total DESC;
```

### 6. Tool leaderboard

```sql
SELECT * FROM tool_rank(30);
```

## Patterns: "how do I actually work"

### 7. Autonomy trend over time (requires `classify`)

```sql
SELECT * FROM autonomy_trend(90);
```

Yields one row per week with the fraction of sessions at each tier
(1 = hand-holding, 3 = autonomous).

### 8. Work category mix (requires `classify`)

```sql
SELECT * FROM work_mix(30);
```

### 9. Success rate per category (requires `classify`)

```sql
SELECT * FROM success_rate_by_work(30);
```

### 10. Todos I open and never close

```sql
SELECT session_id, subject, status, written_at
FROM todo_state_current
WHERE status IN ('pending', 'in_progress')
  AND written_at < current_timestamp - INTERVAL 14 DAY
ORDER BY written_at;
```

## Themes: "group my sessions by topic"

### 11. Top themes in the last month (requires `cluster`)

```sql
SELECT c.cluster_id,
       count(DISTINCT mc.session_id) AS sessions,
       string_agg(DISTINCT ct.term, ', ' ORDER BY ct.rank)
         FILTER (WHERE ct.rank <= 5) AS top_terms
FROM message_clusters mc
JOIN cluster_terms ct USING (cluster_id)
JOIN messages m ON m.uuid = mc.message_uuid
JOIN sessions s ON s.session_id = m.session_id
WHERE s.started_at >= current_timestamp - INTERVAL 30 DAY
  AND NOT mc.is_noise
GROUP BY c.cluster_id
ORDER BY sessions DESC
LIMIT 10;
```

### 12. Session communities with their dominant clusters
(requires `cluster` + `community`)

```sql
SELECT community_id,
       count(DISTINCT session_id) AS size,
       (SELECT string_agg(term, ', ') FROM community_top_topics(community_id, 5)) AS topics
FROM session_communities
GROUP BY community_id
ORDER BY size DESC;
```

## Contradictions: "where did I disagree with myself"

### 13. Stance conflicts, with resolution status
(requires `conflicts`)

```sql
SELECT session_id, stance_a, stance_b, resolution
FROM session_conflicts
ORDER BY session_id;
```

Filter to `resolution = 'abandoned'` to find the unresolved ones worth
revisiting.

## Chaining SQL and semantic search

Find semantic hits, then join back to SQL for context. The
`semantic_search` macro returns a table:

```sql
SELECT s.session_id, s.started_at, m.text_content, hit.distance
FROM semantic_search(
       (SELECT embedding FROM message_embeddings WHERE uuid = '...'),
       20
     ) AS hit
JOIN messages_text m USING (uuid)
JOIN sessions s USING (session_id)
ORDER BY hit.distance;
```

For plain-text queries, `claude-sql search` handles the embed-then-query
round trip automatically.
