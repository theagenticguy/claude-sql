# claude-sql jobs-to-be-done

Multi-step workflows that combine several commands to answer a shape
of question users ask. Each job names the intent in plain English and
maps it to the command sequence.

---

## Job 1: "What did I work on this month?"

**Intent:** Narrative summary of the last N days.

**Sequence:**

1. `claude-sql query "SELECT count(*) FROM sessions WHERE started_at >= current_timestamp - INTERVAL 30 DAY"` — sanity-check the window.
2. Ensure classifications exist: `claude-sql classify --since-days 30 --no-dry-run` (first time only).
3. `claude-sql query "SELECT work_category, count(*) FROM session_classifications sc JOIN sessions s USING(session_id) WHERE s.started_at >= current_timestamp - INTERVAL 30 DAY GROUP BY 1"`.
4. For each category, pull the goal field to build the narrative:
   `SELECT goal FROM session_classifications WHERE work_category = 'coding' ORDER BY session_id`.

**Output:** A brief per-category summary grounded in real classifications.

---

## Job 2: "Where is my Claude money actually going?"

**Intent:** Spend audit with categorical breakdown.

**Sequence:**

1. Total spend by model for the window (recipe #5).
2. Top 20 most expensive sessions (recipe #4, extended).
3. If classifications exist, cross with work_category to show "what was
   I doing when I spent the most":

```sql
SELECT sc.work_category,
       round(sum(cost_estimate(s.session_id)), 2) AS usd
FROM sessions s
JOIN session_classifications sc USING (session_id)
WHERE s.started_at >= current_timestamp - INTERVAL 30 DAY
GROUP BY 1
ORDER BY usd DESC;
```

**Output:** A cost table the user can act on — which *kind* of work
costs the most.

---

## Job 3: "Find every conversation I've had about X"

**Intent:** Semantic recall across the corpus.

**Sequence:**

1. Confirm embeddings are current: `claude-sql query "SELECT max(embedded_at) FROM message_embeddings"`.
2. If stale: `AWS_PROFILE=... claude-sql embed --since-days 30 --no-dry-run`.
3. `claude-sql search "<concept>" --k 25`.
4. Join results back to sessions for the timeline:

```sql
SELECT s.session_id, s.started_at, m.text_content
FROM semantic_search(
       <query_vec>, 25
     ) AS hit
JOIN messages_text m USING (uuid)
JOIN sessions s USING (session_id)
ORDER BY s.started_at DESC;
```

**Output:** Every mention of the concept, ranked by similarity, grouped
by session.

---

## Job 4: "Group my sessions by topic and tell me what's trending"

**Intent:** Thematic view of the whole corpus.

**Sequence:**

1. `claude-sql embed --since-days 90 --no-dry-run` (if needed).
2. `claude-sql cluster` — UMAP + HDBSCAN + c-TF-IDF.
3. `claude-sql community` — Louvain over session centroids.
4. Report top communities with dominant clusters and trend:

```sql
SELECT community_id,
       count(DISTINCT session_id) AS size,
       min(s.started_at) AS first_seen,
       max(s.started_at) AS last_seen
FROM session_communities sc
JOIN sessions s USING (session_id)
GROUP BY community_id
HAVING count(*) >= 3
ORDER BY last_seen DESC;
```

Pair each community with `community_top_topics(community_id, 5)` to
label it.

**Output:** A ranked list of themes with human-readable labels.

---

## Job 5: "Find where I disagreed with myself"

**Intent:** Surface stance conflicts for review.

**Sequence:**

1. `AWS_PROFILE=... claude-sql conflicts --since-days 60 --no-dry-run`.
2. List them, bucketed by resolution:

```sql
SELECT resolution, count(*)
FROM session_conflicts
GROUP BY resolution;
```

3. Pull the unresolved ones with a snippet of stance_a and stance_b so
   the user can decide what to revisit.

**Output:** A short list of decisions the user took two positions on,
plus whether the conflict got resolved.

---

## Job 6: "Which of my todos fell on the floor?"

**Intent:** Find TodoWrite entries that were never closed.

**Sequence:** Recipe #10 (`SELECT ... FROM todo_state_current WHERE
status IN ('pending','in_progress') AND written_at < ... - 14 DAY`).

Then for each stale todo, pull the session it came from so the user
can reopen the context:

```sql
SELECT t.subject, t.status, s.session_id, s.started_at
FROM todo_state_current t
JOIN sessions s USING (session_id)
WHERE t.status IN ('pending', 'in_progress')
  AND t.written_at < current_timestamp - INTERVAL 14 DAY
ORDER BY s.started_at DESC;
```

**Output:** A backlog the user can triage in-place.

---

## Job 7: "Full health check — run everything"

**Intent:** The user hasn't analyzed the corpus in a while and wants a
fresh picture.

**Sequence:**

1. Dry-run first to show cost:
   `claude-sql analyze --since-days 30`.
2. On approval:
   `AWS_PROFILE=... claude-sql analyze --since-days 30 --no-dry-run`.
3. Once done, produce a one-page summary drawing on:
   - `work_mix(30)` for category breakdown
   - `autonomy_trend(30)` for how autonomous the work got
   - `success_rate_by_work(30)` for quality signal
   - top 5 communities with labels
   - any new unresolved conflicts

**Output:** A dashboard-style summary the user can scan in under a
minute.

---

## Cross-cutting habits

- **Dry-run every spender first.** The cost estimate takes seconds and
  the user should see it before the bill hits.
- **Cache awareness.** If a parquet already exists and the window
  hasn't moved, don't re-run — the view is already live.
- **Grounding loop.** When in doubt, run `claude-sql schema` and
  `claude-sql query "DESCRIBE <view>"` instead of guessing columns.
