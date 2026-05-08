---
title: list_avg is per-row, not aggregate — use unnest+pos+groupby for centroid math
track: knowledge
category: api-patterns
module: src/claude_sql/community_worker.py
component: DuckDB SQL aggregations over array columns
severity: info
tags: [duckdb, sql, aggregation, embeddings, centroid]
applies_when:
  - You're computing per-group element-wise mean over a fixed-length array column
  - You see a plan or PRD prescribing `list_avg` for centroid-style aggregation
pattern: |
  `list_avg(arr)` returns a scalar — the mean of one row's array. It is
  **not** an aggregate over rows. To compute per-group element-wise
  means (centroids), unnest with positions, group by (group_key, pos),
  AVG the values, then list-aggregate back ordered by pos.

  ```sql
  WITH unrolled AS (
      SELECT group_key,
             generate_subscripts(emb, 1) AS pos,
             unnest(emb) AS v
        FROM joined
  ),
  agg AS (
      SELECT group_key, pos, avg(v) AS m
        FROM unrolled
       GROUP BY 1, 2
  )
  SELECT group_key, list(m ORDER BY pos) AS centroid
    FROM agg
   GROUP BY 1
   ORDER BY 1
  ```

  Cast `Array(Float32, dim)` to `FLOAT[]` first if you're reading from a
  fixed-size-array parquet (DuckDB sees those as `FLOAT[<dim>]` and the
  unnest path expects a non-fixed list).
example_files:
  - src/claude_sql/community_worker.py
---

# Why this matters

A plan or LLM-suggested SQL snippet that uses
`list_avg(list_value(embedding))` looks plausible and may even compile —
but it produces a scalar per row, not a per-group centroid. Silent
shape mismatch downstream when the result is downcast to a numpy
matrix. The unnest+groupby+list pattern is verbose but correct,
typesafe, and ~10× faster than the equivalent Python loop on 50K rows.

# Example

```python
sql = f"""
    WITH joined AS (
        SELECT CAST(m.session_id AS VARCHAR) AS session_id,
               e.embedding::FLOAT[] AS emb
          FROM read_parquet([{path_literals}]) e
          JOIN messages m
            ON CAST(m.uuid AS VARCHAR) = e.uuid
    ),
    unrolled AS (
        SELECT session_id,
               generate_subscripts(emb, 1) AS pos,
               unnest(emb) AS v
          FROM joined
    ),
    agg AS (
        SELECT session_id, pos, avg(v) AS m
          FROM unrolled
         GROUP BY 1, 2
    )
    SELECT session_id, list(m ORDER BY pos) AS centroid
      FROM agg
     GROUP BY 1
     ORDER BY 1
"""
df = con.execute(sql).pl()
centroids = np.stack([np.asarray(c, dtype=np.float32) for c in df["centroid"].to_list()])
```
