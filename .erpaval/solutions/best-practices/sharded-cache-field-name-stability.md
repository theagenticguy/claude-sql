---
title: Reinterpret path semantics instead of renaming Settings fields when caches shard
track: knowledge
category: best-practices
module: src/claude_sql/parquet_shards.py
component: Settings + worker output paths
severity: info
tags: [parquet, settings, migration, api-stability, sharding]
applies_when:
  - A plan calls for renaming a path field to reflect a single-file → directory shift
  - That field is referenced by tests, CLI flags, env vars, or other modules
  - You want the new behavior with minimal call-site churn
pattern: |
  When migrating a write pattern from "single parquet file" to
  "directory of part-*.parquet shards", the temptation is to rename the
  Settings field (e.g. `embeddings_parquet_path` →
  `embeddings_dir_path`). That's almost always wrong. The rename
  cascades into ~30 call sites across CLI flags, env vars, tests, and
  documentation, and forces a hard break for users with the old env var
  set.

  Better: keep the field name, change its **semantics** in a small
  helper module. The path is now interpreted as a directory by default,
  with a legacy single-file fallback when the path resolves to a file.
  The env var `CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH` keeps working for
  both old (file) and new (dir) values. A `cache migrate` subcommand
  handles the one-time data move.

  Rule of thumb: rename a field only when its **type** or **contract**
  changes in a way that a wrapper helper can't paper over. Path → path
  is not such a change.
example_files:
  - src/claude_sql/parquet_shards.py
  - src/claude_sql/config.py
---

# Why this matters

The original plan called for five field renames. Following it literally
would have changed ~30 call sites, broken users' env vars, forced a
test-fixture rewrite across 5 files, and bumped the major version.
Reinterpreting the path semantics inside `parquet_shards.is_sharded_dir
/ write_part / read_all / iter_part_files` kept the existing API stable
while delivering the same write-throughput win.

# Example

```python
# parquet_shards.py
def is_sharded_dir(path: Path) -> bool:
    return path.is_dir() or (not path.exists() and path.parent.exists())

def write_part(target: Path, df: pl.DataFrame) -> Path:
    if is_sharded_dir(target):
        target.mkdir(parents=True, exist_ok=True)
        out = target / f"part-{time.time_ns()}.parquet"
        df.write_parquet(out)
        return out
    # Legacy single-file path: read+concat+rewrite.
    if target.exists() and target.stat().st_size > 16:
        df = pl.concat([pl.read_parquet(target), df], how="diagonal_relaxed")
    df.write_parquet(target)
    return target

# embed_worker.py — call site stays simple:
write_part(settings.embeddings_parquet_path, df)
```
