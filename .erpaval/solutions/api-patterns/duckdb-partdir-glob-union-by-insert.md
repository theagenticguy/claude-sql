# DuckDB: unioning flat + part-dir transcript globs needs INSERT BY NAME, not glob lists

**Category:** api-patterns
**Tags:** duckdb, read_json, glob, transcripts, part-files
**Session:** session-4fb0fd (T-6-1 collapse-parity fixes)

## Lesson

To enumerate sessions across BOTH layouts —
flat `{root}/projects/{proj}/{sid}.jsonl` and part-dir
`{root}/projects/{proj}/{sid}/part-*.jsonl` (the SDK S3-mirror shape) — in one
DuckDB relation, neither of the obvious approaches works on DuckDB 1.5.4:

- **Brace-globs `{a,b}` are NOT supported** by `read_json`'s glob engine.
- **A LIST of globs** (`read_json(['glob1','glob2'])`) raises `IOException` if
  ANY member matches zero files — unusable when one layout may be absent.

**Working pattern:** register the primary glob normally, then supplement with

```sql
INSERT INTO v_raw_events BY NAME
SELECT ... FROM read_json('{root}/projects/*/*/part-*.jsonl', ...)
```

guarded by a Python-side `glob.glob()` existence check so the INSERT is skipped
when no part-dir sessions exist. `BY NAME` tolerates column-order differences
between the two read_json projections.

Also (same task): the `messages_text` view's `HAVING length(...) >= 32` floor
is an analytics-quality filter — any *reader* surface that must render short
turns ("screenshot?") needs its own no-floor projection over `content_blocks`
or raw `read_json`, never a relaxation of the shared view.
