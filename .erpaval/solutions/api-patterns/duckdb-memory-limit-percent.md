---
title: DuckDB rejects "%" in memory_limit — translate at apply time
track: knowledge
category: api-patterns
module: src/claude_sql/cli.py
component: DuckDB connection PRAGMAs
severity: info
tags: [duckdb, pragma, memory_limit, settings]
applies_when:
  - You're configuring `SET memory_limit` from user-supplied or convention strings
  - You want to support `'70%'`-style relative sizing in addition to absolute units
pattern: |
  DuckDB's `memory_limit` parser only knows `KB / MB / GB / TB` and the
  `KiB / MiB / GiB / TiB` binary variants. A literal `'70%'` raises
  ``Parser Error: Unknown unit for memory: '%'``. To support percentage
  strings without surprising users (a sensible default on a single-user
  devbox), resolve `<n>%` against host total memory before issuing the
  PRAGMA. Use `os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")`
  on POSIX, with a 4 GiB conservative fallback for non-POSIX hosts.
example_files:
  - src/claude_sql/cli.py
---

# Why this matters

If users pass `'70%'` (or your default is one), every connection open
crashes with a Parser Error before any work can happen. The
percentage-resolution helper is small (regex + sysconf math + format
string) but easy to forget when adopting DuckDB tuning PRAGMAs from
"PRAGMA cheatsheet" docs that imply percentages just work.

# Example

```python
_PERCENT_LIMIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")

def _resolve_memory_limit(limit: str) -> str:
    match = _PERCENT_LIMIT_RE.match(limit)
    if match is None:
        return limit.strip()
    fraction = float(match.group(1)) / 100.0
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        total_bytes = 4 * 1024**3
    else:
        total_bytes = page_size * phys_pages
    target_mib = max(1, int((total_bytes * fraction) // (1024 * 1024)))
    return f"{target_mib}MiB"

# Apply:
con.execute(f"SET memory_limit = '{_resolve_memory_limit('70%')}'")
```
