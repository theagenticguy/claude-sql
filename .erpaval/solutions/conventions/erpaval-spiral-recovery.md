---
title: When you're caught in a spiral, ground in installed source — not training data, not docs
track: knowledge
category: conventions
module: .erpaval/
component: ERPAVal methodology
severity: info
tags: [erpaval, methodology, plan-mode, grounding, anti-pattern]
applies_when:
  - You've made multiple edits to defend against an API shape you haven't verified
  - Tests are passing but something feels structurally wrong
  - You're writing compatibility shims for behavior you assumed exists
  - The user says some variant of "you got caught in a spiral"
pattern: |
  Recovery sequence when caught in a spiral over an unverified API:

  1. **Stop editing.** `EnterPlanMode` if available; otherwise just
     stop writing code.

  2. **Read the installed source.** Not the docs, not training data,
     not Context7's mirror — the actual file in
     `.venv/lib/python3.{N}/site-packages/<package>/<module>.py`. The
     docstring + signature + body is the truth.

  3. **Live-probe the behavior.** `uv run python -c "..."` against
     the real installed package. Verify what `repr()` shows, what
     attributes exist, what types come back. Two minutes of probing
     beats two hours of guessing.

  4. **Write down what's actually true** in the plan file. Cite
     line numbers from the source. Make claims auditable.

  5. **Only then** plan edits. The plan should reference the verified
     facts, not guess at them.

  Cost of pausing to verify: ~10 minutes. Cost of NOT pausing:
  hours of edits-on-edits-on-edits, defensive shims accreting around
  shapes that don't exist, tests that pass-by-accident.
example_files:
  - .erpaval/solutions/api-patterns/lancedb-list-tables-pagination-shape.md
  - .erpaval/solutions/api-patterns/duckdb-attach-lance-empty-namespace.md
---

# Why this matters

The lance integration session began with the user noticing slow
queries and asked for performance work. After the initial fixes
landed, the user came back with: *"you got caught in a spiral on
getting lancedb up and running. need you to take a step back, deep
dive into lancedb reference APIs from context7, and get it right.
/plan"*

The intervention was correct and necessary. The code I'd shipped
included:

- A `_table_names()` shim that pattern-matched on a paginated tuple
  shape that didn't exist (relied on pydantic's `__iter__` accident)
- A `register_vss` filesystem-based empty-dataset gate that was
  semantically wrong (caught in
  `duckdb-attach-lance-empty-namespace.md`)
- Defensive `pl.Array` casts that were dead branches (the input was
  already correctly shaped)
- A `tbl.optimize()` debug log of a value that's always `None`

Each was an edit made in isolation, defending against a hypothetical
that I'd never actually verified. Tests still passed because (a) my
test seeded the table, so `_has_table`-equivalent checks always
returned True; (b) my fixture directories were either empty or
contained valid Lance, never the empty-namespace edge case.

The recovery — `EnterPlanMode` → read `.venv/.../lancedb/db.py` →
live-probe `db.list_tables()` → write down what's actually true →
plan deletions, not additions — produced a smaller, sharper PR with
two new tests pinning the API gotchas.

# Counter-pattern: what good grounding looks like

```bash
# 1. Find the source
find .venv/lib/python3.13/site-packages/lancedb -name "db.py"

# 2. Read it
Read /efs/lalsaado/workplace/claude-sql/.venv/lib/python3.13/site-packages/lancedb/db.py:167-220

# 3. Probe
uv run --quiet python -c "
import lancedb
from datetime import timedelta
db = lancedb.connect('/tmp/d', read_consistency_interval=timedelta(0))
print('list_tables:', repr(db.list_tables()))
print('type:', type(db.list_tables()))
print('attrs:', dir(db.list_tables()))
"

# 4. Write the verified table into the plan:
| Question | Answer | Source |
|---|---|---|
| `db.list_tables()` shape | `ListTablesResponse(tables=[...], page_token=None)` | `lancedb/db.py:167-193` + live probe |
```

# When to apply

The trigger isn't "the user pushed back" — it's "I'm writing code
that defends against an API I haven't verified." Common signals:

- Compatibility shim for "older versions" — verify what the *installed*
  version does first
- `getattr(obj, "method", None)` defense for methods that may exist —
  if you have to defend against them, you don't know if they exist
- `try/except (AttributeError, TypeError)` around a single call — same
- Wrapper functions called `_normalize_*` or `_safe_*` — usually
  symptomatic of unverified shape assumptions

# Pinned by

Process discipline, not a test. The lessons in
`api-patterns/lancedb-list-tables-pagination-shape.md` and
`api-patterns/duckdb-attach-lance-empty-namespace.md` are the
artifacts of one such recovery; this lesson captures the meta-pattern
so future-me reaches for it earlier.
