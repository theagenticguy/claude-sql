# claude-sql · Dead code

Analysis method: `vulture 2.16` over `src/` (56 findings at 60% confidence,
none at 100%) cross-checked against `codegraph callers`, whole-tree grep, and
`ruff check --select F401,F811,F841` (passed clean — no unused imports). Every
vulture candidate was triaged against framework dispatch: pydantic v2 model
fields and `@model_validator`, cyclopts `@app.command` registrations,
`@runtime_checkable` port Protocols and their concrete impls, PEP 562 module
`__getattr__`, and the `composition.py` public facade. One genuine finding
survives.

## Unreferenced exports

| Symbol | Path | Last modified |
| --- | --- | --- |
| `ANALYTICS_VIEW_NAMES` | `src/claude_sql/infrastructure/duckdb_views.py:306` | 2026-07-20 |

`ANALYTICS_VIEW_NAMES` is an exported tuple whose docstring
(`src/claude_sql/infrastructure/duckdb_views.py:302`) promises callers
("`claude-sql` subcommands, smoke tests") that never materialized. It has zero
references across `src/` and `tests/`, no `__all__` re-export (the module
declares none), and is not enumerated by
`src/claude_sql/infrastructure/duckdb_connection.py`, which uses the sibling
`VIEW_NAMES` instead (`src/claude_sql/infrastructure/duckdb_connection.py:42`).
The provenance-only schema leftovers flagged by the pre-reshape pass
(`PRReviewSheet`, `Correction`, `PR_REVIEW_SHEET_SCHEMA`) were removed by the
hexagonal reshape — `core/schemas.py` no longer exists and
`src/claude_sql/domain/models.py` carries no such symbols.

## Unreferenced files

_none_

Every module under `src/claude_sql/` has at least one inbound reference. The
optional-extra adapters — the highest-risk orphan candidates post-reshape — are
wired via lazy factory dispatch, not static import: `OllamaEmbedder`
(`src/claude_sql/infrastructure/embedding/ollama.py`) and `OnnxBgeEmbedder`
(`src/claude_sql/infrastructure/embedding/onnx_bge.py`) are imported inside the
`build_embedder` branches at
`src/claude_sql/infrastructure/embedding/__init__.py:50` and `:54`;
`StrandsLunaAnalytics`
(`src/claude_sql/infrastructure/llm_analytics/strands_luna.py`) is imported at
`src/claude_sql/infrastructure/llm_analytics/__init__.py:63`.

## Dead imports

_none_

`ruff check src/ tests/ --select F401,F811,F841` reports all checks passed — no
import is bound without being referenced in the importing file anywhere in the
tree.

## See also

- [claude-sql · Impact analysis](../insights/impact-analysis.md) — 3 shared source citations
- [claude-sql · Data flow](../architecture/data-flow.md) — 2 shared source citations
- [claude-sql · Processes](../behavior/processes.md) — 2 shared source citations
- [claude-sql · Sequences](../diagrams/behavioral/sequences.md) — 2 shared source citations
- [claude-sql · Contract map](../insights/contract-map.md) — 2 shared source citations
