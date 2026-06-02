# claude-sql · Dependency graph

```mermaid
flowchart LR
    classDef external stroke-dasharray: 3 3

    app[app]
    core[core]
    analytics[analytics]
    evals[evals]
    provenance[provenance]

    loguru[(loguru)]:::external
    polars[(polars)]:::external
    duckdb[(duckdb)]:::external
    bedrock[(AWS Bedrock)]:::external
    pydantic[(pydantic)]:::external
    lancedb[(lancedb)]:::external
    pyarrow[(pyarrow)]:::external
    tenacity[(tenacity)]:::external
    anyio[(anyio)]:::external
    numpy[(numpy)]:::external
    cyclopts[(cyclopts)]:::external
    hdbscan[(hdbscan)]:::external
    umap[(umap-learn)]:::external
    leidenalg[(leidenalg)]:::external
    tiktoken[(tiktoken)]:::external

    app --> core
    app --> analytics
    app --> evals
    app --> provenance
    analytics --> core
    evals --> core
    provenance --> core

    app --> cyclopts

    core --> loguru
    core --> polars
    core --> duckdb
    core --> bedrock
    core --> pydantic
    core --> lancedb
    core --> pyarrow
    core --> tenacity

    analytics --> anyio
    analytics --> numpy
    analytics --> tiktoken
    analytics --> hdbscan
    analytics --> umap
    analytics --> leidenalg
```

## Legend (overflow)

Elided external nodes (declared in `packages/*/pyproject.toml` but dropped to fit the 20-node budget), with the count of source files that import each:

| Node | Owning package | Importing files | Note |
|---|---|---|---|
| scikit-learn | analytics | 1 | `terms_worker.py:65` c-TF-IDF CountVectorizer |
| igraph | analytics | 1 (2 sites) | `community_worker.py:70` mutual-kNN graph |
| pydantic-settings | core | 1 | `config.py` settings root |
| scipy | analytics | declared only | `packages/analytics/pyproject.toml:17` |
| pyyaml | core | 1 | `config.py` YAML knobs |
| packaging | app | 1 | version parsing |
| anthropic | core | declared only | `packages/core/pyproject.toml:10`, no source import |

## See also

- [claude-sql · System overview](../../architecture/system-overview.md) — 2 shared source files
