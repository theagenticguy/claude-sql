# claude-sql · Dependency graph

Internal nodes are the four hexagonal layers plus the `composition` facade, ordered by the single `import-linter` layers contract at `pyproject.toml:295-303` (`interfaces > application > infrastructure > domain`). External nodes are the direct runtime dependencies from `pyproject.toml:28-50`, ranked by how many source files import them and anchored to the layer that imports them most. Deps that fell outside the 20-node budget are in the Legend below.

```mermaid
flowchart LR
    ui[interfaces/cli]
    app[application]
    infra[infrastructure]
    domain[domain]
    comp[composition]

    ui --> app
    ui --> infra
    ui --> domain
    app --> infra
    app --> domain
    infra --> domain
    comp --> app
    comp --> infra

    cyclopts[(cyclopts)]:::external
    polars[(polars)]:::external
    duckdb[(duckdb)]:::external
    anyio[(anyio)]:::external
    loguru[(loguru)]:::external
    pyarrow[(pyarrow)]:::external
    pydantic[(pydantic)]:::external
    tenacity[(tenacity)]:::external
    boto3[(boto3)]:::external
    lancedb[(lancedb)]:::external
    numpy[(numpy)]:::external
    hdbscan[(hdbscan)]:::external
    umap[(umap-learn)]:::external
    sklearn[(scikit-learn)]:::external
    leidenalg[(leidenalg)]:::external

    ui --> cyclopts
    app --> polars
    app --> duckdb
    app --> anyio
    infra --> loguru
    infra --> pyarrow
    infra --> pydantic
    infra --> tenacity
    infra --> boto3
    infra --> lancedb
    domain --> numpy
    domain --> hdbscan
    domain --> umap
    domain --> sklearn
    domain --> leidenalg

    classDef external stroke-dasharray: 3 3
```

## Legend (overflow)

Direct dependencies (`pyproject.toml:28-50`) elided from the diagram to hold the 20-node budget. Edge count = number of source files importing the dep.

| Dep | Anchor layer | Edges | Source |
| --- | --- | --- | --- |
| igraph | domain | 1 | `src/claude_sql/domain/structure/community.py` |
| returns | application | 1 | `src/claude_sql/application/ports.py` |
| tiktoken | domain | 1 | `src/claude_sql/domain/dedup.py` |
| pydantic-settings | infrastructure | 1 | `src/claude_sql/infrastructure/settings.py` |
| pyyaml | infrastructure | 1 | `src/claude_sql/infrastructure/skills_fs.py` |
| packaging | domain | 1 | `src/claude_sql/domain/skills.py` |
