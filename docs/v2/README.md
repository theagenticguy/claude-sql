# claude-sql v2 docs

The v2 direction for claude-sql: a hexagonal rewrite that drops the eval and
provenance planes, makes embeddings pluggable (Cohere-on-Bedrock, Ollama, local
ONNX BGE), and keeps the retrieval + clustering plane on Python 3.13.

The shipping package is still v1.2.1. These documents describe where it is
going and how to get there.

## Read in this order

1. **[DESIGN.md](DESIGN.md)**: the target architecture: the four decisions, the
   hexagonal package tree, the ports, the embedding seam, and the biggest risk.
2. **[MIGRATION.md](MIGRATION.md)**: the ordered cut plan (drop → pluggable
   embeddings → hexagonal reshape) and the definition of done.

## Grounding

Both documents synthesize `understanding/`: seven perspective reads of the
v1.2.1 codebase, each grounding every claim in `path:line` via CodeGraph and
`lint-imports`:

| Doc | Perspective |
|---|---|
| [01-architecture.md](understanding/01-architecture.md) | Layering, the import DAG, hexagonal gap analysis, the ports |
| [02-retrieval-clustering.md](understanding/02-retrieval-clustering.md) | The kept plane: SQL views, semantic search, clustering, Leiden communities |
| [03-embedding-seam.md](understanding/03-embedding-seam.md) | The two-function seam, the `EmbeddingProvider` port, the dimension hazard |
| [04-cli-interfaces.md](understanding/04-cli-interfaces.md) | 31 commands: keep 21 / change 3 / drop 7 |
| [05-drop-analysis.md](understanding/05-drop-analysis.md) | Call-graph proof that evals + provenance drop cleanly |
| [06-core-engine.md](understanding/06-core-engine.md) | The DuckDB zero-copy engine, config, schemas, LLM plumbing |
| [07-tests-ci-build-docs.md](understanding/07-tests-ci-build-docs.md) | 826 tests, the 5-gate `mise run check`, the docs worklist |

## Terminology note

The roadmap language used "pagerank" for the graph-intelligence plane. The code
implements no PageRank: graph ranking is Leiden community detection with the CPM
objective, ranked by community medoid and coherence. v2 docs use the correct
names.
