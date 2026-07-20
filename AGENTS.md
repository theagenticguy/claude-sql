## CodeGraph: code intelligence

This repository is indexed by CodeGraph (a `.codegraph/` directory exists at
the repo root). When working in this codebase, reach for CodeGraph BEFORE
grep/find or reading whole files to understand or locate code. It returns the
relevant symbols' verbatim source plus the call paths between them, including
dynamic-dispatch hops that text search cannot follow.

- **MCP tool:** `codegraph_explore` answers most code questions in one call.
  Name a file or symbol in the query to read its current line-numbered source.
  `codegraph_node` returns one symbol's source plus its caller/callee trail.
- **Shell (always works):**
  - `codegraph explore "<symbols or question>"`: same output as the MCP tool.
  - `codegraph query "<term>"`: search for symbols.
  - `codegraph node <name>`: one symbol's source plus its caller/callee trail.
  - `codegraph callers <symbol>` / `codegraph callees <symbol>`: call graph.
  - `codegraph impact <symbol>`: what a change to a symbol affects.
  - `codegraph affected <files...>`: test files affected by changed sources.
  - `codegraph files`: project file structure from the index.

Run `codegraph sync` after pulling new commits so the index stays aligned with
the working tree. `codegraph status` reports index staleness.
