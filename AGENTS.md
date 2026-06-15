## OpenCodeHub MCP Tools

This repository has been indexed by OpenCodeHub. When you are working in this
codebase, prefer the following MCP tools over raw file search — they return
graph-aware results grouped by execution flow and include blast-radius risk
tiers.

- `list_repos` — enumerate repos currently indexed on this machine.
- `query` — hybrid BM25 + vector search over symbols, grouped by process.
- `context` — inbound/outbound refs and participating flows for one symbol.
- `impact` — dependents of a target up to a configurable depth, with a risk tier.
- `detect_changes` — map an uncommitted or committed diff to affected symbols.
- `list_findings` — browse SARIF findings from the latest scan by severity and rule.
- `sql` — read-only SQL against the local temporal store (cochanges + symbol_summaries), 5 s timeout; the node/edge graph is queried via the typed tools or Cypher via the MCP `sql` tool.

Run `codehub analyze` after pulling new commits so the index stays aligned
with the working tree. `codehub status` reports staleness.
