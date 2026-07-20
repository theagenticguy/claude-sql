# claude-sql · Doc tree

Generated codebase docs for `claude-sql` on the `feat/v2-hexagonal` branch.

Prose is generated; structure is mechanical. Cross-references are deterministic. Built against commit `7670b4c`: the completed v2 hexagonal tree — `domain` / `application` / `infrastructure` / `interfaces` layers plus the `composition.py` ClaudeSql facade and the Lean 4 `proofs/` track. The transitional `core` package is dissolved; the import-linter enforces one layers contract (`interfaces > application > infrastructure > domain`). The v2 specs at [docs/v2/DESIGN.md](v2/DESIGN.md) and [docs/v2/MIGRATION.md](v2/MIGRATION.md) are now historical (all phases DONE).

## Architecture

- [System overview](architecture/system-overview.md)
- [Module map](architecture/module-map.md)
- [Data flow](architecture/data-flow.md)

## Reference

- [Public API](reference/public-api.md)
- [CLI](reference/cli.md)

## Behavior

- [Processes](behavior/processes.md)
- [State machines](behavior/state-machines.md)

## Analysis

- [Risk hotspots](analysis/risk-hotspots.md)
- [Ownership](analysis/ownership.md)
- [Dead code](analysis/dead-code.md)

## Diagrams

- [Components](diagrams/architecture/components.md)
- [Dependency graph](diagrams/structural/dependency-graph.md)
- [Sequences](diagrams/behavioral/sequences.md)

## Insights

- [Impact analysis](insights/impact-analysis.md)
- [Debugging guide](insights/debugging-guide.md)
- [Contract map](insights/contract-map.md)
- [Business logic](insights/business-logic.md)
- [Tech debt](insights/tech-debt.md)
