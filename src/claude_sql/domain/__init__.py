"""Domain layer (L-domain) for the v2 hexagonal reshape.

Pure, dependency-free business types: the error hierarchy, and (as later waves
land) the schemas, community/cluster/terms math, simhash/hamming/token-budget,
and friction-regex rules. Nothing here imports duckdb, polars, lancedb, boto3,
or any adapter — the domain is the innermost hexagon and depends on nothing.

This package is additive in MIGRATION Phase C step 1 (T-1-1). The import-linter
contract does not yet name these packages; that lands in T-5.
"""

from __future__ import annotations
