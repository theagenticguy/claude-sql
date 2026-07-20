"""Structure-plane pure math: clustering, c-TF-IDF terms, and community detection.

domain/structure is I/O-free but may import heavy compute deps (numpy, igraph,
sklearn, umap, hdbscan). This is the deliberate exception to the stdlib+pydantic
domain rule — recorded at Gate 1 of session-4fb0fd.

Concretely: these modules take the frozen ``domain.config`` value-objects
(``ClusteringConfig`` / ``CommunityConfig`` / ``TermsConfig``) plus in-memory
numpy matrices and return in-memory results. They never open a DuckDB
connection, read/write parquet, touch LanceDB, or call Bedrock — that
orchestration lives in ``application.use_cases``. The heavy compute deps are
lazy-imported inside the functions that use them so the CLI fast path never
pays the igraph/leidenalg/umap/hdbscan/sklearn import cost.
"""

from __future__ import annotations
