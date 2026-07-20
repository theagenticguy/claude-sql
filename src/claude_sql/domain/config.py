"""Per-adapter / per-pipeline config value-objects (pure, frozen dataclasses).

MIGRATION Phase C step 5 (T-2-4): the v1 god-``Settings`` bundles the
transcript globs, Bedrock model IDs, DuckDB tuning knobs, and the pure-math
hyperparameters into one pydantic object. The clustering / community / terms
math has no business seeing a Bedrock model ID, and the session-text assembler
has no business seeing a UMAP neighbor count.

This module carves the pure-math slices out into small frozen dataclasses that
carry *only* the numbers a given consumer needs. They are stdlib-only (no
pydantic, no duckdb, no adapters) so the domain hexagon stays dependency-free.
``Settings`` keeps every field verbatim — the ``CLAUDE_SQL_*`` env contract is
unchanged — and grows derivation methods (``clustering_config()`` etc.) that
project a ``Settings`` down into these value-objects. Consumers thread the
value-object instead of reaching back into ``Settings`` for individual knobs.

Frozen so a config object handed to a worker can't be mutated mid-run, which
keeps determinism (``seed``) honest and makes the objects hashable for any
future memoization.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClusteringConfig:
    """UMAP + HDBSCAN hyperparameters for ``cluster_worker.run_clustering``.

    Mirrors the ``umap_*`` / ``hdbscan_*`` fields plus ``seed`` on ``Settings``.
    ``seed`` threads into both UMAP ``random_state`` calls so same-seed reruns
    produce byte-identical cluster IDs.
    """

    umap_n_components_50: int
    umap_n_components_2: int
    umap_n_neighbors: int
    umap_min_dist_cluster: float
    umap_min_dist_viz: float
    umap_metric: str
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    seed: int


@dataclass(frozen=True, slots=True)
class CommunityConfig:
    """Leiden+CPM + mutual-kNN hyperparameters for ``community_worker``.

    Mirrors the ``leiden_*`` fields plus ``seed`` on ``Settings``. ``seed``
    keys both ``leidenalg.find_partition(seed=...)`` and the resolution-profile
    bisection RNG (``Optimiser.set_rng_seed``). ``leiden_resolution`` is
    ``None`` when auto-γ should run.
    """

    leiden_knn_k: int
    leiden_edge_floor: float
    leiden_min_community_size: int
    leiden_resolution: float | None
    leiden_resolution_range_lo: float
    leiden_resolution_range_hi: float
    leiden_n_iterations: int
    seed: int


@dataclass(frozen=True, slots=True)
class TermsConfig:
    """c-TF-IDF hyperparameters for ``terms_worker.run_terms``.

    Mirrors the ``tfidf_*`` fields on ``Settings``: ``CountVectorizer``
    ``min_df`` / ``max_df`` / ngram bounds plus the per-cluster top-N cutoff.
    """

    tfidf_min_df: int
    tfidf_max_df: float
    tfidf_ngram_min: int
    tfidf_ngram_max: int
    tfidf_top_n_terms: int


@dataclass(frozen=True, slots=True)
class TranscriptCaps:
    """Character caps for session-text assembly.

    Mirrors ``session_text_total_max_chars`` /
    ``session_text_tool_result_max_chars`` on ``Settings``. The per-tool-result
    clip bounds arbitrarily large Bash / file-read outputs; the total cap keeps
    an assembled session under the model context window.
    """

    session_text_total_max_chars: int
    session_text_tool_result_max_chars: int


__all__ = [
    "ClusteringConfig",
    "CommunityConfig",
    "TermsConfig",
    "TranscriptCaps",
]
