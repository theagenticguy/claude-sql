"""Tests for :mod:`claude_sql.domain.config` and the ``Settings`` projections.

T-2-4 decomposes the god-``Settings`` into frozen per-consumer value-objects.
These tests pin three properties:

1. Each derivation method round-trips the exact ``Settings`` field values.
2. The value-objects are frozen (immutable — determinism / hashability).
3. The determinism ``seed`` threads into the clustering and community configs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from claude_sql.domain.config import (
    ClusteringConfig,
    CommunityConfig,
    TermsConfig,
    TranscriptCaps,
)
from claude_sql.infrastructure.settings import Settings


def test_clustering_config_round_trips_settings_fields() -> None:
    s = Settings()
    cfg = s.clustering_config()
    assert isinstance(cfg, ClusteringConfig)
    assert cfg.umap_n_components_50 == s.umap_n_components_50
    assert cfg.umap_n_components_2 == s.umap_n_components_2
    assert cfg.umap_n_neighbors == s.umap_n_neighbors
    assert cfg.umap_min_dist_cluster == s.umap_min_dist_cluster
    assert cfg.umap_min_dist_viz == s.umap_min_dist_viz
    assert cfg.umap_metric == s.umap_metric
    assert cfg.hdbscan_min_cluster_size == s.hdbscan_min_cluster_size
    assert cfg.hdbscan_min_samples == s.hdbscan_min_samples
    assert cfg.seed == s.seed


def test_community_config_round_trips_settings_fields() -> None:
    s = Settings()
    cfg = s.community_config()
    assert isinstance(cfg, CommunityConfig)
    assert cfg.leiden_knn_k == s.leiden_knn_k
    assert cfg.leiden_edge_floor == s.leiden_edge_floor
    assert cfg.leiden_min_community_size == s.leiden_min_community_size
    assert cfg.leiden_resolution == s.leiden_resolution
    assert cfg.leiden_resolution_range_lo == s.leiden_resolution_range_lo
    assert cfg.leiden_resolution_range_hi == s.leiden_resolution_range_hi
    assert cfg.leiden_n_iterations == s.leiden_n_iterations
    assert cfg.seed == s.seed


def test_community_config_carries_explicit_leiden_resolution() -> None:
    """A set ``leiden_resolution`` (not the ``None`` default) round-trips."""
    s = Settings().model_copy(update={"leiden_resolution": 0.4})
    cfg = s.community_config()
    assert cfg.leiden_resolution == pytest.approx(0.4)


def test_terms_config_round_trips_settings_fields() -> None:
    s = Settings()
    cfg = s.terms_config()
    assert isinstance(cfg, TermsConfig)
    assert cfg.tfidf_min_df == s.tfidf_min_df
    assert cfg.tfidf_max_df == s.tfidf_max_df
    assert cfg.tfidf_ngram_min == s.tfidf_ngram_min
    assert cfg.tfidf_ngram_max == s.tfidf_ngram_max
    assert cfg.tfidf_top_n_terms == s.tfidf_top_n_terms


def test_transcript_caps_round_trips_settings_fields() -> None:
    s = Settings()
    caps = s.transcript_caps()
    assert isinstance(caps, TranscriptCaps)
    assert caps.session_text_total_max_chars == s.session_text_total_max_chars
    assert caps.session_text_tool_result_max_chars == s.session_text_tool_result_max_chars


@pytest.mark.parametrize(
    "factory",
    [
        lambda s: s.clustering_config(),
        lambda s: s.community_config(),
        lambda s: s.terms_config(),
        lambda s: s.transcript_caps(),
    ],
)
def test_config_objects_are_frozen(factory: Callable[[Settings], Any]) -> None:
    """Frozen dataclasses reject attribute assignment (immutability guard)."""
    s = Settings()
    cfg = factory(s)
    field_name = next(iter(dataclasses.fields(cfg))).name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cfg, field_name, object())


def test_seed_threads_into_clustering_and_community_configs() -> None:
    """A non-default seed threads through both determinism-sensitive configs."""
    s = Settings().model_copy(update={"seed": 1234})
    assert s.clustering_config().seed == 1234
    assert s.community_config().seed == 1234


def test_projections_do_not_mutate_settings() -> None:
    """Deriving a config object leaves every source field untouched."""
    s = Settings()
    before = s.model_dump()
    s.clustering_config()
    s.community_config()
    s.terms_config()
    s.transcript_caps()
    assert s.model_dump() == before
