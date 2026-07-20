"""Property tests for the ``Settings`` → frozen value-object projections.

``core.config.Settings`` projects four pure-math slices
(``clustering_config`` / ``community_config`` / ``terms_config`` /
``transcript_caps``) into frozen dataclasses in ``domain.config``. The
projections are plain attribute copies — ``Settings`` carries no field-level
numeric constraints — so for any valid field assignment the projected object
must round-trip the exact values and stay immutable.
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from claude_sql.domain.config import (
    ClusteringConfig,
    CommunityConfig,
    TermsConfig,
    TranscriptCaps,
)
from claude_sql.infrastructure.settings import Settings
from property.strategies import SETTINGS_NUMERIC_FIELDS

# Build a Settings from arbitrary valid values for the numeric knobs. Every
# other field keeps its default (env-independent for the projected slices).
# ``function_scoped_fixture`` is not in play here; suppress the too-slow health
# check since constructing a pydantic-settings object per example is not free.
_settings_strategy = st.builds(Settings, **SETTINGS_NUMERIC_FIELDS)


@given(s=_settings_strategy)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_clustering_config_round_trips(s: Settings) -> None:
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


@given(s=_settings_strategy)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_community_config_round_trips(s: Settings) -> None:
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


@given(s=_settings_strategy)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_terms_config_round_trips(s: Settings) -> None:
    cfg = s.terms_config()
    assert isinstance(cfg, TermsConfig)
    assert cfg.tfidf_min_df == s.tfidf_min_df
    assert cfg.tfidf_max_df == s.tfidf_max_df
    assert cfg.tfidf_ngram_min == s.tfidf_ngram_min
    assert cfg.tfidf_ngram_max == s.tfidf_ngram_max
    assert cfg.tfidf_top_n_terms == s.tfidf_top_n_terms


@given(s=_settings_strategy)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_transcript_caps_round_trips(s: Settings) -> None:
    caps = s.transcript_caps()
    assert isinstance(caps, TranscriptCaps)
    assert caps.session_text_total_max_chars == s.session_text_total_max_chars
    assert caps.session_text_tool_result_max_chars == s.session_text_tool_result_max_chars


@given(s=_settings_strategy)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_projected_objects_are_frozen(s: Settings) -> None:
    """Every projected value-object rejects attribute assignment (immutability)."""
    for cfg in (
        s.clustering_config(),
        s.community_config(),
        s.terms_config(),
        s.transcript_caps(),
    ):
        field_name = next(iter(dataclasses.fields(cfg))).name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(cfg, field_name, object())


@given(s=_settings_strategy)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_projections_do_not_mutate_settings(s: Settings) -> None:
    """Deriving the four config objects leaves every source field untouched."""
    before = s.model_dump()
    s.clustering_config()
    s.community_config()
    s.terms_config()
    s.transcript_caps()
    assert s.model_dump() == before
