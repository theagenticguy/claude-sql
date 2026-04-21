"""Runtime configuration for claude-sql.

Pydantic v2 ``BaseSettings`` populated from env vars prefixed with ``CLAUDE_SQL_``.
Defaults are picked for a single-user devbox install pointing at
``~/.claude/projects/**/*.jsonl``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_glob() -> str:
    # Top-level session transcripts only.  Subagent side-files live one level
    # deeper under ``<session>/subagents/`` and are discovered via SUBAGENT_GLOB.
    return os.path.expanduser("~/.claude/projects/*/*.jsonl")


def _default_subagent_glob() -> str:
    return os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.jsonl")


def _default_subagent_meta_glob() -> str:
    return os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.meta.json")


def _default_embeddings_parquet() -> Path:
    return Path(os.path.expanduser("~/.claude/embeddings.parquet"))


def _default_classifications_parquet() -> Path:
    return Path(os.path.expanduser("~/.claude/session_classifications.parquet"))


def _default_trajectory_parquet() -> Path:
    return Path(os.path.expanduser("~/.claude/message_trajectory.parquet"))


def _default_conflicts_parquet() -> Path:
    return Path(os.path.expanduser("~/.claude/session_conflicts.parquet"))


def _default_clusters_parquet() -> Path:
    return Path(os.path.expanduser("~/.claude/clusters.parquet"))


def _default_cluster_terms_parquet() -> Path:
    return Path(os.path.expanduser("~/.claude/cluster_terms.parquet"))


def _default_communities_parquet() -> Path:
    return Path(os.path.expanduser("~/.claude/session_communities.parquet"))


def _default_user_friction_parquet() -> Path:
    return Path(os.path.expanduser("~/.claude/user_friction.parquet"))


def _default_checkpoint_db() -> Path:
    return Path(os.path.expanduser("~/.claude/claude_sql.duckdb"))


# Model pricing per 1M tokens (in_rate, out_rate).  Mirrors claude-mine/transform.py.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
}


class Settings(BaseSettings):
    """Environment-driven settings for claude-sql.

    All fields are overridable via env vars prefixed ``CLAUDE_SQL_`` (e.g.
    ``CLAUDE_SQL_REGION=us-west-2``) or via ``.env`` in the working directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_SQL_",
        env_file=".env",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Data discovery
    # ------------------------------------------------------------------
    default_glob: str = Field(default_factory=_default_glob)
    subagent_glob: str = Field(default_factory=_default_subagent_glob)
    subagent_meta_glob: str = Field(default_factory=_default_subagent_meta_glob)

    # ------------------------------------------------------------------
    # Bedrock / embedding
    # ------------------------------------------------------------------
    region: str = "us-east-1"
    #: Cohere Embed v4 global CRIS profile. Sustained 223 vec/s with zero
    #: throttling at concurrency=8 in testing; US-only and direct on-demand
    #: both throttle hard at low TPM. No reason to expose the knob.
    model_id: str = "global.cohere.embed-v4:0"

    output_dimension: Literal[256, 512, 1024, 1536] = 1024
    embedding_type: Literal["int8", "float", "uint8", "binary", "ubinary"] = "int8"
    #: Parallel Bedrock calls. Tuned for global CRIS TPM ceiling with the
    #: aggregated-text messages_text view (avg ~470 chars/msg).
    concurrency: int = 2
    batch_size: int = 96

    embeddings_parquet_path: Path = Field(default_factory=_default_embeddings_parquet)

    # ------------------------------------------------------------------
    # VSS / HNSW
    # ------------------------------------------------------------------
    hnsw_metric: Literal["cosine", "l2sq", "ip"] = "cosine"
    hnsw_ef_construction: int = 128
    hnsw_ef_search: int = 64
    hnsw_m: int = 16
    hnsw_m0: int = 32

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    pricing: dict[str, tuple[float, float]] = Field(default_factory=lambda: dict(DEFAULT_PRICING))

    # ------------------------------------------------------------------
    # v2: LLM classification (Bedrock Sonnet 4.6 + output_config.format)
    # ------------------------------------------------------------------
    #: Sonnet 4.6 global CRIS inference profile — CRIS-only, 1M context native,
    #: no beta header. Supports `output_config.format` GA structured output.
    sonnet_model_id: str = "global.anthropic.claude-sonnet-4-6"
    #: (input, output) $/MTok for Sonnet 4.6 on Bedrock us-east-1.
    sonnet_pricing: tuple[float, float] = (3.0, 15.0)
    #: Thinking mode for classification.  ``"adaptive"`` keeps reasoning on;
    #: ``"disabled"`` is the escape hatch if Bedrock 400s on thinking +
    #: output_config (undocumented incompatibility).
    classify_thinking: Literal["adaptive", "disabled"] = "adaptive"
    #: Max output tokens for a single classification call.
    classify_max_tokens: int = 2048
    #: Per-text clip used when assembling session_text — tool_results can be
    #: arbitrarily large (Bash output, file reads).
    session_text_tool_result_max_chars: int = 50_000
    #: Total session_text cap (conservative 800K chars ≈ 200K tokens, leaves
    #: room for the response under the 1M window).
    session_text_total_max_chars: int = 800_000

    # v2 parquet outputs
    classifications_parquet_path: Path = Field(default_factory=_default_classifications_parquet)
    trajectory_parquet_path: Path = Field(default_factory=_default_trajectory_parquet)
    conflicts_parquet_path: Path = Field(default_factory=_default_conflicts_parquet)
    clusters_parquet_path: Path = Field(default_factory=_default_clusters_parquet)
    cluster_terms_parquet_path: Path = Field(default_factory=_default_cluster_terms_parquet)
    communities_parquet_path: Path = Field(default_factory=_default_communities_parquet)
    #: Output of the user-friction classifier (see ``friction_worker.py``).
    #: One row per user message flagged as status_ping, unmet_expectation,
    #: confusion, interruption, correction, frustration, or (sentinel) none.
    #: Backs the ``user_friction`` view and the ``friction_counts`` /
    #: ``friction_rate`` analytics macros.
    user_friction_parquet_path: Path = Field(default_factory=_default_user_friction_parquet)
    #: Short-message cutoff for the friction classifier candidate filter.
    #: Friction signals cluster in short messages ("screenshot?", "wait",
    #: "why?"); long messages are almost always on-topic turns.  300 chars
    #: captures ~95% of the interesting class without bloating Bedrock cost.
    friction_max_chars: int = 300
    #: Per-(session_id, pipeline) checkpoint DuckDB file. See ``checkpointer.py``.
    checkpoint_db_path: Path = Field(default_factory=_default_checkpoint_db)

    # ------------------------------------------------------------------
    # v2: UMAP + HDBSCAN + Louvain hyperparameters
    # ------------------------------------------------------------------
    umap_n_components_50: int = 50
    umap_n_components_2: int = 2
    umap_n_neighbors: int = 30
    umap_min_dist_cluster: float = 0.0
    umap_min_dist_viz: float = 0.1
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 5
    #: Absolute cosine floor below which a pair is never considered related,
    #: regardless of the adaptive search.  Kept conservative so the graph
    #: doesn't collapse into a single giant component on very similar
    #: corpora.
    louvain_edge_threshold: float = 0.55
    #: Target band for the average graph degree.  ``_pick_adaptive_threshold``
    #: picks the cosine cut that puts average degree in ``[low, high]``.
    #: 8-15 is the empirically-tested sweet spot for Louvain on session-
    #: centroid graphs (1K-20K nodes): enough to let community structure
    #: emerge, not enough to produce a hairball.
    louvain_target_avg_degree_low: float = 8.0
    louvain_target_avg_degree_high: float = 15.0
    #: Louvain communities smaller than this get collapsed into the
    #: NOISE_COMMUNITY_ID bucket (-1) so reports stay legible.
    louvain_min_community_size: int = 3
    louvain_resolution: float = 1.0
    seed: int = 42

    # ------------------------------------------------------------------
    # v2: TF-IDF
    # ------------------------------------------------------------------
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_top_n_terms: int = 10

    @property
    def active_model_id(self) -> str:
        """Return the Bedrock embedding model ID (kept as a property for call-site stability)."""
        return self.model_id
