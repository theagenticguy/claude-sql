"""Runtime configuration for claude-sql.

Pydantic v2 ``BaseSettings`` populated from env vars prefixed with ``CLAUDE_SQL_``.
Defaults are picked for a single-user devbox install pointing at
``~/.claude/projects/**/*.jsonl``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from claude_sql.core.home import claude_sql_home


def _default_glob() -> str:
    # Top-level session transcripts only.  Subagent side-files live one level
    # deeper under ``<session>/subagents/`` and are discovered via SUBAGENT_GLOB.
    return os.path.expanduser("~/.claude/projects/*/*.jsonl")


def _default_subagent_glob() -> str:
    return os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.jsonl")


def _default_subagent_meta_glob() -> str:
    return os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.meta.json")


def _default_embeddings_parquet() -> Path:
    # Legacy parquet shard directory. Kept here for one-time migration only —
    # the live embeddings store is now LanceDB (see ``_default_lance_uri``).
    # The field name keeps the ``_parquet_path`` suffix so existing call sites
    # that hand the path to migrators / cache-list helpers stay stable.
    return claude_sql_home() / "embeddings"


def _default_lance_uri() -> Path:
    """LanceDB local dataset directory backing the embeddings store.

    Replaces the parquet-shards + ``hnsw.duckdb`` combo. Lance handles
    storage, versioning, and the IVF_HNSW_SQ index in one place.
    """
    return claude_sql_home() / "embeddings_lance"


def _default_classifications_parquet() -> Path:
    return claude_sql_home() / "session_classifications"


def _default_trajectory_parquet() -> Path:
    return claude_sql_home() / "message_trajectory"


def _default_conflicts_parquet() -> Path:
    return claude_sql_home() / "session_conflicts"


def _default_clusters_parquet() -> Path:
    return claude_sql_home() / "clusters.parquet"


def _default_cluster_terms_parquet() -> Path:
    return claude_sql_home() / "cluster_terms.parquet"


def _default_communities_parquet() -> Path:
    return claude_sql_home() / "session_communities.parquet"


def _default_community_profile_parquet() -> Path:
    return claude_sql_home() / "community_profile.parquet"


def _default_user_friction_parquet() -> Path:
    return claude_sql_home() / "user_friction"


def _default_skills_catalog_parquet() -> Path:
    return claude_sql_home() / "skills_catalog.parquet"


def _default_user_skills_dir() -> Path:
    # Source-of-truth lives under ``~/.claude/skills`` (Claude Code owns it);
    # claude-sql only reads from it.
    return Path(os.path.expanduser("~/.claude/skills"))


def _default_plugins_cache_dir() -> Path:
    # Same story — Claude Code maintains ``~/.claude/plugins/cache``; we read.
    return Path(os.path.expanduser("~/.claude/plugins/cache"))


def _default_checkpoint_db() -> Path:
    # SQLite WAL state file. The legacy ``claude_sql.duckdb`` path is migrated
    # once on first open by ``checkpointer._migrate_from_duckdb_if_present``.
    return claude_sql_home() / "state.db"


def _default_duckdb_temp_dir() -> Path:
    return claude_sql_home() / "duckdb_tmp"


def _default_ingest_stamps_parquet() -> Path:
    return claude_sql_home() / "ingest_stamps"


def _default_duckdb_threads() -> int:
    return os.cpu_count() or 4


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
    #: Team-corpus root.  When set, ``default_glob`` / ``subagent_glob`` /
    #: ``subagent_meta_glob`` are derived from ``<root>/<author>/projects/*``
    #: instead of ``~/.claude/projects/*``.  Replaces (does not union with)
    #: the personal corpus root; an explicit per-glob override always wins.
    team_corpus_root: Path | None = Field(
        default=None,
        description=(
            "If set, default_glob/subagent_glob/subagent_meta_glob derive from "
            "<root>/<author>/projects/* instead of ~/.claude/projects/*. "
            "Replaces (does not union with) the personal corpus root."
        ),
    )

    # ------------------------------------------------------------------
    # S3 transcript source
    # ------------------------------------------------------------------
    #: When any transcript glob is an ``s3://`` URI, claude-sql loads DuckDB's
    #: ``httpfs`` extension and creates a ``credential_chain`` S3 secret so the
    #: existing ``read_json`` view stack reads the remote corpus zero-copy.
    #: Point ``default_glob`` at the ``S3SessionStore`` layout, e.g.
    #: ``CLAUDE_SQL_DEFAULT_GLOB='s3://bucket/prefix/*/*/part-*.jsonl'``.
    #: Credentials come from the standard AWS chain (env / shared config /
    #: instance-role) — never embedded in SQL. The three fields below override
    #: the S3 endpoint for non-AWS stores or a local mock server; leave
    #: ``s3_endpoint`` unset (the default) for real AWS S3.
    s3_endpoint: str | None = Field(
        default=None,
        description=(
            "Custom S3 endpoint host[:port] for non-AWS stores or a local mock "
            "(e.g. MinIO, moto). Unset uses the default AWS S3 endpoint."
        ),
    )
    s3_url_style: Literal["vhost", "path"] = "vhost"
    s3_use_ssl: bool = True

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
    #: Parallel Bedrock calls for Cohere Embed v4 on global CRIS. Sustained
    #: 8 × batch_size 96 in testing without throttling — Cohere's TPM bucket
    #: is the binding constraint and embed v4 is generous on global CRIS.
    embed_concurrency: int = 8
    #: Parallel Bedrock calls for Sonnet 4.6 on global CRIS. 16 is the
    #: sweet spot once system prompts cross the cache threshold — cache
    #: reads don't deduct from the per-model TPM bucket, so 16 parallel
    #: cached calls sustain well below the throttle ceiling. Observed
    #: ~5 calls/sec at concurrency=8 on trajectory's full backfill;
    #: concurrency=16 scales that linearly with negligible throttle.
    #: Drop to 2–4 if a future model has a smaller TPM bucket.
    llm_concurrency: int = 16
    batch_size: int = 96

    embeddings_parquet_path: Path = Field(default_factory=_default_embeddings_parquet)

    # ------------------------------------------------------------------
    # LanceDB embeddings store
    # ------------------------------------------------------------------
    #: Local LanceDB dataset URI. Replaces ``embeddings_parquet_path`` (legacy
    #: kept for one-time migration only) and ``hnsw_db_path`` (removed).
    #: DuckDB reads it back via ``INSTALL lance; LOAD lance; ATTACH (TYPE LANCE)``.
    lance_uri: Path = Field(default_factory=_default_lance_uri)
    #: Distance metric for the IVF_HNSW_SQ index. Cosine matches what the
    #: ``semantic_search`` macro expects.
    hnsw_metric: Literal["cosine", "l2", "dot"] = "cosine"

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
    #: Default thinking mode used by the session-level ``classify`` and
    #: ``conflicts`` pipelines.  ``"adaptive"`` lets Sonnet reason before
    #: emitting structured output; ``"disabled"`` is the escape hatch when
    #: Bedrock 400s on thinking + output_config (rare, undocumented).
    classify_thinking: Literal["adaptive", "disabled"] = "adaptive"
    #: Per-message trajectory classifier thinking mode. Disabled by
    #: default — trajectory is a 3-class enum + 1 boolean; reasoning burns
    #: 5–20× output tokens for no measurable quality gain on this shape.
    trajectory_thinking: Literal["adaptive", "disabled"] = "disabled"
    #: Friction classifier thinking mode. Disabled by default for the same
    #: reason as trajectory: short-message classification doesn't benefit
    #: from reasoning.  Bumps to ``adaptive`` only if quality regresses
    #: in real eval data.
    friction_thinking: Literal["adaptive", "disabled"] = "disabled"
    #: Max output tokens for a single classification call — covers BOTH the
    #: adaptive-thinking budget AND the structured-output answer. The old
    #: 2048 ceiling truncated large `conflicts` sessions: adaptive thinking
    #: consumed the whole budget, the model emitted only a `thinking` block,
    #: `stop_reason` came back `max_tokens`, and the parser raised on the
    #: answer-less payload (see the conflicts retry-queue silent-drop bug,
    #: 2026-06-15). 16000 leaves ample room for thinking + a full conflict
    #: array on the largest sessions.
    classify_max_tokens: int = 16000
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
    #: Resolution-profile sidecar written when ``community`` runs auto-γ.
    #: Holds one row per γ tested by ``leidenalg.Optimiser.resolution_profile``
    #: with columns ``(gamma, n_communities, quality, plateau_length)``. Lets
    #: an agent preview "what γ would give 50 communities" without rerunning
    #: Leiden. Conditional: explicit ``--gamma`` runs do not write it.
    community_profile_parquet_path: Path = Field(default_factory=_default_community_profile_parquet)
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

    #: Catalog of locally-available Skills and slash commands, produced by
    #: ``claude-sql skills sync`` (see :mod:`claude_sql.skills_catalog`).
    #: Backs the ``skills_catalog`` view, the ``skill_usage`` enrichment join,
    #: and the ``unused_skills`` macro.  Walked from :attr:`user_skills_dir`
    #: and :attr:`plugins_cache_dir`.
    skills_catalog_parquet_path: Path = Field(default_factory=_default_skills_catalog_parquet)
    #: Root of user-level skills (each entry has a ``SKILL.md``).
    user_skills_dir: Path = Field(default_factory=_default_user_skills_dir)
    #: Root of the plugins cache maintained by Claude Code.  The walker
    #: expects ``<owner>/<plugin>/<version>/`` underneath, each with a
    #: ``.claude-plugin/plugin.json`` and ``skills/`` / ``commands/`` subdirs.
    plugins_cache_dir: Path = Field(default_factory=_default_plugins_cache_dir)

    #: Per-(session_id, pipeline) checkpoint DuckDB file. See ``checkpointer.py``.
    checkpoint_db_path: Path = Field(default_factory=_default_checkpoint_db)

    #: Sharded parquet cache for ingest stamps (``approx_tokens`` / ``simhash64``
    #: / future per-message metadata). Written by the ingest worker scheduled
    #: in v1.x; reserved here so :func:`claude_sql_home` plumbing covers it
    #: from day one and the field name doesn't churn when the worker lands.
    ingest_stamps_parquet_path: Path = Field(default_factory=_default_ingest_stamps_parquet)

    # ------------------------------------------------------------------
    # v2: UMAP + HDBSCAN + Leiden hyperparameters
    # ------------------------------------------------------------------
    umap_n_components_50: int = 50
    umap_n_components_2: int = 2
    umap_n_neighbors: int = 30
    umap_min_dist_cluster: float = 0.0
    umap_min_dist_viz: float = 0.1
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 5
    #: Mutual-kNN k for the session-centroid graph.  k=15 is the Scanpy /
    #: BERTopic default for embedding-similarity graphs and lands a graph
    #: density that gives Leiden+CPM clean communities without hairballs.
    leiden_knn_k: int = 15
    #: Absolute cosine floor; edges below this are dropped before Leiden.
    #: Lower than the old louvain_edge_threshold (0.55) because mutual-kNN
    #: already constrains degree by construction.
    leiden_edge_floor: float = 0.3
    #: Communities smaller than this collapse to NOISE_COMMUNITY_ID (-1).
    leiden_min_community_size: int = 3
    #: CPM resolution parameter γ.  ``None`` triggers auto-γ via
    #: ``leidenalg.Optimiser.resolution_profile`` + longest-plateau picker.
    #: For cosine weights in [edge_floor, 1.0], γ has direct density
    #: semantics: communities have internal density ≥ γ, external ≤ γ.
    leiden_resolution: float | None = None
    #: Bisection search range for ``Optimiser.resolution_profile``.  Stored
    #: as two scalars rather than a tuple because pydantic-settings env-var
    #: support for tuple fields is awkward.
    leiden_resolution_range_lo: float = 0.05
    leiden_resolution_range_hi: float = 0.95
    #: Iterations for ``leidenalg.find_partition``.  ``-1`` means iterate
    #: until no quality improvement; ``2`` is the leidenalg default.
    leiden_n_iterations: int = -1
    seed: int = 42

    # ------------------------------------------------------------------
    # v2: TF-IDF  # noqa: ERA001 — section header, not commented-out code
    # ------------------------------------------------------------------
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_top_n_terms: int = 10

    # ------------------------------------------------------------------
    # DuckDB engine tuning — applied as PRAGMAs in cli._open_connection_full
    # and cli._open_connection_introspect.
    # ------------------------------------------------------------------
    #: Worker threads. Defaults to ``os.cpu_count()`` so DuckDB uses every
    #: core; agents and CI runners with limited parallelism can override.
    duckdb_threads: int = Field(default_factory=_default_duckdb_threads)
    #: Memory ceiling. ``"70%"`` is permissive for a single-user devbox;
    #: drop on shared hosts via the env var if it pressures other workloads.
    duckdb_memory_limit: str = "70%"
    #: Spill directory. Amazon devboxes ship ``/tmp`` as a 4 GB tmpfs that
    #: thrashes the host once a clustering run starts spilling — point at
    #: ``~/.claude/duckdb_tmp`` (real disk) instead.
    duckdb_temp_dir: Path = Field(default_factory=_default_duckdb_temp_dir)

    @model_validator(mode="after")
    def _derive_team_corpus_globs(self) -> Self:
        """Rewrite the three transcript globs when ``team_corpus_root`` is set.

        Pattern: ``<root>/<author>/projects/<project>/<sid>.jsonl`` (and the
        matching ``subagents/`` siblings).  Replaces — does not union with —
        the personal corpus root, per memo §Coherent Actions #3.

        Per-glob user pins always win: if any of ``default_glob`` /
        ``subagent_glob`` / ``subagent_meta_glob`` differ from their factory
        defaults at validation time, none of them are rewritten (we can't
        cherry-pick a partial rewrite without smuggling intent).
        """
        root = self.team_corpus_root
        if root is None:
            return self
        # Detect "user pinned a glob" by comparing to the factory-provided
        # default rather than literal string equality, so refactors of
        # ``_default_glob()`` and friends don't silently break this path.
        user_pinned = (
            self.default_glob != _default_glob()
            or self.subagent_glob != _default_subagent_glob()
            or self.subagent_meta_glob != _default_subagent_meta_glob()
        )
        if user_pinned:
            return self
        resolved = root.expanduser().resolve()
        object.__setattr__(self, "default_glob", f"{resolved}/*/projects/*/*.jsonl")
        object.__setattr__(
            self,
            "subagent_glob",
            f"{resolved}/*/projects/*/subagents/agent-*.jsonl",
        )
        object.__setattr__(
            self,
            "subagent_meta_glob",
            f"{resolved}/*/projects/*/subagents/agent-*.meta.json",
        )
        return self

    @property
    def active_model_id(self) -> str:
        """Return the Bedrock embedding model ID (kept as a property for call-site stability)."""
        return self.model_id
